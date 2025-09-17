#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibra_global_p2m.py
---------------------------------
Versió 'Args class' (sense argparse) i amb informe detallat al terminal.

- Edita la secció 'Args' de sota amb les teves rutes/params.
- Suporta 'index_offset=None' -> prova 0 i +1 i tria el millor.
- Calcula regressió global (totes les boies) i per-node.
- Imprimeix metres/px global i extrems per node (min/max Z amb y_px i timestep).
- Gestiona Z amplificada: anno_zamp (anotacions), pred_zamp (prediccions).
"""

import json, re, os
from collections import defaultdict
import numpy as np

# ---------------------
# Configuració Arguments (edita aquí)
# ---------------------


class Args:
    pass

args = Args()
args.coco = r"C:\Users\marcp\Desktop\TOT\MODEL Estat de mar\train\_annotations.coco.json"
args.xyz  = r"C:\Users\marcp\Desktop\xyz_per_timestep.txt"
args.nodes = "5033,7297,10457,7929"            # ordre "real" (identitats dels nodes)
args.left_to_right = "5033,7297,10457,7929"    # ordre ESQUERRA→DRETA a la imatge
args.invert_y = True                           # com al teu pipeline: y_up = H - y
args.index_offset = None                       # None → provarà 0 i +1
args.index_regex = r"(?P<idx>\d+)"             # per extreure número (ex: 0118.png -> 118)
args.index_field = "file_name"                 # camp d'on extreure el nom
args.anno_zamp = 4.0                           # anotacions sobre renders Z×4
args.pred_zamp = 2.0                           # prediccions sobre renders Z×2
args.min_detections_per_image = 4
args.out = r"C:\Users\marcp\Desktop\calibration_global.json"


# ---------------------
# Utils
# ---------------------
def dot_get(d, path, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def parse_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    image_meta = {}
    for im in coco.get("images", []):
        image_meta[im["id"]] = {"W": im.get("width"), "H": im.get("height"), "raw": im}
    anns_per_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        x, y, w, h = ann["bbox"]
        cx = x + w/2.0
        cy = y + h/2.0
        anns_per_img[ann["image_id"]].append({"cx": cx, "cy": cy, "bbox": [x,y,w,h]})
    return image_meta, anns_per_img

def parse_xyz(path):
    ts_info = {}
    cur_ts = None
    cur_t = None
    re_ts = re.compile(r"TimeStep\s+(\d+)\s*\|\s*t\s*=\s*([0-9.+-Ee]+)\s*s")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = re_ts.search(line)
                if m:
                    cur_ts = int(m.group(1))
                    cur_t = float(m.group(2))
                    ts_info[cur_ts] = [cur_t, {}]
                continue
            if line.lower().startswith("node_id"):
                continue
            parts = line.split()
            if len(parts) >= 4 and cur_ts is not None:
                node_id = int(parts[0])
                z = float(parts[3])  # Z real (m)
                ts_info[cur_ts][1][node_id] = z
    return {k: (v[0], v[1]) for k, v in ts_info.items()}

def extract_index_from_image(im_raw, field, regex):
    src = dot_get(im_raw, field, None)
    if src is None:
        src = im_raw.get("file_name", "")
    m = re.search(regex, str(src))
    if not m or "idx" not in m.groupdict():
        return None
    try:
        return int(m.group("idx"))
    except Exception:
        return None

def assign_buoys_left_to_right(anns, node_ids_l2r):
    if len(anns) < 4:
        return None
    pick = sorted(anns, key=lambda a: a["cx"])[:4]
    return {node: ann for ann, node in zip(pick, node_ids_l2r)}

# ---------------------
# Core helpers
# ---------------------
def collect_pairs(image_meta, anns_per_img, ts_info, node_ids_l2r, invert_y, index_field, index_regex, index_offset, min_det):
    X_px, Y_m, pairs_dbg = [], [], []
    n_images_used = 0
    for img_id, meta in image_meta.items():
        anns = anns_per_img.get(img_id, [])
        if len(anns) < min_det:
            continue
        im_raw = meta["raw"]
        W, H = meta["W"], meta["H"]
        idx = extract_index_from_image(im_raw, index_field, index_regex)
        if idx is None:
            continue
        timestep = idx + index_offset
        if timestep not in ts_info:
            continue
        t_seconds, z_by_node = ts_info[timestep]
        mapping = assign_buoys_left_to_right(anns, node_ids_l2r)
        if mapping is None:
            continue

        used_any = False
        for node in node_ids_l2r:
            ann = mapping.get(node)
            if ann is None or node not in z_by_node:
                continue
            cy = ann["cy"]
            y_up_px = (H - cy) if invert_y else cy
            z_real = float(z_by_node[node])  # metres reals
            X_px.append(y_up_px)
            Y_m.append(z_real)
            pairs_dbg.append({
                "img_id": img_id, "timestep": timestep, "t": t_seconds,
                "node": node, "y_up_px": y_up_px, "z_real_m": z_real
            })
            used_any = True
        if used_any:
            n_images_used += 1
    return X_px, Y_m, pairs_dbg, n_images_used

def linear_fit(X, Y):
    X = np.array(X, float)
    Y = np.array(Y, float)
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    y_pred = a*X + b
    ss_res = float(np.sum((Y - y_pred)**2))
    ss_tot = float(np.sum((Y - np.mean(Y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2

# ---------------------
# Main runner
# ---------------------
def run(args):
    node_ids = [int(s) for s in args.nodes.split(",") if s.strip()]
    node_ids_l2r = [int(s) for s in args.left_to_right.split(",") if s.strip()]
    if len(node_ids) != 4 or len(node_ids_l2r) != 4:
        raise ValueError("Cal indicar exactament 4 node IDs")

    image_meta, anns_per_img = parse_coco(args.coco)
    ts_info = parse_xyz(args.xyz)

    # Decideix index_offset si és None (prova 0 i +1, tria el que dona més parelles)
    tested_offsets = [args.index_offset] if args.index_offset is not None else [0, 1]
    best = None
    for off in tested_offsets:
        X_px, Y_m, pairs_dbg, n_images_used = collect_pairs(
            image_meta, anns_per_img, ts_info, node_ids_l2r,
            args.invert_y, args.index_field, args.index_regex, off, args.min_detections_per_image
        )
        score = len(X_px)
        if best is None or score > best["score"]:
            best = dict(offset=off, X=X_px, Y=Y_m, dbg=pairs_dbg, nimg=n_images_used, score=score)

    if not best or len(best["X"]) < 4:
        raise RuntimeError("No hi ha prou parelles y_up_px↔z_real (comprova regex/index i mapping).")

    chosen_offset = best["offset"]
    X_px, Y_m, pairs_dbg, n_images_used = best["X"], best["Y"], best["dbg"], best["nimg"]

    # Fit global (referència, Z real)
    a_ref, b_ref, r2_ref = linear_fit(X_px, Y_m)

    # Fit per node (referència)
    by_node = defaultdict(list)
    for it in pairs_dbg:
        by_node[it["node"]].append(it)

    per_node = {}
    for node, items in by_node.items():
        x = [it["y_up_px"] for it in items]
        y = [it["z_real_m"] for it in items]
        a_n, b_n, r2_n = linear_fit(x, y)

        # extrems
        z_vals = y
        y_vals = x
        ts_vals = [it["timestep"] for it in items]
        z_min = min(z_vals); z_max = max(z_vals)
        i_min = z_vals.index(z_min); i_max = z_vals.index(z_max)
        per_node[node] = {
            "count": len(items),
            "a_m_per_px": float(a_n),
            "b_m": float(b_n),
            "r2": float(r2_n),
            "ext_min": {"z_m": float(z_min), "y_px": float(y_vals[i_min]), "timestep": int(ts_vals[i_min])},
            "ext_max": {"z_m": float(z_max), "y_px": float(y_vals[i_max]), "timestep": int(ts_vals[i_max])},
        }

    # Global metres/px com a mitjana de les pendents per-node
    a_ref_mean_nodes = float(np.mean([info["a_m_per_px"] for info in per_node.values()]))

    # Escalat per a prediccions (anno_zamp / pred_zamp)
    scale = float(args.anno_zamp) / float(args.pred_zamp) if args.pred_zamp else 1.0
    a_pred = a_ref * scale
    b_pred = b_ref * scale
    a_pred_mean_nodes = a_ref_mean_nodes * scale

    # Sortida JSON
    out = {
        "about": "Global pixel→meter calibration for buoys",
        "equation": "z_real_m = a * y_up_px + b",
        "n_pairs": len(X_px),
        "n_images_used": n_images_used,
        "config": {
            "nodes_order_world": node_ids,
            "nodes_order_left_to_right_pixels": node_ids_l2r,
            "invert_y": args.invert_y,
            "index_regex": args.index_regex,
            "index_field": args.index_field,
            "index_offset_chosen": chosen_offset,
            "anno_zamp": float(args.anno_zamp),
            "pred_zamp": float(args.pred_zamp),
            "scale_applied_for_predictions": float(scale),
            "coco": os.path.abspath(args.coco),
            "xyz": os.path.abspath(args.xyz),
        },
        "calibration_reference": {
            "global_fit": {"a_m_per_px": float(a_ref), "b_m": float(b_ref), "r2": float(r2_ref)},
            "per_node": per_node,
            "global_m_per_px_mean_nodes": float(a_ref_mean_nodes),
        },
        "calibration_for_predictions": {
            "global_fit": {"a_m_per_px": float(a_pred), "b_m": float(b_pred)},
            "global_m_per_px_mean_nodes": float(a_pred_mean_nodes),
            "derived_scale": float(scale)
        }
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # -------- Pretty print informe (terminal) --------
    print(f"  Nodes (world): {node_ids}")
    print(f"  Nodes L→R (pixels): {node_ids_l2r}")
    print(f"  invert_y: {args.invert_y} | anno_zamp: {args.anno_zamp} | pred_zamp: {args.pred_zamp} | scale(anno/pred): {scale}")
    print(f"✅ Matches imatge↔timestep: {n_images_used} amb index_offset={chosen_offset}")

    # Per-node (referència, Z real)
    for node in node_ids_l2r:
        info = per_node.get(node)
        if not info: 
            continue
        print(f"Node {node}: a={info['a_m_per_px']:.6f} m/px | r²={info['r2']:.3f}")

    print(f"\n📏 metres/px global (mitjana per-node, referència): {a_ref_mean_nodes:.6f}")
    print(f"📏 metres/px global per a PREDICCIONS (escala aplicada): {a_pred_mean_nodes:.6f}")

    print("\n— Extrems per node —")
    for node in node_ids_l2r:
        info = per_node.get(node)
        if not info: 
            continue
        mn = info["ext_min"]; mx = info["ext_max"]
        print(f"  Node {node}: min={mn['z_m']:.3f} m (y_px={mn['y_px']:.1f}, ts={mn['timestep']}) | "
              f"max={mx['z_m']:.3f} m (y_px={mx['y_px']:.1f}, ts={mx['timestep']})")

    # Resum 'clàssic' addicional:
    print("\n——— Resum fit global (referència) ———")
    print(f"a_ref (m/px) = {a_ref:.6g}")
    print(f"b_ref (m)    = {b_ref:.6g}")
    print(f"R² global    = {r2_ref:.5f}")
    print("\n——— Per a PREDICCIONS ———")
    print(f"a_pred (m/px) = {a_pred:.6g}")
    print(f"b_pred (m)    = {b_pred:.6g}")
    print(f"📏 metres/px final (prediccions, mitjana per-node) = {a_pred_mean_nodes:.6f}")
    print(f"\n✅ JSON guardat a: {os.path.abspath(args.out)}")

    return out

# ---------------------
# Execució directa
# ---------------------
if __name__ == "__main__":
    run(args)
