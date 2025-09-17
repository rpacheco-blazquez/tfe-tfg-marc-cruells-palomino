import os, json, warnings, time, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from inference import get_model

# ============ CONFIG ============
# --- Entrada ---
FRAMES_DIR = r"C:\Users\marcp\Desktop\TOT\26 malla 12x2\render"

# --- Sortides (prediccio_<n>_<sufix> amb comptador global) ---
OUTPUT_PARENT_DIR = r"C:\Users\marcp\Desktop\TOT\Prediccions"
OUTPUT_BASENAME = "prediccio"  # carpeta creada com prediccio_1_<sufix>, prediccio_2_<sufix>, ...

FPS = 10.0
API_KEY = "mYivOBO8ahi9EpHOmCml"
MODEL_ID = "estat-de-mar-1-2zsov/5"
CONF_THRES = 0.5
IOU_TRACK_THRES = 0.30          # relaxat
CLASSES_D_BOIA = None
REQUIRED_BOIES = 4
INVERT_Y = True

# --- Calibració px->m via JSON ---
CALIB_JSON = r"C:\Users\marcp\Desktop\calibration_global.json"
CALIB_MODE = "per_node"         # "per_node" (recomanat) o "global"
METERS_PER_PX = None            # Fallback només si NO hi ha JSON vàlid

# --- Escalat Z (opcional) ---
Z_ANNOTATION_AMP = 4.0
Z_PREDICTION_AMP = 2.0

# --- Mode d'operació ---
INTERACTIVE_ENFORCE_FOUR = True

# --- Sortides de la Passada 1 ---
WRITE_ANNOTATED_FRAMES = True            # guarda ann_XXXXX.png (caixes + etiquetes)
WRITE_VIDEO_FIRST_PASS = False           # vídeo provisional (millor deixar False)

# --- Vídeo final (Passada 2) ---
WRITE_VIDEO_FINAL = True
VIDEO_CODEC = "mp4v"
VIDEO_FPS = FPS
Y_MARGIN_FACTOR = 1.20                   # ±(Hmax/2 * marge) -> marge=1.20 és +20%

# --- Overlay gràfica (X/Y fixes, “com un riu”) ---
PLOT_ENABLED = True
PLOT_CENTERED = True         # treu la mitjana de cada boia
PLOT_W, PLOT_H = 1200, 320
PLOT_MARGIN = 6
PLOT_BG = (25, 25, 25)
PLOT_AXIS = (160, 160, 160)
COLOR_ID = {                   # BGR
    1: (0, 0, 255),     # vermell
    2: (0, 255, 0),     # verd
    3: (0, 255, 255),   # groc
    4: (255, 0, 255),   # rosa (magenta)
}
MEAN_COLOR = (255, 255, 255)
LEGEND_HEIGHT = 40
Y_DECIMALS_M = 2
# ================================


# ----------- utilitats sortida incremental -----------
def next_numbered_dir(parent: str, base: str) -> str:
    os.makedirs(parent, exist_ok=True)
    rx = re.compile(rf"^{re.escape(base)}_(\d+)$", re.IGNORECASE)
    max_n = 0
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if os.path.isdir(full):
            m = rx.match(name)
            if m:
                try:
                    max_n = max(max_n, int(m.group(1)))
                except ValueError:
                    pass
    n = max_n + 1
    out_dir = os.path.join(parent, f"{base}_{n}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# >>>>> AFEGIT: helpers per al sufix i la carpeta prediccio_<n>_<sufix>

def _suffix_from_frames_parent(frames_dir: str) -> str:
    """
    Sufix basat en la carpeta pare del FRAMES_DIR:
      - baixa a minúscules
      - treu el número inicial i espais (p. ex. '22 ' )
      - substitueix espais i guions per '_'
      - elimina caràcters que no siguin [a-z0-9_]
      - compacta múltiples '_' i fa strip('_')
    Exemples:
      '...\\22 malla 8 v2\\render' -> 'malla_8_v2'
      '...\\20 malla 8\\render 1'  -> 'malla_8'
      '...\\23 malla 10\\render'   -> 'malla_10'
    """
    frames_dir = (frames_dir or "").rstrip("\\/")
    parent = os.path.basename(os.path.dirname(frames_dir)) or ""
    s = parent.lower()
    s = re.sub(r'^\s*\d+\s*', '', s)      # treu número inicial i espais
    s = re.sub(r'[\s\-]+', '_', s)        # espais/guions -> _
    s = re.sub(r'[^a-z0-9_]+', '', s)     # només a-z0-9_
    s = re.sub(r'_+', '_', s).strip('_')  # compacta i neteja extrems
    return s or "dades"

def next_numbered_dir_with_suffix(parent: str, base: str, suffix: str):
    """
    Crea un directori amb format: base_<n>_<suffix>
    - El comptador <n> és GLOBAL dins 'parent' (mira base_<n>_* ja existents)
    - Si ja existeix el nom concret, incrementa n fins trobar-ne un de lliure
    Retorna (out_dir, n, dirname)
    """
    os.makedirs(parent, exist_ok=True)
    rx = re.compile(rf"^{re.escape(base)}_(\d+)(?:_.+)?$", re.IGNORECASE)
    max_n = 0
    for name in os.listdir(parent):
        full = os.path.join(parent, name)
        if os.path.isdir(full):
            m = rx.match(name)
            if m:
                try:
                    max_n = max(max_n, int(m.group(1)))
                except ValueError:
                    pass
    n = max_n + 1
    while True:
        dirname = f"{base}_{n}_{suffix}" if suffix else f"{base}_{n}"
        out_dir = os.path.join(parent, dirname)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            return out_dir, n, dirname
        n += 1
# <<<<< FI AFEGIT


# ----------- Silenciar avisos innecessaris -----------
os.environ["ORT_DISABLE_GPU"] = "1"
warnings.filterwarnings(
    "ignore",
    message=r"Specified provider .* not in available provider names.*",
    category=UserWarning,
    module=r"onnxruntime.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Your `inference` configuration does not support .*",
    category=Warning,
    module=r"inference.models.utils",
)
for k in [
    "PALIGEMMA_ENABLED","FLORENCE2_ENABLED","QWEN_2_5_ENABLED","CORE_MODEL_SAM_ENABLED",
    "CORE_MODEL_SAM2_ENABLED","CORE_MODEL_CLIP_ENABLED","CORE_MODEL_GAZE_ENABLED",
    "SMOLVLM2_ENABLED","DEPTH_ESTIMATION_ENABLED","MOONDREAM2_ENABLED",
    "CORE_MODEL_TROCR_ENABLED","CORE_MODEL_GROUNDINGDINO_ENABLED","CORE_MODEL_YOLO_WORLD_ENABLED",
    "CORE_MODEL_PE_ENABLED",
]:
    os.environ[k] = "False"


# ----------- Utils inferència/format ----------
def _val(p, k, default=None):
    if isinstance(p, dict): return p.get(k, default)
    return getattr(p, k, default)

def _normalize_predictions(out):
    if isinstance(out, list):
        out = out[0] if out else {}
    if hasattr(out, "model_dump") and callable(out.model_dump):
        try: out = out.model_dump()
        except Exception: pass
    elif hasattr(out, "dict") and callable(out.dict):
        try: out = out.dict()
        except Exception: pass
    if isinstance(out, dict) and "predictions" in out:
        return out["predictions"]
    preds = getattr(out, "predictions", None)
    if preds is not None: return preds
    return out

def _fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60); s = sec - 60*m
    return f"{m:02d}:{s:05.2f}"

def _show_fullscreen(win_name, img):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass
    cv2.imshow(win_name, img)


# ----------- Carrega model -----------
def load_model():
    print("Carregant model de Roboflow...")
    return get_model(model_id=MODEL_ID, api_key=API_KEY)


# ----------- Inferència d’un frame -----------
def process_frame(frame_bgr, model):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = model.infer(frame_rgb, confidence=CONF_THRES)
    preds = _normalize_predictions(out)
    detections = []
    for pred in preds or []:
        cls_name = _val(pred, "class_name", _val(pred, "class", None))
        detections.append({
            "x": float(_val(pred, "x", 0.0)),
            "y": float(_val(pred, "y", 0.0)),
            "width": float(_val(pred, "width", _val(pred, "w", 0.0))),
            "height": float(_val(pred, "height", _val(pred, "h", 0.0))),
            "class": cls_name if cls_name is not None else "boia",
            "confidence": float(_val(pred, "confidence", _val(pred, "score", 0.0))),
            "manual": False,
        })
    if CLASSES_D_BOIA is not None:
        detections = [d for d in detections if d["class"] in CLASSES_D_BOIA]
    return detections


def _draw_boxes(img, boxes, keep_mask=None, info_text=""):
    vis = img.copy()
    bar_h = 40
    bar = np.full((bar_h, vis.shape[1], 3), 30, dtype=np.uint8)
    msg = (info_text or "")
    cv2.putText(bar, msg, (10, int(bar_h*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(255,255,255), 1, cv2.LINE_AA)
    vis = np.vstack([bar, vis])
    yoff = bar_h
    for i, b in enumerate(boxes):
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        color = (0,255,0)
        label = f"{i+1}"
        if keep_mask is not None:
            if keep_mask[i]:
                color = (0, 200, 0); lab2 = "KEEP"
            else:
                color = (0, 0, 200); lab2 = "ELIM"
            label = f"{i+1}:{lab2}"
        cv2.rectangle(vis, (x1, y1+yoff), (x2, y2+yoff), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, max(0, y1+yoff - th - 6)), (x1 + tw + 6, y1+yoff), (0, 0, 0), -1)
        cv2.putText(vis, label, (x1 + 3, y1+yoff - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return vis, yoff


def _clamp_box(x, y, w, h, W, H):
    x1 = max(0, min(W-1, int(x - w/2)))
    y1 = max(0, min(H-1, int(y - h/2)))
    x2 = max(0, min(W-1, int(x + w/2)))
    y2 = max(0, min(H-1, int(y + h/2)))
    x2 = max(x2, x1+1); y2 = max(y2, y1+1)
    w2, h2 = x2 - x1, y2 - y1
    cx, cy = x1 + w2/2.0, y1 + h2/2.0
    return cx, cy, float(w2), float(h2)


# ========== Helpers overlay ==========
def draw_time_series_overlay(
    hist_dict,
    mean_hist,
    size=(1200,320),
    fixed_t_range=None,
    fixed_y_range=None,
    y_label="Alçada (m)",
    y_decimals=2,
    x_tick_step=5.0,
    y_tick_step=0.25,
    line_thickness=1,
    label_pad=10
):
    w, h_no_legend = size
    h = h_no_legend + LEGEND_HEIGHT
    canvas = np.full((h, w, 3), PLOT_BG, np.uint8)
    graph_area = canvas[:h_no_legend, :]
    cv2.rectangle(graph_area, (0,0), (w-1,h_no_legend-1), PLOT_AXIS, 1)

    # marges (espai per etiquetes)
    pad_left, pad_right, pad_top, pad_bottom = 48, 20, 24, 40
    x0, y0 = pad_left, pad_top
    x1, y1 = w - pad_right, h_no_legend - pad_bottom

    # eixos
    cv2.line(graph_area, (x0, y1), (x1, y1), PLOT_AXIS, 1)
    cv2.line(graph_area, (x0, y0), (x0, y1), PLOT_AXIS, 1)

    # Rangs
    if fixed_t_range is not None:
        tmin, tmax = fixed_t_range
    else:
        all_t = []
        for v in hist_dict.values(): all_t += v['t']
        all_t += mean_hist['t']
        tmin, tmax = (min(all_t), max(all_t)) if all_t else (0,1)
    xr = max(1e-6, tmax - tmin)

    if fixed_y_range is not None:
        ymin, ymax = fixed_y_range
    else:
        all_y = []
        for v in hist_dict.values(): all_y += v['y']
        all_y += mean_hist['y']
        ymin, ymax = (min(all_y), max(all_y)) if all_y else (-1.0, 1.0)
        if abs(ymax - ymin) < 1e-6: ymax = ymin + 1.0
    yr = max(1e-9, ymax - ymin)

    def to_polyline(t, y):
        """Desduplica per píxel X per evitar ratlles verticals."""
        if len(t) < 2: return None
        pts = []
        last_x = None
        for ti, yi in zip(t, y):
            if ti < tmin or ti > tmax: continue
            xx = x0 + (ti - tmin) / xr * (x1 - x0)
            yy = y1 - (yi - ymin) / yr * (y1 - y0)
            xi = int(xx + 0.5); yi_i = int(yy + 0.5)
            if last_x is None or xi > last_x:
                pts.append((xi, yi_i)); last_x = xi
            else:
                pts[-1] = (xi, yi_i)
        return np.array(pts, dtype=np.int32).reshape(-1, 1, 2) if len(pts) >= 2 else None

    # Sèries
    for bid, v in sorted(hist_dict.items(), key=lambda kv: kv[0]):
        pts = to_polyline(v['t'], v['y'])
        if pts is not None:
            cv2.polylines(graph_area, [pts], False, v['color'], line_thickness, cv2.LINE_AA)

    # Mitjana
    mpts = to_polyline(mean_hist['t'], mean_hist['y'])
    if mpts is not None:
        cv2.polylines(graph_area, [mpts], False, MEAN_COLOR, line_thickness, cv2.LINE_AA)

    # Ticks X (5 s)
    if x_tick_step and x_tick_step > 0:
        first = tmin - (tmin % x_tick_step) + x_tick_step
        tt = first
        while tt <= tmax + 1e-9:
            xx = int(round(x0 + (tt - tmin) / xr * (x1 - x0)))
            cv2.line(graph_area, (xx, y1), (xx, y1 + 4), PLOT_AXIS, 1)
            label = f"{tt:.0f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(graph_area, label, (xx - tw//2, y1 + th + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
            tt += x_tick_step

    # Ticks Y (0.25 m o personalitzat)
    if y_tick_step and y_tick_step > 0:
        def _round_to_step(v, step): return np.floor(v/step)*step
        yy_val = _round_to_step(ymin, y_tick_step)
        while yy_val <= ymax + 1e-12:
            yy = int(round(y1 - (yy_val - ymin) / yr * (y1 - y0)))
            cv2.line(graph_area, (x0 - 4, yy), (x0, yy), PLOT_AXIS, 1)
            label = f"{yy_val:.{y_decimals}f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(graph_area, label, (max(0, x0 - tw - 8), yy + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
            yy_val += y_tick_step

    # Títols eixos
    (xtw, xth), _ = cv2.getTextSize("Temps (s)", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(graph_area, "Temps (s)", (x1 - xtw, y1 + xth + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)
    (ytw, yth), _ = cv2.getTextSize(y_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(graph_area, y_label, (x0 + 6, y0 - 6 - label_pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)

    # Llegenda
    legend_area = canvas[h_no_legend:, :]
    legend_area[:] = (40, 40, 40)
    x, y = 12, LEGEND_HEIGHT//2 + 5
    for bid, v in sorted(hist_dict.items(), key=lambda kv: kv[0]):
        cv2.rectangle(legend_area, (x, y-10), (x+18, y+2), v['color'], -1)
        cv2.putText(legend_area, f"B{bid}", (x+24,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)
        x += 70
    cv2.rectangle(legend_area, (x, y-10), (x+18, y+2), MEAN_COLOR, -1)
    cv2.putText(legend_area, "Mean", (x+24,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)

    return canvas


# ---------- càlcul alçada màxima d'onada ----------
def _find_local_extrema(y: np.ndarray):
    extrema = []
    n = len(y)
    if n < 3: return extrema
    for i in range(1, n-1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            extrema.append((i, 'peak'))
        elif y[i] < y[i-1] and y[i] < y[i+1]:
            extrema.append((i, 'trough'))
    return extrema

def _max_wave_height_from_series(y: np.ndarray):
    if len(y) == 0:
        return (float('nan'), None, None)
    ex = _find_local_extrema(y)
    if len(ex) < 2:
        j_max = int(np.nanargmax(y))
        j_min = int(np.nanargmin(y))
        H = float(y[j_max] - y[j_min])
        if H >= 0: return (H, j_max, j_min)
        else: return (-H, j_min, j_max)
    H_best = -1.0; peak_best = None; trough_best = None
    for k in range(len(ex)-1):
        i0, t0 = ex[k]; i1, t1 = ex[k+1]
        if t0 == t1: continue
        if t0 == 'trough' and t1 == 'peak':
            H = float(y[i1] - y[i0])
            if H > H_best: H_best, peak_best, trough_best = H, i1, i0
        elif t0 == 'peak' and t1 == 'trough':
            H = float(y[i0] - y[i1])
            if H > H_best: H_best, peak_best, trough_best = H, i0, i1
    if H_best >= 0: return (H_best, peak_best, trough_best)
    j_max = int(np.nanargmax(y)); j_min = int(np.nanargmin(y))
    H = float(y[j_max] - y[j_min])
    if H >= 0: return (H, j_max, j_min)
    else: return (-H, j_min, j_max)


# ---------- Calibració ----------
def _load_calibration(json_path, override_scale=None):
    if not os.path.isfile(json_path):
        return None, {}, []
    with open(json_path, "r", encoding="utf-8") as f:
        J = json.load(f)

    a_g = J.get("calibration_reference", {}).get("global_fit", {}).get("a_m_per_px", None)
    b_g = J.get("calibration_reference", {}).get("global_fit", {}).get("b_m", None)
    global_ref = (float(a_g), float(b_g)) if (a_g is not None and b_g is not None) else None

    per_node_ref = {}
    ref_nodes = J.get("calibration_reference", {}).get("per_node", {})
    for node_str, info in ref_nodes.items():
        try:
            node = int(node_str)
        except:
            continue
        a_n = info.get("a_m_per_px", None)
        b_n = info.get("b_m", None)
        if a_n is None or b_n is None:
            continue
        per_node_ref[node] = (float(a_n), float(b_n))

    scale_json = float(J.get("calibration_for_predictions", {}).get("derived_scale", 1.0))
    scale = float(override_scale) if (override_scale is not None) else scale_json

    global_pred = (global_ref[0]*scale, global_ref[1]*scale) if global_ref is not None else None
    per_node_pred = {node: (a*scale, b*scale) for node, (a, b) in per_node_ref.items()}

    nodes_l2r = J.get("config", {}).get("nodes_order_left_to_right_pixels", [])
    nodes_l2r = [int(n) for n in nodes_l2r] if nodes_l2r else []

    print(f"ℹ️ Escala px→m utilitzada: {scale:.6g}  (origen={'override' if override_scale is not None else 'json'})")
    return global_pred, per_node_pred, nodes_l2r

def _make_tid_to_node_map(dets_sorted_l2r, nodes_l2r):
    mapping = {}
    n = min(len(dets_sorted_l2r), len(nodes_l2r))
    for k in range(n):
        tid, _ = dets_sorted_l2r[k]
        mapping[tid] = nodes_l2r[k]
    return mapping


# ---------- Tracking ----------
def iou_xywh(a, b):
    ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2
    ax2, ay2 = a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2
    bx2, by2 = b[0]+b[2]/2, b[1]+b[3]/2
    interw = max(0, min(ax2,bx2) - max(ax1,bx1))
    interh = max(0, min(ay2,by2) - max(ay1,by1))
    inter = interw * interh
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)  # <-- correcció (abans tenia ay2-ay1)
    return inter / (area_a + area_b - inter + 1e-9)

def track_iou(prev_tracks, detections, iou_th=0.5):
    assigned, new_tracks, matches = set(), {}, []
    for tid, t in prev_tracks.items():
        best, best_j = -1.0, -1
        for j, d in enumerate(detections):
            if j in assigned: continue
            iou = iou_xywh(t["box"], d["box"])
            if iou > best: best, best_j = iou, j
        if best >= iou_th and best_j >= 0:
            d = detections[best_j]
            new_tracks[tid] = {"box": d["box"], "miss": 0, "class": d["class"], "conf": d["conf"]}
            matches.append((tid, d)); assigned.add(best_j)
        else:
            new_tracks[tid] = {**t, "miss": t["miss"]+1}
    next_id = (max(prev_tracks.keys())+1) if prev_tracks else 1
    for j, d in enumerate(detections):
        if j not in assigned:
            new_tracks[next_id] = {"box": d["box"], "miss": 0, "class": d["class"], "conf": d["conf"]}
            matches.append((next_id, d)); next_id += 1
    return {tid: t for tid, t in new_tracks.items() if t["miss"] <= 15}, matches


# ---------- Etiquetes i colors fixes L→R ----------
def _build_rank_maps(nodes_l2r, tid2node):
    node2rank = {node: i+1 for i, node in enumerate(nodes_l2r)}
    tid2rank = {tid: node2rank.get(node, None) for tid, node in tid2node.items()}
    return node2rank, tid2rank

def draw_annotations(img_bgr, tracks, tid2rank):
    annotated = img_bgr.copy()
    for tid, t in tracks.items():
        x, y, w, h = t["box"]
        x1 = int(x - w/2); y1 = int(y - h/2)
        x2 = int(x + w/2); y2 = int(y + h/2)
        rank = tid2rank.get(tid, None)
        color = COLOR_ID.get(rank, (200,200,200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"boia_{rank}" if rank is not None else f"ID {tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return annotated


# ---------- UI d’anotació (igual que abans) ----------
def select_roi_thin(window_name, img_with_overlay):
    base = img_with_overlay.copy()
    start = [None, None]; end = [None, None]
    drawing = [False]; awaiting = [False]; preview = [base.copy()]
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not awaiting[0]:
            start[0], start[1] = x, y; end[0], end[1] = x, y; drawing[0] = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            tmp = base.copy(); cv2.rectangle(tmp, (start[0], start[1]), (x, y), (0,255,255), 1); preview[0] = tmp
        elif event == cv2.EVENT_LBUTTONUP and drawing[0]:
            end[0], end[1] = x, y; drawing[0] = False; awaiting[0] = True
            tmp = base.copy(); cv2.rectangle(tmp, (start[0], start[1]), (end[0], end[1]), (0,255,255), 1); preview[0] = tmp
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try: cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception: pass
    cv2.setMouseCallback(window_name, on_mouse)
    accepted = False
    while True:
        cv2.imshow(window_name, preview[0] if (drawing[0] or awaiting[0]) else base)
        k = cv2.waitKey(50) & 0xFF
        if awaiting[0]:
            if k in (13,10): accepted = True; break
            elif k == 27: accepted = False; break
    cv2.setMouseCallback(window_name, lambda *args: None)
    cv2.destroyWindow(window_name)
    if not accepted or None in start or None in end: return (0,0,0,0)
    x1,y1,x2,y2 = start[0], start[1], end[0], end[1]
    if x2 < x1: x1,x2 = x2,x1
    if y2 < y1: y1,y2 = y2,y1
    return (x1, y1, max(0, x2-x1), max(0, y2-y1))

def _draw_boxes(img, boxes, keep_mask=None, info_text=""):
    # (ja definit abans; mantenim per compatibilitat UI)
    return __draw_boxes_impl(img, boxes, keep_mask, info_text)

def __draw_boxes_impl(img, boxes, keep_mask=None, info_text=""):
    vis = img.copy()
    bar_h = 40
    bar = np.full((bar_h, vis.shape[1], 3), 30, dtype=np.uint8)
    msg = (info_text or "")
    cv2.putText(bar, msg, (10, int(bar_h*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(255,255,255), 1, cv2.LINE_AA)
    vis = np.vstack([bar, vis])
    yoff = bar_h
    for i, b in enumerate(boxes):
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        color = (0,255,0)
        label = f"{i+1}"
        if keep_mask is not None:
            if keep_mask[i]:
                color = (0, 200, 0); lab2 = "KEEP"
            else:
                color = (0, 0, 200); lab2 = "ELIM"
            label = f"{i+1}:{lab2}"
        cv2.rectangle(vis, (x1, y1+yoff), (x2, y2+yoff), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, max(0, y1+yoff - th - 6)), (x1 + tw + 6, y1+yoff), (0, 0, 0), -1)
        cv2.putText(vis, label, (x1 + 3, y1+yoff - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return vis, yoff

def annotate_until_four(img_bgr, dets_raw, required=4):
    H, W = img_bgr.shape[:2]
    cur = list(dets_raw)
    while True:
        while len(cur) < required:
            missing = required - len(cur)
            info = f"Falten {missing} boia/es. Dibuixa 1 ROI (ENTER=OK, ESC=cancel·la)."
            vis, yoff = __draw_boxes_impl(img_bgr, cur, info_text=info)
            win = "Afegir boia (1 a 1)"
            _show_fullscreen(win, vis); cv2.waitKey(1)
            x, y, w, h = select_roi_thin(win, vis)
            if w == 0 or h == 0: continue
            y_adj = max(0, y - yoff)
            cx = float(x + w/2.0); cy = float(y_adj + h/2.0)
            cx, cy, w, h = _clamp_box(cx, cy, float(w), float(h), W, H)
            cur.append({
                "x": cx, "y": cy, "width": float(w), "height": float(h),
                "class": "boia",
                "confidence": 1.0, "manual": True,
            })
        cur_sorted = sort_left_to_right(cur)
        if final_confirm(img_bgr, cur_sorted): return cur_sorted
        else: cur = list(dets_raw)

def refine_excess_topbar(img_bgr, dets_raw, required=4):
    H, W = img_bgr.shape[:2]
    pad_x = 10; pad_y = 8; btn_w = 90; btn_h = 28; gap = 6
    max_per_row = max(1, (W - pad_x*2) // (btn_w + gap))
    rows_btn = int(np.ceil(len(dets_raw) / max_per_row))
    bar_h = pad_y*2 + rows_btn*btn_h + (rows_btn-1)*gap + 44
    def layout_button_rects(n):
        rects = []
        for i in range(n):
            row = i // max_per_row; col = i % max_per_row
            x1 = pad_x + col*(btn_w + gap); y1 = pad_y + row*(btn_h + gap)
            x2 = x1 + btn_w; y2 = y1 + btn_h; rects.append((x1, y1, x2, y2))
        ok_w = 80; ok_h = 28; ok_x2 = W - pad_x; ok_x1 = ok_x2 - ok_w
        ok_y1 = bar_h - pad_y - ok_h; ok_y2 = ok_y1 + ok_h
        return rects, (ok_x1, ok_y1, ok_x2, ok_y2)
    while True:
        keep = [True] * len(dets_raw)
        win = "Selecciona 4 (Topbar)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        try: cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception: pass
        btn_rects, ok_btn = None, None
        clicked_ok = False
        def draw_ui():
            nonlocal btn_rects, ok_btn
            rects, ok_rect = layout_button_rects(len(dets_raw))
            topbar = np.full((bar_h, W, 3), 32, dtype=np.uint8)
            kept = sum(1 for k in keep if k)
            instr = f"Selecciona EXACTAMENT {required} KEEP. KEEP={kept}/{len(dets_raw)} | Clic botons | ENTER/OK per confirmar"
            cv2.putText(topbar, instr, (10, bar_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            for i, (x1,y1,x2,y2) in enumerate(rects):
                state = "KEEP" if keep[i] else "ELIM"
                color = (0,180,0) if keep[i] else (0,0,200)
                cv2.rectangle(topbar, (x1,y1), (x2,y2), color, -1)
                label = f"{i+1}: {state}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                tx = x1 + (btn_w - tw)//2; ty = y1 + (btn_h + th)//2 - 2
                cv2.putText(topbar, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            x1,y1,x2,y2 = ok_rect
            cv2.rectangle(topbar, (x1,y1), (x2,y2), (60,60,60), -1)
            (tw, th), _ = cv2.getTextSize("OK", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(topbar, "OK", (x1 + (x2-x1 - tw)//2, y1 + (y2-y1 + th)//2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            vis_img, _ = __draw_boxes_impl(img_bgr, dets_raw, keep_mask=keep, info_text="")
            full = np.vstack([topbar, vis_img[40:]])
            btn_rects, ok_btn = rects, ok_rect
            return full
        def on_mouse(event, x, y, flags, param):
            nonlocal clicked_ok
            if event == cv2.EVENT_LBUTTONDOWN and y < bar_h:
                for i, (x1,y1,x2,y2) in enumerate(btn_rects):
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        keep[i] = not keep[i]; return
                x1,y1,x2,y2 = ok_btn
                if x1 <= x <= x2 and y1 <= y <= y2:
                    clicked_ok = True
        cv2.setMouseCallback(win, on_mouse)
        while True:
            vis = draw_ui(); cv2.imshow(win, vis)
            key = cv2.waitKey(50) & 0xFF
            kept = sum(1 for k in keep if k)
            if key in (13,10) or clicked_ok:
                if kept == required: break
                clicked_ok = False
            elif key in [ord(str(d)) for d in range(1,10)]:
                idx = int(chr(key)) - 1
                if 0 <= idx < len(keep): keep[idx] = not keep[idx]
        cv2.destroyWindow(win)
        refined = [dets_raw[i] for i, k in enumerate(keep) if k]
        refined_sorted = sort_left_to_right(refined)
        if final_confirm(img_bgr, refined_sorted): return refined_sorted
        else: continue

def sort_left_to_right(dets):
    return sorted(dets, key=lambda d: d["x"])

def handle_mismatch(img_bgr, dets_raw, required=4):
    if len(dets_raw) == required: return sort_left_to_right(dets_raw)
    if len(dets_raw) < required: return annotate_until_four(img_bgr, dets_raw, required)
    else: return refine_excess_topbar(img_bgr, dets_raw, required)

def final_confirm(img_bgr, dets_final_sorted):
    vis, _ = __draw_boxes_impl(img_bgr, dets_final_sorted,
                         info_text="Revisió final: ENTER=Confirmar  |  R=Revisar  |  ESC=Cancel·lar")
    win = "Revisio final"
    _show_fullscreen(win, vis)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (13,10):
            cv2.destroyWindow(win); return True
        elif k in (27, ord('r'), ord('R')):
            cv2.destroyWindow(win); return False


# ========= PASSADA 1: detecció, tracking, CSV/JSON, frames anotats =========
def first_pass_process():
    # >>>>> AFEGIT: calcular sufix i carpeta d'output prediccio_<n>_<sufix>
    suffix = _suffix_from_frames_parent(FRAMES_DIR)
    OUTPUT_DIR, run_index, dirname = next_numbered_dir_with_suffix(
        OUTPUT_PARENT_DIR, OUTPUT_BASENAME, suffix
    )
    OUTPUT_DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_frames")
    os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)

    VIDEO_FIRST_PATH = os.path.join(OUTPUT_DIR, "deteccions_firstpass.mp4")
    CSV_PATH = os.path.join(OUTPUT_DIR, "prediccions.csv")
    JSON_PATH = os.path.join(OUTPUT_DIR, "prediccions.json")
    SUMMARY_PATH = os.path.join(OUTPUT_DIR, "resum_max_onada.csv")
    RESULT_JSON_PATH = os.path.join(OUTPUT_DIR, "Resultat.json")
    FRAMES_LIST_PATH = os.path.join(OUTPUT_DIR, "frames_list.txt")

    print(f"📁 Carpeta de sortida: {os.path.abspath(OUTPUT_DIR)}  (índex={run_index}, sufix='{suffix}')")
    # <<<<< FI AFEGIT

    if not os.path.isdir(FRAMES_DIR):
        print(f"La carpeta FRAMES_DIR no existeix: {FRAMES_DIR}"); return None

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    def _natural_key(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    frame_paths = sorted([os.path.join(FRAMES_DIR, f)
                          for f in os.listdir(FRAMES_DIR)
                          if f.lower().endswith(exts)],
                         key=lambda p: _natural_key(os.path.basename(p)))
    if not frame_paths:
        print(f"No s'han trobat imatges a {FRAMES_DIR}"); return None

    # Desa l’ordre dels frames per a la passada 2
    with open(FRAMES_LIST_PATH, "w", encoding="utf-8") as fw:
        fw.write("\n".join(frame_paths))

    total_frames = len(frame_paths)
    TOTAL_DURATION_S = max(1.0, total_frames / FPS)
    print(f"Total de frames trobats: {total_frames} | Mode: {'INTERACTIU' if INTERACTIVE_ENFORCE_FOUR else 'AUTO'}")

    model, tracks, rows = load_model(), {}, []
    override_scale = None
    if (Z_ANNOTATION_AMP is not None) and (Z_PREDICTION_AMP is not None) and (float(Z_PREDICTION_AMP) != 0.0):
        override_scale = float(Z_ANNOTATION_AMP) / float(Z_PREDICTION_AMP)

    global_pred, per_node_pred, nodes_l2r = _load_calibration(CALIB_JSON, override_scale=override_scale)
    if global_pred is None and not per_node_pred:
        print("⚠️ No s'ha trobat calibració vàlida; es farà servir METERS_PER_PX si està definit.")
    else:
        print("✅ Calibració carregada:",
              f"global={global_pred is not None}, per_node={len(per_node_pred)} nodes")

    writer = None
    processing_time_total = 0.0

    tid2node = {}
    have_mapped = False
    node2rank = {}
    tid2rank = {}

    try:
        for idx, path in enumerate(frame_paths):
            t0 = time.time()
            img = cv2.imread(path)
            if img is None:
                print(f"Avís: no es pot llegir {path}"); continue
            H, W = img.shape[:2]

            if WRITE_VIDEO_FIRST_PASS and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                writer = cv2.VideoWriter(VIDEO_FIRST_PATH, fourcc, VIDEO_FPS, (W, H))

            dets_raw = process_frame(img, model)
            t1 = time.time(); processing_time_total += (t1 - t0)

            if INTERACTIVE_ENFORCE_FOUR and len(dets_raw) != REQUIRED_BOIES:
                print(f"🟡 {os.path.basename(path)}: dets={len(dets_raw)} != {REQUIRED_BOIES}. Obrint UI…")
                dets_raw = handle_mismatch(img, dets_raw, REQUIRED_BOIES)
            else:
                dets_raw = sort_left_to_right(dets_raw)

            t2 = time.time()
            dets = [{"box": (p["x"], p["y"], p["width"], p["height"]),
                     "class": p["class"], "conf": p["confidence"], "manual": p.get("manual", False)}
                    for p in dets_raw]
            tracks, matches = track_iou(tracks, dets, iou_th=IOU_TRACK_THRES)
            t3 = time.time(); processing_time_total += (t3 - t2)

            # Mapatge inicial TID→node (L→R) i rank fix boia_1..4
            cur_l2r = sorted(matches, key=lambda kv: kv[1]["box"][0])
            if nodes_l2r and len(cur_l2r) >= 4 and not have_mapped:
                tid2node = _make_tid_to_node_map(cur_l2r, nodes_l2r)
                have_mapped = True
                node2rank, tid2rank = _build_rank_maps(nodes_l2r, tid2node)
                print("ℹ️ Mapatge TID→node:", tid2node)
                print("ℹ️ Rangs fixes (boia_n):", tid2rank)

            # Guarda files per CSV/JSON
            t_s = idx / FPS
            for tid, d in matches:
                x, y, w, h = d["box"]
                y_val = (H - y) if INVERT_Y else y

                z_m = None
                if CALIB_MODE == "per_node" and have_mapped and (tid in tid2node) and (tid2node[tid] in per_node_pred):
                    a, b = per_node_pred[tid2node[tid]]
                    z_m = a * float(y_val) + b
                elif CALIB_MODE == "global" and global_pred is not None:
                    a, b = global_pred
                    z_m = a * float(y_val) + b
                elif METERS_PER_PX is not None:
                    z_m = float(y_val) * float(METERS_PER_PX)

                rows.append({
                    "time_s": round(t_s, 6), "frame": idx,
                    "img": os.path.basename(path), "buoy_id": tid,
                    "rank": int(tid2rank.get(tid)) if tid in tid2rank and tid2rank[tid] is not None else None,
                    "class": d.get("class"), "conf": d.get("conf"),
                    "manual": d.get("manual", False),
                    "x_px": x, "y_px": y, "y_up_px": y_val, "w_px": w, "h_px": h,
                    "z_m": z_m,
                    "node_id": tid2node.get(tid) if have_mapped else None
                })

            # Dibuixa caixes + etiqueta boia_n i desa ann_XXXXX.png
            annotated = draw_annotations(img, tracks, tid2rank)
            if WRITE_ANNOTATED_FRAMES:
                cv2.imwrite(os.path.join(OUTPUT_DEBUG_DIR, f"ann_{idx:05d}.png"), annotated)
            if WRITE_VIDEO_FIRST_PASS and writer is not None:
                writer.write(annotated)

            avg_time = processing_time_total / (idx + 1)
            frames_left = total_frames - (idx + 1)
            remaining = avg_time * frames_left
            print(f"[{idx+1}/{total_frames}] {os.path.basename(path)} | "
                  f"dets={len(dets_raw)} | tracks={len(tracks)} | ETA ~{_fmt_time(remaining)}")
    finally:
        if WRITE_VIDEO_FIRST_PASS and writer is not None:
            writer.release()
            print(f"🎬 Vídeo (passada 1) guardat a: {os.path.abspath(VIDEO_FIRST_PATH)}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ====== Exportació i mètriques ======
    df = pd.DataFrame(rows)
    if df.empty:
        print("Sense deteccions."); return None

    df.to_csv(CSV_PATH, index=False, float_format="%.6f")
    with open(JSON_PATH, "w", encoding="utf-8") as f: json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"✅ Guardat: {os.path.abspath(CSV_PATH)} i {os.path.abspath(JSON_PATH)}")

    # Selecció sèrie física
    if "z_m" in df.columns and df["z_m"].notna().any():
        series_col = "z_m"; unit = "m"
    else:
        if METERS_PER_PX is not None:
            df["y_up_m"] = df["y_up_px"] * METERS_PER_PX
            series_col = "y_up_m"; unit = "m"
        else:
            series_col = "y_up_px"; unit = "px"

    # FFT + Hs + export gràfics
    per_buoy_results = []
    OUTPUT_DIR = os.path.dirname(CSV_PATH)

    for bid, g in df.groupby("buoy_id"):
        g = g.sort_values("time_s")
        t = g["time_s"].values
        y_abs = g[series_col].values.astype(float)
        if len(y_abs) < 2:
            print(f"Boia {bid}: massa pocs punts per graficar."); continue

        plt.figure(); plt.plot(t, y_abs)
        plt.xlabel("Temps (s)"); plt.ylabel(f"Alçada {unit} (absoluta)")
        plt.title(f"Boia {bid} - sèrie z (calibrada)"); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"serie_buoy_{bid}.png"), dpi=150); plt.close()

        y = y_abs - np.nanmean(y_abs)
        n = len(y)
        if n >= 8:
            dt = 1.0 / FPS; window = np.hanning(n)
            Y = np.fft.rfft(y * window); f = np.fft.rfftfreq(n, d=dt)
            S = (np.abs(Y) ** 2) / n
            Tp = 1.0 / f[np.argmax(S[1:])+1] if len(S) > 1 else float("nan")
            Hs = 4.0 * np.std(y)
            line_txt = f"Boia {int(bid)}: Tp ≈ {Tp:.2f} s | Hs ≈ {Hs:.2f} {unit}"
            print(line_txt)
            per_buoy_results.append({
                "buoy_id": int(bid),
                "Tp_s": float(Tp),
                "Hs": float(Hs),
                "unit": unit,
                "text": line_txt
            })
            pd.DataFrame({"freq_Hz": f, "S": S}).to_csv(
                os.path.join(OUTPUT_DIR, f"espectre_buoy_{bid}.csv"), index=False)
            plt.figure(); plt.plot(f, S)
            plt.xlabel("Freqüència (Hz)"); plt.ylabel("Espectre (arb.)")
            plt.title(f"Boia {bid} - FFT (Tp~{Tp:.2f}s)"); plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"fft_buoy_{bid}.png"), dpi=150); plt.close()
        else:
            print(f"Boia {bid}: massa pocs punts per FFT (n={n})")

    # H_max per boia i global
    summary_rows = []
    best_global = {"H_max": -1.0, "buoy_id": None, "frame": None, "time_s": None,
                   "peak_val": None, "trough_val": None, "peak_frame": None, "trough_frame": None}

    for bid, g in df.groupby("buoy_id"):
        g = g.sort_values("time_s").reset_index(drop=True)
        y_abs = g[series_col].to_numpy(float)
        y = y_abs - np.nanmean(y_abs)
        frames = g["frame"].to_numpy(); times  = g["time_s"].to_numpy()
        H_max, idx_peak, idx_trough = _max_wave_height_from_series(y)
        if idx_peak is not None and idx_trough is not None:
            peak_val = float(y[idx_peak]); trough_val = float(y[idx_trough])
            peak_frame = int(frames[idx_peak]); trough_frame = int(frames[idx_trough])
            t_when = float(times[idx_peak])
            summary_rows.append({
                "buoy_id": int(bid), "H_max": float(H_max), "unit": unit,
                "frame_at_peak": peak_frame, "time_s_at_peak": t_when,
                "peak_val_detrended": peak_val, "trough_val_detrended": trough_val,
                "peak_frame": peak_frame, "trough_frame": trough_frame
            })
            if H_max > best_global["H_max"]:
                best_global.update({"H_max": float(H_max), "buoy_id": int(bid),
                                    "frame": peak_frame, "time_s": t_when,
                                    "peak_val": peak_val, "trough_val": trough_val,
                                    "peak_frame": peak_frame, "trough_frame": trough_frame})
        else:
            summary_rows.append({
                "buoy_id": int(bid), "H_max": float("nan"), "unit": unit,
                "frame_at_peak": None, "time_s_at_peak": None,
                "peak_val_detrended": None, "trough_val_detrended": None,
                "peak_frame": None, "trough_frame": None
            })

    pd.DataFrame(summary_rows).sort_values("buoy_id").to_csv(SUMMARY_PATH, index=False)
    print(f"📄 Resum “alçada màxima d'onada” guardat a: {os.path.abspath(SUMMARY_PATH)}")

    if best_global["buoy_id"] is not None:
        global_text = ("🌊 Màxima global: "
                       f"H_max={best_global['H_max']:.3f} {unit} | "
                       f"Boia {best_global['buoy_id']} | Frame {best_global['frame']} | "
                       f"t={best_global['time_s']:.3f}s "
                       f"(pic frame {best_global['peak_frame']}, vall frame {best_global['trough_frame']})")
        print(global_text)
    else:
        global_text = "No s'ha pogut calcular l'alçada màxima d'onada."
        print(global_text)

    # Mitjanes
    if per_buoy_results:
        tp_vals = [r["Tp_s"] for r in per_buoy_results if np.isfinite(r["Tp_s"])]
        hs_vals = [r["Hs"] for r in per_buoy_results if np.isfinite(r["Hs"])]
        Tp_mean = float(np.mean(tp_vals)) if tp_vals else float("nan")
        Hs_mean = float(np.mean(hs_vals)) if hs_vals else float("nan")
        avg_text = (f"Mitjanes: Tp ≈ {Tp_mean:.2f} s | Hs ≈ {Hs_mean:.2f} {unit} "
                    f"(sobre {len(per_buoy_results)} boies)")
        print(avg_text)
    else:
        Tp_mean, Hs_mean, avg_text = float("nan"), float("nan"), "Mitjanes no disponibles."

    resultat_payload = {
        "per_buoy": per_buoy_results,
        "averages": {
            "Tp_s_mean": None if not np.isfinite(Tp_mean) else float(Tp_mean),
            "Hs_mean": None if not np.isfinite(Hs_mean) else float(Hs_mean),
            "count": len(per_buoy_results),
            "unit": unit,
            "text": avg_text
        },
        "global_max": {
            "H_max": None if best_global["H_max"] is None or not np.isfinite(best_global["H_max"]) else float(best_global["H_max"]),
            "unit": unit,
            "buoy_id": best_global["buoy_id"],
            "frame": best_global["frame"],
            "time_s": best_global["time_s"],
            "peak_frame": best_global["peak_frame"],
            "trough_frame": best_global["trough_frame"],
            "text": global_text
        }
    }
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(resultat_payload, f, ensure_ascii=False, indent=2)
    print(f"🧾 Resultat guardat a: {os.path.abspath(RESULT_JSON_PATH)}")

    # Retorna info per a la passada 2
    return {
        "OUTPUT_DIR": OUTPUT_DIR,
        "OUTPUT_DEBUG_DIR": OUTPUT_DEBUG_DIR,
        "CSV_PATH": CSV_PATH,
        "FRAMES_LIST_PATH": FRAMES_LIST_PATH,
        "TOTAL_DURATION_S": TOTAL_DURATION_S,
        "unit": unit,
        "best_global": best_global
    }


# ========= PASSADA 2: vídeo final amb Y fixa basada en Hmax =========
def second_pass_render_video(pass1_info):
    OUTPUT_DIR = pass1_info["OUTPUT_DIR"]
    OUTPUT_DEBUG_DIR = pass1_info["OUTPUT_DEBUG_DIR"]
    CSV_PATH = pass1_info["CSV_PATH"]
    FRAMES_LIST_PATH = pass1_info["FRAMES_LIST_PATH"]
    TOTAL_DURATION_S = pass1_info["TOTAL_DURATION_S"]
    unit = pass1_info["unit"]
    best_global = pass1_info["best_global"]

    VIDEO_FINAL_PATH = os.path.join(OUTPUT_DIR, "deteccions_final.mp4")

    # Carrega ordre de frames
    if os.path.isfile(FRAMES_LIST_PATH):
        with open(FRAMES_LIST_PATH, "r", encoding="utf-8") as fr:
            frame_paths = [line.strip() for line in fr if line.strip()]
    else:
        print("⚠️ No hi ha frames_list.txt; es tornarà a llistar FRAMES_DIR (pot variar l’ordre).")
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        def _natural_key(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
        frame_paths = sorted([os.path.join(FRAMES_DIR, f)
                              for f in os.listdir(FRAMES_DIR)
                              if f.lower().endswith(exts)],
                             key=lambda p: _natural_key(os.path.basename(p)))

    if not frame_paths:
        print("No hi ha frames per renderitzar el vídeo final.")
        return

    # Carrega dades
    df = pd.read_csv(CSV_PATH)
    use_meters = ("z_m" in df.columns) and df["z_m"].notna().any() and (unit == "m")
    if use_meters:
        series_col = "z_m"
        y_tick_step = 0.25
        y_decimals = Y_DECIMALS_M
        y_label = "Alçada (m)"
    else:
        series_col = "y_up_px"
        y_label = "Alçada (px)"
        y_decimals = 0
        y_tick_step = max(1, int( (df[series_col].max() - df[series_col].min()) // 8 ) )

    # Calcula rang Y fix basat en Hmax global
    if np.isfinite(best_global.get("H_max", np.nan)):
        A = float(best_global["H_max"]) / 2.0
        A = max(A, 1e-6)
        A *= Y_MARGIN_FACTOR
        y_min, y_max = -A, +A
        print(f"🔒 Overlay Y fix ({'m' if use_meters else 'px'}): ({y_min:.3f}, {y_max:.3f}) "
              f"amb pas {y_tick_step} {'m' if use_meters else 'px'}")
    else:
        # Fallback robust si no hi ha Hmax
        y_vals = df[series_col].values.astype(float)
        Y = y_vals - np.nanmean(y_vals) if PLOT_CENTERED else y_vals
        A = np.percentile(np.abs(Y), 99.0)
        A = max(A, 1.0)
        y_min, y_max = -A, +A
        print(f"🔒 Overlay Y fix (fallback): ({y_min:.3f}, {y_max:.3f})")

    # Prepara sèries per rank (boia_1..4) per tenir colors estables
    # Si no hi ha 'rank', usem buoy_id com a clau
    have_rank = "rank" in df.columns and df["rank"].notna().any()
    key_col = "rank" if have_rank else "buoy_id"

    # Precompute sèries completes per clau
    series_by_key = {}
    for k, g in df.groupby(key_col):
        if pd.isna(k): continue
        g = g.sort_values("time_s")
        t = g["time_s"].to_list()
        y_abs = g[series_col].to_numpy(float)
        y = (y_abs - np.nanmean(y_abs)) if PLOT_CENTERED else y_abs
        series_by_key[int(k)] = {"t": t, "y": y.tolist(),
                                 "color": COLOR_ID.get(int(k), (200,200,200))}

    # Precompute mean per temps (usant només claus presents a cada t)
    times_unique = sorted(df["time_s"].unique().tolist())
    mean_series_t = []
    mean_series_y = []
    for tt in times_unique:
        vals = []
        for k, S in series_by_key.items():
            if S["t"] and S["t"][0] <= tt:
                idx = np.searchsorted(S["t"], tt, side="right") - 1
                if 0 <= idx < len(S["y"]) and S["t"][idx] <= tt + 1e-9:
                    vals.append(S["y"][idx])
        if vals:
            mean_series_t.append(tt)
            mean_series_y.append(float(np.mean(vals)))

    mean_series_full = {"t": mean_series_t, "y": mean_series_y}

    # Obrim primer frame per mida
    first_img = None
    for idx in range(len(frame_paths)):
        ann_path = os.path.join(OUTPUT_DEBUG_DIR, f"ann_{idx:05d}.png")
        img_path = ann_path if os.path.isfile(ann_path) else frame_paths[idx]
        first_img = cv2.imread(img_path)
        if first_img is not None: break
    if first_img is None:
        print("No s’ha pogut llegir cap frame per inicialitzar el vídeo final.")
        return
    H, W = first_img.shape[:2]

    writer = None
    if WRITE_VIDEO_FINAL:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(VIDEO_FINAL_PATH, fourcc, VIDEO_FPS, (W, H))

    # Render “riu”: per cada frame, només fins a t_actual
    for idx in range(len(frame_paths)):
        ann_path = os.path.join(OUTPUT_DEBUG_DIR, f"ann_{idx:05d}.png")
        img_path = ann_path if os.path.isfile(ann_path) else frame_paths[idx]
        frame = cv2.imread(img_path)
        if frame is None: continue

        t_current = idx / FPS

        # Construeix hist parcial fins t_current
        hist_plot = {}
        for k, S in series_by_key.items():
            n = np.searchsorted(S["t"], t_current, side="right")
            if n >= 2:
                hist_plot[k] = {"t": S["t"][:n], "y": S["y"][:n], "color": S["color"]}

        # Mean parcial
        n_mean = np.searchsorted(mean_series_full["t"], t_current, side="right")
        mean_partial = {"t": mean_series_full["t"][:n_mean], "y": mean_series_full["y"][:n_mean]}

        if PLOT_ENABLED and hist_plot:
            plot_img = draw_time_series_overlay(
                hist_plot, mean_partial,
                size=(PLOT_W, PLOT_H),
                fixed_t_range=(0.0, TOTAL_DURATION_S),
                fixed_y_range=(y_min, y_max),
                y_label=y_label,
                y_decimals=y_decimals,
                x_tick_step=5.0,
                y_tick_step=y_tick_step,
                line_thickness=1,
                label_pad=10
            )
            ph, pw = plot_img.shape[:2]
            y0 = PLOT_MARGIN; x0 = PLOT_MARGIN
            roi = frame[y0:y0+ph, x0:x0+pw]
            if roi.shape[0] == ph and roi.shape[1] == pw:
                frame[y0:y0+ph, x0:x0+pw] = plot_img
            else:
                frame[y0:y0+ph, x0:x0+pw] = plot_img[:roi.shape[0], :roi.shape[1]]

        if WRITE_VIDEO_FINAL and writer is not None:
            writer.write(frame)

        if (idx+1) % 50 == 0 or idx == len(frame_paths)-1:
            print(f"[Passada 2] {idx+1}/{len(frame_paths)} frames")

    if WRITE_VIDEO_FINAL and writer is not None:
        writer.release()
        print(f"🎬 Vídeo final guardat a: {os.path.abspath(VIDEO_FINAL_PATH)}")


# -----------------------------------------------
def main():
    # Passada 1: detecció + tracking + CSV/JSON + aann frames
    info = first_pass_process()
    if info is None:
        return
    # Passada 2: vídeo final amb Y fixa definida per Hmax global
    second_pass_render_video(info)

if __name__ == "__main__":
    main()
