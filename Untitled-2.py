import re
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# ==== CONFIGURACIÓ ====
MESH_FILE = Path(r"C:\Users\marcp\Desktop\Marc\TFG24\malla 6.gid\malla 6.flavia")
POST_FILE = Path(r"C:\Users\marcp\Desktop\Marc\TFG24\malla 6.post.res")

OUTPUT_JSON = Path("annotations.json")
OUTPUT_CSV = Path("labels.csv")

NODE_IDS = [10457, 7297, 5033, 7929]  # Nodes de les boies
START_TIME = 60.0                     # segons
TIME_STEP = 0.1                        # segons
START_FRAME = 1                        # 0001.png és el primer frame

# ==== 1. Llegir MESH (FreeSurface) ====
free_surface_xy = {}
with open(MESH_FILE, "r") as f:
    mesh_text = f.read()

# Extreure secció FreeSurface/Coordinates
match = re.search(r"FreeSurface;.*?Coordinates\n(.*?)end coordinates", mesh_text, re.S)
if not match:
    raise ValueError("No s'ha trobat la secció FreeSurface/Coordinates al MESH.")

coords_text = match.group(1).strip().splitlines()
for line in coords_text:
    parts = line.split()
    if len(parts) >= 4:
        node_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        free_surface_xy[node_id] = (x, y)

print(f"[MESH] Nodes FreeSurface carregats: {len(free_surface_xy)}")

# ==== 2. Llegir POST (.res) ====
frames_data = {}  # {frame_idx: {node_id: elevation}}
with open(POST_FILE, "r") as f:
    lines = f.readlines()

current_time = None
capture = False
for line in lines:
    # Detectar inici d'un bloc "Total elevation" "Free surface"
    if line.startswith('Result "Total elevation" "Free surface"'):
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                current_time = float(parts[5])
                frame_idx = int(round((current_time - START_TIME) / TIME_STEP)) + START_FRAME
                frames_data[frame_idx] = {}
                capture = True
            except ValueError:
                capture = False
        continue

    if capture:
        if line.startswith("End Values"):
            capture = False
            current_time = None
            continue
        parts = line.strip().split()

        # Saltar línies que no són dades numèriques node_id elevació
        if len(parts) != 2:
            continue
        if not parts[0].isdigit():
            continue

        node_id = int(parts[0])
        elev = float(parts[1])
        frames_data[frame_idx][node_id] = elev

print(f"[POST] Frames processats: {len(frames_data)}")

# ==== 3. Construir annotations ====
images = []
annotations = []
csv_rows = []

for frame_idx in sorted(frames_data.keys()):
    file_name = f"{frame_idx:04d}.png"
    time = START_TIME + (frame_idx - START_FRAME) * TIME_STEP

    images.append({
        "id": frame_idx,
        "file_name": file_name,
        "frame_idx": frame_idx,
        "time": time
    })

    kpts = []
    for nid in NODE_IDS:
        x, y = free_surface_xy[nid]
        z = frames_data[frame_idx].get(nid, 0.0)  # Opció A: z = elevació
        kpts.extend([x, y, z])

        csv_rows.append({
            "frame": frame_idx,
            "time": time,
            "node_id": nid,
            "x": x,
            "y": y,
            "z": z
        })

    annotations.append({
        "image_id": frame_idx,
        "keypoints_3d": kpts,
        "num_keypoints": len(NODE_IDS),
        "node_ids": NODE_IDS,
        "unit": "m",
        "coord_space": "blender"
    })

categories = [{
    "id": 1,
    "name": "buoys",
    "keypoints": ["b1", "b2", "b3", "b4"]
}]

# ==== 4. Guardar JSON ====
dataset = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(dataset, f, indent=2)
print(f"[OUTPUT] Guardat JSON: {OUTPUT_JSON}")

# ==== 5. Guardar CSV ====
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frame", "time", "node_id", "x", "y", "z"])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"[OUTPUT] Guardat CSV: {OUTPUT_CSV}")

# ==== 6. Visualització trajectòries Z ====
plt.figure(figsize=(10, 5))
for nid in NODE_IDS:
    times = [row["time"] for row in csv_rows if row["node_id"] == nid]
    zs = [row["z"] for row in csv_rows if row["node_id"] == nid]
    plt.plot(times, zs, label=f"Node {nid}")
plt.xlabel("Temps (s)")
plt.ylabel("Elevació (m)")
plt.title("Evolució elevació boies")
plt.legend()
plt.grid(True)
plt.show()
