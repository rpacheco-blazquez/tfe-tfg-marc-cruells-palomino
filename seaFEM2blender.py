import bpy
import re
from mathutils import Vector

# --- RUTES --------------------------------------------------------------------
FLAVIA_FILE = r"C:\Users\marcp\Desktop\Marc\TFG24\malla 6.gid\malla 6.flavia"
XYZ_FILE    = r"C:\Users\marcp\Desktop\xyz_per_timestep.txt"

# --- PARÀMETRES ---------------------------------------------------------------
USE_XY_FROM_XYZ = True    # True: usa X,Y del TXT; False: (si cal) podem usar X,Y del .flavia
EXTRA_Z_SCALE   = 4.0     # multiplicador addicional (si vols amplificar/atenuar el Z del TXT)
ADD_Z_OFFSET    = 0.0     # offset vertical addicional
MATERIALS_AND_MARKERS = True  # crea materials i esferes
SELECTED_NODE_IDS = [10457, 7297, 5033, 7929]  # nodes a marcar amb esferes
BAKE_INTERPOLATION = 'CONSTANT'  # 'CONSTANT' | 'LINEAR' | 'BEZIER' (per les boies)

# --- NETEJA ESCENA ------------------------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
for obj_ in list(bpy.data.objects):
    if obj_.type == 'MESH':
        obj_.select_set(True)
bpy.ops.object.delete()

# --- PARSERS ------------------------------------------------------------------
def parse_flavia_freesurface(path_flavia):
    """
    Llegeix FreeSurface -> Coordinates i Elements del .flavia.
    Retorna:
      coords: {node_id: (x,y,z)}
      tris:   [(n1,n2,n3), ...]
    """
    coords = {}
    tris   = []
    in_fs = False
    in_coords = False
    in_elems  = False
    with open(path_flavia, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            low = line.lower()
            if not line:
                continue
            # Entrar/Sortir seccions
            if low.startswith('freesurface'):
                in_fs = True; in_coords = False; in_elems = False; continue
            if in_fs and low.startswith('coordinates'):
                in_coords = True; in_elems = False; continue
            if in_fs and in_coords and low.startswith('end coordinates'):
                in_coords = False; continue
            if in_fs and low.startswith('elements'):
                in_elems = True; continue
            if in_fs and in_elems and low.startswith('end elements'):
                in_elems = False; in_fs = False; continue

            # Llegir dades
            if in_fs and in_coords:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        nid = int(float(parts[0]))
                        x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                        coords[nid] = (x, y, z)
                    except ValueError:
                        pass
            elif in_fs and in_elems:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        _eid = int(float(parts[0]))  # no l'usem
                        n1 = int(float(parts[1])); n2 = int(float(parts[2])); n3 = int(float(parts[3]))
                        tris.append((n1, n2, n3))
                    except ValueError:
                        pass
    return coords, tris

def parse_xyz_per_timestep(path_txt):
    """
    Llegeix el TXT generat (xyz_per_timestep.txt).
    Format esperat:
      # TimeStep i | t=... s
      node_id x y z
      ...
    Retorna:
      times: [t0, t1, ...]
      blocks: [ {nid:(x,y,z)}, ... ] (mateix ordre que times)
    """
    times = []
    blocks = []
    cur = None
    with open(path_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                m = re.search(r'TimeStep\s+(\d+)\s*\|\s*t=([\-0-9.+Ee]+)', line)
                if m:
                    if cur is not None:
                        blocks.append(cur)
                    cur = {}
                    try:
                        t = float(m.group(2))
                    except ValueError:
                        t = float('nan')
                    times.append(t)
                continue
            parts = line.split()
            if len(parts) == 4:
                try:
                    nid = int(float(parts[0]))
                    x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                    if cur is None:
                        cur = {}
                        times.append(float('nan'))
                    cur[nid] = (x, y, z)
                except ValueError:
                    pass
    if cur is not None:
        blocks.append(cur)
    return times, blocks

# --- LLEGIR DADES -------------------------------------------------------------
fs_coords, fs_tris = parse_flavia_freesurface(FLAVIA_FILE)
times, xyz_blocks  = parse_xyz_per_timestep(XYZ_FILE)

if not xyz_blocks:
    raise RuntimeError("No s'han trobat blocs XYZ al fitxer: " + XYZ_FILE)

# Nodes disponibles al TXT (primer bloc com a referència)
nodes_in_txt = set(xyz_blocks[0].keys())
# Filtra triangles que tinguin TOTS els seus nodes al TXT
tris_filtered = [tri for tri in fs_tris if all(n in nodes_in_txt for n in tri)]
if not tris_filtered:
    raise RuntimeError("Cap triangle amb tots els nodes presents al TXT. Revisa NODE_FILTER en el generador del TXT.")

# Defineix ordre de nodes (basat en IDs presents al TXT i necessaris per a la malla)
used_node_ids = sorted(nodes_in_txt)
# Map ID -> index de vèrtex
node_id_to_index = {nid: i for i, nid in enumerate(used_node_ids)}

# Construeix llista de vèrtex del BASIS amb el 1r bloc
basis_coords = []
first_block = xyz_blocks[0]
for nid in used_node_ids:
    if nid not in first_block:
        raise RuntimeError(f"Node {nid} no és al primer bloc del TXT.")
    x, y, z = first_block[nid]
    if not USE_XY_FROM_XYZ and (nid in fs_coords):
        fx, fy, fz = fs_coords[nid]
        x, y = fx, fy
    basis_coords.append((x, y, z * EXTRA_Z_SCALE + ADD_Z_OFFSET))

# Construeix llista de cares amb índexs de vèrtex
faces = []
for (n1, n2, n3) in tris_filtered:
    try:
        faces.append((node_id_to_index[n1], node_id_to_index[n2], node_id_to_index[n3]))
    except KeyError:
        pass

# --- CREAR MALLA I OBJECTE ----------------------------------------------------
mesh = bpy.data.meshes.new("FreeSurfaceMesh")
mesh.from_pydata(basis_coords, [], faces)
mesh.update()

obj = bpy.data.objects.new("FreeSurfaceObject", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# --- SHAPE KEYS PER TIME STEPS ------------------------------------------------
if not obj.data.shape_keys:
    obj.shape_key_add(name="Basis")

# guardem posicions del basis per referència
basis_positions = [v.co.copy() for v in obj.data.vertices]

# crea una shape key per cada bloc (a partir del 2n, ja que el 1r és el basis)
for i, block in enumerate(xyz_blocks):
    if i == 0:
        continue
    key = obj.shape_key_add(name=f"Step_{i}")
    for nid, (x, y, z) in block.items():
        idx = node_id_to_index.get(nid)
        if idx is None:
            continue
        if not USE_XY_FROM_XYZ and (nid in fs_coords):
            x, y, _ = fs_coords[nid]
        key.data[idx].co = Vector((x, y, z * EXTRA_Z_SCALE + ADD_Z_OFFSET))

# --- ANIMACIÓ (1 shape key activa per frame) ----------------------------------
scene = bpy.context.scene

# neteja animació anterior
if obj.data.shape_keys and obj.data.shape_keys.animation_data:
    obj.data.shape_keys.animation_data_clear()

def set_scene_fps_from_times(times):
    # si hi ha temps uniformes, ajusta fps = 1/dt mig, sinó deixa 30
    if len(times) < 2:
        scene.render.fps = 30; scene.render.fps_base = 1.0; return
    dts = [times[i+1]-times[i] for i in range(len(times)-1) if times[i+1] > times[i]]
    if not dts:
        scene.render.fps = 30; scene.render.fps_base = 1.0; return
    mean_dt = sum(dts)/len(dts)
    if mean_dt <= 0: mean_dt = 1/30
    fps = 1.0/mean_dt
    scene.render.fps = max(1, int(round(fps)))
    scene.render.fps_base = 1.0

set_scene_fps_from_times(times)

N = len(xyz_blocks)
frame_numbers = [1 + i for i in range(N)]  # 1 frame per bloc; simple i robust

# inicis a 0
for kb in obj.data.shape_keys.key_blocks:
    if kb.name != "Basis":
        kb.value = 0.0
        kb.keyframe_insert(data_path="value", frame=frame_numbers[0]-1)

for i, frame in enumerate(frame_numbers):
    for j, kb in enumerate(obj.data.shape_keys.key_blocks):
        if kb.name == "Basis":
            continue
        kb.value = 1.0 if (j - 1) == i else 0.0
        kb.keyframe_insert(data_path="value", frame=frame)

scene.frame_start = frame_numbers[0]
scene.frame_end   = frame_numbers[-1]

print(f"✅ Importat {N} time steps del fitxer XYZ. Frames: {scene.frame_start}–{scene.frame_end}")

# --- MATERIALS I MARCADORS (ESFERES multicolor) -------------------------------
if MATERIALS_AND_MARKERS:

    def make_emissive_material(name, rgba, strength=1.8):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            ns = mat.node_tree.nodes; ls = mat.node_tree.links
            for n in list(ns): ns.remove(n)
            out_n = ns.new(type='ShaderNodeOutputMaterial')
            emi_n = ns.new(type='ShaderNodeEmission')
            emi_n.inputs['Color'].default_value = rgba
            emi_n.inputs['Strength'].default_value = strength
            ls.new(emi_n.outputs['Emission'], out_n.inputs['Surface'])
        else:
            # per si vols retocar valors en reexecutar
            emi_n = mat.node_tree.nodes.get("Emission") or \
                    next((n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeEmission'), None)
            if emi_n:
                emi_n.inputs['Color'].default_value = rgba
                emi_n.inputs['Strength'].default_value = strength
        return mat

    # Paleta de colors càlids/vius que contrasten amb el blau del mar (RGBA 0..1)
    # Si hi ha més boies que colors, la paleta es repeteix.
    PALETTE = [
        (1.00, 1.00, 0.00, 1.00),  # groc
        (0.00, 1.00, 0.00, 1.00),  # verd
        (1.00, 0.00, 0.00, 1.00),  # vermell
        (1.00, 0.75, 0.80, 1.00),  # rosa
        
    ]

    # Crea materials per a la paleta
    palette_mats = [make_emissive_material(f"BuoyColor_{i}", rgba, strength=2.0)
                    for i, rgba in enumerate(PALETTE)]

    # ESFERES sobre superfície als nodes seleccionats (lleugerament més petites)
    sphere_objs = {}
    for i, nid in enumerate(SELECTED_NODE_IDS):
        idx = node_id_to_index.get(nid)
        if idx is None:
            print(f"⚠️ Node {nid} no trobat; sense esfera.")
            continue
        x, y, z = basis_coords[idx]
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(x, y, z))  # abans 0.5
        sp = bpy.context.object
        sp.name = f"SurfacePoint_Node_{nid}"

        # Assigna un color diferent a cada boia
        mat = palette_mats[i % len(palette_mats)]
        if sp.data.materials:
            sp.data.materials[0] = mat
        else:
            sp.data.materials.append(mat)

        sphere_objs[nid] = sp

    # --- BOIES (ESFERES) SENSE HANDLERS: BAKE PER FRAME -----------------------
    # Opcional: neteja animacions anteriors de les boies
    for sp in sphere_objs.values():
        if sp.animation_data:
            sp.animation_data_clear()

    # Guarda el frame actual per restaurar-lo al final
    _frame0 = scene.frame_current

    # Mostreja els frames “clau” dels blocs (o canvia per range(...) si vols tots)
    frames_a_bakejar = frame_numbers

    for frame in frames_a_bakejar:
        scene.frame_set(frame)

        # Avaluem la malla amb les shape keys del frame actual
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh_eval = obj_eval.to_mesh()
        try:
            Mw = obj_eval.matrix_world.copy()  # per convertir a espai món si cal
            for nid, sp in sphere_objs.items():
                idx = node_id_to_index.get(nid)
                if idx is None:
                    continue
                co_local = mesh_eval.vertices[idx].co
                co_world = Mw @ co_local  # respecta transformacions de l'objecte
                sp.location = co_world
                sp.keyframe_insert(data_path="location", frame=frame)
        finally:
            obj_eval.to_mesh_clear()

    # Restaura el frame inicial
    scene.frame_set(_frame0)

    # Ajusta la interpolació de les corbes de les boies
    if BAKE_INTERPOLATION:
        for sp in sphere_objs.values():
            ad = sp.animation_data
            if ad and ad.action:
                for fcu in ad.action.fcurves:
                    if fcu.data_path == "location":
                        for kp in fcu.keyframe_points:
                            kp.interpolation = BAKE_INTERPOLATION


