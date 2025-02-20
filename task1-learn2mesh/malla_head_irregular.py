import numpy as np
import os
from stl import mesh  # Importar la librería para leer el archivo STL
from utils import meshing  # Asumimos que tienes una función de guardado
from utils import plotting  # Para visualización (si la necesitas)


def load_stl_mesh(stl_file):
    """Carga un archivo STL y extrae los vértices y las caras (triángulos)."""
    # Cargar el archivo STL
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # Obtener los vértices y las caras
    vertices = stl_mesh.vectors.reshape(
        -1, 3
    )  # Aplanamos para tener una lista de vértices
    elements = []

    # Cada cara del STL está representada por 3 vértices, que corresponden a un triángulo
    for i in range(0, len(vertices), 3):
        elements.append([i, i + 1, i + 2])

    # Convertir a arrays numpy para facilitar el procesamiento
    vertices = np.array(vertices)
    elements = np.array(elements)

    return vertices, elements


if __name__ == "__main__":

    # Obtener la ruta donde está el script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ruta al archivo STL (cambia esta ruta a la correcta en tu sistema)
    stl_file = os.path.join(
        script_dir, "saved-mesh", "Statue Draft.stl"
    )  #  Aquí pones la ruta del archivo STL

    # Cargar la malla desde el archivo STL
    nodes, elements = load_stl_mesh(stl_file)

    # Get the base name of the script without the extension
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Specify the save paths for images relative to the script's directory
    save_img_path = os.path.join(
        script_dir, "docs", "img", f"{script_name}_grid_plot.png"
    )
    save_img_3d_path = os.path.join(
        script_dir, "docs", "img", f"{script_name}_3d_plot.png"
    )

    # Ensure the directory exists
    if not os.path.exists(os.path.dirname(save_img_path)):
        os.makedirs(os.path.dirname(save_img_path))

    # Usar el módulo de trazado para visualizar la malla 3D
    plotting.plot3D(
        nodes, elements, show_nodes=False, save_path=save_img_3d_path
    )  # Save 3D plot

    # Specify the save path for the mesh data
    save_data_path = os.path.join(
        script_dir, "saved-mesh", f"{script_name}_data.py"
    )
    save_data_path_txt = os.path.join(
        script_dir, "saved-mesh", f"{script_name}_data.txt"
    )

    # Save the mesh data (nodes and elements)
    meshing.save_py(nodes, elements, save_data_path)

    # Save mesh data in text format
    meshing.save_txt(nodes, elements, save_path=save_data_path_txt)

    # Add this to the end of your script to prevent immediate termination
    input("\n\nPress Enter to exit and close the terminal...")
