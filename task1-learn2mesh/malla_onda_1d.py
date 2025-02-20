import numpy as np
import os
from utils import plotting  # Import plotting from utils folder
from utils import meshing  # Import meshing module

def generate_grid_3d(length, width, nx, ny, amplitude=1.0):
    """Genera una malla 3D con desplazamiento sinusoidal en la dirección X."""
    # Crear puntos equidistantes en X e Y
    x_vals = np.linspace(0, length, nx + 1)
    y_vals = np.linspace(0, width, ny + 1)
    
    # Crear la matriz nodal (lista de coordenadas [x, y, z])
    nodes = []
    for y in y_vals:
        for x in x_vals:
            z = amplitude * np.sin(2 * np.pi * x / length)  # Desplazamiento sinusoidal en Z
            nodes.append([x, y, z])
    
    # Convertir la lista de nodos a un array numpy
    nodes = np.array(nodes)

    # Crear la matriz de conectividad (elementos triangulares)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n1 + (nx + 1)
            n4 = n3 + 1
            
            # Crear dos triángulos por cada cuadrilátero
            elements.append([n1, n2, n3])  # Primer triángulo
            elements.append([n2, n3, n4])  # Segundo triángulo

    return nodes, np.array(elements)

if __name__ == "__main__":
    # Parámetros definidos por el usuario
    length = 10.0  # Longitud en dirección X
    width = 5.0   # Longitud en dirección Y
    nx = 20  # Número de divisiones en X (más divisiones para ver la onda más detallada)
    ny = 5   # Número de divisiones en Y

    amplitude = 2.0  # Amplitud de la onda sinusoidal

    # Generar los nodos y elementos de la malla
    nodes, elements = generate_grid_3d(length, width, nx, ny, amplitude)

 # Get the path where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

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

    # Use the plotting module to plot and save the 2D and 3D grids
    plotting.plot_grid(nodes, elements, save_path=save_img_path)  # Save 2D plot
    plotting.plot3D(
        nodes, elements, show_nodes=True, save_path=save_img_3d_path
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