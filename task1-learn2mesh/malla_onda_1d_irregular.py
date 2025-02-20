import numpy as np
import os
from utils import plotting  # Import plotting from utils folder
from utils import meshing  # Import meshing module


def generate_grid_3d_irregular(length, width, nx, ny, amplitudes, frequencies):
    """Genera una malla 3D con desplazamiento irregular en la dirección X usando una suma de ondas."""
    # Crear puntos equidistantes en X e Y
    x_vals = np.linspace(0, length, nx + 1)
    y_vals = np.linspace(0, width, ny + 1)

    # Crear la matriz nodal (lista de coordenadas [x, y, z])
    nodes = []

    # Generate random phases for each frequency, once per call
    phases = np.random.uniform(
        0, 2 * np.pi, len(amplitudes)
    )  # Random phases for all frequencies

    for y in y_vals:
        for x in x_vals:
            # Inicializar el desplazamiento z a cero
            z = 0

            # Sumar las contribuciones de cada onda
            for i in range(len(amplitudes)):
                # Frecuencia
                w_i = frequencies[i]
                # Fase aleatoria para la frecuencia i
                phi_i = phases[
                    i
                ]  # Use the pre-generated random phase for this frequency
                # Amplitud
                A_i = amplitudes[i]
                # Sumar la onda correspondiente al desplazamiento en Z
                z += A_i * np.sin(2 * np.pi * w_i * x + phi_i)

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
    width = 5.0  # Longitud en dirección Y
    nx = 20  # Número de divisiones en X (más divisiones para ver la onda más detallada)
    ny = 5  # Número de divisiones en Y

    # Definir las amplitudes y frecuencias de las ondas
    amplitudes = [0.5, 0.3, 0.1]  # Amplitudes para cada frecuencia
    frequencies = [0.1, 0.2, 0.8]  # Frecuencias de las ondas

    # Generar los nodos y elementos de la malla
    nodes, elements = generate_grid_3d_irregular(
        length, width, nx, ny, amplitudes, frequencies
    )

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
    save_data_path = os.path.join(script_dir, "saved-mesh", f"{script_name}_data.py")

    # Save the mesh data (nodes and elements)
    meshing.save_py(nodes, elements, save_data_path)

    # Add this to the end of your script to prevent immediate termination
    input("\n\nPress Enter to exit and close the terminal...")
