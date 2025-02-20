import numpy as np
import os
from utils import plotting  # Import plotting from utils folder
from utils import meshing  # Import meshing module


def generate_grid_2d_irregular(length, width, nx, ny, amplitudes, frequencies):
    """Genera una malla 2D con desplazamiento irregular usando un espectro polar de ondas."""
    # Crear puntos equidistantes en X e Y
    x_vals = np.linspace(0, length, nx + 1)
    y_vals = np.linspace(0, width, ny + 1)

    # Convertir la malla de coordenadas cartesianas a polares
    nodes = []

    # Inicializamos las fases aleatorias para el espectro polar
    random_phases = np.random.uniform(
        0, 2 * np.pi, (ny + 1, nx + 1)
    )  # Fases aleatorias para cada punto

    for y in y_vals:
        for x in x_vals:
            # Inicializar el desplazamiento z a cero
            z = 0

            # Calcular el radio y el ángulo de cada punto
            r = np.sqrt(x**2 + y**2)  # Radio en coordenadas polares
            theta = np.arctan2(y, x)  # Ángulo en coordenadas polares

            # Sumamos las contribuciones de cada onda (amplitud y frecuencia por dirección)
            for i in range(len(amplitudes)):
                for j in range(len(frequencies)):
                    # Obtenemos la amplitud y frecuencia para cada dirección polar
                    A_ij = amplitudes[i, j]  # Amplitud para el par de frecuencias (i,j)
                    w_ij = frequencies[
                        i, j
                    ]  # Frecuencia para el par de frecuencias (i,j)
                    phi_ij = random_phases[i, j]  # Fase aleatoria para cada par (i,j)

                    # Contribución de la onda correspondiente al desplazamiento en Z
                    z += A_ij * np.sin(2 * np.pi * w_ij * r + phi_ij)

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
    width = 10.0  # Longitud en dirección Y
    nx = 20  # Número de divisiones en X
    ny = 20  # Número de divisiones en Y

    # Definir las amplitudes y frecuencias para cada dirección (matrices 2D)
    amplitudes = np.array(
        [[0.5, 0.8], [0.1, 0.05]]
    )  # Matriz de amplitudes para cada par de frecuencias
    frequencies = np.array(
        [[0.1, 0.2], [0.5, 0.8]]
    )  # Matriz de frecuencias para cada par de direcciones

    # Generar los nodos y elementos de la malla
    nodes, elements = generate_grid_2d_irregular(
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
