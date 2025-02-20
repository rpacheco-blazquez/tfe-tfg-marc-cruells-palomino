import numpy as np
import os
from utils import plotting  # Import plotting from utils folder
from utils import meshing  # Import meshing module


def generate_grid(length, width, nx, ny):
    """Generates a 2D grid of points and connectivity matrix (triangular mesh)."""
    # Create linearly spaced points
    x_vals = np.linspace(0, length, nx + 1)
    y_vals = np.linspace(0, width, ny + 1)

    # Create nodal matrix (list of [x, y] coordinates)
    nodes = np.array([[x, y] for y in y_vals for x in x_vals])

    # Create connectivity matrix (triangular elements)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n1 + (nx + 1)
            n4 = n3 + 1

            # Create two triangles from each quadrilateral (n1, n2, n3) and (n2, n3, n4)
            elements.append([n1, n2, n3])  # First triangle
            elements.append([n2, n3, n4])  # Second triangle

    return nodes, np.array(elements)


if __name__ == "__main__":
    # User-defined parameters
    length = 10.0  # X direction length
    width = 5.0  # Y direction width
    nx = 4  # Number of divisions in X
    ny = 2  # Number of divisions in Y

    nodes, elements = generate_grid(length, width, nx, ny)

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