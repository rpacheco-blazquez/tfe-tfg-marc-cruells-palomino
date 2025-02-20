import numpy as np
import matplotlib.pyplot as plt
import os


def plot_grid(nodes, elements, save_path=None):
    """Plots the generated 2D grid using matplotlib and optionally saves it."""
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()

    # Plot the mesh edges
    for element in elements: 
        x_coords = nodes[element, 0]
        y_coords = nodes[element, 1]
        x_coords = np.append(x_coords, x_coords[0])  # Close the quadrilateral
        y_coords = np.append(y_coords, y_coords[0])
        ax.plot(x_coords, y_coords, "k")  # Black grid lines

    # Plot the nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], color="red", s=10, label="Nodes")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("2D Grid")
    ax.legend()
    ax.grid(True)

    # Set aspect ratio to 'equal' to ensure same scale for both axes
    ax.set_aspect("equal", adjustable="box")

    # Show the plot
    plt.show()

    # Save the plot if a path is provided
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(
                os.path.dirname(save_path)
            )  # Create directory if it doesn't exist
        fig.savefig(save_path, format="png")  # Save the plot as PNG
        print(f"Plot saved to {save_path}")


def plot3D(nodes, elements, show_nodes=True, save_path=None):
    """Plots the 3D representation of the mesh and optionally saves it."""
    plt.ion()  # Enable interactive mode

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # If nodes have only 2 columns (x, y), add z=0
    if nodes.shape[1] == 2:
        nodes = np.hstack((nodes, np.zeros((nodes.shape[0], 1))))

    # Plot elements (the mesh connections)
    for element in elements:
        x_coords = nodes[element, 0]
        y_coords = nodes[element, 1]
        z_coords = nodes[element, 2]

        # Close the shape by appending the first node at the end
        x_coords = np.append(x_coords, x_coords[0])
        y_coords = np.append(y_coords, y_coords[0])
        z_coords = np.append(z_coords, z_coords[0])

        ax.plot(x_coords, y_coords, z_coords, "k")  # Black edges

    # Plot nodes if the flag is True
    if show_nodes:
        ax.scatter(
            nodes[:, 0], nodes[:, 1], nodes[:, 2], color="red", s=20, label="Nodes"
        )

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Mesh Visualization")
    ax.legend()

    # Show the plot
    plt.show()

    # Save the plot if a path is provided
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(
                os.path.dirname(save_path)
            )  # Create directory if it doesn't exist
        fig.savefig(save_path, format="png")  # Save the plot as PNG
        print(f"3D Plot saved to {save_path}")

    # Keep the plot open without blocking the execution
    plt.draw()  # Redraw the plot
