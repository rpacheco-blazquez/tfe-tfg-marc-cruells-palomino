import numpy as np
import os


def save_py(nodes, elements, save_path="saved-mesh/mesh.py"):
    """Saves nodal and connectivity matrix to a specified path."""

    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write("import numpy as np\n\n")
        f.write(f"nodes = np.array({repr(nodes.tolist())})\n")
        f.write(f"elements = np.array({repr(elements.tolist())})\n")
    print(f"Mesh data saved in {save_path}")

def save_txt(nodes, elements, save_path="saved-mesh/mesh.txt"):
    """Saves the nodal and connectivity matrices in a human-readable text format."""
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        # Writing the nodal matrix
        f.write("nodal matrix\n")
        f.write("begin values\n")
        for i, row in enumerate(nodes):
            f.write(f"{i}, " + ", ".join(map(str, row)) + "\n")
        f.write("end values\n\n")

        # Writing the connectivity matrix
        f.write("connectivity matrix\n")
        f.write("begin values\n")
        for i, row in enumerate(elements):
            f.write(f"{i}, " + ", ".join(map(str, row)) + "\n")
        f.write("end values\n")

    print(f"Mesh data saved in {save_path}")