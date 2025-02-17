import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import a_1_1_malla_simple_data as mesh_data

# Definir nodes i elements (correcció de l'assignació)
nodes = np.array(mesh_data.nodes)
elements = np.array(mesh_data.elements)

# Crear la figura
fig, ax = plt.subplots()

# Dibuixar elements com a quadrilàters
for elem in elements:
    x = np.append(nodes[elem][:, 0], nodes[elem][0, 0])  # per tancar el polígon
    y = np.append(nodes[elem][:, 1], nodes[elem][0, 1]) 
    polygon = patches.Polygon(np.c_[x, y], edgecolor='black', facecolor='lightgray', alpha=0.5)
    ax.add_patch(polygon)

# Dibuixar nodes
ax.scatter(nodes[:, 0], nodes[:, 1], color='red', zorder=3)

# Ajustar aspecte
ax.set_aspect('equal')
plt.grid(True)
plt.show()
