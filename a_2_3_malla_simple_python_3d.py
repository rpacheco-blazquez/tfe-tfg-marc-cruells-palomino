import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import a_1_1_malla_simple_data as mesh_data

# Definir nodes i elements (correcció de l'assignació)
nodes = np.array(mesh_data.nodes)
elements = np.array(mesh_data.elements)

nodes_3d = np.hstack([nodes, np.zeros((nodes.shape[0], 1))])  # Afegir columna de 0's per a Z

# Crear la figura i els eixos 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for elem in elements:
    # Obtenir les coordenades dels nodes de cada element (quadrilàter)
    x = np.append(nodes_3d[elem][:, 0], nodes_3d[elem][0, 0])  # Tancar el quadrilàter
    y = np.append(nodes_3d[elem][:, 1], nodes_3d[elem][0, 1])
    z = np.append(nodes_3d[elem][:, 2], nodes_3d[elem][0, 2])  # Aquí Z serà 0 per a tots els nodes

    # Dibuixar el quadrilàter en 3D
    ax.plot(x, y, z, color='black')

#Dibuixar els nodes en 3D
ax.scatter(nodes_3d[:, 0], nodes_3d[:, 1], nodes_3d[:, 2], color='red')


# Ajustar aspecte
plt.grid(True)
plt.show()
