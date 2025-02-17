import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Importar les dades de la malla des de malla_simple_data
import b_1_1_malla_simple_tri_data as mesh_data 

# Extreure nodes i connectivitats, o sigui, elements
nodes = np.array(mesh_data.nodes)  
elements = np.array(mesh_data.elements)  

# Afegir coordenades Z com a 0, ja que la malla només és 2D
nodes_3d = np.hstack([nodes, np.zeros((nodes.shape[0], 1))])  # Afegir columna de 0's per a Z

# Crear la triangulació (en 2D, però representarem en 3D)
triangulacio = tri.Triangulation(nodes_3d[:, 0], nodes_3d[:, 1], elements)

# Crear la figura per al gràfic 3D
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Dibuixar els elements (triangles) en 3D
for i in range(len(elements)):
    # Obtenir les coordenades dels tres nodes de cada triangle
    x = nodes_3d[elements[i], 0]
    y = nodes_3d[elements[i], 1]
    z = nodes_3d[elements[i], 2]  # Aquí Z serà 0 per tots els nodes
    
    # Tancar el triangle afegint el primer node al final
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    z = np.append(z, z[0])
    
    # Dibuixar el triangle
    ax.plot(x, y, z, color='black')

# Dibuixar els nodes (també en 3D)
ax.scatter(nodes_3d[:, 0], nodes_3d[:, 1], nodes_3d[:, 2], color='red')

# Ajustar les etiquetes dels eixos
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Mostrar la figura
plt.show()