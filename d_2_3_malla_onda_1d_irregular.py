import numpy as np
import matplotlib.pyplot as plt

# Importar les dades de la malla des de malla_simple_data
import d_1_1_malla_onda_1d_irregular_data as mesh_data 

# Extreure nodes i connectivitats, o sigui, elements
nodes = np.array(mesh_data.nodes)  
elements = np.array(mesh_data.elements)  

# Crear la figura per al gràfic 3D
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Dibuixar els elements (triangles) en 3D
for i in range(len(elements)):
    # Obtenir les coordenades dels tres nodes de cada triangle
    x = nodes[elements[i], 0]
    y = nodes[elements[i], 1]
    z = nodes[elements[i], 2] 
    
    # Tancar el triangle afegint el primer node al final
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    z = np.append(z, z[0])
    
    # Dibuixar el triangle
    ax.plot(x, y, z, color='black')

# Dibuixar els nodes (també en 3D)
ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='red')

# Ajustar les etiquetes dels eixos
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Mostrar la figura
plt.show()