# Importar llibreries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Importar les dades de la malla des de malla_simple_data
import b_1_1_malla_simple_tri_data as mesh_data 

# Extreure nodes i connectivitats, o sigui, elements
nodes = np.array(mesh_data.nodes)  
elements = np.array(mesh_data.elements)  

# Crear la triangulació
triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

# Crear (plot) la malla
plt.figure()
plt.triplot(triangulation)
plt.show()
