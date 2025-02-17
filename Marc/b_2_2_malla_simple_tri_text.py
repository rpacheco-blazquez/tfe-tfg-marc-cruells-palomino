# Importar llibreries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Definir la funció per llegir el fitxer
def llegir_malla():
    # Obrir el fitxer per llegir-lo
    with open("b_1_2_malla_simple_tri_data.txt", 'r') as file:   
        lines = file.readlines()

    nodes = []  # Crear llista buida per als nodes
    elements = []  # Crear llista buida per als elements
    llegir_nodes = False
    llegir_elements = False

    # Processar cada línia del fitxer
    for line in lines:
        line = line.strip()  # Perquè la línia estigui neta

        # Detectar les seccions del fitxer
        if line.startswith('nodal matrix'):
            llegir_nodes = True
            llegir_elements = False
            continue
        elif line.startswith('connectivity matrix'):
            llegir_nodes = False
            llegir_elements = True
            continue
        elif line.startswith('begin values') or line.startswith('end values'):
            continue  # Ometre els límits de les llistes

        # Llegir nodes
        if llegir_nodes:
            parts = line.split(',')  # Separar per comes
            if len(parts) >= 3:  # Assegurar que hi ha prou dades
                x, y = float(parts[1]), float(parts[2])  # Extraure coordenades
                nodes.append([x, y])  # Afegir a la llista de nodes

        # Llegir els elements de la connectivitat
        if llegir_elements:
            parts = line.split(',')  # Separar per comes
            if len(parts) >= 4:  # Assegurar que hi ha prou dades
                n1, n2, n3 = int(parts[1]), int(parts[2]), int(parts[3])  # Índexs dels nodes
                elements.append([n1, n2, n3])  # Afegir a la llista d'elements

    # Convertir les llistes a arrays de NumPy
    nodes_array = np.array(nodes)
    elements_array = np.array(elements)

    return nodes_array, elements_array

# Llegir la malla des del fitxer
nodes, elements_tri = llegir_malla()

# Crear la triangulació
triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements_tri)

# Crear (plot) la malla
plt.figure()
plt.triplot(triangulation)  
plt.show()
