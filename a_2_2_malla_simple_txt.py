# Importar llibreries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Definir la funció per llegir el fitxer
def llegir_malla():
    # Obrir el fitxer per llegir-lo
    with open("a_1_2_malla_simple_data.txt", 'r') as file:   
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
                n1, n2, n3, n4 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])  # Índexs dels nodes
                elements.append([n1, n2, n3, n4])  # Afegir a la llista d'elements

    # Convertir les llistes a arrays de NumPy
    nodes_array = np.array(nodes)
    elements_array = np.array(elements)

    return nodes_array, elements_array

nodes,elements = llegir_malla()

# Crear la figura
fig, ax = plt.subplots()

# Dibuixar elements com a quadrilàters
for elem in elements:
    x = np.append(nodes[elem, 0], nodes[elem][0, 0])  # per tancar el polígon
    y = np.append(nodes[elem, 1], nodes[elem][0, 1]) 
    polygon = patches.Polygon(np.c_[x, y], edgecolor='black', facecolor='lightgray', alpha=0.5)
    ax.add_patch(polygon)

# Dibuixar nodes
ax.scatter(nodes[:, 0], nodes[:, 1], color='red', zorder=3)

# Ajustar aspecte
ax.set_aspect('equal')
plt.grid(True)
plt.show()