
# Solver de Richards 2D FEM linear (triângulos), sem FEniCS
import numpy as np
import meshio
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Parâmetros do problema
Ksat = 1e-6  # condutividade hidráulica saturada [m/s]
g = 9.81  # gravidade [m/s^2]

# Ler malha 2D (malha_dominio.xdmf)
malha = meshio.read("malha_dominio.xdmf")
points = malha.points
triangles = None
for cell_block in malha.cells:
    if cell_block.type == "triangle":
        triangles = cell_block.data
        break
if triangles is None:
    raise RuntimeError("Malha não contém elementos triangulares!")
Nnodes = points.shape[0]
Nelems = triangles.shape[0]

# Condições de contorno: exemplo - h=0 no topo (y=max), h=-2 na base (y=min)
y = points[:,1]
h_top = 0.0
h_bottom = -2.0
tol = 1e-8
nodes_top = np.where(np.abs(y - y.max()) < tol)[0]
nodes_bottom = np.where(np.abs(y - y.min()) < tol)[0]

# Montagem da matriz global
A = lil_matrix((Nnodes, Nnodes))
b = np.zeros(Nnodes)

for elem in triangles:
    node_ids = elem
    coords = points[node_ids, :2]
    # Matriz de área do triângulo
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area = 0.5 * np.abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    # Gradientes das funções de forma
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    B = np.array([[b1, b2, b3], [c1, c2, c3]]) / (2*area)
    # Matriz local de rigidez (K constante)
    k_local = Ksat * area * (B.T @ B)
    for i in range(3):
        for j in range(3):
            A[node_ids[i], node_ids[j]] += k_local[i, j]
    # Fonte (zero neste exemplo)

# Vetor de carga (fonte nula)
# b já é zero

# Aplicar condições de contorno de Dirichlet
fixed_nodes = np.concatenate([nodes_top, nodes_bottom])
fixed_values = np.concatenate([
    np.full(nodes_top.shape, h_top),
    np.full(nodes_bottom.shape, h_bottom)
])

for idx, node in enumerate(fixed_nodes):
    A[node, :] = 0
    A[node, node] = 1
    b[node] = fixed_values[idx]

# Resolver sistema linear
h = spsolve(A.tocsr(), b)

# Exportar resultado para VTU
cells = [("triangle", triangles)]
mesh = meshio.Mesh(points, cells, point_data={"h": h})
mesh.write("saida_solver2.vtu")
print("Simulação finalizada. Resultado salvo em saida_solver2.vtu.")
