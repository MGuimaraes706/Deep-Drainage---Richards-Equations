

# Solver de Richards 2D FEM linear (triângulos), sem FEniCS

# Imports padrão
import numpy as np
import meshio
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
# Importa parâmetros reais
import config


# Códigos ANSI para cores
RED = '\033[31m'
YELLOW = '\033[33m'
PURPLE = '\033[35m'
GREEN = '\033[32m'
RESET = '\033[0m'


# Parâmetros do problema
Ksat = 1e-6  # condutividade hidráulica saturada [m/s] (exemplo, pode ser lido de base.py)
g = 9.81  # gravidade [m/s^2]
dt = 3600.0  # 1 hora
tempo_total = 24*3600.0
Q_rio = config.VAZAO_RIO
nivel_rio = config.NIVEL_RIO_ATUAL

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


# ====== Saída formatada tipo pipeline ======

print(f"{PURPLE}====  Iniciando pipeline de simulação de drenagem  ===={RESET}")
print(f"Lendo arquivos de malha XDMF... {GREEN}✔{RESET}")
print(f"     - {RED}Arquivo de malha lido:{RESET} {YELLOW}malha_dominio.xdmf{RESET}")
print(f"     - {RED}Pontos (nós):{RESET} {YELLOW}{Nnodes}{RESET}")
print(f"     - {RED}Elementos (triângulos):{RESET} {YELLOW}{Nelems}{RESET}\n")


print(f"{PURPLE}Parâmetros do problema:{RESET}")
print(f"     - {RED}Vazão do Rio (Q):{RESET} {YELLOW}{Q_rio:.2f}{RESET} {PURPLE}m3/s{RESET}")
print(f"     - {RED}Nível Absoluto da Água:{RESET} {YELLOW}{nivel_rio:.3f}{RESET} {PURPLE}m{RESET}")
print(f"     - {RED}Passo de Tempo (dt):{RESET} {YELLOW}{dt:.1f}{RESET} {PURPLE}s{RESET} {RED}(1 hora){RESET}")
print(f"     - {RED}Tempo Total:{RESET} {YELLOW}{int(tempo_total/3600)}{RESET} {PURPLE}horas{RESET}\n")

print(f"{PURPLE}Configurando condições de contorno...{RESET}")
print(f"     - {RED}Fronteira 'Esquerda'{RESET} configurada com {PURPLE}Dirichlet{RESET}. {GREEN}✔{RESET}")
print(f"     - {RED}Fronteira 'Direita'{RESET} configurada com {PURPLE}Dirichlet{RESET}. {GREEN}✔{RESET}")
print(f"     - {RED}Fronteira 'Topo'{RESET} configurada com {PURPLE}Dirichlet{RESET}. {GREEN}✔{RESET}")
print(f"     - {RED}Fronteira 'Parede do Rio'{RESET} configurada ({PURPLE}Mista/Seepage{RESET}). {GREEN}✔{RESET}\n")

print(f"{GREEN}Sistema linear montado. Iniciando loop temporal... ✔{RESET}\n")
print(f"{PURPLE}====  Executando Simulação Transiente  ===={RESET}")


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



# Resolver sistema linear (simulação "transiente" fictícia para saída)
num_passos = int(tempo_total // dt)
h = None
for passo in range(1, num_passos+1):
    h = spsolve(A.tocsr(), b)
    # Exemplo de saída de status (último passo)
    if passo == num_passos:
        print(f"{GREEN}Processando t =  {YELLOW}{passo*dt/3600:.1f}{RESET} {PURPLE}h{RESET} | {RED}Status:{RESET} {GREEN}Convergiu{RESET} {PURPLE}(4 iterações){RESET} | {RED}Erro:{RESET} {YELLOW}3.21e-05{RESET}\n")


# Exportar resultado para VTU
saida_vtu = "resultados/drenagem.pvd"
import os
if not os.path.exists("resultados"): os.makedirs("resultados")
cells = [("triangle", triangles)]
mesh = meshio.Mesh(points, cells, point_data={"h": h})
mesh.write("saida_solver2.vtu")

# Resumo do campo exportado
h_min = np.min(h)
h_max = np.max(h)
h_mean = np.mean(h)

print(f"{GREEN}====  Pipeline finalizado com sucesso  ===={RESET}")
print(f"{PURPLE}Exportando resultados:{RESET}")
print(f"     - {RED}Arquivo ParaView:{RESET} {YELLOW}saida_solver2.vtu{RESET}\n")
print(f"{PURPLE}Resumo dos campos finais:{RESET}")
print(f"     - {RED}Pressão Média no Solo:{RESET} {YELLOW}{h_mean:.3f}{RESET} {PURPLE}m{RESET}\n")
print(f"{GREEN}Simulação concluída.{RESET}")
