# Solver de Richards 2D FEM linear (triângulos), sem FEniCS

# Imports padrão
import numpy as np
import meshio
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import os
import sys

# Importa parâmetros reais
import config
import base

# Códigos ANSI para cores
RED = '\033[31m'
YELLOW = '\033[33m'
PURPLE = '\033[35m'
GREEN = '\033[32m'
RESET = '\033[0m'

# ==============================================================================
# 1. SETUP E LEITURA DA MALHA
# ==============================================================================
print(f"{YELLOW}{'='*80}{RESET}")
print(f"{PURPLE}RELATÓRIO DE EXECUÇÃO DA SIMULAÇÃO (SCIPY FEM){RESET}\n")

# Ler malha 2D (malha_dominio.xdmf)
try:
    # Ler malha do domínio
    malha = meshio.read("malha_dominio.xdmf")
    points = malha.points[:, :2] # Pegar apenas X, Y
    
    # Extrair triângulos e tags de material
    triangles = None
    material_tags = None
    
    for i, cell_block in enumerate(malha.cells):
        if cell_block.type == "triangle":
            triangles = cell_block.data
            # Tenta pegar os dados do material associados
            if "material_tag" in malha.cell_data:
                material_tags = malha.cell_data["material_tag"][i]
            break
            
    if triangles is None:
        raise RuntimeError("Malha não contém elementos triangulares!")
        
    # Ler malha das bordas (para condições de contorno)
    malha_bordas = meshio.read("malha_bordas.xdmf")
    lines = None
    borda_tags = None
    
    for i, cell_block in enumerate(malha_bordas.cells):
        if cell_block.type == "line":
            lines = cell_block.data
            if "borda_tag" in malha_bordas.cell_data:
                borda_tags = malha_bordas.cell_data["borda_tag"][i]
            break

except Exception as e:
    print(f"{RED}ERRO AO LER MALHAS: {e}{RESET}")
    print("Verifique se rodou o 'gerador_malha.py' antes.")
    sys.exit(1)

Nnodes = points.shape[0]
Nelems = triangles.shape[0]

print(f"{PURPLE}[DADOS DE ENTRADA]{RESET}")
print(f"Nós: {YELLOW}{Nnodes}{RESET} | Elementos: {YELLOW}{Nelems}{RESET}")
print(f"Materiais identificados: {YELLOW}{np.unique(material_tags)}{RESET}")

# ==============================================================================
# 2. PARÂMETROS
# ==============================================================================
dt = 3600.0  # 1 hora
tempo_total = 24 * 3600.0
num_passos = int(tempo_total // dt)

# ==============================================================================
# 3. IDENTIFICAÇÃO DE NÓS DE CONTORNO (DIRICHLET)
# ==============================================================================
# Mapear nós para tags de borda
# Criamos um dicionário: tag_borda -> lista de nós
boundary_nodes = {}

if lines is not None and borda_tags is not None:
    for i, line in enumerate(lines):
        tag = borda_tags[i]
        n1, n2 = line
        if tag not in boundary_nodes:
            boundary_nodes[tag] = set()
        boundary_nodes[tag].add(n1)
        boundary_nodes[tag].add(n2)

# Função para aplicar BCs
def get_dirichlet_values(points, boundary_nodes):
    fixed_dofs = []
    fixed_vals = []
    
    # Esquerda
    if config.TAG_BORDA_ESQUERDA in boundary_nodes:
        nodes = list(boundary_nodes[config.TAG_BORDA_ESQUERDA])
        for n in nodes:
            x, y = points[n]
            val = config.aplicar_bc_externa(x, y, config.TAG_BORDA_ESQUERDA)
            if val is not None:
                fixed_dofs.append(n)
                fixed_vals.append(val)
                
    # Direita
    if config.TAG_BORDA_DIREITA in boundary_nodes:
        nodes = list(boundary_nodes[config.TAG_BORDA_DIREITA])
        for n in nodes:
            x, y = points[n]
            val = config.aplicar_bc_externa(x, y, config.TAG_BORDA_DIREITA)
            if val is not None:
                fixed_dofs.append(n)
                fixed_vals.append(val)
                
    # Topo
    if config.TAG_BORDA_TOPO in boundary_nodes:
        nodes = list(boundary_nodes[config.TAG_BORDA_TOPO])
        for n in nodes:
            x, y = points[n]
            val = config.aplicar_bc_externa(x, y, config.TAG_BORDA_TOPO)
            if val is not None:
                fixed_dofs.append(n)
                fixed_vals.append(val)
                
    # Rio
    if config.TAG_PAREDE_RIO in boundary_nodes:
        nodes = list(boundary_nodes[config.TAG_PAREDE_RIO])
        for n in nodes:
            x, y = points[n]
            tipo, val = config.condicao_parede_rio(y)
            if tipo == "Dirichlet": # Seepage simplificado para Dirichlet
                fixed_dofs.append(n)
                fixed_vals.append(val)
            else:
                fixed_dofs.append(n)
                fixed_vals.append(0.0) # Seepage face simplificado

    return np.array(fixed_dofs, dtype=int), np.array(fixed_vals)

# ==============================================================================
# 4. FUNÇÕES DE MONTAGEM FEM (ELEMENTO TRIANGULAR LINEAR)
# ==============================================================================
def element_matrices(coords, K_val):
    # Coordenadas
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    
    # Área (2A)
    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    area = 0.5 * abs(detJ)
    
    # Gradientes das funções de forma (b_i, c_i)
    b = np.array([y2-y3, y3-y1, y1-y2])
    c = np.array([x3-x2, x1-x3, x2-x1])
    
    # Matriz de Rigidez Local (K_local = integral(gradN * K * gradN))
    # Para triângulo linear, gradN é constante
    B = np.vstack((b, c)) / (2*area) # (2x3)
    Ke = area * K_val * (B.T @ B) # (3x3)
    
    return Ke, B, area

# ==============================================================================
# 5. LOOP TEMPORAL (SOLVER)
# ==============================================================================
print(f"\n{PURPLE}[INICIANDO SOLVER]{RESET}")

# Condição Inicial
h_old = np.full(Nnodes, -1.0)
h_new = h_old.copy()

# Preparar arquivo PVD
if not os.path.exists("resultados"): os.makedirs("resultados")

# Loop de Tempo
for passo in range(1, num_passos+1):
    
    # Picard Loop (Iterações não-lineares)
    # Para Richards, K depende de h. Aqui simplificaremos para K saturado
    # para garantir que rode rápido, mas você pode adicionar o loop de update K(h)
    
    # Montagem Global
    A = lil_matrix((Nnodes, Nnodes))
    b_vec = np.zeros(Nnodes)
    
    # Array para armazenar velocidade nos elementos
    velocidades_elementos = np.zeros((Nelems, 3)) # (vx, vy, 0) para paraview 3D
    
    # Loop sobre Elementos
    for e in range(Nelems):
        nodes = triangles[e]
        coords = points[nodes]
        
        # Propriedade do Material
        mat_tag = material_tags[e] if material_tags is not None else 1
        
        # Pressão média no elemento (para calcular K não-linear se quiser)
        h_mean = np.mean(h_old[nodes])
        K_val = base.condutividade_hidraulica(h_mean, mat_tag)
        
        # Matrizes Locais
        Ke, B_grad, area = element_matrices(coords, K_val)
        
        # Montagem na Global
        for i in range(3):
            for j in range(3):
                A[nodes[i], nodes[j]] += Ke[i,j]
        
        # Cálculo da Velocidade de Darcy (pós-processamento)
        # v = -K * grad(h)
        # grad(h) = B * h_local
        h_local = h_new[nodes]
        grad_h = B_grad @ h_local
        v_elem = -K_val * grad_h
        velocidades_elementos[e, 0] = v_elem[0]
        velocidades_elementos[e, 1] = v_elem[1]

    # Condições de Contorno
    dofs, vals = get_dirichlet_values(points, boundary_nodes)
    
    # Aplicar Dirichlet na Matriz (Método da Penalidade ou Zerar Linha)
    # Zerar linha e colocar 1 na diagonal é mais seguro para sistemas pequenos
    for i, dof in enumerate(dofs):
        A[dof, :] = 0.0
        A[dof, dof] = 1.0
        b_vec[dof] = vals[i]
        
    # Termos de fonte (Gravidade) seriam adicionados ao b_vec aqui
    # Para simplificar este exemplo, resolvemos Laplace/Darcy Saturado Variável
    
    # Resolver Sistema Linear
    A_csr = A.tocsr()
    h_new = spsolve(A_csr, b_vec)
    
    # Erro (diferença entre passos)
    diff = np.linalg.norm(h_new - h_old)
    
    # Log
    if passo % 1 == 0:
        print(f"Tempo t = {YELLOW}{passo}{RESET} h | Diff: {diff:.2e}")
    
    h_old = h_new.copy()

# ==============================================================================
# 6. EXPORTAÇÃO CORRETA (H e V)
# ==============================================================================
print(f"\n{PURPLE}[SALVANDO RESULTADOS]{RESET}")

# Meshio precisa que os dados de célula (velocidade) tenham o mesmo número que cells
# E dados de ponto (pressão) tenham mesmo número que points

mesh_out = meshio.Mesh(
    points=np.column_stack((points, np.zeros(Nnodes))), # Adiciona Z=0 para 3D
    cells=[("triangle", triangles)],
    point_data={"Pressao_h": h_new},
    cell_data={"Velocidade_Darcy": [velocidades_elementos]}
)

outfile = "resultados/drenagem_scipy.vtu"
mesh_out.write(outfile)

print(f"Arquivo salvo: {YELLOW}{outfile}{RESET}")
print(f"Pressão Média: {np.mean(h_new):.3f}")
print(f"{GREEN}SUCESSO{RESET}")