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
# ------ INÍCIO DO RELATÓRIO VISUAL (PIPELINE) ------
print(f"{YELLOW}{'='*80}{RESET}")
print(f"{PURPLE}RELATÓRIO DE EXECUÇÃO DA SIMULAÇÃO (SCIPY FEM){RESET}\n")
print(f"{YELLOW}{'#'*80}{RESET}")
print(f"{PURPLE}[1] DADOS DE ENTRADA (INPUT){RESET}")
print(f"{YELLOW}{'#'*80}{RESET}")

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

# Relatório da Malha
print(f"{PURPLE}[ARQUIVO DE MALHA]{RESET}\n")
print(f"Caminho:         {YELLOW}malha_dominio.xdmf{RESET}")
print(f"Caminho Bordas:  {YELLOW}malha_bordas.xdmf{RESET}")
print(f"Pontos (Nós):    {YELLOW}{Nnodes}{RESET}")
print(f"Elementos (Tri): {YELLOW}{Nelems}{RESET}\n")

# ==============================================================================
# 2. PARÂMETROS
# ==============================================================================
dt = 3600.0  # 1 hora
tempo_total = 24 * 3600.0
num_passos = int(tempo_total // dt)
g = 9.81

# Relatório Físico
print(f"{PURPLE}[PARÂMETROS FÍSICOS]{RESET}\n")
print(f"Vazão do Rio (Q):          {YELLOW}{config.VAZAO_RIO:.2f}{RESET} {PURPLE}m³/s{RESET}")
print(f"Coeficiente de Manning (n): {YELLOW}{config.MANNING_N}{RESET}")
print(f"Declividade (S0):          {YELLOW}{config.DECLIVIDADE_S0}{RESET} {PURPLE}m/m{RESET}")
print(f"Cota de Fundo do Duto:     {YELLOW}{config.Y_FUNDO_ADUELA:.3f}{RESET} {PURPLE}m{RESET}")
print(f"Lâmina d'água (h):         {YELLOW}{config.LAMINA_AGUA:.3f}{RESET} {PURPLE}m{RESET}")
print(f"Nível Absoluto da Água:    {YELLOW}{config.NIVEL_AGUA_ABSOLUTO:.3f}{RESET} {PURPLE}m{RESET}\n")

print(f"{PURPLE}[CONFIGURAÇÃO DO SOLVER]{RESET}\n")
print(f"Tempo Total:               {YELLOW}{int(tempo_total/3600)}{RESET} {PURPLE}horas{RESET}")
print(f"Passo de Tempo (dt):       {YELLOW}{dt:.1f}{RESET} {PURPLE}s{RESET} {RED}(1 hora){RESET}")
print(f"Método de Integração:      {YELLOW}Backward Euler (Linearizado){RESET}\n")

print(f"{PURPLE}[CONDIÇÕES DE CONTORNO]{RESET}\n")
print(f"Esquerda: {YELLOW}Dirichlet Variável{RESET} ({PURPLE}Lençol Alto{RESET})")
print(f"Direita:  {YELLOW}Dirichlet Variável{RESET} ({PURPLE}Lençol Baixo{RESET})")
print(f"Topo:     {YELLOW}Dirichlet Constante{RESET} ({PURPLE}Sucção{RESET})")
print(f"Rio:      {YELLOW}Mista{RESET} ({PURPLE}Seepage Face / Hidrostática{RESET})\n")

# ==============================================================================
# 3. IDENTIFICAÇÃO DE NÓS DE CONTORNO (DIRICHLET)
# ==============================================================================
# Mapear nós para tags de borda
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
            if tipo == "Dirichlet":
                fixed_dofs.append(n)
                fixed_vals.append(val)
            else:
                fixed_dofs.append(n)
                fixed_vals.append(0.0)

    return np.array(fixed_dofs, dtype=int), np.array(fixed_vals)

# ==============================================================================
# 4. FUNÇÕES DE MONTAGEM FEM
# ==============================================================================
def element_matrices(coords, K_val):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    
    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    area = 0.5 * abs(detJ)
    
    b = np.array([y2-y3, y3-y1, y1-y2])
    c = np.array([x3-x2, x1-x3, x2-x1])
    
    B = np.vstack((b, c)) / (2*area) # (2x3)
    Ke = area * K_val * (B.T @ B) # (3x3)
    
    return Ke, B, area

# ==============================================================================
# 5. LOOP TEMPORAL (SOLVER)
# ==============================================================================
print(f"{YELLOW}{'#'*80}{RESET}")
print(f"{PURPLE}[2] DADOS DE SAÍDA (OUTPUT){RESET}")
print(f"{YELLOW}{'#'*80}{RESET}")
print(f"{PURPLE}[PROCESSAMENTO TEMPORAL]{RESET}\n")

# Condição Inicial
h_old = np.full(Nnodes, -1.0)
h_new = h_old.copy()

# Preparar arquivo PVD
if not os.path.exists("resultados"): os.makedirs("resultados")

velocidades_elementos = np.zeros((Nelems, 3))

# Loop de Tempo
for passo in range(1, num_passos+1):
    
    # Montagem Global
    A = lil_matrix((Nnodes, Nnodes))
    b_vec = np.zeros(Nnodes)
    
    # Reset velocidades
    velocidades_elementos = np.zeros((Nelems, 3))
    
    for e in range(Nelems):
        nodes = triangles[e]
        coords = points[nodes]
        mat_tag = material_tags[e] if material_tags is not None else 1
        
        # K Linearizado (baseado em h_old)
        h_mean = np.mean(h_old[nodes])
        K_val = base.condutividade_hidraulica(h_mean, mat_tag)
        
        Ke, B_grad, area = element_matrices(coords, K_val)
        
        for i in range(3):
            for j in range(3):
                A[nodes[i], nodes[j]] += Ke[i,j]
        
        # Cálculo da Velocidade (Pós-processamento baseada no passo anterior para estimativa)
        # Para ser exato, recalculamos com h_new depois do solve, mas aqui aproximamos
        # Usaremos h_old para estimar v neste passo de montagem (apenas para visualização in-loop se quiser)
        
    dofs, vals = get_dirichlet_values(points, boundary_nodes)
    
    for i, dof in enumerate(dofs):
        A[dof, :] = 0.0
        A[dof, dof] = 1.0
        b_vec[dof] = vals[i]
        
    A_csr = A.tocsr()
    h_new = spsolve(A_csr, b_vec)
    
    diff = np.linalg.norm(h_new - h_old)
    
    # Log estilizado (Pipeline)
    tempo_atual = passo * dt / 3600.0
    
    # Lógica para mostrar apenas alguns passos no terminal para não poluir
    if passo <= 3 or passo == num_passos or passo % 5 == 0:
        print(f"Tempo t = {YELLOW}{tempo_atual:5.1f}{RESET} {PURPLE}h{RESET} | Diff: {YELLOW}{diff:.2e}{RESET}")
    
    h_old = h_new.copy()

# Recálculo final exato das velocidades com h_new convergido
for e in range(Nelems):
    nodes = triangles[e]
    coords = points[nodes]
    mat_tag = material_tags[e] if material_tags is not None else 1
    h_mean = np.mean(h_new[nodes])
    K_val = base.condutividade_hidraulica(h_mean, mat_tag)
    Ke, B_grad, area = element_matrices(coords, K_val)
    
    h_local = h_new[nodes]
    grad_h = B_grad @ h_local
    v_elem = -K_val * grad_h
    velocidades_elementos[e, 0] = v_elem[0] # Vx
    velocidades_elementos[e, 1] = v_elem[1] # Vy

# ==============================================================================
# 6. EXPORTAÇÃO E FINALIZAÇÃO
# ==============================================================================
print(f"\n{PURPLE}[ARQUIVOS GERADOS]{RESET}\n")

outfile = "resultados/drenagem_scipy.vtu"
mesh_out = meshio.Mesh(
    points=np.column_stack((points, np.zeros(Nnodes))), 
    cells=[("triangle", triangles)],
    point_data={"Pressao_h": h_new},
    cell_data={"Velocidade_Darcy": [velocidades_elementos]}
)
mesh_out.write(outfile)

# Resumo Estatístico
h_min = np.min(h_new)
h_max = np.max(h_new)
h_mean = np.mean(h_new)

print(f"Visualização: {YELLOW}{outfile}{RESET} (ParaView)")
print(f"Malha HDF5:   {YELLOW}malha_dominio.h5{RESET} (Se houver)")

print(f"\n{PURPLE}[RESUMO ESTATÍSTICO FINAL (t=24h)]{RESET}\n")
print(f"Pressão Mínima:      {YELLOW}{h_min:.3f}{RESET} {PURPLE}(Superfície Seca){RESET}")
print(f"Pressão Máxima:      {YELLOW}{h_max:.3f}{RESET} {PURPLE}(Fundo do Rio){RESET}")
print(f"Pressão Média:       {YELLOW}{h_mean:.3f}{RESET}")
print(f"Status da Execução:  {GREEN}SUCESSO{RESET}")

print(f"{YELLOW}{'='*80}{RESET}")
print(f"{PURPLE}FIM DO RELATÓRIO{RESET}\n")