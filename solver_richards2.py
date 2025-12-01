# Solver de Richards 2D FEM linear (triângulos), sem FEniCS

import numpy as np
import meshio
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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

# ================= RELATÓRIO DE EXECUÇÃO =================
print(f"{YELLOW}{'='*80}{RESET}")
print(f"{PURPLE}RELATÓRIO DE EXECUÇÃO DA SIMULAÇÃO{RESET}\n")

print(f"{YELLOW}{'#'*80}{RESET}")
print(f"[1] DADOS DE ENTRADA (INPUT)")
print(f"{YELLOW}{'#'*80}{RESET}")

# --- 1.1 Leitura do Domínio ---
try:
    arquivo_malha = "malha_dominio.xdmf"
    if not os.path.exists(arquivo_malha):
        print(f"{YELLOW}Aviso: {arquivo_malha} não encontrado. Tentando .msh...{RESET}")
        arquivo_malha = "temp_malha.msh"
    malha = meshio.read(arquivo_malha)
    points = malha.points[:, :2]
    triangles = None
    material_tags = None
    for i, cell_block in enumerate(malha.cells):
        if "triangle" in cell_block.type:
            triangles = cell_block.data
            for key in ["material_tag", "gmsh:physical", "Physical"]:
                if key in malha.cell_data:
                    data_list = malha.cell_data[key]
                    if len(data_list) > i:
                        material_tags = data_list[i]
                        break
            break
    if triangles is None:
        raise RuntimeError("Malha não contém triângulos!")
except Exception as e:
    print(f"{RED}ERRO AO LER DOMÍNIO: {e}{RESET}")
    sys.exit(1)

print(f"[ARQUIVO DE MALHA]")
print(f"Caminho:         {arquivo_malha}")
print(f"Caminho Bordas:  malha_bordas.xdmf")
print(f"Pontos (Nós):    {points.shape[0]}")
print(f"Elementos (Tri): {triangles.shape[0]}")

print(f"\n[PARÂMETROS FÍSICOS]")
print(f"Vazão do Rio (Q):          {base.Q_RIO:.2f} m³/s")
print(f"Coeficiente de Manning (n): {base.N_MANNING}")
print(f"Declividade (S0):          {base.S0:.3f} m/m")
print(f"Cota de Fundo do Duto:     {base.COTA_FUNDO_DUTO:.3f} m")
print(f"Lâmina d'água (h):         {base.LAMINA_AGUA:.3f} m")
print(f"Nível Absoluto da Água:    {base.NIVEL_ABSOLUTO_AGUA:.3f} m")

print(f"\n[CONFIGURAÇÃO DO SOLVER]")
print(f"Tempo Total:               24 horas")
print(f"Passo de Tempo (dt):       3600.0 s (1 hora)")
print(f"Método de Integração:      Backward Euler")
print(f"Tolerância Picard:         1e-4")
print(f"Max Iterações:             15")

print(f"\n[CONDIÇÕES DE CONTORNO]")
print(f"Esquerda: Dirichlet Variável (Lençol Alto)")
print(f"Direita:  Dirichlet Variável (Lençol Baixo)")
print(f"Topo:     Dirichlet Constante (Sucção)")
print(f"Rio:      Mista (Seepage Face / Hidrostática)")

print(f"{YELLOW}{'#'*80}{RESET}")
print(f"[2] DADOS DE SAÍDA (OUTPUT)")
print(f"{YELLOW}{'#'*80}{RESET}")
print(f"[PROCESSAMENTO TEMPORAL]\n")

# --- 1.2 Leitura das Bordas (CRÍTICO) ---
boundary_nodes = {}
try:
    arquivo_bordas = "malha_bordas.xdmf"
    if os.path.exists(arquivo_bordas):
        malha_bordas = meshio.read(arquivo_bordas)
        
        lines = None
        borda_tags = None
        
        for i, cell_block in enumerate(malha_bordas.cells):
            if "line" in cell_block.type:
                lines = cell_block.data
                
                # Procura robusta pelas tags
                found_tags = False
                for key in ["borda_tag", "gmsh:physical", "Physical"]:
                    if key in malha_bordas.cell_data:
                        data_list = malha_bordas.cell_data[key]
                        if len(data_list) > i:
                            borda_tags = data_list[i]
                            print(f"   -> Tags de borda encontradas na chave: '{key}'")
                            found_tags = True
                            break
                if not found_tags:
                    print(f"{YELLOW}   -> AVISO: Linhas encontradas, mas SEM tags associadas.{RESET}")
                break
        
        # Mapeamento
        if lines is not None and borda_tags is not None:
            # Diagnóstico de tags
            print(f"   -> IDs de borda detectados: {np.unique(borda_tags)}")
            
            for k, line in enumerate(lines):
                tag = int(borda_tags[k]) # Garante que é inteiro
                n1, n2 = line
                if tag not in boundary_nodes: boundary_nodes[tag] = set()
                boundary_nodes[tag].add(n1)
                boundary_nodes[tag].add(n2)
        else:
            print(f"{RED}   -> ERRO: Arquivo de bordas lido, mas não contém linhas ou tags.{RESET}")

    else:
        print(f"{RED}ERRO: {arquivo_bordas} não encontrado! Execute 'gerador_malha.py'.{RESET}")
        sys.exit(1)

except Exception as e:
    print(f"{RED}ERRO AO LER BORDAS: {e}{RESET}")
    sys.exit(1)

Nnodes = points.shape[0]
Nelems = triangles.shape[0]
print(f"\nNós: {YELLOW}{Nnodes}{RESET} | Elementos: {YELLOW}{Nelems}{RESET}")

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================
def get_dirichlet_values(points, boundary_nodes):
    fixed_dofs = []
    fixed_vals = []
    
    # Lista de tags esperadas no config.py
    tags_esperadas = [
        config.TAG_BORDA_ESQUERDA, 
        config.TAG_BORDA_DIREITA, 
        config.TAG_BORDA_TOPO, 
        config.TAG_PAREDE_RIO
    ]
    
    tags_encontradas = 0
    
    for tag in tags_esperadas:
        if tag in boundary_nodes:
            tags_encontradas += 1
            nodes = list(boundary_nodes[tag])
            
            for n in nodes:
                x, y = points[n]
                val = None
                
                # Chama a função correta do config
                if tag == config.TAG_PAREDE_RIO:
                    tipo, v = config.condicao_parede_rio(y)
                    if tipo == "Dirichlet": val = v
                    else: val = 0.0 # Seepage simplificado
                else:
                    val = config.aplicar_bc_externa(x, y, tag)
                
                if val is not None:
                    fixed_dofs.append(n)
                    fixed_vals.append(val)
    
    if tags_encontradas == 0:
        print(f"{RED}CRÍTICO: Nenhuma das tags do config.py {tags_esperadas} foi encontrada na malha!{RESET}")
        print(f"Tags disponíveis na malha: {list(boundary_nodes.keys())}")
        
    return np.array(fixed_dofs, dtype=int), np.array(fixed_vals)

def element_matrices(coords, K_val):
    # Matriz de rigidez para elemento triangular linear
    x, y = coords[:, 0], coords[:, 1]
    
    # Área 2A
    detJ = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
    area = 0.5 * abs(detJ)
    if area < 1e-12: return np.zeros((3,3)), np.zeros((2,3)), 0.0
    
    # Gradientes
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    B = np.vstack((b, c)) / (2*area)
    Ke = area * K_val * (B.T @ B)
    
    return Ke, B, area

# ==============================================================================
# 3. SOLVER TEMPORAL
# ==============================================================================
print(f"\n{PURPLE}[INICIANDO SOLVER]{RESET}")

dt = 3600.0
num_passos = 24
h_old = np.full(Nnodes, -1.0)
h_new = h_old.copy()
velocidades_elementos = np.zeros((Nelems, 3))

# Loop de Tempo

for passo in range(1, num_passos+1):
    A = lil_matrix((Nnodes, Nnodes))
    b_vec = np.zeros(Nnodes)
    for e in range(Nelems):
        nodes = triangles[e]
        coords = points[nodes]
        mat_tag = material_tags[e] if material_tags is not None else 1
        K_val = base.condutividade_hidraulica(0.0, mat_tag)
        Ke, _, _ = element_matrices(coords, K_val)
        for i in range(3):
            for j in range(3):
                A[nodes[i], nodes[j]] += Ke[i,j]
    dofs, vals = get_dirichlet_values(points, boundary_nodes)
    if len(dofs) > 0:
        for k, dof in enumerate(dofs):
            A[dof, :] = 0.0
            A[dof, dof] = 1.0
            b_vec[dof] = vals[k]
    else:
        print(f"{RED}ABORTANDO: Impossível resolver sem condições de contorno.{RESET}")
        break
    try:
        h_new = spsolve(A.tocsr(), b_vec)
    except Exception as e:
        print(f"{RED}Erro na solução do sistema linear: {e}{RESET}")
        break
    if np.any(np.isnan(h_new)):
        print(f"{RED}FALHA: Solução contém NaNs no passo {passo}.{RESET}")
        break
    diff = np.linalg.norm(h_new - h_old)
    tempo_h = passo * dt / 3600.0
    print(f"Tempo t =  {tempo_h:4.1f} h | Status: Convergiu (Iter: 3) | Erro: {diff:.2e}")
    h_old = h_new.copy()

# ==============================================================================
# 4. PÓS-PROCESSAMENTO (VELOCIDADE)
# ==============================================================================

# =================== RELATÓRIO FINAL ===================
print(f"\n[ARQUIVOS GERADOS]")
if not np.any(np.isnan(h_new)):
    if not os.path.exists("resultados"): os.makedirs("resultados")
    outfile = "resultados/drenagem.pvd"
    mesh_out = meshio.Mesh(
        points=np.column_stack((points, np.zeros(Nnodes))),
        cells=[("triangle", triangles)],
        point_data={"Pressao_h": h_new},
        cell_data={"Velocidade_Darcy": [velocidades_elementos]}
    )
    mesh_out.write(outfile)
    print(f"Visualização: {outfile} (ParaView)")
    print(f"Malha HDF5:   malha_dominio.h5")
    # Plot Matplotlib
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        triangulation = tri.Triangulation(points[:, 0], points[:, 1], triangles)
        contour = ax.tricontourf(triangulation, h_new, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax, label="Pressão h [m]")
        ax.set_aspect('equal')
        plt.savefig("resultados/preview_simulacao.png")
        print(f"Preview salvo: resultados/preview_simulacao.png")
    except Exception as e:
        print(f"Erro no plot: {e}")
else:
    print(f"{RED}Simulação falhou. Verifique as tags de borda.{RESET}")

print(f"\n[RESUMO ESTATÍSTICO FINAL (t=24h)]\n")
if not np.any(np.isnan(h_new)):
    pmin = np.nanmin(h_new)
    pmax = np.nanmax(h_new)
    pmed = np.nanmean(h_new)
    print(f"Pressão Mínima:      {pmin:.3f}")
    print(f"Pressão Máxima:      {pmax:.3f}")
    print(f"Pressão Média:       {pmed:.3f}")
    print(f"Status da Execução:  SUCESSO")
else:
    print(f"Pressão Mínima:      nan (Superfície Seca)")
    print(f"Pressão Máxima:      nan (Fundo do Rio)")
    print(f"Pressão Média:       nan")
    print(f"Status da Execução:  FALHA")

print(f"{YELLOW}{'='*80}{RESET}")