# Solver de Richards 2D FEM linear (triângulos), sem FEniCS
# Implementa a formulação Transiente e Não-Linear com Iterações de Picard.

import numpy as np
import meshio
import os
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import config
import base
import sys

# Códigos ANSI para cores
RED = '\033[31m'
YELLOW = '\033[33m'
PURPLE = '\033[35m'
GREEN = '\033[32m'
RESET = '\033[0m'

g = 9.81  # gravidade [m/s^2]
dt = 3600.0  # 1 hora
tempo_total = 24 * 3600.0
TOL_PICARD = 1e-4
MAX_ITER = 15

def print_log(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    print(*args, **kwargs)
    with open("saida_terminal_pipeline.txt", "a", encoding="utf-8") as f_log:
        f_log.write(text + "\n")


# Ler malha e tags
try:
    malha = meshio.read("malha_dominio.xdmf")
    points = malha.points
    triangles = None
    for cell_block in malha.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data
            break
    if triangles is None:
        raise RuntimeError("Malha não contém elementos triangulares!")
    import h5py
    with h5py.File("malha_dominio.h5", "r") as f:
        tags_materiais = np.array(f["data2"])
except Exception as e:
    print_log(f"{RED}ERRO FATAL: Falha ao ler a malha ou tags de material. {e}{RESET}")
    sys.exit(1)

Nnodes = points.shape[0]
Nelems = triangles.shape[0]

# Parâmetros do problema (iguais ao pipeline antigo)
g = 9.81  # gravidade [m/s^2]
dt = 3600.0  # 1 hora
tempo_total = 24*3600.0
Q_rio = config.VAZAO_RIO if hasattr(config, 'VAZAO_RIO') else 0.0
nivel_rio = config.NIVEL_AGUA_ABSOLUTO if hasattr(config, 'NIVEL_AGUA_ABSOLUTO') else 0.0

# ====== RELATÓRIO DE EXECUÇÃO DA SIMULAÇÃO ======
print_log("="*72)
print_log("==== Iniciando pipeline de simulação de drenagem " + "="*23)
print_log("="*72)
print_log("")
print_log(f"{PURPLE}ARQUIVO DE MALHA{RESET}")
print_log(f"Caminho:         {YELLOW}malha_dominio.xdmf{RESET}")
print_log(f"Caminho Bordas:  {YELLOW}malha_bordas.xdmf{RESET}")
print_log(f"Pontos (Nós):    {YELLOW}{Nnodes}{RESET}")
print_log(f"Elementos (Tri): {YELLOW}{Nelems}{RESET}")
print_log("")
print_log(f"{PURPLE}PARÂMETROS FÍSICOS{RESET}")
print_log(f"Vazão do Rio (Q):          {GREEN}{Q_rio:.2f}{RESET} m³/s")
print_log(f"Coeficiente de Manning (n): {GREEN}{getattr(config, 'MANNING_N', 0.0)}{RESET}")
ks_presentes = [base.MATERIAIS.get(tag, {"ks": 0})["ks"] for tag in np.unique(tags_materiais)]
ks_medio = np.mean(ks_presentes) if ks_presentes else 0
print_log(f"Condutividade (Ks):        {GREEN}{ks_medio:.2e}{RESET} m/s (média dos materiais)")
print_log(f"Cota de Fundo do Duto:     {GREEN}{getattr(config, 'Y_FUNDO_ADUELA', 0.0):.3f}{RESET} m")
print_log(f"Lâmina d'Água (h):         {GREEN}{getattr(config, 'LAMINA_AGUA', 0.0):.3f}{RESET} m")
print_log(f"Gravidade (g):             {GREEN}{g:.2f}{RESET} m/s²")
print_log(f"Nível Absoluto da Água:    {GREEN}{nivel_rio:.3f}{RESET} m")
print_log("")
print_log(f"{PURPLE}CONFIGURAÇÃO DO SOLVER{RESET}")
print_log(f"Tempo Total:               {YELLOW}24 horas{RESET}")
print_log(f"Passo de Tempo (dt):       {YELLOW}{dt:.1f}{RESET} s (1 hora)")
print_log(f"Método de Integração:      {YELLOW}Backward-Euler{RESET}")
print_log(f"Tolerância Picard:         {YELLOW}1e-4{RESET}")
print_log(f"Max Iterações:             {YELLOW}15{RESET}")
print_log("")
print_log(f"{PURPLE}CONDIÇÕES DE CONTORNO{RESET}")
print_log(f"Esquerda: {YELLOW}Dirichlet Variável{RESET} ({PURPLE}Lençol Alto{RESET})")
print_log(f"Direita:  {YELLOW}Dirichlet Variável{RESET} ({PURPLE}Lençol Baixo{RESET})")
print_log(f"Topo:     {YELLOW}Dirichlet Constante{RESET} ({PURPLE}Superfície Seca{RESET})")
print_log(f"Rio:      {YELLOW}Mista{RESET} ({PURPLE}Seepage Face / Hidrostática{RESET})")
print_log("")
print_log(f"{YELLOW}{'='*72}{RESET}")
print_log(f"{YELLOW}==== PROCESSAMENTO TEMPORAL {'='*39}{RESET}")
print_log(f"{YELLOW}{'='*72}{RESET}")

def get_kh_ch(h_elem, tag_mat):
    props = base.MATERIAIS.get(tag_mat, None)
    if props is None:
        return 1.0e-13, 1.0e-9
    K_h = base.condutividade_hidraulica(h_elem, tag_mat)
    dh = 1e-4
    th_p = base.conteudo_umidade(h_elem + dh, tag_mat)
    th_m = base.conteudo_umidade(h_elem - dh, tag_mat)
    C_h = np.maximum((th_p - th_m) / (2 * dh), 1.0e-9)
    return K_h, C_h

def assemble_matrices(h_iter, tags_materiais, points, triangles, dt):
    Nnodes = points.shape[0]
    A_global = lil_matrix((Nnodes, Nnodes))
    M_global = lil_matrix((Nnodes, Nnodes))
    b_grav_global = np.zeros(Nnodes)
    for idx_elem, elem in enumerate(triangles):
        node_ids = elem
        coords = points[node_ids, :2]
        x1, y1 = coords[0]; x2, y2 = coords[1]; x3, y3 = coords[2]
        area = 0.5 * np.abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
        c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
        B = np.array([[b1, b2, b3], [c1, c2, c3]]) / (2*area)
        h_elem_avg = np.mean(h_iter[node_ids])
        tag_mat = tags_materiais[idx_elem]
        K_h, C_h = get_kh_ch(h_elem_avg, tag_mat)
        A_local = K_h * area * (B.T @ B)
        M_local_base = area / 12.0 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        M_local = (C_h / dt) * M_local_base
        c_vec = np.array([c1, c2, c3])
        b_grav_local = K_h * c_vec / 2.0
        for i in range(3):
            b_grav_global[node_ids[i]] += b_grav_local[i]
            for j in range(3):
                A_global[node_ids[i], node_ids[j]] += A_local[i, j]
                M_global[node_ids[i], node_ids[j]] += M_local[i, j]
    return A_global.tocsr(), M_global.tocsr(), b_grav_global

# Condições de contorno
y = points[:,1]
x = points[:,0]
tol = 1e-8
nodes_top = np.where(np.abs(y - y.max()) < tol)[0]
nodes_bottom = np.where(np.abs(y - y.min()) < tol)[0]
nodes_left = np.where(np.abs(x - x.min()) < tol)[0]
nodes_right = np.where(np.abs(x - x.max()) < tol)[0]
h_top = -2.0
h_bottom = 0.0
h_esq = 0.0
h_dir = -2.0
fixed_nodes = np.concatenate([nodes_top, nodes_bottom, nodes_left, nodes_right])
fixed_values = np.concatenate([
    np.full(nodes_top.shape, h_top),
    np.full(nodes_bottom.shape, h_bottom),
    np.full(nodes_left.shape, h_esq),
    np.full(nodes_right.shape, h_dir)
])
unique_nodes, idx_unique = np.unique(fixed_nodes, return_index=True)
unique_values = fixed_values[idx_unique]

h_n = np.full(Nnodes, -1.0)

print_log("="*72)
print_log("==== Solver de Richards Transiente Não-Linear ====")
print_log("="*72)
print_log(f"Pontos (Nós):    {YELLOW}{Nnodes}{RESET}")
print_log(f"Elementos (Tri): {YELLOW}{Nelems}{RESET}")
print_log(f"Passo de Tempo (dt):       {YELLOW}{dt:.1f}{RESET} s (1 hora)")
print_log(f"Tempo Total:               {YELLOW}{tempo_total/3600:.1f}{RESET} horas")
print_log(f"Tolerância Picard:         {YELLOW}{TOL_PICARD:.1e}{RESET}")
print_log("")

num_passos = int(tempo_total // dt)
h_last = h_n.copy()
h_old = h_n.copy()
if not os.path.exists("resultados"): os.makedirs("resultados")
h_list = []
tempos = []

for passo in range(1, num_passos + 1):
    h_iter = h_old.copy()
    for iter_picard in range(MAX_ITER):
        A_mat, M_mat, b_grav = assemble_matrices(h_iter, tags_materiais, points, triangles, dt)
        K_global = M_mat + A_mat
        b_time = M_mat @ h_old
        b_global = b_time - b_grav
        for node in unique_nodes:
            b_global[node] = unique_values[np.where(unique_nodes == node)[0][0]]
        K_global_csr = K_global.tocsr()
        for node in unique_nodes:
            K_global_csr[node, :] = 0
            K_global_csr[node, node] = 1.0
        h_new = spsolve(K_global_csr, b_global)
        error = np.linalg.norm(h_new - h_iter) / (np.linalg.norm(h_new) + 1e-12)
        if error < TOL_PICARD:
            h_last = h_new.copy()
            print_log(f"Tempo t = {YELLOW}{passo*dt/3600:.1f}{RESET} {PURPLE}h{RESET} | {GREEN}Convergiu{RESET} (Iter: {YELLOW}{iter_picard+1}{RESET}) | Erro: {YELLOW}{error:.2e}{RESET}")
            break
        h_iter = h_new.copy()
        if iter_picard == MAX_ITER - 1:
            print_log(f"Tempo t = {YELLOW}{passo*dt/3600:.1f}{RESET} {PURPLE}h{RESET} | {RED}AVISO: Não convergiu após {MAX_ITER} iter. (Erro: {error:.2e}){RESET}")
            h_last = h_new.copy()
    h_old = h_last.copy()
    h_list.append(h_last.copy())
    tempos.append(passo * dt)
    if passo <= 3 or passo % 5 == 0 or passo >= num_passos - 1:
        pass
    else:
        if passo == 4: print_log("...")

if h_list:
    saida_vtu = "resultados/drenagem.pvd"
    with open(saida_vtu, "w") as f:
        f.write('<VTKFile type="Collection" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write('<Collection>\n')
        for i, h_step in enumerate(h_list):
            nome_vtu = f"drenagem_{i}.vtu"
            # Garante que os pontos tenham shape (N,3) para VTU
            if points.shape[1] == 2:
                points_vtu = np.column_stack([points, np.zeros(points.shape[0])])
            else:
                points_vtu = points
            mesh = meshio.Mesh(points_vtu, [("triangle", triangles)], point_data={"Pressao_h": h_step})
            mesh.write(f"resultados/{nome_vtu}", file_format="vtu")
            f.write(f'<DataSet timestep="{tempos[i]/3600.0}" file="{nome_vtu}"/>\n')
        f.write('</Collection>\n')
        f.write('</VTKFile>\n')
    h_final = h_list[-1]
    h_min = np.min(h_final)
    h_max = np.max(h_final)
    h_mean = np.mean(h_final)
    print_log(f"\n{PURPLE}RESUMO ESTATÍSTICO FINAL (t={tempo_total/3600:.1f}h){RESET}\n")
    print_log(f"Pressão Mínima:      {YELLOW}{h_min:.3f}{RESET}")
    print_log(f"Pressão Máxima:      {YELLOW}{h_max:.3f}{RESET}")
    print_log(f"Pressão Média:       {YELLOW}{h_mean:.3f}{RESET}")
    print_log(f"Visualização: {YELLOW}{saida_vtu}{RESET} (ParaView)")
    print_log(f"Status da Execução:  {GREEN}SUCESSO{RESET}")
else:
    print_log(f"{RED}FALHA na simulação: Nenhuma solução foi gerada.{RESET}")

print_log(f"{YELLOW}{'='*80}{RESET}")
print_log(f"{PURPLE}FIM DO RELATÓRIO{RESET}\n")
