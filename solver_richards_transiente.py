"""Solver de Richards Transiente Não-Linear (v3) em 2D."""
import os
import sys

import h5py
import meshio
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import base
import config

# Paleta ANSI usada no relatório
def _ansi(code: str) -> str:
    return f"\033[{code}m"

RED = _ansi("31")
YELLOW = _ansi("33")
PURPLE = _ansi("35")
GREEN = _ansi("32")
RESET = _ansi("0")

# Parâmetros do problema (ajustados para reduzir tempo de execução)
g = 9.81  # gravidade [m/s²]
dt = 600.0  # 10 minutos
tempo_total = 24.0 * 3600.0  # 24 horas
TOL_PICARD = 1.0e-4
MAX_ITER = 15


def print_log(*args, **kwargs):
    """Imprime na tela e armazena no arquivo de log compartilhado."""
    text = " ".join(str(a) for a in args)
    print(*args, **kwargs)
    with open("saida_terminal_pipeline.txt", "a", encoding="utf-8") as f_log:
        f_log.write(text + "\n")


# ---------------------------------------------------------------------------
# 1) Leitura da malha e das tags de material
# ---------------------------------------------------------------------------
try:
    malha = meshio.read("malha_dominio.xdmf")
    points = malha.points
    triangles = next((cell.data for cell in malha.cells if cell.type == "triangle"), None)
    if triangles is None:
        raise RuntimeError("Malha não contém elementos triangulares!")
except Exception as exc:
    print_log(f"{RED}ERRO FATAL ao ler a malha: {exc}{RESET}")
    sys.exit(1)

try:
    with h5py.File("malha_dominio.h5", "r") as f:
        tags_materiais = np.array(f["data2"])
except Exception as exc:
    print_log(f"{RED}ERRO FATAL ao ler tags de material: {exc}{RESET}")
    sys.exit(1)

Nnodes = points.shape[0]
Nelems = triangles.shape[0]


# ---------------------------------------------------------------------------
# 2) Propriedades hidráulicas
# ---------------------------------------------------------------------------
def get_kh_ch(h_elem: float, tag_mat: int):
    props = base.MATERIAIS.get(tag_mat)
    if props is None:
        return 1.0e-13, 1.0e-9
    K_h = base.condutividade_hidraulica(h_elem, tag_mat)
    dh = 1e-4
    th_p = base.conteudo_umidade(h_elem + dh, tag_mat)
    th_m = base.conteudo_umidade(h_elem - dh, tag_mat)
    C_h = np.maximum((th_p - th_m) / (2.0 * dh), 1.0e-8)
    return K_h, C_h


def assemble_matrices(h_iter, tags_materiais, points, triangles, dt):
    """Monta as matrizes A/M e o vetor de gravidade usando h_iter."""
    Nnodes = points.shape[0]
    A_global = lil_matrix((Nnodes, Nnodes))
    M_global = lil_matrix((Nnodes, Nnodes))
    b_grav_global = np.zeros(Nnodes)
    K_diag = []
    C_diag = []
    for elem_idx, elem in enumerate(triangles):
        node_ids = elem
        coords = points[node_ids, :2]
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1
        B = np.array([[b1, b2, b3], [c1, c2, c3]]) / (2.0 * area)
        h_elem_avg = np.mean(h_iter[node_ids])
        tag_mat = tags_materiais[elem_idx]
        K_h, C_h = get_kh_ch(h_elem_avg, tag_mat)
        K_diag.append(K_h)
        C_diag.append(C_h)
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
    return A_global.tocsr(), M_global.tocsr(), b_grav_global, np.array(K_diag), np.array(C_diag)


# ---------------------------------------------------------------------------
# 3) Condições de contorno fixas
# ---------------------------------------------------------------------------
y = points[:, 1]
x = points[:, 0]
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
    np.full(nodes_right.shape, h_dir),
])
unique_nodes, idx_unique = np.unique(fixed_nodes, return_index=True)
unique_values = fixed_values[idx_unique]

h_n = np.full(Nnodes, -10.0)
h_n[unique_nodes] = unique_values


# ---------------------------------------------------------------------------
# 4) Relatório inicial
# ---------------------------------------------------------------------------
print_log("=" * 72)
print_log("==== Solver de Richards Transiente Não-Linear (v3) ====")
print_log("=" * 72)
print_log(f"Pontos (Nós):    {YELLOW}{Nnodes}{RESET}")
print_log(f"Elementos (Tri): {YELLOW}{Nelems}{RESET}")
print_log(f"Passo de Tempo (dt):       {YELLOW}{dt:.1f}{RESET} s (10 minutos)")
print_log(f"Tempo Total:               {YELLOW}{tempo_total/3600:.1f}{RESET} horas")
print_log(f"Tolerância Picard:         {YELLOW}{TOL_PICARD:.1e}{RESET}")
print_log(f"{PURPLE}Condição Inicial: h = -10.0 m (seco){RESET}\n")


# ---------------------------------------------------------------------------
# 5) Loop temporal
# ---------------------------------------------------------------------------
num_passos = int(tempo_total // dt)
h_last = h_n.copy()
h_old = h_n.copy()
if not os.path.exists("resultados"):
    os.makedirs("resultados")
h_list = []
tempos = []

for passo in range(1, num_passos + 1):
    h_iter = h_old.copy()
    for iter_picard in range(MAX_ITER):
        A_mat, M_mat, b_grav, K_diag, C_diag = assemble_matrices(
            h_iter, tags_materiais, points, triangles, dt
        )
        if iter_picard == 0 and (passo <= 2 or passo % 72 == 0 or passo == num_passos):
            print_log(
                "T={:.1f}h | C_h min/max: {:.2e}/{:.2e} | K_h min/max: {:.2e}/{:.2e}".format(
                    passo * dt / 3600.0,
                    np.min(C_diag),
                    np.max(C_diag),
                    np.min(K_diag),
                    np.max(K_diag),
                )
            )
        K_global = (M_mat + A_mat).tolil()
        b_time = M_mat @ h_old
        b_global = b_time - b_grav
        for idx, node in enumerate(unique_nodes):
            h_bc = unique_values[idx]
            K_global[node, :] = 0.0
            K_global[node, node] = 1.0
            b_global[node] = h_bc
        h_new = spsolve(K_global.tocsr(), b_global)
        error = np.linalg.norm(h_new - h_iter) / (np.linalg.norm(h_new) + 1e-12)
        if error < TOL_PICARD:
            h_last = h_new.copy()
            if passo <= 2 or passo % 72 == 0 or passo == num_passos:
                print_log(
                    f"Tempo t = {YELLOW}{passo*dt/3600:.1f}{RESET}h | {GREEN}Convergiu{RESET} "
                    f"(Iter: {YELLOW}{iter_picard+1}{RESET}) | Erro: {YELLOW}{error:.2e}{RESET}"
                )
            break
        h_iter = h_new.copy()
        if iter_picard == MAX_ITER - 1:
            print_log(
                f"Tempo t = {YELLOW}{passo*dt/3600:.1f}{RESET}h | "
                f"{RED}AVISO:{RESET} não convergiu ({error:.2e})"
            )
            h_last = h_new.copy()
    h_old = h_last.copy()
    if passo % 36 == 0:
        h_list.append(h_last.copy())
        tempos.append(passo * dt)
    if passo == 3:
        print_log("...")


# ---------------------------------------------------------------------------
# 6) Saída em VTU e relatório final
# ---------------------------------------------------------------------------
if h_list:
    saida_vtu = "resultados/drenagem.pvd"
    points_vtk = points if points.shape[1] == 3 else np.column_stack((points, np.zeros((Nnodes, 1))))
    with open(saida_vtu, "w", encoding="utf-8") as f:
        f.write('<VTKFile type="Collection" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write("<Collection>\n")
        for i, h_step in enumerate(h_list):
            nome_vtu = f"drenagem_{i}.vtu"
            mesh = meshio.Mesh(points_vtk, [("triangle", triangles)], point_data={"Pressao_h": h_step})
            mesh.write(f"resultados/{nome_vtu}", file_format="vtu")
            f.write(f'<DataSet timestep="{tempos[i]/3600.0}" file="{nome_vtu}"/>\n')
        f.write("</Collection>\n")
        f.write("</VTKFile>\n")
    h_final = h_list[-1]
    print_log(f"\n{PURPLE}RESUMO ESTATÍSTICO FINAL (t={tempo_total/3600:.1f}h){RESET}\n")
    print_log(f"Pressão Mínima:      {YELLOW}{np.min(h_final):.3f}{RESET}")
    print_log(f"Pressão Máxima:      {YELLOW}{np.max(h_final):.3f}{RESET}")
    print_log(f"Pressão Média:       {YELLOW}{np.mean(h_final):.3f}{RESET}")
    print_log(f"Visualização: {YELLOW}{saida_vtu}{RESET} (ParaView)")
    print_log(f"Status da Execução:  {GREEN}SUCESSO{RESET}")
else:
    print_log(f"{RED}FALHA na simulação: nenhuma solução armazenada.{RESET}")

print_log(f"{YELLOW}{'=' * 80}{RESET}")
print_log(f"{PURPLE}FIM DO RELATÓRIO{RESET}\n")
