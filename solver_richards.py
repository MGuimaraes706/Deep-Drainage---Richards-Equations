import numpy as np
import meshio
from dolfin import *
from base import MATERIAIS, condutividade_hidraulica, calcular_se
from config import aplicar_bc_externa, condicao_parede_rio, TAG_BORDA_ESQUERDA, TAG_BORDA_DIREITA, TAG_BORDA_BASE, TAG_BORDA_TOPO, TAG_PAREDE_RIO
import os

# ===================== 1. Carregar a malha =====================
mesh = Mesh()
with XDMFFile("malha_dominio.xdmf") as infile:
    infile.read(mesh)

mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("malha_bordas.xdmf") as infile:
    infile.read(mvc, "borda_tag")
boundaries = MeshFunction("size_t", mesh, mvc)

# ===================== 2. Espaço de funções =====================
V = FunctionSpace(mesh, "CG", 1)
h = Function(V)  # Solução (carga de pressão)
h_n = Function(V)  # Solução no passo anterior

# ===================== 3. Parâmetros de tempo =====================
dt = 3600.0  # 1 hora
T = 24*3600.0  # 1 dia
t = 0.0

# ===================== 4. Condição inicial =====================
h_init = Constant(-1.0)
h.interpolate(h_init)
h_n.assign(h)

# ===================== 5. Condições de contorno =====================
def bc_externa(x, on_boundary):
    # Exemplo: Dirichlet nas bordas esquerda/direita
    # (Ajuste conforme seu problema e tags)
    return on_boundary

bcs = [DirichletBC(V, Constant(0.0), bc_externa)]

# ===================== 6. Parâmetros do solo (exemplo: areia fina) =====================
mat_tag = 1
props = MATERIAIS[mat_tag]
ks = props["ks"]
theta_s = props["theta_s"]
theta_r = props["theta_r"]
alpha = props["alpha"]
n = props["n"]

def theta(h_val):
    Se = calcular_se(h_val, alpha, n)
    return theta_r + (theta_s - theta_r) * Se

def C(h_val):
    dh = 1e-6
    return (theta(h_val + dh) - theta(h_val - dh)) / (2*dh)

def K(h_val):
    return condutividade_hidraulica(h_val, mat_tag)

# ===================== 7. Solver de Richards (Backward Euler + Picard) =====================
h_ = Function(V)
if not os.path.exists("saida_richards"): os.makedirs("saida_richards")
for step in range(int(T/dt)):
    t += dt
    h_k = h_n.copy(deepcopy=True)
    for picard in range(5):
        Ck = project(C(h_k), V)
        Kk = project(K(h_k), V)
        F = (Ck*(h_ - h_n)/dt)*TestFunction(V)*dx + dot(Kk*grad(h_ + Constant(1.0)), grad(TestFunction(V)))*dx
        solve(lhs(F) == rhs(F), h_, bcs)
        if np.linalg.norm(h_.vector().get_local() - h_k.vector().get_local(), np.inf) < 1e-4:
            break
        h_k.assign(h_)
    h_n.assign(h_)
    # Salvar resultado
    vtkfile = File(f"saida_richards/sol_{step:04d}.pvd")
    vtkfile << h_

print("Simulação finalizada. Resultados em 'saida_richards/'.")
