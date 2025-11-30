import numpy as np
import meshio
import os
from mpi4py import MPI
import dolfinx
import dolfinx.fem as fem
import dolfinx.io as io
import ufl
from petsc4py import PETSc
from base import MATERIAIS, condutividade_hidraulica, calcular_se
from config import aplicar_bc_externa, condicao_parede_rio, TAG_BORDA_ESQUERDA, TAG_BORDA_DIREITA, TAG_BORDA_BASE, TAG_BORDA_TOPO, TAG_PAREDE_RIO

# ===================== 1. Carregar a malha =====================
with io.XDMFFile(MPI.COMM_WORLD, "malha_dominio.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

# ===================== 2. Espaço de funções =====================
V = fem.FunctionSpace(mesh, ("CG", 1))
h = fem.Function(V)
h_n = fem.Function(V)

# ===================== 3. Parâmetros de tempo =====================
dt = 3600.0  # 1 hora
T = 24*3600.0  # 1 dia
t = 0.0

# ===================== 4. Condição inicial =====================
h.x.array[:] = -1.0
h_n.x.array[:] = -1.0

# ===================== 5. Condições de contorno =====================
def dirichlet_boundary(x):
    # Exemplo: Dirichlet em toda a fronteira (ajuste para seu caso real)
    return np.full(x.shape[1], True)

bc = fem.dirichletbc(PETSc.ScalarType(0.0), fem.locate_dofs_geometrical(V, dirichlet_boundary), V)
bcs = [bc]

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
if not os.path.exists("saida_richards") and MPI.COMM_WORLD.rank == 0:
    os.makedirs("saida_richards")

h_ = fem.Function(V)
num_steps = int(T/dt)
for step in range(num_steps):
    t += dt
    h_k = fem.Function(V)
    h_k.x.array[:] = h_n.x.array[:]
    for picard in range(5):
        # Avaliação local para C e K (aproximação)
        Ck = C(h_k.x.array)
        Kk = K(h_k.x.array)
        v = ufl.TestFunction(V)
        u = h_  # solução atual
        F = (Ck*(u - h_n)/dt)*v*ufl.dx + ufl.dot(Kk*ufl.grad(u + 1.0), ufl.grad(v))*ufl.dx
        problem = fem.petsc.NonlinearProblem(F, h_, bcs)
        solver = fem.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
        solver.solve(h_)
        if np.linalg.norm(h_.x.array - h_k.x.array, np.inf) < 1e-4:
            break
        h_k.x.array[:] = h_.x.array[:]
    h_n.x.array[:] = h_.x.array[:]
    # Salvar resultado
    with io.XDMFFile(MPI.COMM_WORLD, f"saida_richards/sol_{step:04d}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(h_)

if MPI.COMM_WORLD.rank == 0:
    print("Simulação finalizada. Resultados em 'saida_richards/'.")
