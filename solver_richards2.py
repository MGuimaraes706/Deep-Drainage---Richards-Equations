def theta(h):
def C(h):
def K(h):

# Solver de Richards 1D com elementos finitos lineares (FEM), sem FEniCS
import numpy as np
import meshio
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Parâmetros do problema
Ksat = 1e-6  # condutividade hidráulica saturada [m/s]
theta_s = 0.4  # teor de água na saturação
theta_r = 0.1  # teor de água residual
alpha = 0.8  # parâmetro de Van Genuchten [1/m]
n = 1.5  # parâmetro de Van Genuchten
m = 1 - 1/n
g = 9.81  # gravidade [m/s^2]

# Domínio 1D
L = 1.0  # comprimento [m]
Nx = 50
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]

# Condições de contorno
h_top = 0.0  # pressão no topo [m]
h_bottom = -2.0  # pressão na base [m]

# Funções constitutivas
def theta(h):
    return theta_r + (theta_s - theta_r) / (1 + (alpha * np.abs(h))**n)**m

def C(h):
    dh = 1e-6
    return (theta(h + dh) - theta(h - dh)) / (2 * dh)

def K(h):
    Se = (theta(h) - theta_r) / (theta_s - theta_r)
    return Ksat * Se**0.5 * (1 - (1 - Se**(1/m))**m)**2

# Inicialização
h0 = -1.0 * np.ones(Nx)
h0[0] = h_top
h0[-1] = h_bottom
h = h0.copy()

# Parâmetros de tempo
T = 3600.0 * 24  # tempo total [s]
dt = 60.0  # passo de tempo [s]
nt = int(T / dt)

# Montagem da matriz global (FEM linear 1D)
for step in range(nt):
    K_local = np.zeros(Nx-1)
    C_local = np.zeros(Nx)
    for e in range(Nx-1):
        h_e = 0.5 * (h[e] + h[e+1])
        K_local[e] = K(h_e)
    for i in range(Nx):
        C_local[i] = C(h[i])

    A = lil_matrix((Nx, Nx))
    M = lil_matrix((Nx, Nx))
    for e in range(Nx-1):
        # Elemento [e, e+1]
        ke = K_local[e] / dx
        me = 0.5 * dx * np.array([[C_local[e], 0],[0, C_local[e+1]]])
        # Rigidez
        A[e, e] += ke
        A[e, e+1] -= ke
        A[e+1, e] -= ke
        A[e+1, e+1] += ke
        # Massa
        M[e, e] += me[0,0]
        M[e, e+1] += me[0,1]
        M[e+1, e] += me[1,0]
        M[e+1, e+1] += me[1,1]

    # Sistema linear: (M + dt*A) h^{n+1} = M h^n
    lhs = M + dt*A
    rhs = M @ h

    # Condições de contorno de Dirichlet
    lhs[0,:] = 0
    lhs[0,0] = 1
    rhs[0] = h_top
    lhs[-1,:] = 0
    lhs[-1,-1] = 1
    rhs[-1] = h_bottom

    h_new = spsolve(lhs.tocsr(), rhs)
    h = h_new
    if step % 100 == 0:
        print(f"Passo {step}/{nt}")

# Exportar resultado para VTK (VTU)
points = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
cells = [("line", np.array([[i, i+1] for i in range(Nx-1)]))]
mesh = meshio.Mesh(points, cells, point_data={"h": h})
mesh.write("saida_solver2.vtu")
print("Simulação finalizada. Resultado salvo em saida_solver2.vtu.")
