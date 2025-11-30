import numpy as np
import os
from mpi4py import MPI
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem import petsc  # Importação direta do módulo petsc
import ufl
import basix.ufl 
from petsc4py import PETSc

# Importa módulos locais
import base
import config

# ==============================================================================
# 1. CARREGAR MALHAS
# ==============================================================================
comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    print(f"--> Iniciando Solver (Modo Manual PETSc). Vazão: {config.VAZAO_RIO} m3/s")

# Ler Malhas
try:
    with io.XDMFFile(comm, "malha_dominio.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(domain, name="Grid")
    with io.XDMFFile(comm, "malha_bordas.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(domain, name="Grid")
except Exception as e:
    if rank == 0: print(f"ERRO CRÍTICO ao ler malhas: {e}")
    exit()

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)

# ==============================================================================
# 2. ESPAÇOS DE FUNÇÃO
# ==============================================================================
# P1 (Pressão)
P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
V = fem.functionspace(domain, P1)

# DG0 (Propriedades Físicas)
DG0 = basix.ufl.element("DG", domain.topology.cell_name(), 0)
Q = fem.functionspace(domain, DG0)

# ==============================================================================
# 3. CAMPOS E FUNÇÕES
# ==============================================================================
h = fem.Function(V, name="Pressao_Atual")           
h_n = fem.Function(V, name="Pressao_Passo_Anterior") 
h_iter = fem.Function(V, name="Pressao_Auxiliar")    
h_new = fem.Function(V, name="Pressao_Nova_Iteracao") # Variável extra para o solver

K_field = fem.Function(Q, name="Condutividade_K")
C_field = fem.Function(Q, name="Capacidade_C")

# ==============================================================================
# 4. ATUALIZAÇÃO DA FÍSICA
# ==============================================================================
def update_material_properties(h_func):
    material_tags = ct.values
    expr = fem.Expression(h_func, Q.element.interpolation_points())
    h_vals = expr.eval(domain, list(range(domain.topology.index_map(domain.topology.dim).size_local))).flatten()
    
    k_vals = np.zeros_like(h_vals)
    c_vals = np.zeros_like(h_vals)
    dh = 1e-4
    
    for tag in np.unique(material_tags):
        cells = (material_tags == tag)
        if np.any(cells):
            h_loc = h_vals[cells]
            ks = np.array([base.condutividade_hidraulica(val, tag) for val in h_loc])
            k_vals[cells] = ks
            
            th_p = np.array([base.conteudo_umidade(val + dh, tag) for val in h_loc])
            th_m = np.array([base.conteudo_umidade(val - dh, tag) for val in h_loc])
            cs = (th_p - th_m) / (2 * dh)
            c_vals[cells] = np.maximum(cs, 1e-9)

    K_field.x.array[:] = k_vals
    C_field.x.array[:] = c_vals

# ==============================================================================
# 5. CONDIÇÕES DE CONTORNO
# ==============================================================================
bcs = []
fdim = domain.topology.dim - 1

# Bordas Externas
bc_map = {config.TAG_BORDA_ESQUERDA: "Esquerda", config.TAG_BORDA_DIREITA: "Direita", config.TAG_BORDA_TOPO: "Topo"}
for tag, nome in bc_map.items():
    facets = ft.find(tag)
    if len(facets) > 0:
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        def bc_wrapper(x):
            vals = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                val = config.aplicar_bc_externa(x[0,i], x[1,i], tag)
                if val is not None: vals[i] = val
            return vals
        bc_func = fem.Function(V)
        bc_func.interpolate(bc_wrapper)
        bcs.append(fem.dirichletbc(bc_func, dofs))

# Parede do Rio
rio_facets = ft.find(config.TAG_PAREDE_RIO)
if len(rio_facets) > 0:
    rio_dofs = fem.locate_dofs_topological(V, fdim, rio_facets)
    def rio_wrapper(x):
        vals = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            tipo, val = config.condicao_parede_rio(x[1,i])
            vals[i] = val 
        return vals
    rio_bc_func = fem.Function(V)
    rio_bc_func.interpolate(rio_wrapper)
    bcs.append(fem.dirichletbc(rio_bc_func, rio_dofs))

# ==============================================================================
# 6. FORMULAÇÃO E PREPARAÇÃO DO SOLVER MANUAL (KSP)
# ==============================================================================
dt = fem.Constant(domain, default_scalar_type(3600.0))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Resíduo
F = (C_field * (u - h_n) / dt) * v * ufl.dx \
  + ufl.dot(K_field * ufl.grad(u), ufl.grad(v)) * ufl.dx \
  + K_field * ufl.grad(v)[1] * ufl.dx 

a_form = fem.form(ufl.lhs(F)) # Bilinear (Lado Esquerdo)
L_form = fem.form(ufl.rhs(F)) # Linear (Lado Direito)

# Criar Matriz (A) e Vetor (b) vazios
A = petsc.create_matrix(a_form)
b = petsc.create_vector(L_form)

# Configurar o Solver Linear PETSc (KSP)
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY) # Solver direto
solver.getPC().setType(PETSc.PC.Type.LU) # Fatoração LU

# ==============================================================================
# 7. LOOP DE TEMPO
# ==============================================================================
T_total = 24 * 3600
t = 0.0
h_n.x.array[:] = -1.0 

if not os.path.exists("resultados"): os.makedirs("resultados")
vtk = io.VTKFile(comm, "resultados/drenagem.pvd", "w")
vtk.write_function(h_n, 0.0)

while t < T_total:
    t += dt.value
    if rank == 0: print(f"=== Tempo: {t/3600:.1f} h ===")
    
    h_iter.x.array[:] = h_n.x.array[:] # Chute inicial = passo anterior
    
    # Loop de Picard
    for i in range(15):
        # 1. Atualizar propriedades físicas
        update_material_properties(h_iter)
        
        # 2. Montar Sistema Linear (Assemble)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()
        
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_form)
        
        # Aplicar Lifting (BCs de Dirichlet)
        petsc.apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # 3. Resolver A * x = b
        solver.solve(b, h_new.vector)
        h_new.x.scatter_forward()
        
        # 4. Verificar Convergência
        diff = h_new.x.array - h_iter.x.array
        error = np.linalg.norm(diff) / (np.linalg.norm(h_new.x.array) + 1e-6)
        
        h_iter.x.array[:] = h_new.x.array[:]
        
        if error < 1e-4:
            if rank == 0: print(f"   Convergiu na iteração {i+1} (Erro: {error:.2e})")
            break
    else:
        if rank == 0: print("   AVISO: Picard não convergiu neste passo.")

    h_n.x.array[:] = h_iter.x.array[:]
    vtk.write_function(h_n, t)

vtk.close()
if rank == 0: print("Fim. Verifique a pasta 'resultados'.")