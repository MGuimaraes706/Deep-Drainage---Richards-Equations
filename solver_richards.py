import numpy as np
import os
from mpi4py import MPI
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
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
    print("--> Iniciando Solver de Richards...")
    print(f"--> Vazão do Rio Configurada: {config.VAZAO_RIO} m3/s")

# Ler Malha do Domínio (Triângulos)
# O nome "Grid" deve bater com o que foi salvo no mesh_converter.py
with io.XDMFFile(comm, "malha_dominio.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid") # Tags dos Materiais

# Ler Malha das Bordas (Linhas)
with io.XDMFFile(comm, "malha_bordas.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(domain, name="Grid") # Tags das Fronteiras

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)

# ==============================================================================
# 2. ESPAÇOS DE FUNÇÃO (Correção para FEniCSx v0.8+)
# ==============================================================================

# Elemento P1 (Lagrange grau 1) para a Pressão (h) - Contínuo
P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
V = fem.functionspace(domain, P1)

# Elemento DG0 (Discontinuous Galerkin grau 0) para Propriedades Físicas (K e C)
# DG0 é constante dentro de cada elemento, essencial para materiais com saltos bruscos
DG0 = basix.ufl.element("DG", domain.topology.cell_name(), 0)
Q = fem.functionspace(domain, DG0)

# ==============================================================================
# 3. CAMPOS E FUNÇÕES
# ==============================================================================

h = fem.Function(V, name="Pressao_Atual")           # Solução do passo atual (k)
h_n = fem.Function(V, name="Pressao_Passo_Anterior") # Solução do tempo anterior (n)
h_iter = fem.Function(V, name="Pressao_Auxiliar_Picard") # Auxiliar para iteração

# Propriedades Materiais (Variáveis no espaço e tempo)
K_field = fem.Function(Q, name="Condutividade_K")
C_field = fem.Function(Q, name="Capacidade_C")

# ==============================================================================
# 4. LÓGICA DE ATUALIZAÇÃO DA FÍSICA (Loop sobre Materiais)
# ==============================================================================
def update_material_properties(h_func):
    """
    Atualiza os campos K e C em toda a malha com base na pressão h_func.
    Lê as tags do arquivo XDMF para aplicar as propriedades de Areia, Argila ou Brita.
    """
    # 1. Obter tags dos materiais
    material_tags = ct.values
    
    # 2. Interpolar a pressão h para o centro dos elementos (DG0 precisa disso)
    expr = fem.Expression(h_func, Q.element.interpolation_points())
    h_vals = expr.eval(domain, list(range(domain.topology.index_map(domain.topology.dim).size_local))).flatten()
    
    # Arrays temporários
    k_vals = np.zeros_like(h_vals)
    c_vals = np.zeros_like(h_vals)
    
    dh = 1e-4 # Delta h para derivada numérica da capacidade
    
    # 3. Loop sobre cada material existente na malha
    for tag in np.unique(material_tags):
        # Máscara: quais células são deste material?
        cells = (material_tags == tag)
        
        if np.any(cells):
            h_loc = h_vals[cells]
            
            # --- Calcular K (Condutividade) ---
            # Chama a função do base.py vetorizada
            ks = np.array([base.condutividade_hidraulica(val, tag) for val in h_loc])
            k_vals[cells] = ks
            
            # --- Calcular C (Capacidade = dTheta/dh) ---
            # Diferenças finitas centradas usando a função conteudo_umidade do base.py
            th_p = np.array([base.conteudo_umidade(val + dh, tag) for val in h_loc])
            th_m = np.array([base.conteudo_umidade(val - dh, tag) for val in h_loc])
            cs = (th_p - th_m) / (2 * dh)
            
            # Evitar zero absoluto para estabilidade numérica
            cs = np.maximum(cs, 1e-9)
            c_vals[cells] = cs
            
    # 4. Atualiza as funções do FEniCS
    K_field.x.array[:] = k_vals
    C_field.x.array[:] = c_vals

# ==============================================================================
# 5. CONDIÇÕES DE CONTORNO
# ==============================================================================
bcs = []
fdim = domain.topology.dim - 1

# Mapeamento do config.py para nomes legíveis
bc_map = {
    config.TAG_BORDA_ESQUERDA: "Esquerda",
    config.TAG_BORDA_DIREITA: "Direita",
    config.TAG_BORDA_TOPO: "Topo"
}

# --- A. Bordas Externas ---
for tag, nome in bc_map.items():
    facets = ft.find(tag)
    if len(facets) > 0:
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        
        # Função wrapper para interpolar valores do config.py
        def bc_wrapper(x):
            vals = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                val = config.aplicar_bc_externa(x[0,i], x[1,i], tag)
                if val is not None: vals[i] = val
            return vals
            
        bc_func = fem.Function(V)
        bc_func.interpolate(bc_wrapper)
        bcs.append(fem.dirichletbc(bc_func, dofs))

# --- B. Parede do Rio (Condição Mista) ---
rio_facets = ft.find(config.TAG_PAREDE_RIO)
if len(rio_facets) > 0:
    rio_dofs = fem.locate_dofs_topological(V, fdim, rio_facets)
    
    def rio_wrapper(x):
        vals = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            tipo, val = config.condicao_parede_rio(x[1,i])
            # Simplificação robusta para Picard: Impomos o valor retornado.
            vals[i] = val 
        return vals

    rio_bc_func = fem.Function(V)
    rio_bc_func.interpolate(rio_wrapper)
    bcs.append(fem.dirichletbc(rio_bc_func, rio_dofs))

# ==============================================================================
# 6. FORMULAÇÃO VARIACIONAL E SOLVER
# ==============================================================================
dt = fem.Constant(domain, default_scalar_type(3600.0)) # Passo de tempo: 1 hora
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Resíduo da Equação de Richards (Discretização Backward Euler no tempo)
# Termo Temporal: C * (h - h_n) / dt
# Termo Difusivo: div(K * grad(h)) -> K * dot(grad(u), grad(v)) na forma fraca
# Termo Gravidade: div(K * ez) -> K * d(v)/dy na forma fraca

F = (C_field * (u - h_n) / dt) * v * ufl.dx \
  + ufl.dot(K_field * ufl.grad(u), ufl.grad(v)) * ufl.dx \
  + K_field * ufl.grad(v)[1] * ufl.dx 

problem = LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=bcs, 
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# ==============================================================================
# 7. LOOP DE TEMPO
# ==============================================================================
T_total = 24 * 3600 # 1 dia
t = 0.0

# Condição Inicial: Solo não saturado (-1.0m)
h_n.x.array[:] = -1.0 
h_iter.x.array[:] = -1.0 

# Preparar arquivo de saída
if not os.path.exists("resultados"): os.makedirs("resultados")
vtk = io.VTKFile(comm, "resultados/drenagem.pvd", "w")
vtk.write_function(h_n, 0.0)

while t < T_total:
    t += dt.value
    if rank == 0: print(f"=== Tempo: {t/3600:.1f} h ===")
    
    # Início do Picard
    h_iter.x.array[:] = h_n.x.array[:]
    
    for i in range(15):
        # 1. Atualizar física com base no 'chute' atual
        update_material_properties(h_iter)
        
        # 2. Resolver sistema linear
        try:
            h_new = problem.solve()
        except Exception as e:
            if rank==0: print(f"Erro no solver linear: {e}")
            break
            
        # 3. Calcular erro
        diff = h_new.x.array - h_iter.x.array
        error = np.linalg.norm(diff) / (np.linalg.norm(h_new.x.array) + 1e-6)
        
        h_iter.x.array[:] = h_new.x.array[:]
        
        if error < 1e-4:
            if rank == 0: print(f"   Convergiu na iteração {i+1} (Erro: {error:.2e})")
            break
    else:
        if rank == 0: print("   AVISO: Picard não convergiu neste passo.")

    # Atualiza passo de tempo
    h_n.x.array[:] = h_iter.x.array[:]
    
    # Salva resultado
    vtk.write_function(h_n, t)

vtk.close()
if rank == 0: print("Simulação Concluída. Abra 'resultados/drenagem.pvd' no ParaView.")