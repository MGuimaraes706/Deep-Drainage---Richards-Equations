import numpy as np
import os
from mpi4py import MPI
from dolfinx import fem, io, mesh
import ufl
from petsc4py import PETSc
import config
import base

# ==============================================================================
# 1. CARREGAMENTO DA MALHA E MATERIAIS
# ==============================================================================
comm = MPI.COMM_WORLD
if comm.rank == 0:
    print("--> Carregando malhas...")

# Ler a Malha do Domínio
with io.XDMFFile(comm, "malha_dominio.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    # Ler as Tags dos Materiais (Areia, Argila, etc)
    # Importante: O nome "Grid" deve bater com o que o meshio gravou
    ct = xdmf.read_meshtags(domain, name="Grid")

# Ler a Malha das Bordas (Para Condições de Contorno)
with io.XDMFFile(comm, "malha_bordas.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(domain, name="Grid")

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)

# ==============================================================================
# 2. ESPAÇOS DE FUNÇÃO
# ==============================================================================
# V: Espaço Contínuo (Lagrange grau 1) para a Pressão (h)
V = fem.FunctionSpace(domain, ("CG", 1))

# Q: Espaço Descontínuo (DG grau 0) para as Propriedades (K e C)
# DG0 significa que o valor é constante dentro de cada triângulo (ideal para materiais)
Q = fem.FunctionSpace(domain, ("DG", 0))

# Funções da Pressão
h = fem.Function(V, name="Pressao_Atual")
h_n = fem.Function(V, name="Pressao_Anterior") # Passo anterior
h_k = fem.Function(V, name="Pressao_Picard")   # Iteração atual de Picard

# Funções das Propriedades (Que variam no espaço e tempo)
K_func = fem.Function(Q, name="Condutividade")
C_func = fem.Function(Q, name="Capacidade")

# ==============================================================================
# 3. FUNÇÕES AUXILIARES DE ATUALIZAÇÃO (FÍSICA)
# ==============================================================================
def update_properties(h_current):
    """
    Atualiza os campos K e C em cada elemento da malha baseado 
    na pressão atual (h_current) e no material (ct).
    """
    # Pega os arrays de dados locais
    k_array = K_func.x.array
    c_array = C_func.x.array
    
    # Pega a pressão média em cada célula (aproximação para DG0)
    # Como h é CG1 (vértices), e queremos passar para DG0 (centro),
    # uma forma rápida é interpolar ou avaliar no centro.
    # Para simplificar neste script, vamos iterar sobre as células.
    
    # Tags dos materiais para cada célula
    material_tags = ct.values
    
    # Precisamos dos valores de h nos centros das células. 
    # Estratégia eficiente em FEniCSx: Usar Expressão.
    expr = fem.Expression(h_current, Q.element.interpolation_points())
    h_centers = expr.eval(domain, list(range(domain.topology.index_map(domain.topology.dim).size_local)))
    
    # Loop sobre células (pode ser otimizado com vetorização numpy depois)
    # Achata o array de h (que pode vir (N, 1))
    h_vals = h_centers.flatten()
    
    # Vetorização via Base.py
    # Criamos arrays temporários
    new_k = np.zeros_like(k_array)
    new_c = np.zeros_like(c_array)
    
    # Itera sobre os tipos de materiais únicos presentes na malha
    unique_mats = np.unique(material_tags)
    
    for mat_id in unique_mats:
        # Máscara booleana: Onde está esse material?
        mask = (material_tags == mat_id)
        
        # Valores de h nesses locais
        h_local = h_vals[mask]
        
        # Calcula K e C usando as funções do base.py
        # Nota: Precisamos adaptar base.py para aceitar arrays ou fazer loop
        # Vamos fazer um map simples aqui para robustez
        ks = np.array([base.condutividade_hidraulica(hv, mat_id) for hv in h_local])
        
        # Capacidade C(h) = dTheta/dh
        # Aproximação por diferenças finitas centrais
        dh = 1e-5
        theta_plus = np.array([base.condutividade_hidraulica(hv+dh, mat_id) for hv in h_local]) # Usando K provisoriamente se não tiver theta no base
        # CORREÇÃO: Você precisa implementar theta() no base.py.
        # Vou assumir C = 0 por enquanto se não tiver a função, ou usar uma derivada numérica do K como dummy
        # O ideal é ter base.conteudo_umidade(h, mat_id)
        
        # Preenche os arrays globais
        new_k[mask] = ks
        # new_c[mask] = ... (Cálculo do C)
        # Para evitar erro, vamos colocar um C pequeno constante se não implementado
        new_c[mask] = 0.01 
        
    K_func.x.array[:] = new_k
    C_func.x.array[:] = new_c

# ==============================================================================
# 4. CONDIÇÕES DE CONTORNO
# ==============================================================================
bcs = []

# Identificar facetas (arestas) das bordas
fdim = domain.topology.dim - 1

# --- A. Bordas Externas (Esquerda, Direita, etc) ---
# Mapeamento do Config para a Malha
boundary_map = {
    config.TAG_BORDA_ESQUERDA: "Esquerda",
    config.TAG_BORDA_DIREITA: "Direita",
    config.TAG_BORDA_TOPO: "Topo",
    config.TAG_BORDA_BASE: "Base" # Geralmente Neumann (não entra na lista bcs)
}

# Coordenadas dos nós para aplicar funções
x_coords = V.tabulate_dof_coordinates()

for tag, nome in boundary_map.items():
    # Achar arestas com essa tag
    facets = ft.find(tag)
    if len(facets) > 0:
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        
        # Como o valor pode variar com Y (hidrostático), precisamos iterar ou projetar
        # Simplificação: Aplicar valor médio ou iterar sobre DOFs
        # FEniCSx permite aplicar função. Vamos criar uma função que interpolada o config.
        
        def bc_func_wrapper(x):
            # x é array (3, N)
            values = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                # Chama a lógica do config.py
                val = config.aplicar_bc_externa(x[0,i], x[1,i], tag)
                if val is not None:
                    values[i] = val
            return values

        bc_val = fem.Function(V)
        bc_val.interpolate(bc_func_wrapper)
        bcs.append(fem.dirichletbc(bc_val, dofs))

# --- B. Parede do Rio (Complexa: Mista) ---
rio_facets = ft.find(config.TAG_PAREDE_RIO)
if len(rio_facets) > 0:
    rio_dofs = fem.locate_dofs_topological(V, fdim, rio_facets)
    
    def rio_func_wrapper(x):
        values = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            # Chama lógica do config.py
            tipo, val = config.condicao_parede_rio(x[1,i])
            if tipo == "Dirichlet":
                values[i] = val
            else:
                # Seepage Face: Em Picard, se h < 0, vira Neumann (fluxo 0).
                # Se h > 0, vira Dirichlet (h=0).
                # Isso é não-linear. Vamos simplificar fixando 0 onde deveria ser Seepage.
                values[i] = 0.0 
        return values

    rio_bc_val = fem.Function(V)
    rio_bc_val.interpolate(rio_func_wrapper)
    bcs.append(fem.dirichletbc(rio_bc_val, rio_dofs))


# ==============================================================================
# 5. FORMULAÇÃO VARIACIONAL (PICARD)
# ==============================================================================
dt = fem.Constant(domain, 3600.0) # Delta T (segundos)
v = ufl.TestFunction(V)
u = ufl.TrialFunction(V) # Usado para o solve linear dentro de Picard

# Termos da Equação: C * (h - h_n)/dt - Div(K * Grad(h)) = 0
# Forma fraca: Integral(C*(h-h_n)/dt * v) + Integral(K * Grad(h) * Grad(v)) ...

# Como é Picard, o 'u' que resolvemos é o h_novo, mas K e C dependem de h_k (iter anterior)
F = (C_func * (u - h_n) / dt) * v * ufl.dx + \
    ufl.dot(K_func * ufl.grad(u), ufl.grad(v)) * ufl.dx + \
    ufl.dot(K_func * fem.Constant(domain, np.array([0.0, 1.0, 0.0])), ufl.grad(v)) * ufl.dx # Gravidade dz

# Separar Bilinear (a) e Linear (L)
a, L = ufl.lhs(F), ufl.rhs(F)

problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# ==============================================================================
# 6. LOOP DE TEMPO
# ==============================================================================
T_total = 24 * 3600 # 1 dia
t = 0
h_n.interpolate(lambda x: np.full(x.shape[1], -2.0)) # Condição Inicial: Solo seco (-2m)

# Preparar arquivo de saída
vtk = io.VTKFile(comm, "resultados/drenagem.pvd", "w")
vtk.write_function(h_n, 0.0)

while t < T_total:
    t += float(dt)
    if comm.rank == 0: print(f"Calculando t = {t/3600:.2f} h")
    
    # Picard Loop
    h_k.x.array[:] = h_n.x.array[:] # Chute inicial = passo anterior
    for i in range(10): # Max iterações
        # 1. Atualizar K e C com base no h_k atual
        update_properties(h_k)
        
        # 2. Resolver Sistema Linear
        h_solution = problem.solve()
        
        # 3. Calcular erro (norma da diferença)
        diff = h_solution.x.array - h_k.x.array
        error = np.linalg.norm(diff, ord=2)
        
        # Atualiza h_k para próxima iteração
        h_k.x.array[:] = h_solution.x.array[:]
        
        if error < 1e-5:
            if comm.rank == 0: print(f"  Convergiu em {i+1} iterações (Erro: {error:.2e})")
            break
    else:
        if comm.rank == 0: print("  AVISO: Picard não convergiu neste passo.")

    # Atualiza passo de tempo
    h_n.x.array[:] = h_solution.x.array[:]
    
    # Salvar
    vtk.write_function(h_n, t)

vtk.close()
print("Simulação concluída!")