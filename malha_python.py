import gmsh
import meshio
import numpy as np
import os

# ===============================
# 1. PARÂMETROS GEOMÉTRICOS
# ===============================
L_SOLO = 10.0
H_TOTAL = 6.0
H_INTERFACE = 2.5
W_DUTO_INT = 2.0
H_DUTO_INT = 1.5
ESP_PAREDE = 0.20
ESP_BRITA = 0.30
X_CENTRO = L_SOLO / 2.0
Y_FUNDO_INT = 1.0
LC_GROSSO = 0.4
LC_FINO = 0.08

# ===============================
# 2. INICIALIZAÇÃO GMSH
# ===============================
gmsh.initialize()
gmsh.model.add("Drenagem_Profunda")
occ = gmsh.model.occ

# ===============================
# 3. CRIAÇÃO DAS FORMAS
# ===============================
r_solo_inf = occ.addRectangle(0, 0, 0, L_SOLO, H_INTERFACE)
r_solo_sup = occ.addRectangle(0, H_INTERFACE, 0, L_SOLO, H_TOTAL - H_INTERFACE)
x_brita = X_CENTRO - (W_DUTO_INT/2 + ESP_PAREDE + ESP_BRITA)
y_brita = Y_FUNDO_INT - ESP_PAREDE - ESP_BRITA
w_brita = W_DUTO_INT + 2*(ESP_PAREDE + ESP_BRITA)
h_brita = H_DUTO_INT + 2*(ESP_PAREDE + ESP_BRITA)
r_brita_bruto = occ.addRectangle(x_brita, y_brita, 0, w_brita, h_brita)
x_conc = X_CENTRO - (W_DUTO_INT/2 + ESP_PAREDE)
y_conc = Y_FUNDO_INT - ESP_PAREDE
w_conc = W_DUTO_INT + 2*ESP_PAREDE
h_conc = H_DUTO_INT + 2*ESP_PAREDE
r_conc_bruto = occ.addRectangle(x_conc, y_conc, 0, w_conc, h_conc)
x_int = X_CENTRO - W_DUTO_INT/2
y_int = Y_FUNDO_INT
r_buraco = occ.addRectangle(x_int, y_int, 0, W_DUTO_INT, H_DUTO_INT)
occ.synchronize()

# ===============================
# 4. OPERAÇÕES BOOLEANAS
# ===============================
res_conc, _ = occ.cut([(2, r_conc_bruto)], [(2, r_buraco)], removeTool=False)
id_concreto = res_conc[0][1]
res_brita, _ = occ.cut([(2, r_brita_bruto)], [(2, r_conc_bruto)], removeTool=False)
id_brita = res_brita[0][1]
solos_originais = [(2, r_solo_inf), (2, r_solo_sup)]
ferramenta_corte = [(2, r_brita_bruto)]
res_solos, _ = occ.cut(solos_originais, ferramenta_corte, removeTool=False)
for dim, tag in res_solos:
    mass = occ.getCenterOfMass(dim, tag)
    y_c = mass[1]
    if y_c < H_INTERFACE:
        id_areia = tag
    else:
        id_argila = tag
occ.remove([(2, r_buraco)], recursive=True)
occ.synchronize()

# ===============================
# 5. GRUPOS FÍSICOS
# ===============================
TAG_AREIA = 1
TAG_ARGILA = 2
TAG_BRITA = 3
TAG_CONCRETO = 4
gmsh.model.addPhysicalGroup(2, [id_areia], TAG_AREIA, "Solo_Inferior")
gmsh.model.addPhysicalGroup(2, [id_argila], TAG_ARGILA, "Solo_Superior")
gmsh.model.addPhysicalGroup(2, [id_brita], TAG_BRITA, "Brita")
gmsh.model.addPhysicalGroup(2, [id_concreto], TAG_CONCRETO, "Concreto")
eps = 1e-3
def get_lines_in_box(xmin, ymin, xmax, ymax):
    return gmsh.model.getEntitiesInBoundingBox(xmin-eps, ymin-eps, -eps, xmax+eps, ymax+eps, eps, dim=1)
lines_left = get_lines_in_box(0, 0, 0, H_TOTAL)
tags_left = [tag for dim, tag in lines_left]
gmsh.model.addPhysicalGroup(1, tags_left, 101, "Esquerda")
lines_right = get_lines_in_box(L_SOLO, 0, L_SOLO, H_TOTAL)
tags_right = [tag for dim, tag in lines_right]
gmsh.model.addPhysicalGroup(1, tags_right, 102, "Direita")
lines_base = get_lines_in_box(0, 0, L_SOLO, 0)
tags_base = [tag for dim, tag in lines_base]
gmsh.model.addPhysicalGroup(1, tags_base, 103, "Base")
lines_top = get_lines_in_box(0, H_TOTAL, L_SOLO, H_TOTAL)
tags_top = [tag for dim, tag in lines_top]
gmsh.model.addPhysicalGroup(1, tags_top, 104, "Topo")
lines_rio = get_lines_in_box(x_int, y_int, x_int+W_DUTO_INT, y_int+H_DUTO_INT)
tags_rio = [tag for dim, tag in lines_rio]
gmsh.model.addPhysicalGroup(1, tags_rio, 105, "Rio")

# ===============================
# 6. REFINAMENTO DE MALHA
# ===============================
points_aduela = gmsh.model.getBoundary([(2, id_concreto)], combined=False, recursive=True)
p_tags = [tag for dim, tag in points_aduela if dim==0]
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", p_tags)
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", LC_FINO)
gmsh.model.mesh.field.setNumber(2, "SizeMax", LC_GROSSO)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)
gmsh.model.mesh.field.setNumber(2, "DistMax", 3.0)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

# ===============================
# 7. GERAÇÃO E EXPORTAÇÃO
# ===============================
gmsh.model.mesh.generate(2)
gmsh.write("temp_malha.msh")
gmsh.finalize()
print("--> Malha gerada pelo Gmsh. Convertendo para XDMF...")
msh = meshio.read("temp_malha.msh")
tri_cells = [c for c in msh.cells if c.type == "triangle"]
if tri_cells:
    tri_data = []
    for i, c in enumerate(msh.cells):
        if c.type == "triangle":
            tri_data.append(msh.cell_data["gmsh:physical"][i])
    triangle_mesh = meshio.Mesh(
        points=msh.points[:, :2],
        cells=tri_cells,
        cell_data={"material_tag": tri_data}
    )
    meshio.write("malha_dominio.xdmf", triangle_mesh)
    print("    [OK] malha_dominio.xdmf")
line_cells = [c for c in msh.cells if c.type == "line"]
if line_cells:
    line_data = []
    for i, c in enumerate(msh.cells):
        if c.type == "line":
            line_data.append(msh.cell_data["gmsh:physical"][i])
    line_mesh = meshio.Mesh(
        points=msh.points[:, :2],
        cells=line_cells,
        cell_data={"borda_tag": line_data}
    )
    meshio.write("malha_bordas.xdmf", line_mesh)
    print("    [OK] malha_bordas.xdmf")
if os.path.exists("temp_malha.msh"):
    os.remove("temp_malha.msh")
print("--> Malha pronta! Execute apenas 'python malha_2D.py' para rodar todo o pipeline.")
