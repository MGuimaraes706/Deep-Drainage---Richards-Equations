import meshio
import matplotlib.pyplot as plt
import numpy as np

# Lê o arquivo VTU exportado pelo solver
mesh = meshio.read("saida_solver2.vtu")
points = mesh.points[:, :2]  # 2D
triangles = mesh.cells_dict["triangle"]
pressure = mesh.point_data["h"]

plt.figure(figsize=(10,4))
tri = plt.tricontourf(points[:,0], points[:,1], triangles, pressure, levels=30, cmap="viridis")
cbar = plt.colorbar(tri)
cbar.set_label("Pressão h (m)")
plt.title("Campo de Pressão - Solução de Richards 2D")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.tight_layout()
plt.show()
