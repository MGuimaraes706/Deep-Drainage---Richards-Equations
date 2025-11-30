import dolfinx
import dolfinx.mesh
import dolfinx.fem as fem

mesh = dolfinx.mesh.create_unit_square(dolfinx.MPI.comm_world, 8, 8)
V = fem.FunctionSpace(mesh, ("Lagrange", 1))
u = fem.Function(V)
print("Dolfinx está funcionando!")
