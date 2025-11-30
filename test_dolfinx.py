import dolfinx
import dolfinx.mesh
import dolfinx.fem as fem
from mpi4py import MPI

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
V = fem.FunctionSpace(mesh, ("Lagrange", 1))
u = fem.Function(V)
print("Dolfinx está funcionando!")
