""" This program optimises the control of the Navier-Stokes equation """
from fenics import *
from mshr import *

# Create a rectangle with a circular hole.
rect = Rectangle(Point(0.0, 0.0), Point(30.0, 10.0))
circ = Circle(Point(10, 5), 2.5)
domain = rect - circ
N = 50  # Mesh resolution
mesh = generate_mesh(domain, N)
plot(mesh)

# Define function spaces (P2-P1)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) 
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
DG1 = FiniteElement("DG", mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, P2)  # Velocity
Q = FunctionSpace(mesh, P1)  # Pressure
D = FunctionSpace(mesh, DG1) # Control space
W = FunctionSpace(mesh, P2*P1)

# Define test and solution functions
v, q = TestFunctions(W)
s = Function(W)
u, p = split(s)

# Set parameter values
nu = Constant(1.0)   # Viscosity coefficient
f = Function(D)      # Control

# Define boundary conditions
noslip = DirichletBC(W.sub(0), (0, 0), "on_boundary && x[0] > 0.0 && x[0] < 30")
p0 = Expression("a*(30.0 - x[0])/30.0", degree=1, a=1.0)
bcu = [noslip]
n = FacetNormal(mesh)

# Define the indicator function for the control area
xcoor = SpatialCoordinate(mesh)
chi = conditional(xcoor[1] >= 5, 1, 0)

# Define the variational formulation of the Navier-Stokes equations
F = (inner(grad(u)*u, v)*dx +                 # Advection term
     nu*inner(grad(u), grad(v))*dx -          # Diffusion term
     inner(p, div(v))*dx +                    # Pressure term
     inner(chi*f*u, v)*dx +                   # Sponge term
     div(u)*q*dx +                            # Divergence term
     p0*dot(v, n)*ds
)

# Solve the Navier-Stokes equations
solve(F == 0, s, bcs=bcu)

# Save results
u, p = s.split(deepcopy=True)
File("u.pvd") << u
File("p.pvd") << p
