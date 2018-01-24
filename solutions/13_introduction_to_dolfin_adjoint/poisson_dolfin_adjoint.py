from fenics import *
from dolfin_adjoint import *

# Define mesh and finite element space
mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define basis functions and parameters
u = TrialFunction(V)
v = TestFunction(V)
m = interpolate(Constant(1.0), V)
nu = Constant(1.0)

# Define variational problem
a = nu*inner(grad(u), grad(v))*dx
L = m*v*dx
bc = DirichletBC(V, 0.0, "on_boundary")

# Solve variational problem
u = Function(V)
solve(a == L, u, bc)
plot(u, title="u")

# Define 'observations'
u_d = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=4)

# Assemble the 'misfit' functional
J = assemble(0.5*(u - u_d)*(u - u_d)*dx)

# Define the control variable
ctrl = Control(m)

# Compute gradient of J with respect to m (dJ/dm):
dJdm = compute_gradient(J, ctrl, options={"riesz_representation": "L2"})
File("dJdm.pvd") << dJdm

# Run Taylortest
R = ReducedFunctional(J, ctrl)
taylor_test(R, m, m)

# Compute Hessian in direction 1:
direction = interpolate(Constant(1.0), V)
Hdir = compute_hessian(J, ctrl, direction, options={"riesz_representation": "L2"})
File("Hdir.pvd") << Hdir

# Run optimization
m_opt = minimize(R)
File("m_opt.pvd") << m_opt

