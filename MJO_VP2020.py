import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from dedalus.extras import flow_tools
import time
import requests
from scipy.special import erf
from dedalus.core.operators import GeneralFunction
import pathlib
import subprocess
import h5py
import shutil
from datetime import datetime
import logging
import os 
import pandas as pd
shutil.rmtree('analysis', ignore_errors=True)
plt.rcParams['text.usetex'] = True
days = 24*60*60


########################################################################################
# Set Path 
short_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
path = f'/Users/luitbald/CODE/movies/MJO_{short_datetime}'
########################################################################################
# Set Parameters

Nx, Ny = 128, 128 # Resolution
tau = 900.0 #Condensation timescale - timestep is limited to TIMESTEP_CAP*tau
tau_r = 2* days  # radiative relaxation timescale. "is of the order of a few days" VP2020 
tau_e = 2.4* days #evaporative timescale, analagous to lambda_, from VP reply
nu_ = 1*10**5.5# multiplied by 2 for nu_q, below
gamma =8 #  (L*H*Q_a)/ (q_0*c_p*T_0) # ~8.5 in VP2020, 5 in reply
alpha = 60 # L / (Rv*T_0) # ~20 in VP2020, 2 in reply

TIMESTEP_CAP = 0.8 # factor of tau for maximum timestep 
RUN_TIME = 30 # simulation stop time
########################################################################################
########################################################################################


### Domain 
Lx = 1.0e7 
Ly = 1.0e7 
xmin, xmax = -Lx/2, Lx/2
ymin, ymax = -Ly/2, Ly/2
f0 = 0
f_ = 4.0e-4 # For f-plane 
beta = 2.0e-11 # For beta plane 


### Simulation Parameters 
g = 10.0 #g
H = 30.0 #Mean height

h_0 = 0 #VP2020
q_0 = 1.0 # "Taken to be unity"
q_g = 1.0
L = 2.4*10**6 # Latent heat of condensation
Q_a = 0.035 # Tropical surfaace saturated specific humidy
c_p = 1004 # c_p 
T_0 = 300 
Rv = 462 # Water vapor gas constant

nu_u =nu_ # Viscosities, VP 2020 #raised significantly per Keaton Burns  - defined above
nu_h =nu_
nu_q =2*nu_

########################################################################################
# Dedalus setup

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype = np.float64)
x_basis = d3.RealFourier(coords['x'] , size = Nx, bounds = (xmin, xmax), dealias=3/2)
# y_basis = d3.RealFourier(coords['y'] , size = Ny, bounds = (ymin, ymax), dealias=3/2)
y_basis = d3.Chebyshev(coords['y'] , size = Ny, bounds = (ymin, ymax), dealias=3/2)
tau_basis = y_basis.derivative_basis(1)
x = dist.local_grid(x_basis)
y = dist.local_grid(y_basis)

U = dist.VectorField(coords, name='U', bases=(x_basis,y_basis))
h = dist.Field(name = 'h', bases = (x_basis, y_basis))
q = dist.Field(name = 'q', bases = (x_basis, y_basis))
f = dist.Field(name='f', bases=(y_basis))

# f['g'] = f_ # f-plane
f['g'] = f0 + beta*y  #beta plane

grad = lambda A: d3.Gradient(A)
lap = lambda A: d3.Laplacian(A)
div = lambda A: d3.Divergence(A)
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
zcross = lambda A: d3.Skew(A)
trace = lambda A: d3.Trace(A)
dot = lambda A,B: d3.DotProduct(A,B)

k_heaviside = 1e7

HeavisideTheta = lambda x: 0.5 + 0.5* erf(k_heaviside*x)
lift = lambda A: d3.Lift(A, tau_basis, -1)
norm = lambda A: np.sqrt(A@A)

########################################################################################
# tau terms for rigid wall BCs

tau_Uy_1 = dist.VectorField(coords, name='tau_Uy_1', bases=(x_basis))
tau_Uy_2 = dist.VectorField(coords,name='tau_Uy_2', bases=(x_basis))

tau_h_1 = dist.Field(name='tau_h_1', bases=(x_basis, ))
tau_h_2 = dist.Field(name='tau_h_2', bases=(x_basis,))

tau_q_1 = dist.Field(name='tau_q_1', bases=(x_basis,))
tau_q_2 = dist.Field(name='tau_q_2', bases=(x_basis,))

########################################################################################
# Initial Conditions

x = dist.local_grid(x_basis, scale = 1)
y = dist.local_grid(y_basis, scale= 1)

gaussian = lambda x,y,sigma: np.exp(-0.5*((x)**2 + (y)**2) / sigma**2)
h['g'] = 0.5*gaussian((x),(y),0.2*np.sqrt(Lx*Ly))# Gaussian bump
q['g'] =0.7*q_0 # background moisture near saturation   

# ### Random Noise 
# rng = np.random.default_rng(42)                
# h_noise = rng.normal(loc=0.0, scale=1.0, size=h['g'].shape).astype(h['g'].dtype)
# h['g'] += 0.05 * h_noise

########################################################################################
# Build Solver

E = lambda A:  HeavisideTheta(q_g - A) * (q_g - A) / tau_e # No velocity-dependence
q_sat = lambda h_: q_0 * np.exp(-1*alpha * h_ / H) # Saturation specific humidity
C = lambda h_,q_: HeavisideTheta(q_-q_sat(h_))*(q_-q_sat(h_))/tau # condensation

problem = d3.IVP([U, h, q, tau_Uy_1, tau_Uy_2, tau_h_1, tau_h_2, tau_q_1, tau_q_2], namespace=locals()) 
# problem = d3.IVP([U, h, q], namespace=locals()) 

ex, ey = coords.unit_vector_fields(dist) 
U_x = U @ ex # x velocity component
U_y = U @ ey # y velocity component 
grad_u =d3.grad(U) + ey*lift(tau_Uy_1) # first-order reduction, U
grad_h = d3.grad(h) + ey*lift(tau_h_1) # first-order reduction, h
grad_q = d3.grad(q) + ey*lift(tau_q_1) # first-order reduction, q
dy_Ux = d3.Differentiate(U_x, coords['y'])  # shear


### beta plane
problem.add_equation("dt(U) + f*zcross(U) + g*grad_h + lift(tau_Uy_2)-nu_u*div(grad_u)=0")
problem.add_equation("dt(h) + H*trace(grad_u) -(h_0-h)/tau_r + lift(tau_h_2) - nu_h*div(grad_h) = -1*gamma*C(h,q)")
problem.add_equation("dt(q) + lift(tau_q_2) -nu_q*div(grad_q)  = E(q)-C(h,q) - q*trace(grad_u) - dot(U, grad_q)")

#Boundary Conditions - free-slip walls
problem.add_equation("(U_y)(y =-Ly/2)= 0")
problem.add_equation("(U_y)(y = Ly/2)= 0")
problem.add_equation("dy_Ux(y = -Ly/2) = 0")
problem.add_equation("dy_Ux(y =Ly/2)= 0")
problem.add_equation("dy(h)(y=Ly/2)=0")
problem.add_equation("dy(h)(y=-Ly/2)=0")
problem.add_equation("dy(q)(y=Ly/2)=0")
problem.add_equation("dy(q)(y=-Ly/2)=0")

solver = problem.build_solver('SBDF3') 
# solver.print_subproblem_ranks()

########################################################################################
# IVP Setup -- remove lists when parallelized

u_list = []
h_list = []
t_list = []
C_list = []
E_list = []
q_list = []

ux_s  = []
uy_s  = []
h_s  = []
q_s  = []

init_timestep = 0.1

KE = []
PE_g = []
PE_q = []

kinetic_energy = lambda h,u: d3.Integrate(d3.Integrate(0.5 * h * u@u,'x'),'y') / ((xmax-xmin)*(ymax-ymin))
potential_energy = lambda h:  d3.Integrate(d3.Integrate(0.5 * g * h**2,'x'),'y') / ((xmax-xmin)*(ymax-ymin))
########################################################################################
## Run IVP 

logger = logging.getLogger(__name__)
os.mkdir(path)

with open(path+'/params.txt', 'w') as file:
    file.write(f"dt limit = {TIMESTEP_CAP*tau}s\n")
    file.write(f"Nx = {Nx}\n")
    file.write(f"Ny = {Ny}\n")
    file.write(f"tau = {tau}\n")
    file.write(f"tau_r = {tau_r}\n")
    file.write(f"tau_e = {tau_e}\n")
    file.write(f"q_0, g_g = {q_0}, {q_g}\n")
    file.write(f"gamma = {gamma}\n")
    file.write(f"alpha = {alpha}\n")
    file.write(f"nu_u, nu_h, nu_q = {nu_u}, {nu_h}, {nu_q}\n")


#simulation time- stopped if blow-up occurs
solver.stop_sim_time = 24*60*60*RUN_TIME

#Analysis
filename = path+ f'/MJO_{alpha}a{gamma}g{Ny}r{nu_u}nu{tau_r}tr{tau_e}te' 
analysis = solver.evaluator.add_file_handler(f'{filename}', iter=10, max_writes=400)
analysis.add_tasks(solver.state, layout='g')
analysis.add_task(C(h,q), layout='g', name='C')

CFL = flow_tools.CFL(solver, initial_dt=init_timestep, cadence=10, safety=0.3, max_change=1.5)

CFL.add_velocity(U) 

logger.info('Starting loop')
start_time = time.time()
while solver.proceed:
    timestep = np.minimum(CFL.compute_timestep(), tau*TIMESTEP_CAP) #minimum of CFL condition, factor of condensation timescale 
    solver.step(timestep)

    t_list.append(solver.sim_time)
    h_list.append(np.copy(h['g']))
    q_list.append(np.copy(q['g']))
    C_list.append(np.copy(C(h,q)['g']))
    KE.append(np.sum((U['g'][0]**2 + U['g'][1]**2)*0.5))
    PE_g.append(np.sum(g*h['g']))
    PE_q.append(np.sum((q['g']-q_sat(h)['g'])*gamma*g))
    h_s.append(np.copy(h['c']))
    q_s.append(np.copy(q['c']))
    uy_s.append(np.copy(U['c'][1]))
    ux_s.append(np.copy(U['c'][0]))


    if solver.iteration % 10 == 0:
        if np.max(h['g']>20): break
        if np.any(np.isnan(h['g'])) or np.any(np.isnan(q['g'])):break # Check for nans/blow-ups
        print('Completed iteration {}, time {} days, dt {}'.format(solver.iteration, t_list[-1]/60/60/24, timestep))
end_time = time.time()
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
