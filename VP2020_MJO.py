import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from dedalus.extras import flow_tools
import time
import shutil
from datetime import datetime
from scipy.special import erf
import h5py
import os
import sys
import pandas as pd
import matplotlib.colors as mcolors
import logging
import re
from mpi4py import MPI
plt.switch_backend('Agg') 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = logging.getLogger(__name__)

""" 
Vallis-Penn 2020 MJO Model
    • Paper: "Convective organization and eastward propagating equatorial disturbances in a simple excitable system"
    • Runs simulation (mpi-parallel)

    Model: 
        ∂u/∂t + f × u + g∇u = ν_u ∇²u

        ∂h/∂t + H(∇·u) + γC - (h_0 - h)/τ_r = ν_h ∇²h

        ∂q/∂t + ∇·(qu) - E + C = ν_q ∇²q

    With: 
        C = [𝚯(q - q*)(q - q*)] / τ
        q* = q_0 * exp(-αh / H)
        E = [𝚯(q_g - q)(q_g - q)] / τ_e

    • To run: mpiexec -n [number of cores] python VP2020_MJO.py
    • Modify following parameters: 
        --> F plane (double-periodic, Fourier bases)
        --> β plane (channel w/ rigid-wall BCs in y, Fourier x + Chebyshev y bases)
        * Resolution Nx, Ny (min. 256 typically required to generate disturbance)
        * Run time (days)
        * Interval between saved frames 
        * Parameters γ, ⍺, tau_e, tau_r, lambda_
        * Options to add: stochastic forcing, nonlinear terms
        * Viscosity/diffusivity values 
        * Contact: leopold.li1312@gmail.com w/ inquiries  
            * Code including hyperviscosity (not necessary for disturbances) available by request
    • Initial Conditions: 
        * Small height disturbance off equator
        * Uniform moisture field just below saturation (q = q_0 - ϵ)

    Other Details:
        - Heaviside 𝚯 implemented via erfs for stability
        - Timestep is set to min[CFL timestep, 0.95*tau_e]
        - Stochastic forcing (when applied) is white in time, annulus k-space with random phase
"""

##################################################################
##################################################################
path = "[input path here]"
NAME = f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}" # Name file

Lx = 1.0e7 
Ly = 1.0e7
xmin, xmax = -Lx/2, Lx/2
ymin, ymax = -Ly/2, Ly/2
f_ = 4.0e-4
f0 = 0
Nx, Ny = 256, 256
days = 24*60*60


RUN_TIME = 300 # in days
save_every = 10 # Iterations between saved frames
F_PLANE = False # if True --> F-plane used


if F_PLANE:
    
    STOCHASTIC_FORCING_ON = 0.0 # applied if STOCHASTIC_FORCING_ON == 1
    TIMESTEP_CAP = 0.95 # Maximum timestep is TIMESTEP_CAP * condensation timescale
    NONLINEAR = 0.0 # Add nonlinear terms, default off

    tau_r = 9e4 # relaxation timescale
    lambda_ = 0.42  # 1/tau_e --> evporation rate
    nu_scale = Nx/128 # Scaling of viscosity, diffusivity by resolution
    nu_ = (1*10**4) / (nu_scale**2) 
    gamma = 5 # meters
    alpha =60.0 

    # If stochastic forcing applied: 
    forcing_wavenumber = 5.0 
    forcing_packet_width = 3.0 
    forcing_amplitude = 1e-11

    dealias = 3/2

    mask_width = (Ly*0.3)

else: # beta plane
   
    STOCHASTIC_FORCING_ON = 0.0 # applied if STOCHASTIC_FORCING_ON == 1
    TIMESTEP_CAP = 0.95 # Maximum timestep is TIMESTEP_CAP * condensation timescale
    NONLINEAR = 0.0 # Add nonlinear terms, default off

    beta = 2.0e-11 
    gamma = 5 # meters
    alpha =60.0 
    tau_r = 9e4 # relaxation timescale
    lambda_ = 0.42  # 1/tau_e --> evporation rate
    
    nu_scale = Nx/128 # Scaling of viscosity, diffusivity by resolution
    nu_ = (1*10**4) / (nu_scale**2) # Viscosity/diffusivity 
        # [see below] diffusivities for q, h set to 2*nu_
    
    # If stochastic forcing applied: 
    forcing_wavenumber = 5.0 
    forcing_packet_width = 3.0 
    forcing_amplitude = 1e-11

    dealias = 3/2

    mask_width = (Ly*0.3)
##################################################################
##################################################################


g = 10.0 
H = 30.0 # meters
tau = 900.0 # seconds
h_0 = 0 
q_0 = 3.0
q_g = 3.0  
Q_a = 0.035 
c_p = 1004  
T_0 = 300 
Rv = 462 

nu_u = nu_ 
nu_h = 2*nu_
nu_q = 2* nu_

# Save text file containing variables for each run
if rank == 0:
    if not os.path.exists(path): os.mkdir(path)
    with open(path+'/params.txt', 'w') as file:
        file.write(f"MJO VP2020\n")
        file.write("F-PLANE\n" if F_PLANE else "BETA PLANE\n")
        file.write(f"gamma = {gamma} m\n")
        file.write(f"alpha = {alpha}\n")
        file.write(f"q_0 = {q_0}\n")
        file.write(f"Nx, Ny = {Nx}, {Ny}\n")
        file.write(f"nu = {nu_}\n")
        file.write(f"tau = {tau}\n")
        file.write(f"tau_e = {tau_e}\n")
        file.write(f"tau_r = {tau_r}\n\n")
        if STOCHASTIC_FORCING_ON ==1.0:
            file.write(f"\n\n STOCHASTIC FORCING ON \n")
            file.write(f"Forcing wavenumber = {forcing_wavenumber}\n")
            file.write(f"Forcing wavenumber = {forcing_packet_width}\n")
            file.write(f"Forcing Amplitude = {forcing_amplitude}\n")

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)
x_basis = d3.RealFourier(coords['x'], size=Nx, bounds=(xmin, xmax), dealias=dealias)

if F_PLANE: 
    y_basis = d3.RealFourier(coords['y'], size=Ny, bounds=(ymin, ymax), dealias=dealias)
else:
    y_basis = d3.Chebyshev(coords['y'], size=Ny, bounds=(ymin, ymax), dealias=dealias)
    tau_basis = y_basis.derivative_basis(1)

x = dist.local_grid(x_basis)
y = dist.local_grid(y_basis)

U = dist.VectorField(coords, name='U', bases=(x_basis, y_basis))
h = dist.Field(name='h', bases=(x_basis, y_basis))
q = dist.Field(name='q', bases=(x_basis, y_basis))
f = dist.Field(name='f', bases=(y_basis))
stoch_forcing = dist.Field(name='stoch_forcing', bases=(x_basis, y_basis))

if F_PLANE: 
    f['g'] = f_ #f plane 
else: 
    
    f['g'] = f0+beta*y #beta plane 

grad = lambda A: d3.Gradient(A)
div = lambda A: d3.Divergence(A)
lap = lambda A: d3.Laplacian(A)
zcross = lambda A: d3.Skew(A)
ex, ey = coords.unit_vector_fields(dist)
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
k_heaviside = 2/(Ly/Ny)
HeavisideTheta = lambda x: 0.5 + 0.5 * erf(k_heaviside*x)
HeavisideSharp = lambda x: 0.5 + 0.5 * erf(10**7*x)

if not F_PLANE:
    tau_Uy_1 = dist.VectorField(coords, name='tau_Uy_1', bases=(x_basis))
    tau_Uy_2 = dist.VectorField(coords, name='tau_Uy_2', bases=(x_basis))
    tau_h_1 = dist.Field(name='tau_h_1', bases=(x_basis,))
    tau_h_2 = dist.Field(name='tau_h_2', bases=(x_basis,))
    tau_q_1 = dist.Field(name='tau_q_1', bases=(x_basis,))
    tau_q_2 = dist.Field(name='tau_q_2', bases=(x_basis,))
    
    variables = [U, h, q, tau_Uy_1, tau_Uy_2, tau_h_1, tau_h_2, tau_q_1, tau_q_2]
else:
    variables = [U, h, q]

if not F_PLANE: 
    lift = lambda A: d3.Lift(A, tau_basis, -1) 
    
    grad_u = d3.grad(U) + ey*lift(tau_Uy_1)
    grad_h = d3.grad(h) + ey*lift(tau_h_1)
    grad_q = d3.grad(q) + ey*lift(tau_q_1)

mask = dist.Field(name='mask', bases=(y_basis)) 
mask['g'] = d3.exp(- (y/mask_width)**2) 
q_sat = lambda h_: q_0 * np.exp(-1*alpha * h_ / H)
L  = lambda A: ((A@ex)**2 + (A@ey)**2)**0.5*1.0 

delta = lambda h_, q_: q_ - q_sat(h_)
C = lambda h_,q_: HeavisideSharp(q_ - q_sat(h_)) * (q_ - q_sat(h_)) / tau * mask
E = lambda A, u: HeavisideSharp(q_g - A) * (q_g - A) * L(u)/ tau_e 

U_x = U @ ex
U_y = U @ ey

problem = d3.IVP(variables, namespace=locals())

if F_PLANE:
    problem.add_equation("dt(U)+ f*zcross(U) + g*grad(h) - nu_u*lap(U) = 0")
    problem.add_equation("dt(h)+ H*div(U) -(h_0-h)/tau_r - nu_h*lap(h) = -gamma*C(h,q)")
    problem.add_equation("dt(q)- nu_q*lap(q) = E(q, U) - C(h,q) - div(q*U) + stoch_forcing*STOCHASTIC_FORCING_ON")
else:
    problem.add_equation("dt(U) + f*zcross(U) + g*grad(h)- nu_u*div(grad_u) + lift(tau_Uy_2) = 0 - U@grad(U)*NONLINEAR")
    problem.add_equation("dt(h) + H*div(U) +h/tau_r - nu_h*div(grad_h) + lift(tau_h_2) = -gamma*C(h,q) + h_0/tau_r - U@grad(h)*NONLINEAR")
    problem.add_equation("dt(q) - nu_q*div(grad_q) + lift(tau_q_2) = E(q, U) - C(h,q) - U@grad(q) + stoch_forcing*STOCHASTIC_FORCING_ON")

    problem.add_equation("U_y(y=-Ly/2) = 0")
    problem.add_equation("U_y(y=Ly/2)  = 0")
    problem.add_equation("dy(U_x)(y=-Ly/2) = 0")
    problem.add_equation("dy(U_x)(y=Ly/2)  = 0")
    problem.add_equation("dy(h)(y=-Ly/2) = 0") 
    problem.add_equation("dy(h)(y=Ly/2)  = 0")
    problem.add_equation("dy(q)(y=-Ly/2) = 0") 
    problem.add_equation("dy(q)(y=Ly/2)  = 0")

solver = problem.build_solver('SBDF2')
solver.stop_sim_time = days * RUN_TIME

x = dist.local_grid(x_basis, scale = 1)
y = dist.local_grid(y_basis, scale= 1)
y_envelope_width = Ly / 4.0
gaussian_y = np.exp(-(y / y_envelope_width)**2)

### Initial Conditions 
lump_scale = 5.0e5 
h['g'] = -0.5 * np.exp(-(np.sqrt((x+1.e6)**2 + (y +1e6)**2)/lump_scale)**2)
q['g'] = q_0 - 1e-4

init_timestep = 0.1

 # Stochastic forcing 
def generate_stochastic_forcing(kt, timestep):
    stoch_forcing.change_scales(1)

    c_view = stoch_forcing['c'] 
    slices = stoch_forcing.layout.slices(stoch_forcing.domain, scales=1)

    kx_global = np.array(x_basis.wavenumbers)
    ky_global = np.arange(y_basis.size)
    kx_indices = slices[0]
    ky_indices = slices[1]
    kx_local = kx_global[kx_indices]
    coeff_shape_y = stoch_forcing.layout.global_shape(stoch_forcing.domain, scales=1)[1]
    if len(ky_global) != coeff_shape_y:
        ky_global = ky_global[:coeff_shape_y]
        
    ky_local = ky_global[ky_indices]
    kx = kx_local[:, None]
    ky = ky_local[None, :]
    if not F_PLANE:
        ky = ky * np.pi / Ly
    K_mag = np.sqrt(kx**2 + ky**2)
    F_spectrum = np.exp(-(K_mag - kt)**2 / forcing_packet_width**2)
    shape = stoch_forcing['c'].shape
    noise = np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape)
    stoch_forcing['c'] = noise * F_spectrum
    local_sum_sq = np.sum(stoch_forcing['g']**2)
    local_count = stoch_forcing['g'].size
    
    global_sum_sq = comm.allreduce(local_sum_sq, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    
    var_ave = global_sum_sq / (global_count + 1e-16)
    normalization = np.sqrt(forcing_amplitude / (var_ave + 1e-16))
    if F_PLANE:
        gs = 1.0
    else:
        gs = np.exp(-(y)**2 / (Ly/4)**2) # If on beta plane: stochastic forcing multiplied
                                         # by gaussian profile to avoid interference with b.c.s
    stoch_forcing['g'] *= (normalization * gs / np.sqrt(timestep))
    
    return stoch_forcing['g']

analysis = solver.evaluator.add_file_handler(path + '/data', iter=10, max_writes=100)
analysis.add_task(h, name='h')
analysis.add_task(q, name='q')
analysis.add_task(U, name='U') 
analysis.add_task(C(h,q), name='C')

KE = []
t_list = []

CFL = flow_tools.CFL(solver, initial_dt=init_timestep, cadence=10, safety=0.5, max_change=1.2, max_dt=tau*TIMESTEP_CAP)
CFL.add_velocity(U)

if rank == 0:
    print("Starting simulation ")
start_sim_time = time.time()

while solver.proceed:
    timestep = CFL.compute_timestep()
    
    if STOCHASTIC_FORCING_ON != 0:
        stoch_forcing['g'] = generate_stochastic_forcing(forcing_wavenumber, timestep)
    else:
        stoch_forcing['g'] = 0.0
    
    solver.step(timestep)

    if solver.iteration % save_every == 0: 
        u_sq = U['g'][0]**2 + U['g'][1]**2
        local_KE = np.sum(u_sq * 0.5)
        global_KE = comm.allreduce(local_KE, op=MPI.SUM)
        
        local_max_h = np.max(h['g'])
        global_max_h = comm.allreduce(local_max_h, op=MPI.MAX)
        
        local_nan = np.any(np.isnan(h['g'])) or np.any(np.isnan(q['g']))
        global_nan = comm.allreduce(local_nan, op=MPI.LOR)

        if rank == 0:
            t_list.append(solver.sim_time)
            KE.append(global_KE)
            logger.info(f"Iter: {solver.iteration}, Time: {solver.sim_time/days:.4f} days, dt: {timestep:.4f}, KE: {global_KE:.2e}")

        if global_max_h > 1e2: 
            break
        if global_nan:
            break
if rank == 0:
    print("Simulation complete.")