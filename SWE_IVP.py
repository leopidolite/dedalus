"""
Shallow Water Equations on a Double-Tanh Plane

February 2026
Authors: Leopold Li, Brad Marston

---------------------------------------------------------------------------
    This script solves the IVP for the shallow water equations 
    initialized with an eigenmode of a chosen frequency and horizontal 
    wavenumber on a double-tanh plane: 

        ∂ₜu + ∂ₓh - fv = 0
        ∂ₜv + ∂ᵧh + fu = 0
        ∂ₜh + ∂ₓu + ∂ᵧv = 0

        f = tanh(⍺(y-y₀)) - tanh(⍺(y+y₀)) + 1
---------------------------------------------------------------------------
"""

# Import Packages
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pathlib
import time
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import dedalus.public as d3
import dedalus.core as dec
from dedalus.tools import post 
import logging
logger = logging.getLogger(__name__)
plt.rcParams['text.usetex'] = True
from dedalus.core.operators import GeneralFunction
from dedalus.extras import flow_tools
import shutil
from mpi4py import MPI


path = '[your path]'

# Domain Parameters 
Nx = 64  # x resolution
Ny = 128  # y resolution
Lx = 10   # meridional dimension
Ly = 20*np.pi # zonal dimension
y0 = Ly/4 # Offset for tanh equator 
f0 = 1.0 # f₀ for beta-plane

"""
Select parameters 
"""

target_omega = [-0.7] # Choose ⍵ for desired eigenmode to seed IVP
alphas = [10] # alpha parameter for 'sharpness' double tanh equator
horizontal_wavenumber = [2*(2*np.pi/Lx)] # Choose horizontal wavenumber for desired eigenmode to seed IVP

h_g = None
u_g = None
v_g = None
eig_sel = None
kx  = None
h_gs = []
u_gs = []
v_gs = []
y = None

### Solve EVP for given horizontal wavenumber, frequency, alpha, f0
def EVP_solve(k_x, target_omega, alph, f0):
    alpha = alph
    y = d3.Coordinate('y')
    dist = d3.Distributor(y, dtype=np.complex128)
    Y = d3.ComplexFourier(y, size=Ny, bounds=(-Ly/2, Ly/2))

    _dy = lambda A: d3.Differentiate(A, y)
    _dx = lambda A: (-1j*k_x)*A           
    _dt = lambda A: ( 1j)*omega*A         

    u = dist.Field(name='u', bases=Y)
    v = dist.Field(name='v', bases=Y)
    h = dist.Field(name='h', bases=Y)
    omega = dist.Field(name='omega')        

    yy = dist.local_grids(Y)[0]
    f  = dist.Field(name='f', bases=Y)

    f['g'] = f0*(np.tanh(alph*(yy-y0)) -np.tanh(alph*(yy+y0)) +1) # Double-tanh equator

    problem = d3.EVP([u, v, h], eigenvalue=omega, namespace=locals())
    problem.add_equation("_dt(u) + _dx(h) - f*v = 0")
    problem.add_equation("_dt(v) + _dy(h) + f*u = 0")
    problem.add_equation("_dt(h) + _dx(u) + _dy(v) = 0")

    solver = problem.build_solver()

    solver.solve_dense(solver.subproblems[0])

    order = np.argsort(solver.eigenvalues.real) # Indices ordered by eigenvalues 
    eigs_ordered  = solver.eigenvalues[order] # Eigenmodes in order
    print("First few eigenvalues:\n", eigs_ordered[:5]) 

    # Pick eigenvalues - finds eigenmode nearest to selected ⍵ 
    omega_idx = np.argmin(np.abs(eigs_ordered.real - target_omega)) 
    omega_idx_unsorted = np.argmin(np.abs(solver.eigenvalues.real - target_omega))
    eig_sel = eigs_ordered[omega_idx]

    print(f"Alpha = {alph} Omega:", eig_sel.real)

    solver.set_state(omega_idx_unsorted)
    y1d = dist.local_grids(Y, scales=1)[0]
    h_g = h['g'].copy()
    u_g = u['g'].copy()
    v_g = v['g'].copy()

# Plot selected eigenmode
    plt.figure(figsize=(9, 6))
    plt.plot(y1d, np.real(h_g), color = 'black', lw = 2)
    plt.tick_params('both', size = 8, width = 1.5, direction = 'in')
    plt.xlabel('$$y$$', fontsize = 25, color = 'dimgray')
    plt.ylabel('$$h$$', fontsize = 25, color = 'dimgray')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('dimgray')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'$$\\alpha={alph}, k_x = {k_x:.3f},  \\omega = {eig_sel.real:.3f}, f_0 = {f0}$$',  fontsize = 25, color = 'dimgray')
    ax.tick_params(axis='both', colors='dimgray') 

    # Store grids for initializing IVP
    h_gs.append(np.real(h_g))
    u_gs.append(np.real(u_g))
    v_gs.append(np.real(v_g))

# Run EVP
for i in range(len(alphas)): 
    EVP_solve(horizontal_wavenumber[i], target_omega[i], alphas[i], f0)

# Set desired kx: (Set to zero here)
kx = horizontal_wavenumber[0]
alpha = alphas[0] 


"""
IVP 
"""
# Pick initial conditions from collected eigenmodes
eigenmode_h = h_gs[0]  # selects height field of indexed eigenmode
eigenmode_u =  u_gs[0] # selects meriodional velocity field of indexed eigenmode
eigenmode_v = v_gs[0] # selects zonal velocity field of indexed eigenmode

# Dedalus Domain parameters 
coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=np.float64)

x_basis = d3.RealFourier(coords['x'], Nx, bounds=[-Lx/2, Lx/2], dealias=1.0)
y_basis = d3.RealFourier(coords['y'], Ny, bounds=[-Ly/2, Ly/2], dealias = 1.0)

# Dedalus fields
u = dist.Field(name='u', bases=[x_basis, y_basis])
v = dist.Field(name='v', bases=[x_basis, y_basis])
h = dist.Field(name='h', bases=[x_basis, y_basis])
f = dist.Field(name='f', bases=[y_basis])

dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

x, y = dist.local_grids(x_basis, y_basis)
f['g'] = np.tanh(alpha*(y - y0)) - np.tanh(alpha*(y + y0)) + 1.0 # Double-tanh plane
# f['g'] =np.sin(2*np.pi*y/Ly) # sin coriolis parameter 

# Initial conditions
u['g'] = eigenmode_u.real*np.cos(kx*x)
v['g'] = eigenmode_v.real*np.cos(kx*x)
h['g'] = eigenmode_h.real*np.cos(kx*x)


# IVP Equations 
problem = d3.IVP([u, v, h], namespace=locals())
problem.add_equation("dt(u) + dx(h) - f*v = 0")
problem.add_equation("dt(v) + dy(h) + f*u  = 0")
problem.add_equation("dt(h) + dx(u) + dy(v) = 0")

solver = problem.build_solver('RK222') 

# Run IVP

solver.stop_sim_time = 50
solver.stop_wall_time = np.inf
solver.stop_iteration = 1000

# Set up CFL 
vel = d3.VectorField(dist,coordsys =coords, bases=(x_basis, y_basis), name='vel')
init_dt = 0.001
CFL = flow_tools.CFL(solver, initial_dt=init_dt, cadence=10, safety=0.3, max_change=1.5)
CFL.add_velocity(vel)

# Lists for accumulating grids 
u_max= []
u_list = []
h_list = []
t_list = []

logger.info('Starting loop')
start_time = time.time()
dt = 0.005 # Initial dt

while solver.proceed:
    solver.step(dt)

    vel['g'][0] = u['g'].real
    vel['g'][1] = v['g'].real

    dt = CFL.compute_timestep()
    t_list.append(solver.sim_time)
    u_list.append(np.copy(u['g']))
    h_list.append(np.copy(h['g']))
    
    
    if solver.iteration % 10 == 0:
        print('Completed iteration {}, time {}, dt {}'.format(solver.iteration, t_list[-1], dt))

end_time = time.time()


logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

# Produce animation 
import matplotlib.colors as mcolors
from matplotlib import animation 

xm, ym = np.meshgrid(x,y)

fig, axis = plt.subplots(figsize=(10,5),num="Selected eigenmode")

lim  = np.nanmax(np.abs(h_list))
norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
p = axis.pcolormesh(xm, ym, np.array(h_list[0]).T,norm = norm, cmap='RdBu_r', shading='gouraud')
axis.set_title(rf'$\alpha = {alpha},\ t = {t_list[0]:6.2f}$', fontsize = 20)
axis.set_xlabel('x',fontsize=20)
axis.set_ylabel('y',fontsize=20)
cbar = fig.colorbar(p,ax=axis)
cbar.set_label('h', fontsize = 20)
cbar.ax.tick_params(labelsize=20)
u_all = np.array(u_list)
h_all = np.array(h_list)

def init():
    p.set_array(np.ravel(np.array(h_list[0]).T))
    return p

def animate(i): 
    if i % 10 == 0:
        print(f"Rendering frame {i}...")
    p.set_array(np.ravel(np.array(h_list[(i+1)*10]).T))
    axis.set_title(fr'$\alpha = {alpha},\; t = {t_list[(i+1)*10]:6.2f}$')

    return p

ani = animation.FuncAnimation(fig, animate, frames=int(len(t_list)/10-1))
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='L'), bitrate=2000)
print("Saving animation ...")
ani.save(path + 'swe_ivp.mp4', writer=writer)
print(f"\n \n Animation ('swe_ivp.mp4') saved to: {path}")