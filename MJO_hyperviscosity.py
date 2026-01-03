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
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors
import logging
import re
plt.switch_backend('Agg') 


shutil.rmtree('analysis', ignore_errors=True)

days = 24*60*60


"""
PATH 
"""
path = f'/Users/luitbald/CODE/movies/MJO_{datetime.now().strftime("%Y-%m-%d-%H-%M")}'

"""
Domain Parameters 
"""
Lx = 1.0e7 
Ly = 1.0e7
xmin, xmax = -Lx/2, Lx/2
ymin, ymax = -Ly/2, Ly/2
f0 = 0
f_ = 4.0e-4
beta = 2.0e-11 
Nx, Ny = 128, 128 

""" 
Simulation Parameters
"""
ADAPTIVE_COLORMAP = False  # Plot with adaptive colormap / not

STOCHASTIC_FORCING_ON = 1.0 # 1.0 for stochastic forcing, 0 otherwise
TIMESTEP_CAP = 0.95
RUN_TIME = 2 # days
F_PLANE = False 
NONLINEAR = 1.0 #0 to ommit nonlinear SWE terms, 1.0 otherwise

if F_PLANE:
    """
    F Plane
    """
    tau_r = 9e4
    tau_e  = 800*15
    nu_8 = (1*10**21)
    nu_ = (1*10**5)   
    gamma = 2.0  
    alpha = 20 
    forcing_wavenumber = 20.0 #wavenumber for stochastic forcing
    forcing_packet_width = 2.0 #half-width of wavenumber packet for forcing
    epsilon = 1e-7
    dealias = 3/2 
else: 
    """
    β Plane
    """ 
    tau_r =   9e6
    tau_e  = 0.2*days
    nu_scale = 1.0 
    nu_8 = (1*10**20) / nu_scale
    nu_ = (1*10**4) / nu_scale
    gamma = 2.5
    alpha = 35
    forcing_wavenumber = 5.0 #wavenumber for stochastic forcing
    forcing_packet_width = 3.0 #half-width of wavenumber packet for forcing
    epsilon = 1e-7
    dealias = 3/2

"""
Fixed simulation parameters
"""
g = 10.0 
H = 30.0 
tau = 900.0 
h_0 = 0 
q_0 = 3.0
q_g = 3.0  
Q_a = 0.035 
c_p = 1004  
T_0 = 300 
Rv = 462 

nu_u = nu_ 
nu_h = nu_
nu_q = nu_


"""
Domain
"""

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype = np.float64)
x_basis = d3.RealFourier(coords['x'] , size = Nx, bounds = (xmin, xmax), dealias=dealias)

if (F_PLANE): 
    y_basis = d3.RealFourier(coords['y'] , size   = Ny, bounds = (ymin, ymax), dealias = dealias)
else:
    y_basis = d3.Chebyshev(coords['y'] , size = Ny, bounds = (ymin, ymax), dealias = dealias)
    tau_basis = y_basis.derivative_basis(1)

x = dist.local_grid(x_basis)
y = dist.local_grid(y_basis)

# Physical Fields
U = dist.VectorField(coords, name='U', bases=(x_basis,y_basis))
h = dist.Field(name = 'h', bases = (x_basis, y_basis))
q = dist.Field(name = 'q', bases = (x_basis, y_basis))
f = dist.Field(name='f', bases=(y_basis))

if F_PLANE: 
    f['g'] = f_
else: 
    f['g'] = f0 + beta*y 

grad = lambda A: d3.Gradient(A)
div = lambda A: d3.Divergence(A)
lap = lambda A: d3.Laplacian(A)
zcross = lambda A: d3.Skew(A)
ex, ey = coords.unit_vector_fields(dist)
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
k_heaviside = 1e4
HeavisideTheta = lambda x: 0.5 + 0.5* erf(k_heaviside*x)

if not F_PLANE:
    ### Tau Terms for ∇^8
    tau_Uy_1 = dist.VectorField(coords, name='tau_Uy_1', bases=(x_basis))
    tau_Uy_2 = dist.VectorField(coords, name='tau_Uy_2', bases=(x_basis))
    tau_h_1 = dist.Field(name='tau_h_1', bases=(x_basis, ))
    tau_h_2 = dist.Field(name='tau_h_2', bases=(x_basis,))
    tau_q_1 = dist.Field(name='tau_q_1', bases=(x_basis,))
    tau_q_2 = dist.Field(name='tau_q_2', bases=(x_basis,))
    Lap1_U = dist.VectorField(coords, name='Lap1_U', bases=(x_basis,y_basis))
    Lap2_U = dist.VectorField(coords, name='Lap2_U', bases=(x_basis,y_basis))
    Lap3_U = dist.VectorField(coords, name='Lap3_U', bases=(x_basis,y_basis))
    Lap1_h = dist.Field(name='Lap1_h', bases=(x_basis,y_basis))
    Lap2_h = dist.Field(name='Lap2_h', bases=(x_basis,y_basis))
    Lap3_h = dist.Field(name='Lap3_h', bases=(x_basis,y_basis))
    Lap1_q = dist.Field(name='Lap1_q', bases=(x_basis,y_basis))
    Lap2_q = dist.Field(name='Lap2_q', bases=(x_basis,y_basis))
    Lap3_q = dist.Field(name='Lap3_q', bases=(x_basis,y_basis))

    tau_Lap1U_1 = dist.VectorField(coords, name='tau_Lap1U_1', bases=(x_basis))
    tau_Lap1U_2 = dist.VectorField(coords, name='tau_Lap1U_2', bases=(x_basis))
    tau_Lap1h_1 = dist.Field(name='tau_Lap1h_1', bases=(x_basis,))
    tau_Lap1h_2 = dist.Field(name='tau_Lap1h_2', bases=(x_basis,))
    tau_Lap1q_1 = dist.Field(name='tau_Lap1q_1', bases=(x_basis,))
    tau_Lap1q_2 = dist.Field(name='tau_Lap1q_2', bases=(x_basis,))
    tau_Lap2U_1 = dist.VectorField(coords, name='tau_Lap2U_1', bases=(x_basis))
    tau_Lap2U_2 = dist.VectorField(coords, name='tau_Lap2U_2', bases=(x_basis))
    tau_Lap2h_1 = dist.Field(name='tau_Lap2h_1', bases=(x_basis,))
    tau_Lap2h_2 = dist.Field(name='tau_Lap2h_2', bases=(x_basis,))
    tau_Lap2q_1 = dist.Field(name='tau_Lap2q_1', bases=(x_basis,))
    tau_Lap2q_2 = dist.Field(name='tau_Lap2q_2', bases=(x_basis,))


    tau_Lap3U_1 = dist.VectorField(coords, name='tau_Lap3U_1', bases=(x_basis))
    tau_Lap3U_2 = dist.VectorField(coords, name='tau_Lap3U_2', bases=(x_basis))
    tau_Lap3h_1 = dist.Field(name='tau_Lap3h_1', bases=(x_basis,))
    tau_Lap3h_2 = dist.Field(name='tau_Lap3h_2', bases=(x_basis,))
    tau_Lap3q_1 = dist.Field(name='tau_Lap3q_1', bases=(x_basis,))
    tau_Lap3q_2 = dist.Field(name='tau_Lap3q_2', bases=(x_basis,))

    variables = [U, h, q, Lap1_U, Lap2_U, Lap3_U, Lap1_h, Lap2_h, Lap3_h, Lap1_q, Lap2_q, Lap3_q,tau_Uy_1, tau_Uy_2, tau_h_1, tau_h_2, tau_q_1, tau_q_2,tau_Lap1U_1, tau_Lap1U_2, tau_Lap1h_1, tau_Lap1h_2,
                  tau_Lap1q_1, tau_Lap1q_2,tau_Lap2U_1, tau_Lap2U_2, tau_Lap2h_1, tau_Lap2h_2, tau_Lap2q_1, tau_Lap2q_2,tau_Lap3U_1, tau_Lap3U_2, tau_Lap3h_1, tau_Lap3h_2, tau_Lap3q_1, tau_Lap3q_2]
else:
    variables = [U, h, q]

if not F_PLANE: 
    lift = lambda A: d3.Lift(A, tau_basis, -1) 
    
    grad_u = d3.grad(U) + ey*lift(tau_Uy_1)
    grad_h = d3.grad(h) + ey*lift(tau_h_1)
    grad_q = d3.grad(q) + ey*lift(tau_q_1)

    grad_Lap1U = d3.grad(Lap1_U) + ey*lift(tau_Lap1U_1)
    grad_Lap1h = d3.grad(Lap1_h) + ey*lift(tau_Lap1h_1)
    grad_Lap1q = d3.grad(Lap1_q) + ey*lift(tau_Lap1q_1)

    grad_Lap2U = d3.grad(Lap2_U) + ey*lift(tau_Lap2U_1)
    grad_Lap2h = d3.grad(Lap2_h) + ey*lift(tau_Lap2h_1)
    grad_Lap2q = d3.grad(Lap2_q) + ey*lift(tau_Lap2q_1)
    
    grad_Lap3U = d3.grad(Lap3_U) + ey*lift(tau_Lap3U_1)
    grad_Lap3h = d3.grad(Lap3_h) + ey*lift(tau_Lap3h_1)
    grad_Lap3q = d3.grad(Lap3_q) + ey*lift(tau_Lap3q_1)

q_sat = lambda h_: q_0 * np.exp(-1*alpha * h_ / H)
# L  = lambda A: ((A@ex)**2 + (A@ey)**2)**0.5*1.0 
C = lambda h_,q_: HeavisideTheta(q_-q_sat(h_))*(q_-q_sat(h_))/tau
E = lambda A, u: HeavisideTheta(q_g - A) * (q_g - A) / tau_e ## No velocity dependence 


U_x = U @ ex
U_y = U @ ey

problem = d3.IVP(variables, namespace=locals())

if F_PLANE:
    problem.add_equation("dt(U) + f*zcross(U) + g*grad(h) - nu_u*lap(U) + nu_8*lap(lap(lap(lap(U)))) = 0")
    problem.add_equation("dt(h) + H*div(U) -(h_0-h)/tau_r - nu_h*lap(h) + nu_8*lap(lap(lap(lap(h)))) = -gamma*C(h,q)")
    problem.add_equation("dt(q) - nu_q*lap(q) + nu_8*lap(lap(lap(lap(q)))) = E(q, U) - C(h,q) - div(q*U)")
else:
    """
    Equations for hyperviscosity, ∇^8
    """
    problem.add_equation("Lap1_U -div(grad_u)+ lift(tau_Lap1U_2) = 0")
    problem.add_equation("Lap1_h -div(grad_h)+ lift(tau_Lap1h_2) = 0")
    problem.add_equation("Lap1_q -div(grad_q) +lift(tau_Lap1q_2) = 0")

    problem.add_equation("Lap2_U -div(grad_Lap1U) + lift(tau_Lap2U_2) = 0")
    problem.add_equation("Lap2_h -div(grad_Lap1h) + lift(tau_Lap2h_2) = 0")
    problem.add_equation("Lap2_q -div(grad_Lap1q) + lift(tau_Lap2q_2) = 0")

    problem.add_equation("Lap3_U -div(grad_Lap2U)+ lift(tau_Lap3U_2) = 0")
    problem.add_equation("Lap3_h -div(grad_Lap2h) + lift(tau_Lap3h_2) = 0")
    problem.add_equation("Lap3_q -div(grad_Lap2q)+ lift(tau_Lap3q_2) = 0")

    problem.add_equation("dt(U) + f*zcross(U) + g*grad(h)- nu_u*div(grad_u) + lift(tau_Uy_2) + nu_8*div(grad_Lap3U) = 0 - U@grad(U)*NONLINEAR")
    problem.add_equation("dt(h) + H*div(U) +h/tau_r - nu_h*div(grad_h) + lift(tau_h_2)+ nu_8*div(grad_Lap3h) = -gamma*C(h,q) + h_0/tau_r - U@grad(h)*NONLINEAR")
    problem.add_equation("dt(q) - nu_q*div(grad_q) + lift(tau_q_2) + nu_8*div(grad_Lap3q) = E(q, U) - C(h,q) - div(q*U)")

    ### Free slip BCs
    problem.add_equation("U_y(y=-Ly/2) = 0")
    problem.add_equation("U_y(y=Ly/2)  = 0")
    problem.add_equation("dy(U_x)(y=-Ly/2) = 0")
    problem.add_equation("dy(U_x)(y=Ly/2)  = 0")
    problem.add_equation("dy(h)(y=-Ly/2) = 0")
    problem.add_equation("dy(h)(y=Ly/2)  = 0")
    problem.add_equation("dy(q)(y=-Ly/2) = 0")
    problem.add_equation("dy(q)(y=Ly/2)  = 0")


    ### Hyperviscosity 
    problem.add_equation("Lap1_U(y=-Ly/2) = 0")
    problem.add_equation("Lap1_U(y=Ly/2)  = 0")
    problem.add_equation("Lap1_h(y=-Ly/2) = 0")
    problem.add_equation("Lap1_h(y=Ly/2)  = 0")
    problem.add_equation("Lap1_q(y=-Ly/2) = 0")
    problem.add_equation("Lap1_q(y=Ly/2)  = 0")

    problem.add_equation("Lap2_U(y=-Ly/2) = 0")
    problem.add_equation("Lap2_U(y=Ly/2)  = 0")
    problem.add_equation("Lap2_h(y=-Ly/2) = 0")
    problem.add_equation("Lap2_h(y=Ly/2)  = 0")
    problem.add_equation("Lap2_q(y=-Ly/2) = 0")
    problem.add_equation("Lap2_q(y=Ly/2)  = 0")

    problem.add_equation("Lap3_U(y=-Ly/2) = 0")
    problem.add_equation("Lap3_U(y=Ly/2)  = 0")
    problem.add_equation("Lap3_h(y=-Ly/2) = 0")
    problem.add_equation("Lap3_h(y=Ly/2)  = 0")
    problem.add_equation("Lap3_q(y=-Ly/2) = 0")
    problem.add_equation("Lap3_q(y=Ly/2)  = 0")

"""
Solve IVP 
"""
solver = problem.build_solver('SBDF2')
solver.stop_sim_time = days * RUN_TIME

# Initial Conditions
x = dist.local_grid(x_basis, scale = 1)
y = dist.local_grid(y_basis, scale= 1)

"""
Original VP 2020 initial bump 
lump_scale = y*0.05
h['g'] = 0.01* np.exp(-(np.sqrt((x+1.e6)**2 + (y+3.0e6)**2)/lump_scale)**2) *10
q['g'] = q_0-1*1e-4 # background moisture near saturation   
"""

lump_scale = 5.0e5 
h['g'] = -0.5 * np.exp(-(np.sqrt((x+1.e6)**2 + (y+1.0e6)**2)/lump_scale)**2)
q['g'] = q_0 - 1e-4


init_timestep = 0.1
ts = init_timestep


"""
Stochastic Forcing, gaussian 
"""
F =  dist.Field(name = 'F', bases = (x_basis, y_basis))
F.change_scales(dealias)
current_scale = 1.0
current_time_step = init_timestep
def Forcing(kt = 30):
    time_step = ts
    # epsilon =1e-8
    aliasscale = dealias
    kxs = np.arange(-Nx*aliasscale/2, Nx*aliasscale/2)
    kys = np.arange(-Ny*aliasscale/2, Ny*aliasscale/2)
    kxx, kyy = np.meshgrid(kxs, kys, indexing = 'ij')
    k_ = np.sqrt(kxx**2 + kyy**2) #wavevector 
    # wavelength = 1e4 #realistic size of moisture disturbance?
    k_target = kt #Lx/wavelength
    k_width = forcing_packet_width 
    F_spectrum = np.exp(-(k_-k_target)**2/k_width**2)
    Fh_temp = np.sqrt(F_spectrum) * np.exp( 2*np.pi*1j * np.random.uniform(0,1,F_spectrum.shape))
    F_phy_temp = np.real(np.fft.ifft2(np.fft.ifftshift(Fh_temp)))  
    phys_variance = F_phy_temp**2
    var_ave = phys_variance.mean()
    amp_temp= epsilon / var_ave       
    F_phy = np.sqrt(amp_temp) * F_phy_temp 
    gssian = np.exp(-(np.linspace(-Ly, Ly, int(Ny*aliasscale)))**2/(Ly/4)**2) # gaussian profile in y on beta plane to preserve BCs
    if F_PLANE: gs = 1.0 # doubly periodic on f-plane
    else: gs = gssian 
    return F_phy/np.sqrt(time_step)*(gs) 

def add_stochastic_forcing(x, out):
    out[:] = (x+ Forcing(forcing_wavenumber)*STOCHASTIC_FORCING_ON)
    return out
forcing_operator = lambda field: d3.UnaryGridFunction(add_stochastic_forcing, field)

"""
Save Parameters
"""
if not os.path.exists(path): os.mkdir(path)
with open(path+'/params.txt', 'w') as file:
    file.write(f"MSWE with nabla^8 Hyperviscosity\n")
    if F_PLANE: file.write("F-PLANE\n")
    else: file.write("BETA PLANE\n")
    file.write(f"gamma = {gamma} m\n")
    file.write(f"alpha = {alpha}\n")
    file.write(f"q_0 = {q_0}\n")
    file.write(f"q_g = {q_g}\n")
    file.write(f"tau_r = {tau_r/days} days\n")
    file.write(f"tau_e = {tau_e/days} days\n")
    file.write(f"tau = {tau} s\n")
    file.write(f"Nx, Ny = {Nx}, {Ny}\n")
    file.write(f"Max timestep: {tau*TIMESTEP_CAP}\n")
    file.write(f"Kinematic viscosity: {nu_:.4e}\n")
    file.write(f"h, q diffusivity: {nu_q:.4e}\n")
    file.write(f"hyperviscosity coefficient: {nu_8:.4e}\n")

    if STOCHASTIC_FORCING_ON !=0:
        file.write(f"\n STOCHASTIC FORCING ON: \n")
        file.write(f"Forcing wavenumber: {forcing_wavenumber}\n")
        file.write(f"Wavenumber packet half width: {forcing_packet_width}\n")
        file.write(f"epsilon: {epsilon}\n")


analysis = solver.evaluator.add_file_handler(path + '/data', iter=10, max_writes=100)
analysis.add_task(h, name='h')
analysis.add_task(q, name='q')
analysis.add_task(U, name='U') 
analysis.add_task(C(h,q), name='C')
KE = []
t_list = []

"""
RUN IVP 
"""
CFL = flow_tools.CFL(solver, initial_dt=init_timestep, cadence=10, safety=1.0, max_change=2.0, max_dt=tau*TIMESTEP_CAP)
CFL.add_velocity(U)
print("Starting simulation ")
start_sim_time = time.time()

while solver.proceed:
    timestep = CFL.compute_timestep()
    tau = timestep/0.8
    ts = timestep
    q['g']= forcing_operator(q)['g']
    solver.step(timestep)

    if solver.iteration % 20 == 0: ### save time, KE, check nans
        t_list.append(solver.sim_time)
        # h_list.append(np.copy(h['g']))
        # q_list.append(np.copy(q['g']))
        # C_list.append(np.copy(C(h,q)['g']))
        KE.append(np.sum((U['g'][0]**2 + U['g'][1]**2)*0.5))
 
        print(f"\n {epsilon*np.sqrt(ts)}\n")
        if np.max(h['g']>1e2): break
        if np.any(np.isnan(h['g'])) or np.any(np.isnan(q['g'])):break
        print(f"Iter: {solver.iteration}, Time: {solver.sim_time/days:.4f} days, dt: {timestep:.4f}")

print("Simulation complete.")


"""
PLOTTING
"""

"""
Plot KE
"""
kinetic = np.array(KE)
time_list = np.array(t_list)

plt.figure()
plt.xlabel('Time [days]', fontsize = 15, color = 'dimgray')
plt.ylabel('[m^2 s^-2]', fontsize = 15, color = 'dimgray')
plt.plot(time_list/days, kinetic, label = 'KE', color = 'royalblue', lw = 2)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('dimgray')
ax.tick_params(axis='both', colors='dimgray')  
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(frameon=False, bbox_to_anchor=(1.05, 1))
plt.savefig(path + '/KE__MJO.png', dpi = 200)

""" 
Plot Movie
"""
filename = path + '/data'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def read_files():  
    dirpath = filename 
    df = pd.DataFrame()
    files = sorted([f for f in os.listdir(dirpath) if f.endswith(".h5")], key=natural_sort_key)
    
    for filepath in files:
        file = h5py.File(dirpath+"/"+filepath, mode="r")
        current_file_df = pd.DataFrame()
        for task in file['tasks']:
            current_file_df[task] = np.array(file['tasks'][task]).tolist()
        df = pd.concat([df,current_file_df], ignore_index=True)
        file.close()
    return df

def series_to_ndarray(series):
    if series.size == 0:
        return np.array([])
    element_shape = np.array(series[0]).shape
    shape = np.concatenate([[series.size],element_shape])
    ndarray = np.zeros(shape)
    for i in range(shape[0]):
        ndarray[i] = np.array(series[i])
    return ndarray

def update_u_h_frame(frame, up, hp, ux, uy, h, res, adaptive):
    h_current = h[frame][::res,::res]
    hp.set_array(np.transpose(h_current).flatten())
    
    if adaptive:
        hp.set_clim(np.nanmin(h_current), np.nanmax(h_current))
    return up, hp

def update_scalar_frame(frame, val_plot, val, res, adaptive):
    val_current = val[frame][::res,::res]
    val_plot.set_array(np.transpose(val_current).flatten())
    
    if adaptive:
        val_plot.set_clim(np.nanmin(val_current), np.nanmax(val_current))
    return val_plot,

def update_frame_combined(frame, up, hp, qp, Cp, ux, uy, h, q, C, res, adaptive):
    if frame % 10 == 0:
        n = len(ux)
        print("Rendering frame {} out of {}".format(frame,n))
        
    up, hp = update_u_h_frame(frame,up,hp,ux,uy,h,res, adaptive)
    qp, = update_scalar_frame(frame,qp,q,res, adaptive)
    Cp, = update_scalar_frame(frame, Cp,C, res,adaptive)
    
    current_time_days = frame * (solver.stop_sim_time / len(ux)) / (24*60*60)
    hp.figure.suptitle(f"Time = {current_time_days:.2f} days")

    return up, hp, qp, Cp

def plot_combined(data):
   
    u = series_to_ndarray(data['U'])
    ux = u[:,0,:,:] 
    uy = u[:,1,:,:]
    
    h = series_to_ndarray(data['h'])
    q = series_to_ndarray(data['q'])
    C = series_to_ndarray(data['C'])
    
    nframes, nx_grid, ny_grid = ux.shape
    res = 1 
    
    xgrid = np.linspace(xmin, xmax, nx_grid)[::res]
    ygrid = np.linspace(ymin, ymax, ny_grid)[::res]
    X, Y = np.meshgrid(xgrid, ygrid)

    if not ADAPTIVE_COLORMAP:
        h_abs_max = max(abs(np.nanmin(h)), abs(np.nanmax(h)))
        h_vmin, h_vmax = -0.05, 0.1
        
        q_vmin, q_vmax = np.nanmin(q)*0.8, np.nanmax(q)*0.8
        C_vmin, C_vmax = np.nanmin(C)*0.8, np.nanmax(C)*0.8
    else:
        h_vmin, h_vmax = None, None
        q_vmin, q_vmax = None, None
        C_vmin, C_vmax = None, None

    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(18, 5))

    hp = ax1.pcolormesh(X, Y, np.zeros_like(X), cmap='RdBu', shading='auto', vmin=h_vmin, vmax=h_vmax)
    up = None
    ax1.set_aspect("equal")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(hp, ax=ax1, label="$h$")
    ax1.set_title("h")

    qp = ax2.pcolormesh(X, Y, np.zeros_like(X), cmap='viridis', shading='auto', vmin=q_vmin, vmax=q_vmax)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(qp, ax=ax2, label="$q$")
    ax2.set_title("q")

    Cp = ax3.pcolormesh(X, Y, np.zeros_like(X), cmap='Blues', shading='auto', vmin=C_vmin, vmax=C_vmax)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    plt.colorbar(Cp, ax=ax3, label="C")
    ax3.set_title("Rainfall")

    frameslist = np.arange(nframes, step=1)
    
    anim = FuncAnimation(
        fig, 
        update_frame_combined, 
        fargs=(up, hp, qp, Cp, ux, uy, h, q, C, res, ADAPTIVE_COLORMAP), 
        frames=frameslist, 
        blit=False, 
        repeat=False
    )
    return anim

data = read_files()
combined_anim = plot_combined(data)
save_loc = path + "/MJO_combined.gif"
combined_anim.save(save_loc, writer=PillowWriter(fps=15)) 
plt.close('all')

stop_time = time.time()
print("Total runtime: ", stop_time - start_sim_time)