import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import h5py
import os
import re
 

""" 
Plot output from VP2020_MJO.py 
    • Input folder name(s) containing h5py files 
    • Returns movie + power spectrum saved in each folder
"""

base_path = '/Users/luitbald/CODE/movies/gamma_experiments/' 

# list of folders --> to plot multiple runs, input list of folder names
folders = [ 
    'gamma5_long'
]

Nx = 256
RES = 1    # Increase for lower resolution movie     
ADAPTIVE_COLORMAP = False # True for adaptive colomap
xmin, xmax = 0, 10     
ymin, ymax = 0, 10
target_time = 150 * 24 * 60 * 60  
grey_color = 'dimgray'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def read_files(dirpath):  
    if not os.path.exists(dirpath): 
        return {} 
    files = sorted([f for f in os.listdir(dirpath) if f.endswith(".h5") and not f.startswith("._")], key=natural_sort_key)
    
    if not files:
        return {}

    total_frames = 0
    for filepath in files:
        try:
            with h5py.File(os.path.join(dirpath, filepath), mode="r") as f:
                total_frames += f['tasks']['h'].shape[0]
        except OSError:
            continue
            
    if total_frames == 0:
        return {}

    # downsampled grid
    with h5py.File(os.path.join(dirpath, files[0]), mode="r") as f:
        nx_down = f['tasks']['h'].shape[1] // RES
        ny_down = f['tasks']['h'].shape[2] // RES

    print(f"Found: {total_frames} frames at {nx_down}x{ny_down} resolution \n")

    data = {
        'U': np.zeros((total_frames, 2, nx_down, ny_down), dtype=np.float32),
        'h': np.zeros((total_frames, nx_down, ny_down), dtype=np.float32),
        'q': np.zeros((total_frames, nx_down, ny_down), dtype=np.float32),
        'C': np.zeros((total_frames, nx_down, ny_down), dtype=np.float32),
        'sim_time': np.zeros(total_frames, dtype=np.float32)
    }

    current_idx = 0
    for filepath in files:
        try:
            with h5py.File(os.path.join(dirpath, filepath), mode="r") as f:
                n_frames = f['tasks']['h'].shape[0]
                end_idx = current_idx + n_frames
                
                data['U'][current_idx:end_idx] = f['tasks']['U'][:, :, ::RES, ::RES]
                data['h'][current_idx:end_idx] = f['tasks']['h'][:, ::RES, ::RES]
                data['q'][current_idx:end_idx] = f['tasks']['q'][:, ::RES, ::RES]
                data['C'][current_idx:end_idx] = f['tasks']['C'][:, ::RES, ::RES]
                
                if 'scales/sim_time' in f:
                    data['sim_time'][current_idx:end_idx] = f['scales']['sim_time'][:]
                    
                current_idx = end_idx
        except OSError:
            continue
            
    return data

def update_u_h_frame(frame, up, hp, ux, uy, h, adaptive):
    h_current = h[frame]
    hp.set_array(np.transpose(h_current).flatten())
    if adaptive:
        hp.set_clim(np.nanmin(h_current), np.nanmax(h_current))
    return up, hp

def update_scalar_frame(frame, val_plot, val, adaptive):
    val_current = val[frame]
    val_plot.set_array(np.transpose(val_current).flatten())
    if adaptive:
        val_plot.set_clim(np.nanmin(val_current), np.nanmax(val_current))
    return val_plot,

def update_frame_combined(frame, up, hp, qp, Cp, ux, uy, h, q, C, adaptive, sim_time):
    if frame % 10 == 0:
        n = len(ux)
        print("Rendering frame {} out of {}".format(frame, n))
    
    up, hp = update_u_h_frame(frame, up, hp, ux, uy, h, adaptive)
    qp, = update_scalar_frame(frame, qp, q, adaptive)
    Cp, = update_scalar_frame(frame, Cp, C, adaptive)
    
    current_time_days = sim_time[frame] / (24*60*60)
    hp.figure.suptitle(f"Time = {current_time_days:.2f} days")

    return up, hp, qp, Cp

def plot_combined(data):
    if not data: return None

    u = data['U']
    ux = u[:,0,:,:] 
    uy = u[:,1,:,:]
    
    h = data['h']
    q = data['q']
    C = data['C']
    sim_time = data['sim_time']
    
    nframes, nx_grid, ny_grid = ux.shape
    
    xgrid = np.linspace(xmin, xmax, nx_grid)
    ygrid = np.linspace(ymin, ymax, ny_grid)
    X, Y = np.meshgrid(xgrid, ygrid)

    if not ADAPTIVE_COLORMAP:
        h_last = h[-1]
        q_last = q[-1]
        C_last = C[-1]

        h_vmin, h_vmax = np.nanmin(h_last), np.nanmax(h_last)
        q_vmin, q_vmax = np.nanmin(q_last), np.nanmax(q_last)
        C_vmin, C_vmax = np.nanmin(C_last), np.nanmax(C_last)

        if h_vmax == h_vmin: h_vmax += 1e-5
        if q_vmax == q_vmin: q_vmax += 1e-5
        if C_vmax == C_vmin: C_vmax += 1e-20
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
        fargs=(up, hp, qp, Cp, ux, uy, h, q, C, ADAPTIVE_COLORMAP, sim_time), 
        frames=frameslist, 
        blit=False, 
        repeat=False
    )
    return anim, fig

def get_radial_spectrum(data):
    ny, nx = data.shape
    f_k = np.fft.fftshift(np.fft.fft2(data))
    psd_2d = np.abs(f_k)**2 
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
    KX, KY = np.meshgrid(kx, ky)
    
    K = np.sqrt(KX**2 + KY**2)
    K_int = np.floor(K).astype(int)
    tbin = np.bincount(K_int.ravel(), weights=psd_2d.ravel())
    k_axis = np.arange(len(tbin))
    mask = tbin > 0
    return k_axis[mask], tbin[mask]


# Loop through input folders:  
for folder in folders:
    path = os.path.join(base_path, folder)
    data_path = os.path.join(path, 'data')
    
    print(f"Plotting folder: {folder}")
    
    if not os.path.exists(data_path):
        print(f"Folder {data_path} not found.")
        continue
    
    # Generate gif 
    data = read_files(data_path)
    if data:
        result = plot_combined(data)
        if result:
            combined_anim, fig = result
            save_loc = os.path.join(path, "MJO_movie.gif")
            print(f"Saving animation to {save_loc}.")
            combined_anim.save(save_loc, writer=PillowWriter(fps=15))
            plt.close(fig)

    savepath = os.path.join(path, "power_spectrum.png")
    
    files = sorted([f for f in os.listdir(data_path) if f.endswith(".h5") and not f.startswith("._")], key=natural_sort_key)

    best_file = None
    best_index = 0
    best_time = 0
    min_diff = float('inf')

    for fname in files:
        path_file = os.path.join(data_path, fname)
        try:
            with h5py.File(path_file, 'r') as f:
                if 'scales/sim_time' not in f:
                    continue
                times = np.array(f['scales']['sim_time'])
                idx = (np.abs(times - target_time)).argmin()
                current_diff = np.abs(times[idx] - target_time)
                if current_diff < min_diff:
                    min_diff = current_diff
                    best_file = fname
                    best_index = idx
                    best_time = times[idx]
                if times[0] > target_time and current_diff > min_diff:
                    break
        except OSError:
            continue
                
    if best_file:
        with h5py.File(os.path.join(data_path, best_file), 'r') as f:
            h_frame = np.array(f['tasks']['h'][best_index])
            q_frame = np.array(f['tasks']['q'][best_index])
            C_frame = np.array(f['tasks']['C'][best_index])

        kh, P_h = get_radial_spectrum(h_frame)
        kq, P_q = get_radial_spectrum(q_frame)
        kc, P_c = get_radial_spectrum(C_frame)
        
        plt.figure(figsize=(6, 5))
        ax = plt.gca()

        plt.loglog(kh[1:], P_h[1:], color='k', linewidth=1.5, label='h')
        plt.loglog(kq[1:], P_q[1:], color='royalblue', linewidth=1.5, label='q')
        plt.loglog(kc[1:], P_c[1:], color='firebrick', linewidth=1.5, label='C')
        plt.axvline(x=Nx/2*np.sqrt(2), color='k', linestyle='--', linewidth=1.5, alpha=1)

        plt.xlabel("k", fontsize=12, color=grey_color)
        plt.ylabel(r"$P(k)$", fontsize=12, color=grey_color)

        ax.tick_params(axis='x', colors=grey_color, which='both')
        ax.tick_params(axis='y', colors=grey_color, which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor(grey_color)

        legend = plt.legend(loc='lower left')
        plt.setp(legend.get_texts(), color=grey_color)
        label_text = f"Power Spectrum, t = {best_time/24/60/60:.2f} days"
        plt.text(0.2, 0.05, label_text, transform=ax.transAxes, fontsize=12, color='k')
        plt.text(60, np.max(P_h)/10.0, f"k = {Nx/2*np.sqrt(2):.3f}", color='k')
        plt.grid(True, which="both", ls="-", alpha=0.2, color=grey_color)
        plt.tight_layout()
        plt.ylim(1e-15,) 
        
        print(f"Saving power spectrum to {savepath}.")
        plt.savefig(savepath, dpi=300, bbox_inches = 'tight')

        plt.close('all')
