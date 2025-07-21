import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars

# ======================================================================
# Constants
# ======================================================================
GAM = 42.58e6  # Gyromagnetic ratio [Hz/T]
M0 = 1.0       # Equilibrium magnetization (normalized)
B0 = 3.0       # Main magnetic field [T]

# ======================================================================
# Simulation Parameters
# ======================================================================
FOV = 0.2     # Field of view [m]
Nx, Ny, Nz = 32, 32, 1  # Grid size (reduced for speed)
x = np.linspace(-FOV/2, FOV/2, Nx)
y = np.linspace(-FOV/2, FOV/2, Ny)
z = np.linspace(-FOV/2, FOV/2, Nz)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

# Time parameters
dt = 1e-5      # Time step [s]
Nt = 1024      # Total time steps
t = np.arange(Nt) * dt
TR = 20e-3     # Repetition time [s] (shortened for demo)

# Relaxation times
T1, T2 = 0.8, 0.1  # [s]

# ======================================================================
# Pulse Sequence
# ======================================================================
# RF pulse (90° slice-selective sinc pulse)
flip_angle = 90  # [deg]
rf_dur = 2e-3    # RF duration [s]
rf_steps = int(rf_dur / dt)
t_rf = np.linspace(-rf_dur/2, rf_dur/2, rf_steps)
B1_amp = flip_angle / (360 * GAM * rf_dur)  # Approx amplitude for 90°

# Sinc pulse with 2 lobes
B1_env = np.sinc(2 * t_rf / rf_dur)  
B1 = np.zeros(Nt, dtype=complex)
B1[:rf_steps] = B1_amp * B1_env * (1 + 1j)  # Circularly polarized

# Slice selection parameters
slice_thickness = 0.01  # [m]
Gz_slice = 0.02         # Slice gradient [T/m]
BW = GAM * Gz_slice * slice_thickness  # Pulse bandwidth [Hz]
print(f"RF Bandwidth: {BW/1e3:.1f} kHz")

# Gradients
Gx = np.zeros(Nt)
Gy = np.zeros(Nt)
Gz = np.zeros(Nt)

# Slice-select gradient during RF
Gz[:rf_steps] = Gz_slice

# Readout gradient
readout_start, readout_end = 400, 900
Gx[readout_start:readout_end] = 0.02  # [T/m]

# Phase encoding blip
pe_time = 350
Gy[pe_time] = 0.03  # Max amplitude [T/m]

# ======================================================================
# Main Simulation
# ======================================================================
# Initialize magnetization
Mz = M0 * np.ones((Nx, Ny, Nz))
Mx = np.zeros_like(Mz)
My = np.zeros_like(Mz)

# K-space
kspace = np.zeros((Nx, Ny), dtype=complex)
Gy_amps = np.linspace(-0.03, 0.03, Ny)  # Phase encode steps

for n_pe, Gy_amp in enumerate(tqdm(Gy_amps)):
    # Reset gradients for this TR
    Gy[:] = 0
    Gy[pe_time] = Gy_amp
    
    # Reset magnetization (except for steady-state effects)
    Mz = M0 * (1 - np.exp(-TR/T1)) + Mz * np.exp(-TR/T1)
    Mx, My = Mx * np.exp(-TR/T2), My * np.exp(-TR/T2)
    
    for i in range(Nt):
        # ------------------------------------------------------------------
        # Effective B-field in rotating frame (B0 subtracted)
        # ------------------------------------------------------------------
        Bx_rf = np.real(B1[i])
        By_rf = np.imag(B1[i])
        Bz_off = B0 + Gx[i]*xx + Gy[i]*yy + Gz[i]*zz  # Off-resonance
        
        # ------------------------------------------------------------------
        # Resonance condition: Only flip spins within the slice
        # ------------------------------------------------------------------
        df = GAM * Bz_off  # Off-resonance frequency [Hz]
        rf_mask = np.abs(df) < BW/2  # Mask for spins in resonance
        
        # ------------------------------------------------------------------
        # Bloch rotation (only for resonant spins)
        # ------------------------------------------------------------------
        B_eff = np.stack([Bx_rf * np.ones_like(xx), 
                         By_rf * np.ones_like(yy), 
                         Bz_off], axis=-1)
        
        B_norm = np.linalg.norm(B_eff, axis=-1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            b = np.where(B_norm > 0, B_eff / B_norm, 0)
        
        angle = -2 * np.pi * GAM * B_norm[..., 0] * dt
        c, s = np.cos(angle), np.sin(angle)
        
        # Vectorized Rodrigues' rotation
        M = np.stack([Mx, My, Mz], axis=-1)
        dot = np.sum(b * M, axis=-1, keepdims=True)
        M_rot = M * c[..., np.newaxis] + \
                s[..., np.newaxis] * np.cross(b, M) + \
                (1 - c[..., np.newaxis]) * dot * b
        
        # Apply rotation only to resonant spins
        M_new = np.where(rf_mask[..., np.newaxis], M_rot, M)
        Mx, My, Mz = M_new[..., 0], M_new[..., 1], M_new[..., 2]
        

        # ------------------------------------------------------------------
        # Relaxation (skip during RF pulse for accurate flip angles)
        # ------------------------------------------------------------------
        if i >= rf_steps:
            Mx *= np.exp(-dt/T2)
            My *= np.exp(-dt/T2)
            Mz = Mz * np.exp(-dt/T1) + M0 * (1 - np.exp(-dt/T1))

        cMx = Mx[11, 11, 0]
        cMy = My[11, 11, 0]
        cMz = Mz[11, 11, 0]
        print(f"i: {i}, Mx: {cMx:.25f}, My: {cMy:.10f}, Mz: {cMz:.10f}")
        
        # ------------------------------------------------------------------
        # Signal acquisition (readout period)
        # ------------------------------------------------------------------
        if readout_start <= i < readout_end:
            kx = GAM * np.sum(Gx[:i]) * dt
            ky = GAM * np.sum(Gy[:i]) * dt
            
            # K-space index
            kx_idx = int((kx / (1/FOV)) + Nx//2)
            ky_idx = int((ky / (1/FOV)) + Ny//2)
            
            if 0 <= kx_idx < Nx and 0 <= ky_idx < Ny:
                # Sum signal across slice (center slice for 2D)
                kspace[kx_idx, ky_idx] = np.sum(Mx + 1j*My) * dt

# ======================================================================
# Reconstruction and Plotting
# ======================================================================
# Reconstruct image
image = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(np.abs(kspace), cmap='gray')
plt.title("k-Space")

plt.subplot(122)
plt.imshow(np.abs(image), cmap='gray', vmax=0.1*np.max(np.abs(image)))
plt.title("Reconstructed Image")
plt.colorbar()
plt.tight_layout()
plt.show()