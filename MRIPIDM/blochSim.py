'''Mx = 0
My = 0
Mz = 1
M0 = 3

Bx = 1 * 10^-6 #microTesla
By = 1 * 10^-6 #microTesla
Bz = 1.5 #T

T1 = 1000 # mili s
T2 = 100 # mili s

delta_t = 0.01
gryo_ratio = 42.58e-6 #2.58 MHz / T


for i in range(0, 1000):
    Mx = Mx + delta_t*(gryo_ratio*(My*Bz - Mz*By) - (Mx/T2))
    My = My + delta_t*(gryo_ratio*(Mz*Bx - Mx*Bz) - (My/T2))
    Mz = Mz + delta_t*(gryo_ratio*(Mx*By - My*Bx) - ((Mz-M0)/T1))

print(Mx, My, Mz)'''


import numpy as np

# Define constants
GAM = 4257.59  # [Hz/G]
dt = 1e-6  # [s], example RF time step
num_rf = 128  # number of RF samples

# Generate example RF and gradient waveforms
t = np.arange(num_rf)
vec_RF = np.exp(1j * 2 * np.pi * 0.01 * t)  # Example RF waveform (complex)
vec_Gx = np.sin(2 * np.pi * 0.01 * t)  # Example gradient waveform Gx
vec_Gy = np.cos(2 * np.pi * 0.01 * t)  # Example gradient waveform Gy
vec_Gz = np.ones_like(t) * 0.5  # Constant gradient along Z
vec_RFfrq = np.ones_like(t) * 100  # Example off-resonance RF frequency [Hz]

# Define helper functions
def calcBhat(Bx, By, Bz, dt):
    tol = 1e-14
    Blen = np.sqrt(Bx**2 + By**2 + Bz**2)
    Blen = np.where(Blen <= tol, tol, Blen)
    bx = Bx / Blen
    by = By / Blen
    bz = Bz / Blen
    ang = -2 * np.pi * GAM * Blen * dt
    ang = np.where(Blen == tol, 0, ang)
    return bx, by, bz, ang

def calcRRF(Vx, Vy, Vz, kx, ky, kz, theta):
    crsKV_x = ky * Vz - kz * Vy
    crsKV_y = kz * Vx - kx * Vz
    crsKV_z = kx * Vy - ky * Vx
    dotKV = kx * Vx + ky * Vy + kz * Vz
    dotKV_x = dotKV * kx
    dotKV_y = dotKV * ky
    dotKV_z = dotKV * kz
    c = np.cos(theta)
    s = np.sin(theta)
    Rx = c * Vx + s * crsKV_x + (1 - c) * dotKV_x
    Ry = c * Vy + s * crsKV_y + (1 - c) * dotKV_y
    Rz = c * Vz + s * crsKV_z + (1 - c) * dotKV_z
    return Rx, Ry, Rz

# --- Simulation block 1: posZ and velZ (Flow encoding) ---
Mx = 0; My = 0; Mz = 1
pz = np.linspace(-3, 3, 401)  # mm
vz = np.arange(-100, 100.5, 0.5)  # cm/s
mm_to_cm = 0.1
velZ, posZ = np.meshgrid(vz, pz * mm_to_cm, indexing='ij')

for cnt in range(len(vec_RF)):
    Bx = np.real(vec_RF[cnt])
    By = np.imag(vec_RF[cnt])
    Gz = vec_Gz[cnt]
    Bz = (posZ + (cnt + 0.5) * velZ * dt) * Gz
    bx, by, bz, ang = calcBhat(Bx, By, Bz, dt)
    Mx, My, Mz = calcRRF(Mx, My, Mz, bx, by, bz, ang)

Mxy_flow = np.abs(Mx + 1j * My)

# --- Simulation block 2: off-resonance + frequency offsets ---
Mx = 0; My = 0; Mz = 1
dfz = np.arange(-1200, 1200.1, 0.1)
dBz = dfz / GAM

for cnt in range(len(vec_RF)):
    Bx = np.real(vec_RF[cnt])
    By = np.imag(vec_RF[cnt])
    B1z = vec_RFfrq[cnt] / GAM
    Bz = B1z + dBz
    bx, by, bz, ang = calcBhat(Bx, By, Bz, dt)
    Mx, My, Mz = calcRRF(Mx, My, Mz, bx, by, bz, ang)

Mxy_offres = np.abs(Mx + 1j * My)

# --- Simulation block 3: posY and posZ spatial encoding ---
Mx = 0; My = 0; Mz = 1
y = np.linspace(-50, 50, 301)  # mm
z = np.linspace(-3, 3, 301)  # mm
posY, posZ = np.meshgrid(y * mm_to_cm, z * mm_to_cm, indexing='ij')

for cnt in range(len(vec_RF)):
    Gy = vec_Gy[cnt]
    Gz = vec_Gz[cnt]
    Bx = np.real(vec_RF[cnt])
    By = np.imag(vec_RF[cnt])
    Bz = Gy * posY + Gz * posZ
    bx, by, bz, ang = calcBhat(Bx, By, Bz, dt)
    Mx, My, Mz = calcRRF(Mx, My, Mz, bx, by, bz, ang)

Mxy_space = np.abs(Mx + 1j * My)

print(Mxy_flow.shape, Mxy_offres.shape, Mxy_space.shape)
