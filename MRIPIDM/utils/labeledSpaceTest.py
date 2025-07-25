import numpy as np
import random 
 
xSize = 3
ySize = 3
zSize = 3

Mx = np.zeros((5, 5, 2), dtype=np.float32)
My = np.zeros((5, 5, 2), dtype=np.float32)
Mz = np.ones((5, 5, 2), dtype=np.float32)
T1 = np.zeros((5, 5, 2), dtype=np.float32)
T2 = np.zeros((5, 5, 2), dtype=np.float32)
T2star = np.zeros((5, 5, 2), dtype=np.float32)
Rho = np.zeros((5, 5, 2), dtype=np.float32)

for i, val in np.ndenumerate(Mx):
    T1[i] = random.random()
    T2[i] = random.random()
    

outpath = "D:/Projects/MRIPIDMoutput/test.npz"

np.savez_compressed(outpath,
                        Gyro=4257.59,
                        xSize=xSize,
                        ySize=ySize,
                        zSize=zSize,
                        Mx=Mx,
                        My=My,
                        Mz=Mz,
                        T1=T1,
                        T2=T2,
                        T2star=T2star,
                        Rho=Rho)