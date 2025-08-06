import numpy as np
import os

Matrix = []

for i in range(5000):
    Matrix.append(np.ones((16, 16), dtype = "float32") * i)


outpath = "D:/Projects/MRIPIDMoutput/test.npz"

np.savez_compressed(outpath,
                        M=Matrix
                        )