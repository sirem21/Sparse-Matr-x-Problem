import numpy as np
from scipy.io import mmread
import scipy.sparse.linalg as sla
import time

#read the matrices from files
A = mmread('poisson3Da.mtx')
f = mmread('poisson3Da_b.mtx')

#spsolve (sparse LU) with the following fill-in reducing orderings
for i in ['NATURAL', 'MMD_ATA', 'COLAMD']:
    start1 = time.time()
    x=sla.spsolve( A, f, permc_spec=i)
    end1 = time.time()
    time_1 = end1 - start1
    residual_1 = np.linalg.norm(A.dot(x)-f) / np.linalg.norm(f)
    print(f"Time  of {i}:", time_1)
    print(f"Residual  of {i}:", residual_1)
print("==============================================================================")

# QR decomposition
start2 = time.time()
Q, R = np.linalg.qr(A.todense())
y = np.dot(Q.T, f)
x2 = np.linalg.solve(R, y)
end2 = time.time()
time_2 = end2 - start2
residual_2 = np.linalg.norm(A.dot(x2)-f) / np.linalg.norm(f)
print("Time of QR Decomposition: ", time_2)
print("Residual of QR Decomposition: ", residual_2)
print("==============================================================================")

# Bicg-stab
start3 = time.time()
x3,_ = sla.bicgstab(A, f, maxiter=1000)
end3 = time.time()
time_3 = end3 - start3
residual_3 = np.linalg.norm(A.dot(x3)-f) / np.linalg.norm(f)
print("Time of Bicg-Stab: ", time_3)
print("Residual of Bicg-Stab:", residual_3)
print("==============================================================================")

# Bicgstab  preconditioning ILU
start4 = time.time()
B = sla.spilu(A, drop_tol=1e-12, fill_factor=1)
Mz = lambda r: B.solve(r)
Minv = sla.LinearOperator(A.shape, Mz)
x4,_ = sla.bicgstab(A, f, maxiter=1000, M=Minv, tol=1e-5)
end4 = time.time()
time_4 = end4 - start4
residual_4 = np.linalg.norm(A.dot(x4)-f) / np.linalg.norm(f)
print("Time of ILU preconditioned Bicg-Stab: ", time_4)
print("Residual of ILU preconditioned Bicg-Stab: ", residual_4)
print("==============================================================================")