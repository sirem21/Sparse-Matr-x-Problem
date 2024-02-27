from scipy.io import mmread
import scipy.sparse.linalg as sla
import time
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve

A = mmread('cage10.mtx')
A = A.tocsr()
m, n = A.shape
f = np.random.rand(m) 

def cimmino(A, f, K, iter):
    m, n = A.shape
    #find the number of rows in each K parts
    kpart = m // K
    sum_d= 0
    for t in range(iter):
        x = np.zeros(n)
        x1 = np.zeros_like(m)
        for k in range(K):
            #find the indexes of start and end row for each part
            start_row = k * kpart
            end_row = (k + 1) * kpart
            #divide the A anf to K parts
            Ak = A[start_row:end_row, :]
            fk = f[start_row:end_row]
            #find the residual and delta for each part
            r = fk - Ak.dot(x)
            delta = Ak.T.dot(spsolve(Ak.dot(Ak.T), r))
            #find the sum of deltas
            sum_d += delta
        x1 = x1 + sum_d
        return x1
#find the estimated time and relative residuals for each K part for Cimmino method
def time_residual_cimmino(A, f, K):
    start_time = time.time()
    x = cimmino(A, f, K, 1000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    residual = norm(A.dot(x) - f) / norm(f)
    return elapsed_time, residual

for K in [2, 4, 6, 8]:
    t,res=time_residual_cimmino(A, f, K)
    print(f"Cimmino Method with K={K}  Residual: {res} Time:{t}")
def least_square(A,f):
    start_time = time.time()
    result_lsqr = sla.lsqr(A, f)
    end_time = time.time()
    estimated_time = end_time - start_time
    return result_lsqr[0], estimated_time
result_lstq,time_lstq = least_square(A,f)
residual_lstq = norm(A.dot(result_lstq - f) / norm(f))
print(f"Least Square Method   Residual:{residual_lstq}   Time:{time_lstq}")

#Fit a polynomial for your results in block Cimmino without K=6, and estimate the time result for K=6.
residuals = []
times = []
for K in [2, 4, 8]:
    start_time = time.time()
    residual = norm(A.dot(cimmino(A, f, 6, 1000)) - f) / norm(f)
    end_time = time.time()
    #time and residual values of K=6 are collected in the times and residuals lists.
    residuals.append(residual)
    times.append(end_time - start_time)
#fit a polynomial of degree 2 because it estimates the relationship between time and residuals.
results = np.polyfit(times, residuals, 2)
polynom_6 = np.poly1d(results)
#estimate time for K=6 which is found by the fitted polynomial.
estimated_timeK6 = polynom_6(0)
print(f"Estimated time for K=6: {estimated_timeK6} ")

#Fit a polynomial for your results in block Cimmino including all results
residuals = []
times = []
for K in [2, 4, 6, 8]:
        start_time = time.time()
        residual = norm(A.dot(cimmino(A, f, 16, 1000)) - f) / norm(f)
        end_time = time.time()
        residuals.append(residual)
        times.append(end_time - start_time)
results = np.polyfit(times, residuals, 2)
polynom_16 = np.poly1d(results)
estimated_timeK16 = polynom_16(0)
print(f"Estimated time for K=16: {estimated_timeK16}")

# measure the actual time for K=16
start_K16 = time.time()
cimmino(A, f,16, 1000)
end_K16 = time.time()
actual_K16 = end_K16 - start_K16
print(f"Actual time for K=16: {actual_K16}")

