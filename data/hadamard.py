import numpy as np
import h5py

def circshift_r(inimg, shift):
    return np.roll(inimg, shift, axis = 0)

def isprime(num):
    if num == 1:
        return False

    for i in range(2, num // 2 + 1):
        if num % i == 0:
            return False
    return True

def hadamard_s(n):
    if isprime(n) == 0:
        print('error! n must be a prime\n')
        circ_S = 0
        inv_circ_S = 0
        return False
    if (n - 3) % 4 != 0:
        print('error! n must be 4m + 3\n')
        circ_S = 0
        inv_circ_S = 0
        return False

    A_k = np.array(range(1,(n-1)//2+1))**2
    A_k = A_k % n
    
    S_1 = np.zeros([1,n])
    S_1[0,0]=1.
    S_1[0,A_k]=1.
    
    circ_S=np.zeros([n,n])
    circ_S[0,:]=S_1

    for i in range(2,n+1):
        circ_S[i-1,:]=circshift_r(circ_S[i-2,:],-1)

    pos_measure = circ_S>0
    pos_measure_non = circ_S==0
    pos_measure_non_matr = True - circ_S
    pos_measure_matr = circ_S
    pos_measure_matr[pos_measure_non] = 0

    inv_circ_S = (pos_measure_matr - pos_measure_non_matr)/(n+1)*2

    return circ_S, inv_circ_S

