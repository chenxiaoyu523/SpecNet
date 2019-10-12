import numpy as np
from hadamard import hadamard_s
import random
import h5py
import time
import torch

def tensorshift_r(inimg, shift):
    return torch.roll(inimg, shift, 0)
def circshift_r(inimg, shift):
    return np.roll(inimg, shift, axis = 0)

N = 127

[S_N,inv_S] = hadamard_s(N)
S_N = S_N.astype(np.float)

L = 32
iter_num = 500

fcoded = h5py.File('H5data/coded/000100379.h5','r')
fgt = h5py.File('H5data/gt/000100379.h5','r')

coded_spectrum_measure_S = np.array(fcoded['data']).astype(np.float)
X_S = np.array(fgt['data']).astype(np.float)

X_S_Intensity = np.sum(X_S, 2)

X_S = torch.from_numpy(X_S)
X_S_Intensity = torch.from_numpy(X_S_Intensity)
S_N = torch.from_numpy(S_N)
coded_spectrum_measure_S = torch.from_numpy(coded_spectrum_measure_S)
re_1 = torch.matmul(torch.inverse(X_S_Intensity * S_N), coded_spectrum_measure_S)

# re_1 = np.matmul(np.linalg.inv(X_S_Intensity * S_N), coded_spectrum_measure_S)

re = torch.zeros_like(re_1)
for shift_idx in range(N):
    re[shift_idx,:] = tensorshift_r(re_1[shift_idx,:],-shift_idx)

print(re.detach().cpu().numpy()[0])

f = re[:,:103]
f_gt = torch.sum(X_S, 0)
print(torch.mean(abs(f-f_gt)))

print(xxxxx)

re = np.zeros_like(re_1)
for shift_idx in range(N):
    re[shift_idx,:] = circshift_r(re_1[shift_idx,:],-shift_idx)

f = re[:,:103]
f_gt = np.sum(X_S, 0)
loss = f_gt - f
    
print(np.mean(abs(loss)))

'''
np.set_printoptions(precision=True)
t1 = time.time()
with h5py.File('paviaU.h5','r') as f:
    data = np.array(f['data']).transpose(2,1,0)
    row, col, spec_length = data.shape
    row_i = random.randint(0, row-N)
    col_j = random.randint(0, col-N)
    

    X_S = data[row_i:row_i + N,col_j:col_j + N,:]
    X_S_Intensity = np.sum(X_S, 2)
    X_S_Spec = np.sum(X_S, 0)
    spec_length = X_S.shape[2]
    vec_length = spec_length + N
    coded_spectrum_measure_S = np.zeros([N,vec_length])

    x1 = np.expand_dims(X_S_Spec, 0).repeat(127,0)
    x2 = np.expand_dims(X_S_Intensity, 2).repeat(103,2)
    std_spectrum = x1*x2
    
    
    for S_idx in range(N):
        data_cube = np.zeros([N,vec_length])
        for spec_idx in range(N):
            s_idx = std_spectrum[S_idx, spec_idx]
            data_cube[spec_idx,:spec_length] = s_idx
            data_cube[spec_idx,:] = circshift_r(data_cube[spec_idx,:],spec_idx)
        coded_spectrum_measure_S[S_idx,:] = np.matmul(S_N[S_idx:S_idx+1,:], data_cube)
    
    print(X_S_Intensity.shape, S_N.shape, coded_spectrum_measure_S.shape)
    ## loss function
    re_1 = np.matmul(np.linalg.inv(X_S_Intensity * S_N), coded_spectrum_measure_S)
    re = np.zeros_like(re_1)
    for shift_idx in range(N):
        re[shift_idx,:] = circshift_r(re_1[shift_idx,:],-shift_idx)

    f = re[:,:spec_length]
    f_gt = np.sum(X_S, 0)
    print(f.shape, f_gt.shape)
    loss = f_gt - f
    print(time.time()-t1)
    
    print(np.mean(abs(loss)))
'''
