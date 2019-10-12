import numpy as np
from hadamard import hadamard_s
import random
import h5py
import time
import torch

def circshift_r(inimg, shift):
    return np.roll(inimg, shift, axis = 0)

name = 'paviaU'
N = 127
[S_N,inv_S] = hadamard_s(N)
start_band = 0
end_band = 103

np.set_printoptions(precision=True)
t1 = time.time()
path = name +'.h5'
with h5py.File(path,'r') as f:
    data = np.array(f['data']).transpose(2,1,0)
    #data = torch.nn.functional.interpolate(torch.Tensor(data).unsqueeze(0), scale_factor=2, mode='bilinear').squeeze(0).permute(1,2,0).detach().cpu().numpy()

    data = data[:,:,:end_band]

    row, col, spec_length = data.shape
    # row_i = random.randint(0, row-N)
    # col_j = random.randint(0, col-N)
	
    for row_i in range(0, row-N+1, 10):
        for col_j in range(0, col-N+1, 10):

            row_i = min(random.randint(0, 9), row-N-row_i) + row_i
            col_j = min(random.randint(0, 9), col-N-col_j) + col_j
    

            X_S = data[row_i:row_i + N,col_j:col_j + N,:]
            X_S_Intensity = np.sum(X_S, 2)
            X_S_Spec = np.sum(X_S, 0)
            spec_length = X_S.shape[2]
            vec_length = spec_length + N
            coded_spectrum_measure_S = np.zeros([N,vec_length])

            x1 = np.expand_dims(X_S_Spec, 0).repeat(127,0)
            x2 = np.expand_dims(X_S_Intensity, 2).repeat((end_band-start_band),2)
            std_spectrum = x1*x2
    
    
            for S_idx in range(N):
                data_cube = np.zeros([N,vec_length])
                for spec_idx in range(N):
                    s_idx = std_spectrum[S_idx, spec_idx]
                    data_cube[spec_idx,:spec_length] = s_idx
                    data_cube[spec_idx,:] = circshift_r(data_cube[spec_idx,:],spec_idx)
                coded_spectrum_measure_S[S_idx,:] = np.matmul(S_N[S_idx:S_idx+1,:], data_cube)
            
            c_outpath = 'H5data' + '/coded/' + '%09d.h5'%(col_j * (row-N+1) + row_i)
            g_outpath = 'H5data' + '/gt/' + '%09d.h5'%(col_j * (row-N+1) + row_i)
            f = h5py.File(c_outpath,'w')
            f.create_dataset('data',data=coded_spectrum_measure_S)

            f = h5py.File(g_outpath,'w')
            f.create_dataset('data',data=X_S)
            print((col_j * (row-N+1) + row_i),'/',(row-N)*(col-N))
            
    
            
            ## loss function
            re_1 = np.matmul(np.linalg.inv(X_S_Intensity * S_N), coded_spectrum_measure_S)
            re = np.zeros_like(re_1)
            for shift_idx in range(N):
                re[shift_idx,:] = circshift_r(re_1[shift_idx,:],-shift_idx)

            f = re[:,:spec_length]
            f_gt = np.sum(X_S, 0)
            loss = f_gt - f
    
            print(np.mean(abs(loss)))
            
    
    
