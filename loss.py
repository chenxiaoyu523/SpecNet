import torch
import torch.nn as nn
from utils import tensorshift_r
import matplotlib.pyplot as plt

class ReconsLoss(nn.Module):

    def __init__(self, S_N):
        super(ReconsLoss, self).__init__()
        self.S_N = torch.Tensor(S_N).cuda().double()
        self.N = self.S_N.shape[0]
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, coded_spectrum_measure_S, XS_Int_pred, X_S):

        num, rows, cols = coded_spectrum_measure_S.shape
        spec_length = cols-rows
        re_1 = torch.zeros_like(coded_spectrum_measure_S)
        for n in range(num):
            re_1[n] = torch.mm(torch.inverse(XS_Int_pred[n] * self.S_N), coded_spectrum_measure_S[n])

        re = torch.zeros_like(re_1)
        for shift_idx in range(self.N):
            re[:,shift_idx,:] = tensorshift_r(re_1[:,shift_idx,:],-shift_idx)

        f = re[:,:,:spec_length]
        f_gt = torch.sum(X_S, 1)
        #loss = f_gt - f
        mse = self.criterion(f_gt, f)
        snr = self.snr_cal(f, mse)

        '''
        plt.figure('pred')
        plt.imshow(f[0].cpu().detach().numpy())
        plt.figure('gt_view')
        plt.imshow(f_gt[0].cpu().detach().numpy())
        plt.show()
        '''

        return 100-snr.mean(), snr.mean()

    def snr_cal(self, pred, mse):
        upper = (pred**2).sum(2)
        lower = mse.sum(2)
        snr = 10*torch.log10(upper/lower)
        return snr


