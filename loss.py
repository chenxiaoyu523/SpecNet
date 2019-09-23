import torch
import torch.nn as nn
from utils import tensorshift_r

class ReconsLoss(nn.Module):

    def __init__(self, S_N):
        super(ReconsLoss, self).__init__()
        self.S_N = torch.Tensor(S_N).cuda()
        self.N = self.S_N.shape[0]
        self.criterion = nn.MSELoss()

    def forward(self, coded_spectrum_measure_S, XS_Int_pred, X_S):

        num, rows, cols = coded_spectrum_measure_S.shape
        spec_length = rows-cols
        re_1 = torch.zeros_like(coded_spectrum_measure_S)
        for n in range(num):
            re_1[n] = torch.mm(torch.inverse(XS_Int_pred[n] * self.S_N), coded_spectrum_measure_S[n])
        re = torch.zeros_like(re_1)
        for shift_idx in range(self.N):
            re[:,shift_idx,:] = tensorshift_r(re_1[:,shift_idx,:],-shift_idx)

        f = re[:,:,:spec_length]
        f_gt = torch.sum(X_S, 1)
        #loss = f_gt - f
        loss = self.criterion(f_gt, f)
        return loss


