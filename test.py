import torch
import matplotlib.pyplot as plt

class Test:

    def __init__(self, model, data_loader, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device

    def run_epoch(self, iteration_loss=False):

        self.model.eval()
        epoch_loss = 0.0
        epoch_snr = 0.0
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device).unsqueeze(1)
            labels = batch_data[1].to(self.device)

            with torch.no_grad():
                # Forward propagation
                # mask = (inputs>0).float()
                outputs = self.model(inputs)

                '''
                gt_view = torch.sum(labels, 3)
                plt.figure('pred')
                plt.imshow(outputs[0,0].cpu().detach().numpy())
                plt.figure('gt_view')
                plt.imshow(gt_view[0].cpu().detach().numpy())
                plt.show()
                '''
                             

                # Loss computation
                loss, snr = self.criterion(inputs.squeeze(1), outputs.squeeze(1), labels)
                #gt_view = torch.sum(labels, 3).unsqueeze(1)
                #loss = self.criterion(outputs, gt_view)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()
            epoch_snr += snr.item()

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), epoch_snr / len(self.data_loader)
