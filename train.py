import torch
import matplotlib.pyplot as plt
from args import get_arguments
import sys
args = get_arguments()

class Train:

    def __init__(self, model, data_loader, optim, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.device = device

    def run_epoch(self, lr_updater, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        epoch_snr = 0.0
        for step, batch_data in enumerate(self.data_loader):

            # Get the inputs and labels
            inputs = batch_data[0].to(self.device).unsqueeze(1)
            labels = batch_data[1].to(self.device)
            gt_view = torch.sum(labels, 3).unsqueeze(1)

            # Forward propagation
            # mask = (inputs>0).float()
            outputs = self.model(inputs)

            '''
            print(labels.shape)
            plt.figure('pred')
            plt.imshow(outputs[0,0].cpu().detach().numpy())
            plt.figure('gt_view')
            plt.imshow(gt_view[0].cpu().detach().numpy())
            plt.show()
            '''

            # Loss computation
            loss, snr = self.criterion(inputs.squeeze(1), outputs.squeeze(1), labels)
            loss_b = torch.nn.functional.mse_loss(outputs, gt_view)

            # Backpropagation
            self.optim.zero_grad()
            if args.step1:
              loss_b.backward()
            elif args.step2:
              loss.backward()
            else:
              print('please set which step for training')
              sys.exit(0)
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss_b.item()
            epoch_snr += snr.item()

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), epoch_snr / len(self.data_loader)
