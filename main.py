import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import transforms.transforms as ext_transforms
from models.erfnet import ERFNet
from train import Train
from test import Test
from args import get_arguments
import utils
from data import H5Loader as dataset
from loss import ReconsLoss
from hadamard import hadamard_s
import numpy as np
import visdom

# Get the arguments
args = get_arguments()

device = torch.device(args.device)


def load_dataset(dataset):
    print("\nLoading dataset...\n")

    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    # image_transform = ext_transforms.RandomCrop(336)
    image_transform = transforms.ToTensor()
    val_transform = transforms.ToTensor()

    train_set = dataset(
        args.dataset_dir,
        transform=image_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Load the validation set as tensors
    val_set = dataset(
        args.dataset_dir,
        transform=val_transform,
        mode='val')
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        transform=val_transform,
        mode='test')
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    return train_loader, val_loader, test_loader


def train(train_loader, val_loader, circ_S):
    print("\nTraining...\n")

    model = ERFNet(1).to(device).double()
    #criterion = nn.MSELoss()
    criterion = ReconsLoss(circ_S)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_loss, best_snr = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean loss = {1:.4f} Best mean snr = {1:.4f}".format(start_epoch, best_snr))
    else:
        start_epoch = 0
        best_loss = 0
        best_snr = 0

    if args.visdom:
        vis = visdom.Visdom()

        loss_win = vis.line(X=np.column_stack((np.array(start_epoch),np.array(start_epoch))),
                            Y=np.column_stack((np.array(best_loss),np.array(best_loss))),
                            opts=dict(legend=['train', 'test'],
                                      xlabel='epoch',
                                      ylabel='loss',
                                      title='Loss'))
        snr_win = vis.line(X=np.column_stack((np.array(start_epoch),np.array(start_epoch))),
                           Y=np.column_stack((np.array(0.),np.array(0.))),
                           opts=dict(legend=['train', 'test'],
                                     xlabel='epoch',
                                     ylabel='snr',
                                     title='SNR'))

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, device)
    val = Test(model, val_loader, criterion, device)
    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_loss, epoch_snr = train.run_epoch(lr_updater, args.print_step)
        lr_updater.step()
        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} Avg. snr: {2:.4f} Lr: {3:f}".
              format(epoch, epoch_loss, epoch_snr, lr_updater.get_lr()[0]))

        if (epoch + 1) % 1 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, snr = val.run_epoch(args.print_step)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} Avg. snr: {2:.4f}".
                  format(epoch, loss, snr))

            # Save the model if it's the best thus far
            if snr < best_snr:
                print("\nBest model thus far. Saving...\n")
                best_snr = snr
                utils.save_checkpoint(model, optimizer, epoch + 1, best_loss, best_snr,
                                      args)
        if args.visdom:
            vis.line(
                X=np.column_stack((np.array(epoch),np.array(epoch))),
                Y=np.column_stack((np.array(epoch_loss),np.array(loss))),
                win=loss_win,
                update='append')

            vis.line(
                X=np.column_stack((np.array(epoch),np.array(epoch))),
                Y=np.column_stack((np.array(epoch_snr),np.array(snr))),
                win=snr_win,
                update='append')

    return model


def test(model, test_loader, circ_S):
    print("\nTesting...\n")

    #criterion = nn.MSELoss()
    criterion = ReconsLoss(circ_S)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, device)

    print(">>>> Running test dataset")
    loss, (iou, miou) = test.run_epoch(args.print_step)
    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))


# Run only if this module is being run directly
if __name__ == '__main__':

    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    train_loader, val_loader, test_loader = load_dataset(dataset)
    circ_S, _ = hadamard_s(args.matrix_size)

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader, circ_S)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Intialize a new ERFNet model
            model = ERFNet(1).to(device).double()

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ERFNet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                      args.name)[0]

        test(model, test_loader, circ_S)
