import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle


from vae.modules import ResNet_VAE, convtrans2D_output_size
from data.data import SingleImageLabeledDataset

data_dir = "/home/xxd9704/evdata/2020_hackathon"
train_package_txt = "/home/xxd9704/Workspace/hacktiff2020transferlearning/eighty_list.txt"
val_package_txt = "/home/xxd9704/Workspace/hacktiff2020transferlearning/twenty_list.txt"
size_cutoff = 697

train_whole = True
inference = True

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 256     # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability


# training parameters
epochs = 100        # training epochs
batch_size = 50
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# save model
save_model_path = './results_evtech'



def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def loss_function(recon_x, x, mu, logvar):
    # MSE = F.mse_loss(recon_x, x, reduction='sum')
    MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()
    if inference:
        model.load_state_dict(torch.load('/home/xxd9704/Workspace/hacktiff2020transferlearning/results_cifar10/model_epoch24.pth'))
        model.eval()

    losses = []
    all_y, all_z, all_mu, all_logvar, all_id = [], [], [], [], []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, observation in enumerate(train_loader):
        X = observation['image']
        # distribute data to device
        X = X.to(device)
        #angle = observation['angle']
        y = observation['measurements']
        N_count += X.size(0)

        optimizer.zero_grad()
        X_reconst, z, mu, logvar = model(X)  # VAE
        loss = loss_function(X_reconst, X, mu, logvar)
        losses.append(loss.item())
        if not inference:
            loss.backward()
            optimizer.step()

        all_z.extend(z.data.cpu().numpy())
        all_y.extend(y.data.numpy())
        all_mu.extend(mu.data.cpu().numpy())
        all_logvar.extend(logvar.data.cpu().numpy())
        #all_angle.extend(angle)
        all_id.extend(observation['id'])

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))

    all_y = np.stack(all_y, axis=0)
    all_z = np.stack(all_z, axis=0)
    all_mu = np.stack(all_mu, axis=0)
    all_logvar = np.stack(all_logvar, axis=0)
    #all_angle = np.stack(angle)
    all_id = np.stack(all_id)

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return X_reconst.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, losses, all_id


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y, all_z, all_mu, all_logvar = [], [], [], []
    with torch.no_grad():
        for observation in test_loader:
            # distribute data to device
            X = observation['image']
            X = X.to(device)
            y = observation['measurements']
            X_reconst, z, mu, logvar = model(X)

            loss = loss_function(X_reconst, X, mu, logvar)
            test_loss += loss.item()  # sum up batch loss

            all_y.extend(y.data.numpy())
            all_z.extend(z.data.cpu().numpy())
            all_mu.extend(mu.data.cpu().numpy())
            all_logvar.extend(logvar.data.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    all_z = np.stack(all_z, axis=0)
    all_mu = np.stack(all_mu, axis=0)
    all_logvar = np.stack(all_logvar, axis=0)

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(len(test_loader.dataset), test_loss))
    return X_reconst.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, test_loss


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if use_cuda else {}
# transform = transforms.Compose([transforms.Resize([res_size, res_size]),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

# cifar10 dataset (images and labels)
# cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# cifar10_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_dataset = SingleImageLabeledDataset(data_dir, train_package_txt,
                                          transform, size_cutoff)

if not  train_whole:
    val_dataset = SingleImageLabeledDataset(data_dir, val_package_txt,
                                            transform, size_cutoff)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
if not  train_whole:
    valid_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Create model
resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

print("Using", torch.cuda.device_count(), "GPU!")
model_params = list(resnet_vae.parameters())
optimizer = torch.optim.Adam(model_params, lr=learning_rate)


# record training process
epoch_train_losses = []
epoch_test_losses = []
check_mkdir(save_model_path)

# start training
for epoch in range(epochs):
    # train, test model
    X_reconst_train, y_train, z_train, mu_train, logvar_train, train_losses, ids = train(log_interval, resnet_vae, device, train_loader, optimizer, epoch)
    if not train_whole:
        X_reconst_test, y_test, z_test, mu_test, logvar_test, epoch_test_loss = validation(resnet_vae, device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    if not train_whole:
        epoch_test_losses.append(epoch_test_loss)

    # save all train test results
    A = np.array(epoch_train_losses)
    C = np.array(epoch_test_losses)
    
    np.save(os.path.join(save_model_path, 'ResNet_VAE_training_loss.npy'), A)
    np.save(os.path.join(save_model_path, 'y_evtech_train_epoch{}.npy'.format(epoch + 1)), y_train)
    np.save(os.path.join(save_model_path, 'z_evtech_train_epoch{}.npy'.format(epoch + 1)), z_train)
    np.save(os.path.join(save_model_path, 'id_evtech_train_epoch{}.npy'.format(epoch + 1)), ids)
    
 
    if inference:
        np.save(os.path.join(save_model_path, 'Xrecon_evtech_train_epoch{}.npy'.format(epoch + 1)), X_reconst_train)
    if not train_whole:
        np.save(os.path.join(save_model_path, 'z_evtech_test_epoch{}.npy'.format(epoch + 1)), z_test)
        np.save(os.path.join(save_model_path, 'y_evtech_test_epoch{}.npy'.format(epoch + 1)), y_test)
