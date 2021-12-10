import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection

def convert_labels(scores_labels, normal_cls):
    # we use 1 for anomalies, and 0 for normal samples https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/datasets/mnist.py#L34
    scores_labels = [0 if y in normal_cls else 1 for y in scores_labels]
    return scores_labels

def get_equal_AD_classes(x_norm, y_norm, x_anom, y_anom):
    """
    to avoid the possible issues and bias of unbalanced datasets, an equal number of normal and anomalous samples
    will be considered in the validation and test sets (ie. a sampling over all classes is done in the majority AD class)
    """
    if x_anom.shape[0]>x_norm.shape[0]:
        nbr_samples_needed = x_norm.shape[0]
        shuffled_idx = np.arange(x_anom.shape[0])
        np.random.shuffle(shuffled_idx)
        x_anom = x_anom[shuffled_idx[:nbr_samples_needed]]
        y_anom = y_anom[shuffled_idx[:nbr_samples_needed]]
    elif x_anom.shape[0]<x_norm.shape[0]:
        nbr_samples_needed = x_anom.shape[0]
        shuffled_idx = np.arange(x_norm.shape[0])
        np.random.shuffle(shuffled_idx)
        x_norm = x_norm[shuffled_idx[:nbr_samples_needed]]
        y_norm = y_norm[shuffled_idx[:nbr_samples_needed]]
    else:
        # as many normal samples as there are anomalous samples, no modification needed
        pass
    x = np.concatenate((x_norm, x_anom))
    y = np.concatenate((y_norm, y_anom))
    shuffled_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffled_idx)
    x = x[shuffled_idx]
    y = y[shuffled_idx]
    return x, y

def get_dataloaders_MNIST_FashionMNIST(batch_size, normal_classes, dataset_name, testvalid_ratio=0.5, seed=1):

    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = MNIST(root='./data/', train=True, download=True, transform=transform)
        test_set = MNIST(root='./data/', train=False, download=True, transform=transform)
        x_train = train_set.data.numpy()
        y_train = train_set.targets.numpy()
        x_test = test_set.data.numpy()
        y_test = test_set.targets.numpy()
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = FashionMNIST(root='./data/', train=True, download=True, transform=transform)
        test_set = FashionMNIST(root='./data/', train=False, download=True, transform=transform)
        x_train = train_set.data.numpy()
        y_train = train_set.targets.numpy()
        x_test = test_set.data.numpy()
        y_test = test_set.targets.numpy()
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = CIFAR10(root='./data/', train=True, download=True, transform=transform)
        test_set = CIFAR10(root='./data/', train=False, download=True, transform=transform)
        x_train = train_set.data
        y_train = np.asarray(train_set.targets)
        x_test = test_set.data
        y_test = np.asarray(test_set.targets)
    else:
        raise ValueError('Dataset not implemented in get_dataloaders_MNIST_FashionMNIST() !')

    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=testvalid_ratio, random_state=seed)

    idx_norm_train = np.zeros_like(y_train)
    idx_norm_valid = np.zeros_like(y_valid)
    idx_norm_test = np.zeros_like(y_test)
    for normcls in normal_classes:
        idx_norm_train += np.where(y_train == normcls, 1, 0)
        idx_norm_valid += np.where(y_valid == normcls, 1, 0)
        idx_norm_test += np.where(y_test == normcls, 1, 0)

    x_train = x_train[idx_norm_train!=0] # != 0 for conversion to boolean array https://stackoverflow.com/questions/20373039/how-do-i-convert-a-numpy-matrix-into-a-boolean-matrix
    y_train = y_train[idx_norm_train!=0]

    x_valid_norm = x_valid[idx_norm_valid != 0]
    y_valid_norm = y_valid[idx_norm_valid != 0]
    x_test_norm = x_test[idx_norm_test != 0]
    y_test_norm = y_test[idx_norm_test != 0]

    x_valid_anom = x_valid[idx_norm_valid == 0]
    y_valid_anom = y_valid[idx_norm_valid == 0]
    x_test_anom = x_test[idx_norm_test == 0]
    y_test_anom = y_test[idx_norm_test == 0]

    x_valid, y_valid = get_equal_AD_classes(x_valid_norm, y_valid_norm, x_valid_anom, y_valid_anom)
    x_test, y_test = get_equal_AD_classes(x_test_norm, y_test_norm, x_test_anom, y_test_anom)

    train_set = TensorDataset(torch.from_numpy(x_train/255).float(), torch.from_numpy(y_train))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    valid_set = TensorDataset(torch.from_numpy(x_valid/255).float(), torch.from_numpy(y_valid))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    test_set = TensorDataset(torch.from_numpy(x_test/255).float(), torch.from_numpy(y_test))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader, x_train, y_train, x_valid, y_valid, x_test, y_test


def init_network_weights_from_pretraining(net, dataset, normal_cls, seed):
    """
    Initialize the DeepSVDD or DeepMSVDD encoder network weights from the encoder weights of the pretraining CAE.
    Similar to https://github.com/lukasruff/Deep-SAD-PyTorch/blob/master/src/DeepSAD.py#L116
    """
    net_dict = net.state_dict()
    ae_net_dict = torch.load('./trained_models/{}_{}_{}.pt'.format(dataset, normal_cls, seed))
    # Filter out decoder network keys
    ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
    # Overwrite values in the existing state_dict
    net_dict.update(ae_net_dict)
    # Load the new state_dict
    net.load_state_dict(net_dict)
    return net

def save_pretrained_weights(net, dataset, normal_cls, seed):
    """
    save CAE to provide DeepSVDD and DeepMSVDD with pretrained weights, as indicated in the research papers
    """
    torch.save(net.state_dict(),'./trained_models/{}_{}_{}.pt'.format(dataset, normal_cls, seed))

def plot_distribution(data_loader, net, hyperspheres_center, device, status, normal_cls):
    net.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            try:
                complete_outputs = torch.cat((complete_outputs, outputs), dim=0)
                complete_labels = torch.cat((complete_labels, labels), dim=0)
            except UnboundLocalError:
                complete_outputs = outputs
                complete_labels = labels

    if len(hyperspheres_center.size())==1:
        hyperspheres_center = hyperspheres_center.unsqueeze(0)
    center_labels = torch.ones((hyperspheres_center.size()[0])).to(device)*10
    complete_outputs = torch.cat([complete_outputs, hyperspheres_center], dim=0)
    complete_labels = torch.cat([complete_labels, center_labels], dim=0)

    inputs = complete_outputs.cpu().detach().numpy()
    n_components = 2
    latent_2Ds = []
    latent_2D_names = []
    latent_2Ds.append(TSNE(n_components=n_components).fit_transform(inputs))
    latent_2D_names.append("TSNE")
    latent_2Ds.append(PCA(n_components=n_components).fit_transform(inputs))
    latent_2D_names.append("PCA")
    latent_2Ds.append(SparseRandomProjection(n_components=n_components).fit_transform(inputs))
    latent_2D_names.append("SRP")
    latent_2Ds.append(Isomap(n_components=n_components).fit_transform(inputs))
    latent_2D_names.append("ISOMAP")
    latent_2Ds.append(LocallyLinearEmbedding(n_components=n_components).fit_transform(inputs))
    latent_2D_names.append("LLE")

    c_dict = {0: 'deeppink', 1: 'red', 2: 'orangered', 3: 'chocolate', 4: 'yellowgreen',
              5: 'chartreuse', 6: 'green', 7: 'dodgerblue', 8: 'blue', 9: 'darkviolet', 10: 'black'}

    # red for anomaly, green for normal, black for centroid
    c_dict_AD = {0: 'red', 1: 'red', 2: 'red', 3: 'red', 4: 'red', 5: 'red', 6: 'red', 7: 'red', 8: 'red',
                 9: 'red', 10: 'black'}
    label_dict_AD = {0: '1', 1: '1', 2: '1', 3: '1', 4: '1', 5: '1', 6: '1', 7: '1', 8: '1', 9: '1', 10: '2'}
    for normal_class in normal_cls:
        c_dict_AD[normal_class] = 'green'
        label_dict_AD[normal_class] = '0'

    fig, axs = plt.subplots(1, 5, figsize=(25, 10))
    for fig_idx in range(len(latent_2Ds)):
        x = latent_2Ds[fig_idx][:,0]
        y = latent_2Ds[fig_idx][:,1]
        for label in range(11):
            bool_array = (complete_labels == label).cpu().numpy()
            if label == 10:
                axs[fig_idx].scatter(x[bool_array], y[bool_array], c=c_dict[label], label=label, alpha=1, s=100)
            else:
                axs[fig_idx].scatter(x[bool_array], y[bool_array], c=c_dict[label], label=label, alpha=0.1, s=25)
        axs[fig_idx].legend()
        axs[fig_idx].set_title("{} - Normal: {} - {}".format(latent_2D_names[fig_idx], normal_cls, status))
    plt.show()

    fig, axs = plt.subplots(1, 5, figsize=(25, 10))
    for fig_idx in range(len(latent_2Ds)):
        x = latent_2Ds[fig_idx][:, 0]
        y = latent_2Ds[fig_idx][:, 1]
        for label in range(11):
            bool_array = (complete_labels == label).cpu().numpy()
            if label == 10:
                axs[fig_idx].scatter(x[bool_array], y[bool_array], c=c_dict_AD[label], label=label_dict_AD[label], alpha=1, s=100)
            else:
                axs[fig_idx].scatter(x[bool_array], y[bool_array], c=c_dict_AD[label], label=label_dict_AD[label], alpha=0.1, s=25)
        axs[fig_idx].legend()
        axs[fig_idx].set_title("{} - Normal: {} - {}".format(latent_2D_names[fig_idx], normal_cls, status))
    plt.show()