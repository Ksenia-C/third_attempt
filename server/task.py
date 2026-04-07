"""flower-tutorial: A Flower / PyTorch app."""

import pickle
import io
import PIL
import ray
import copy
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, Lambda
from PIL import Image
from typing import Optional, List
import numpy as np
import partitioners as parts
from flwr_datasets.visualization import plot_label_distributions
from datasets import Dataset
from pathlib import Path
import pandas as pd

TEST_SIZE = 0.2

def get_partiotioner(name, params):
    sized_distribution=params.get('sized_distribution', 'norm')
    random_state=params.get('random_state', 42)
    if name == 'random_uniform':
        return lambda *args, **kwargs : parts.RandomPartitioner(*args, sized_distribution=sized_distribution, random_state=random_state, sized_dist_params=params, **kwargs)
    elif name == 'rare_on_rare':
        rare_clients = params.get('rare_clients', 10)
        rare_data_count = params.get('rare_data_count', 2)
        fillage_percent = params.get('fillage_percent', 0.9)
        return lambda *args, **kwargs: parts.RareOnRarePartitioner(*args, sized_distribution=sized_distribution,random_state=random_state,rare_clients=rare_clients, rare_data_count=rare_data_count, fillage_percent=fillage_percent,sized_dist_params=params, **kwargs)
    elif name == 'rare_on_often':
        often_clients = params.get('often_clients', 10)
        rare_data_count = params.get('rare_data_count', 2)
        fillage_percent = params.get('fillage_percent', 0.1)
        return lambda *args, **kwargs: parts.RareOnOftenPartitioner(*args,sized_distribution=sized_distribution,random_state=random_state, often_clients=often_clients, rare_data_count=rare_data_count, fillage_percent=fillage_percent,sized_dist_params=params, **kwargs)
    elif name == 'often_on_often':
        often_clients = params.get('often_clients', 10)
        often_data_cnt = params.get('often_data_cnt', 2)
        fillage_percent = params.get('fillage_percent', 0.9)
        return lambda *args, **kwargs: parts.OftenOnOftenPartitioner(*args,sized_distribution=sized_distribution,random_state=random_state, often_clients=often_clients, often_data_cnt=often_data_cnt, fillage_percent=fillage_percent, sized_dist_params=params,**kwargs)
    elif name == 'often_everywhere':
        often_clients = params.get('often_clients', 10)
        often_data_cnt = params.get('often_data_cnt', 2)
        constant_on_often = params.get('constant_on_often', 200)
        percentage_on_others = params.get('percentage_on_others', 0.2)
        return lambda *args, **kwargs: parts.OftenEverywherePartitioner(*args,sized_distribution=sized_distribution,random_state=random_state, often_clients=often_clients, often_data_cnt=often_data_cnt, constant_on_often=constant_on_often, percentage_on_others=percentage_on_others,sized_dist_params=params, **kwargs)
    
    else:
        raise RuntimeError("name for partitioner not implemented")


pytorch_transforms = Compose([Resize((128, 128)), Lambda(lambda x: np.array(x))])
import torchvision.models as models

import albumentations as alb

import matplotlib.pyplot as plt
from pathlib import Path

def plot_augmented_distribution(augmented_data_class, saver_directory: Path, mode, filename="augmentation_distribution.png",):
    """
    Creates a bar plot of the augmented class sizes and saves it to saver_directory/filename.
    """
    if (mode != 'train'):
        return
    saver_directory.mkdir(parents=True, exist_ok=True)
    
    # Sort classes for consistent ordering
    classes = sorted(augmented_data_class.keys())
    sizes = [augmented_data_class[cls] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, sizes, color='skyblue')
    plt.xlabel('Class Label')
    plt.ylabel('Augmented Size')
    plt.title('Augmented Data Distribution per Class')
    plt.xticks(classes)  # Ensure all class labels are shown
    plt.tight_layout()
    
    save_path = saver_directory / filename
    plt.savefig(save_path, dpi=150)
    plt.close()  # Close to free memory
    print(f"Augmentation distribution plot saved to {save_path}")

class DataFrameImageDataset(Dataset):
    def __init__(self, dataframe, saver_directory, image_path_col='image', label_col='label', scale_aggresive=6, scale_mild=2, consider_small=0.1, consider_mild = 0.2, augemntation_pipeline=None, mode='test'):
        self.augemntation_pipeline = augemntation_pipeline

        self.image_path_col = image_path_col
        self.label_col = label_col
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        self.transform = Compose([
            ToTensor(),
            preprocess
        ])
        if self.augemntation_pipeline is None:
            self.dataframe = dataframe

            label_grouped = dataframe.groupby([self.label_col]).count().reset_index()
            class_sizes_map = {}
            for i in range(dataframe.shape[0]):
                label = dataframe.iloc[i][self.label_col]
                if label not in class_sizes_map:
                    class_sizes_map[label] = label_grouped[label_grouped[self.label_col] == label][self.image_path_col].values[0]
            augemented_data_class = {}
            for label in class_sizes_map.keys():
                augemented_data_class[label] = class_sizes_map[label]
            
            plot_augmented_distribution(augemented_data_class, saver_directory, mode)
        else:
            global_ind = 0
            label_grouped = dataframe.groupby([self.label_col]).count().reset_index()
            result_dataframe = []
            class_sizes_map = {}
            all_class_number = 0
            for i in range(dataframe.shape[0]):
                label = dataframe.iloc[i][self.label_col]
                if label not in class_sizes_map:
                    class_sizes_map[label] = label_grouped[label_grouped[self.label_col] == label][self.image_path_col].values[0]
                    all_class_number += class_sizes_map[label]
            augemented_data_class = {}
            for label in class_sizes_map.keys():
                if class_sizes_map[label] / all_class_number < consider_small:
                    augemented_data_class[label] = scale_aggresive * class_sizes_map[label]
                elif class_sizes_map[label] / all_class_number < consider_mild:
                    augemented_data_class[label] = scale_mild * class_sizes_map[label]
                else:
                    augemented_data_class[label] = class_sizes_map[label]
            
            plot_augmented_distribution(augemented_data_class, saver_directory, mode)

            for i in range(dataframe.shape[0]):
                label = dataframe.iloc[i][self.label_col]
                if class_sizes_map[label] / all_class_number < consider_small:
                    result_dataframe += [[label, dataframe.iloc[i][self.image_path_col], -1]]
                    result_dataframe += [[label, None, global_ind]] * (scale_aggresive - 1)
                    global_ind += scale_aggresive
                elif class_sizes_map[label] / all_class_number < consider_mild:
                    result_dataframe += [[label, dataframe.iloc[i][self.image_path_col], -1]]
                    result_dataframe += [[label, None, global_ind]] * (scale_mild - 1)
                    global_ind += scale_mild
                else:
                    result_dataframe += [[label, dataframe.iloc[i][self.image_path_col], -1]]
                    global_ind += 1
            self.dataframe = pd.DataFrame(result_dataframe, columns=[self.label_col, self.image_path_col, "index_of_original"])

                    
                
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if (isinstance(idx, List)):
            return self.__getitems__(idx)
        image = self.dataframe.iloc[idx][self.image_path_col]
        label = self.dataframe.iloc[idx][self.label_col]

        if self.augemntation_pipeline is not None:
            if self.dataframe.iloc[idx]["index_of_original"] != -1:
                image = self.dataframe.iloc[self.dataframe.iloc[idx]["index_of_original"]][self.image_path_col]
                image = PIL.Image.fromarray(self.augemntation_pipeline(image=np.asarray(image.convert()))['image'])
        if self.transform:
            image = self.transform(image)
        return image, label

    def __getitems__(self, idxes: List) -> List:
        
        images = []
        labels = self.dataframe.iloc[idxes][self.label_col].values
        if self.augemntation_pipeline is not None:
            for idx in idxes:
                image = self.dataframe.iloc[idx][self.image_path_col]
                if self.dataframe.iloc[idx]["index_of_original"] != -1:
                    image = self.dataframe.iloc[self.dataframe.iloc[idx]["index_of_original"]][self.image_path_col]
                    image = PIL.Image.fromarray(self.augemntation_pipeline(image=np.asarray(image.convert()))['image'])
                if self.transform:
                    image = self.transform(image)
                images.append(image)
        else:
            images = [self.transform(image) if self.transform else image for image in self.dataframe.iloc[idxes][self.image_path_col].values]
        return [[image, label] for image, label in zip(images, labels)] 

@ray.remote
class PartitionerActor:
    def __init__(self):
        pass

    def create_partitioner(self, partitioner_name, partitioner_params, num_partitions, run_id, saver_directory, label_coumn):
        with open('/home/path_to_data/dataset_isic.pkl', 'rb') as f:
            dataset = pickle.load(f)
        partitioner = get_partiotioner(partitioner_name, partitioner_params)
        partitioner = partitioner(num_partitions=num_partitions)
        dataset_fragmanet = dataset
        maybe_distribution_image_path = saver_directory / 'distribution_plot.png'
        if not False:
            df_ = dataset_fragmanet.drop(columns=['image']).copy()
            dataset_ = Dataset.from_pandas(df_)
            partitioner_with_data = get_partiotioner(partitioner_name, partitioner_params)
            partitioner_with_data = partitioner_with_data(num_partitions=num_partitions)
            partitioner_with_data.dataset = dataset_
            fig, ax, df = plot_label_distributions(
                partitioner_with_data,
                label_name=label_coumn,
                plot_type="bar",
                size_unit="absolute",
                partition_id_axis="x",
                legend=True,
                verbose_labels=True,
                title="Per Partition Labels Distribution",
            )
            fig.savefig(maybe_distribution_image_path, bbox_inches='tight')
        partitioner.dataset=dataset_fragmanet
        self.partitioner = partitioner
        self.label_coumn = label_coumn
        return len(dataset_fragmanet[label_coumn].unique())
    def get_partiotion(self, cid: int):
        return self.partitioner.load_partition(partition_id=cid)

    def get_disbalance_metric(self, class_number, random_states_init):
        # we calculate N+2 values, N=number of classes
        # size_std: how client sizes are different. 0 = all the same. large = differ a lot
        # class_std: how each class is different on clients. 0 = all the same, large = differ a lot
        # local_std: how each client's data different. Second momentum of values. 0 = the balanced, large = disbalalanced
        sizes_for_disp = []
        for_disp_per_class = {}
        ginis = []
        all_data_cnt = 0
        for cid in range(self.partitioner.num_partitions):
            train_data, _ = train_test_split(self.get_partiotion(cid), test_size=TEST_SIZE, random_state=random_states_init.get_random_state(cid))
            sizes_for_disp.append(train_data.shape[0])
            all_data_cnt += train_data.shape[0]
            p_for_gini = []
            for class_ in range(class_number):
                class_presence = train_data[train_data[self.label_coumn] == class_].shape[0]
                for_disp_per_class[class_] = for_disp_per_class.get(class_, []) + [class_presence / train_data.shape[0]]
                p_for_gini.append(class_presence / train_data.shape[0])
            ginis.append(sum([p * (1 - p) for p in p_for_gini]))

        stats = {
            'size_std': np.std([x / all_data_cnt for x in sizes_for_disp]),
            'class_std': {class_: np.std(values) for class_, values in for_disp_per_class.items()},
            'local_std': sum([(1 - x)**2 for x in ginis]) / self.partitioner.num_partitions
        }
        area_elements = [stats['size_std']] + [stats['local_std']] + sorted([value for key, value in stats['class_std'].items()])
        area = 0
        for ind in range(1, len(area_elements)):
            area += (area_elements[ind - 1] + area_elements[ind]) / 2
        stats['area'] = area 
        return stats


def load_partition(partition, partition_id: int, num_partitions: int, run_id: str, label_col, augemntation_pipeline, saver_directory, random_state):    
    train_data, test_data = train_test_split(partition, test_size=TEST_SIZE, random_state=random_state)
    train_dataset = DataFrameImageDataset(train_data,saver_directory,  'image', label_col, augemntation_pipeline=augemntation_pipeline, mode='train')
    val_dataset = DataFrameImageDataset(test_data,saver_directory,  'image', label_col, augemntation_pipeline=None)
    
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=64)
    return trainloader, testloader

from focal_loss import FocalLoss
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def print_confustion_matrix(targets, predictions, path_to_save, class_names):
    cm = confusion_matrix(targets, predictions, labels=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Number of Images'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')

from models import FLAlgorithm

def train(net, trainloader, valloader, epochs, lr, device, saver_directory, client_modification, class_names):
    """Train the model on the training set."""
    net.to(device)
    saver_directory.mkdir(parents=True, exist_ok=True)
    # criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    criterion = FocalLoss(gamma=2.0, reduction='sum')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    for _ in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        for bathc_ind, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            if client_modification.method == FLAlgorithm.FED_PROX:
                proximal_term = 0
                for local_weights, global_weights in zip(net.parameters(), client_modification.global_params):
                    proximal_term += (local_weights - global_weights).norm(2)
                loss = criterion(F.softmax(net(images), dim=1), labels) + (client_modification.proximal_mu / 2) * proximal_term
            else:
                loss = criterion(F.softmax(net(images), dim=1), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if bathc_ind % 10 == 1:
                print(bathc_ind, 'out of', len(trainloader), running_loss / bathc_ind / labels.size(0))

    avg_trainloss = running_loss / len(trainloader.dataset)

    net.eval()
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for (images, labels) in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            y_true_val += labels.cpu().numpy().tolist()
            y_pred_val += outputs.max(1)[1].cpu().numpy().tolist()

    print_confustion_matrix(y_true_val, y_pred_val, saver_directory / 'confusion_matrix_val.png', class_names=class_names)

    return avg_trainloss

def scaffold_train(net, c_local, c_global, eta_local, trainloader, valloader, epochs, lr, device, saver_directory, client_modification, class_names):
    """Train the model on the training set."""
    net.to(device)
    saver_directory.mkdir(parents=True, exist_ok=True)
    # criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    criterion = FocalLoss(gamma=2.0, reduction='sum')

    optimizer = torch.optim.SGD(net.parameters(), lr=eta_local)
    net.train()
    K_calculate = 0
    for _ in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        for bathc_ind, (images, labels) in enumerate(trainloader):
            K_calculate += 1
            images, labels = images.to(device), labels.to(device)
            loss = criterion(F.softmax(net(images), dim=1), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            with torch.no_grad():
                for param, loc, glob in zip(net.parameters(), c_local, c_global):
                    delta = eta_local * (loc - glob)
                    param.add_(delta)
            if bathc_ind % 10 == 1:
                print(bathc_ind, 'out of', len(trainloader), running_loss / bathc_ind / labels.size(0))

    avg_trainloss = running_loss / len(trainloader.dataset)

    net.eval()
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for (images, labels) in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            y_true_val += labels.cpu().numpy().tolist()
            y_pred_val += outputs.max(1)[1].cpu().numpy().tolist()

    print_confustion_matrix(y_true_val, y_pred_val, saver_directory / 'confusion_matrix_val.png', class_names)

    return avg_trainloss, K_calculate

from sklearn.metrics import f1_score

def test(net, testloader, device, saver_directory: Path, class_names):
    saver_directory.mkdir(parents=True, exist_ok=True)
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion_cross_entropy = torch.nn.CrossEntropyLoss(reduction='sum')
    criterion = FocalLoss(gamma=2.0, reduction='sum')
    correct, loss, loss_cross_entropy = 0, 0.0, 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss_cross_entropy += criterion_cross_entropy(outputs, labels).item()
            loss += criterion(F.softmax(outputs, dim=1), labels).item()
            correct += outputs.max(1)[1].eq(labels).sum().item()
            y_true += labels.cpu().numpy().tolist()
            y_pred += outputs.max(1)[1].cpu().numpy().tolist()
    print_confustion_matrix(y_true, y_pred, saver_directory / 'confusion_matrix.png', class_names)

    accuracy = correct / len(testloader.dataset)
    loss_cross_entropy = loss_cross_entropy / len(testloader.dataset)
    loss = loss / len(testloader.dataset)
    f1 = f1_score(y_true, y_pred, average='weighted')

    extra_for_research = {}


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for one_class in class_names:
        mask = (y_true == one_class)
        count = np.sum(y_pred[mask] == one_class)
        extra_for_research['evil_' +str(one_class) + '_TP'] = count.item()
        extra_for_research['evil_' +str(one_class) + '_all_for_precision'] = np.sum(y_pred == one_class).item()
        extra_for_research['evil_' +str(one_class) + '_all_for_recall'] = np.sum(y_true == one_class).item()
    return loss, extra_for_research | {'accuracy': accuracy, 'f1': f1, 'loss_cross_entropy': loss_cross_entropy}
