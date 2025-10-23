import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import random
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


def download_dataset(path, opt_label):
    # for EDA we can use more of datasets
    dataset = datasets.load_dataset(path)
    if opt_label in dataset:
        return pd.DataFrame(dataset[opt_label])
    return pd.DataFrame(dataset)

def oned_label_distribution(data):
    pass

def twod_label_distribution(data, label_col, group_col, labels_names):
    categories_size = data.groupby([group_col, label_col]).size().reset_index(name='count')
    sns_feed = categories_size.pivot(index=group_col, columns=label_col, values='count')
    ax = sns.heatmap(sns_feed, xticklabels=labels_names)
    plt.show()

    plt.stackplot(sns_feed.index, sns_feed.T.values, labels=labels_names, colors=sns.color_palette(n_colors=len(sns_feed.columns)))
    plt.legend(loc='upper left')
    plt.xlabel('group')
    plt.ylabel('stacked count')
    plt.show()  

def find_mean_photo(img_generator):
    result = None
    all_size = 0
    for img in img_generator:
        array = np.asarray(img.convert("RGB")).astype(float)
        all_size += 1
        if result is None:
            result = array.copy()
        else:
            result = (result + array).copy()
    if all_size == 0:
        return np.asarray(Image.fromarray(np.zeros((224, 224)))).astype(float)
    result = (result / all_size).astype(np.uint8)
    return result
        
def all_generator(dataframe_, image_col, random=False):
    dataframe = dataframe_
    if random:
        # TODO: use shuffle(indices)
        dataframe = dataframe_.sample(frac=1)
    for i in range(dataframe.shape[0]):
        yield dataframe[image_col].iloc[i]

def single_generator(dataframe, image_col, group_col, label_col, spec_group, spec_label, random=False):
    yield from all_generator(dataframe[(dataframe[group_col] == spec_group) & (dataframe[label_col] == spec_label)], image_col, random)


def draw_image_over_groups(dataframe, get_image, group_enum, label_enum, group_col, label_col, image_col, random=False):
    global_image = get_image(all_generator(dataframe, image_col, random=random))
    plt.imshow(global_image)

    fig, axes = plt.subplots(nrows=len(label_enum), ncols=len(group_enum), figsize=(20, 20))
    axes = axes.flatten()

    axes_ind = 0
    for label, label_name in enumerate(label_enum):
        for group in group_enum:
            ax = axes[axes_ind]
            ax.set_title("{}-{}".format(label_name, group))
            ax.imshow(get_image(single_generator(dataframe, image_col, group_col, label_col, group, label, random=random)))
            ax.axis('off')
            axes_ind += 1


def find_eigenphoto(img_generator, limit = 100):
    result = []
    all_size = 0
    for img in img_generator:
        array = np.asarray(img.convert("L")).astype(float)
        all_size += 1
        result.append(array.copy())
        if all_size > limit:
            break
    if len(result) == 0:
        return np.asarray(Image.fromarray(np.zeros((224, 224)))).astype(float)

    random.shuffle(result)
    images = np.array(result[:limit])
    images -= np.mean(images, axis = 0)
    n_samples, height, width = images.shape
    images_flat = images.reshape(n_samples, -1)
    pca = PCA(n_components=None, svd_solver="randomized", whiten=False)
    pca.fit(images_flat)
    def percentile_scale(arr):
        p2 = np.percentile(arr, 2)
        p98 = np.percentile(arr, 98)
        scaled = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255)
        return scaled.astype(np.uint8)

    # Eigen-pictures are the principal components
    eigen_pictures = pca.components_.reshape((n_samples, height, width))
    return percentile_scale(eigen_pictures[0]).astype(np.uint8)


# TODO: save necessary names in the dataframe info and pass as one structure
def see_samples(dataframe, labels_enum, group_enum, samples_per_rows, image_col, group_col, label_col,):
    for label, label_name in enumerate(labels_enum):
        for group in group_enum:
            fig, axes = plt.subplots(nrows=1, ncols=max(2, samples_per_rows), figsize=(20, 20))
            axes = axes.flatten()

            axes_ind = 0
            images_for_iteration = single_generator(dataframe, image_col, group_col, label_col, group, label, random = True)
            for num in range(samples_per_rows):
                img = next(images_for_iteration)
                ax = axes[axes_ind]
                ax.set_title("{}-{}".format(label_name, group))
                ax.imshow((img).resize((500, 500)))
                ax.axis('off')
                axes_ind += 1
            plt.show()
            plt.close(fig)


def get_colors(img_generator):
    red_to_blue = []
    blue_to_green = []
    green_to_red = []
    for img in img_generator:
        array = np.asarray(img.convert()).astype(float)
        red = np.mean(array[:, :, 0])
        green = array[:, :, 1].mean()
        blue = array[:, :, 2].mean()
        red_to_blue.append(red / max(1, blue))
        blue_to_green.append(blue / max(1, green))
        green_to_red.append(green / max(1, red))
        
    return red_to_blue, blue_to_green, green_to_red


def draw_color_distrib(red_to_blue, blue_to_green, green_to_red):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def draw_channel(channel, ax, color):
        sns.histplot(channel, bins=50, log_scale=True, color=color, alpha=0.7, edgecolor='black', ax=ax)
        ax.set_title(f'Distribution of relativeness {color} Channel')
        ax.set_xlabel(f'Relativeness of mean {color} Value')
        ax.set_ylabel('Number of Images')
        ax.axvline(1, color='dark'+color, linestyle='--', label='Zero: 10^0')
        ax.legend()
    draw_channel(red_to_blue, axes[0], 'red')
    draw_channel(green_to_red, axes[1], 'green')  
    draw_channel(blue_to_green, axes[2], 'blue')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

# Code below is generated and for EDA purpose (I wanted results quick plus didn't have any saved code for it) only on the weak laptop

class DataFrameImageDataset(Dataset):
    def __init__(self, dataframe, image_path_col='image', label_col='label', transform=None):
        self.dataframe = dataframe
        self.image_path_col = image_path_col
        self.label_col = label_col
        self.transform = transform
                
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx][self.image_path_col]
        label = self.dataframe.iloc[idx][self.label_col]
                
        if self.transform:
            image = self.transform(image)
            
        return image, label
    

def create_cpu_friendly_dataloaders(df, transform, image_path_col='image', label_col='label'):
    # Use smaller dataset for quick iterations
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # TODO: prepare and uncomment it
    # Start with small subset for testing
    if len(train_df) > 1000:
        train_df = train_df.sample(1000, random_state=42)
    if len(val_df) > 200:
        val_df = val_df.sample(200, random_state=42)
    
    train_dataset = DataFrameImageDataset(train_df, image_path_col, label_col, transform)
    val_dataset = DataFrameImageDataset(val_df, image_path_col, label_col, transform)
    
    # Small batch size for CPU
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    
    return train_loader, val_loader

def cpu_friendly_train(device, model, train_loader, val_loader, epochs=5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Gradient accumulation to simulate larger batch size
    accumulation_steps = 4
    batch_count = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss
            loss.backward()
            
            # Statistics
            running_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            batch_count += 1
            
            # Update weights every accumulation_steps
            if batch_count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if batch_idx % 10 == 0:
                acc = 100. * correct / total
                print(f'  Batch {batch_idx}, Loss: {running_loss/(batch_idx+1):.3f}, Acc: {acc:.2f}%')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1} Complete:')
        print(f'  Train Loss: {running_loss/len(train_loader):.3f}, Train Acc: {100.*correct/total:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%')
        print('-' * 50)

def evaluate_model(model, test_loader, device):
    """
    Generate predictions and confusion matrix
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)


def plot_confusion_matrix(targets, predictions, class_names):
    """
    Create a beautiful confusion matrix visualization
    """
    cm = confusion_matrix(targets, predictions)
    
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
    plt.show()
    
    return cm


def visualize_errors(test_dataset, targets, predictions, probabilities, class_names, num_samples=8):
    """
    Display images that the model got wrong, with predictions and confidence
    """
    # Find misclassified examples
    misclassified_indices = np.where(predictions != targets)[0]
    
    print(f"Found {len(misclassified_indices)} misclassified examples")
    
    if len(misclassified_indices) == 0:
        print("No errors found!")
        return
    
    # Select random misclassified examples
    selected_indices = np.random.choice(misclassified_indices, 
                                      size=min(num_samples, len(misclassified_indices)), 
                                      replace=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i, idx in enumerate(selected_indices):
        # Get image, true label, and prediction
        image = test_dataset['image'][idx]
        true_label = test_dataset['label'][idx]
        pred_label = predictions[idx]
        confidence = probabilities[idx][pred_label]
        
        # Plot
        axes[i].imshow(image)
        axes[i].set_title(f'True: {class_names[true_label]}\n'
                         f'Pred: {class_names[pred_label]}\n'
                         f'Conf: {confidence:.3f}', 
                         fontsize=10)
        axes[i].axis('off')
        
        # Make title red for errors
        axes[i].title.set_color('red')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Misclassified Examples', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()