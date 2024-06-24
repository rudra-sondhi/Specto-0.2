
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import numpy as np
import cv2
import os
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

YAML_PATH = '/Users/rudrasondhi/Desktop/Specto-0.2/Specto-0.2/Data/IR_and_MS_Plots/IR_Found/data_in_folder.yaml'
DATA_DIR = '/Users/rudrasondhi/Desktop/Specto-0.2/Specto-0.2/Data/Special_IR'

def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-label classification model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads for self-attention')
    #parser.add_argument('--data_dir', type=str, required=True, default= '/Users/rudrasondhi/Desktop/Specto-0.2/Specto-0.2/Data/Special IR' , help='Directory for dataset')
    #parser.add_argument('--yaml_path', type=str, required=True, default=  '/Users/rudrasondhi/Desktop/Specto-0.2/Specto-0.2/Data/IR and MS Plots/IR Found/data_in_folder.yaml', help='Path to the YAML file with labels')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_1-normal-ish', help='Directory to save checkpoints')
    return parser.parse_args()

def load_data(yaml_path, base_dir, all_labels, label_to_index):
    with open(yaml_path, 'r') as file:
        functional_groups_data = yaml.safe_load(file)
    return create_multilabels(functional_groups_data, base_dir, all_labels, label_to_index)

def create_multilabels(cas_data, base_dir, all_labels, label_to_index):
    multilabels = []
    image_paths = []

    for cas_id, data in cas_data.items():
        func_groups = data['Functional Groups']
        label_vector = [0] * len(all_labels)

        for group in func_groups:
            if group in label_to_index:
                label_vector[label_to_index[group]] = 1

        image_path = os.path.join(base_dir, f"{cas_id}.png")
        image_paths.append(image_path)
        multilabels.append(label_vector)

    return image_paths, np.array(multilabels)

def remove_classes_with_no_functional_groups(image_paths, multilabels, all_labels, top_n=8):
    label_sums = np.sum(multilabels, axis=0)
    top_indices = np.argsort(label_sums)[-top_n:]
    new_all_labels = [all_labels[idx] for idx in top_indices]
    new_multilabels = multilabels[:, top_indices]
    return image_paths, new_multilabels, new_all_labels

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale format

        if image is None:
            raise FileNotFoundError(f"Image not found or unable to read: {image_path}")

        # Resize while maintaining aspect ratio
        target_size = 352  # New target size
        h, w = image.shape
        if h > w:
            new_h = target_size
            new_w = int(w * (new_h / h))
        else:
            new_w = target_size
            new_h = int(h * (new_w / w))

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Pad the image to make it target_size x target_size
        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        #padded_image = padded_image.astype(np.float32) / 255.0

        # Convert to a 1-channel image for consistency with potential transformations
        padded_image = np.expand_dims(padded_image, axis=2)

        if self.transform:
            transformed = self.transform(image=padded_image)  # Transform for model input
            padded_image = transformed['image']

        label = self.labels[idx]
        return padded_image, label  # Return only the required values for training


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.features(x)
        return x

class SelfAttentionFC(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionFC, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        attn_scores = self.attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        attended_features = x * attn_weights
        return attended_features

class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        x = x.view(batch_size, C, -1)
        x = x.permute(0, 2, 1)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(0, 2, 1)
        attn_output = attn_output.reshape(batch_size, C, width, height)
        out = self.gamma * attn_output + x.reshape(batch_size, C, width, height)
        return out

class CustomModel(nn.Module):
    def __init__(self, num_heads=8, num_labels=8):
        super(CustomModel, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor()
        self.cnn = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(1024, num_heads),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(512, num_heads),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(256, num_heads),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            SelfAttention(128, num_heads),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.attention_fc1 = SelfAttentionFC(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.attention_fc2 = SelfAttentionFC(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.attention_fc3 = SelfAttentionFC(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 128)
        self.attention_fc4 = SelfAttentionFC(128)
        self.dropout4 = nn.Dropout(0.5)
        self.output = nn.Linear(128, num_labels)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cnn(x)
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = self.attention_fc1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.attention_fc2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.attention_fc3(x)
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.attention_fc4(x)
        x = self.dropout4(x)
        
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x



def weighted_bce_loss(outputs, targets, weights):
    return F.binary_cross_entropy(outputs, targets, weight=weights, reduction='mean')

def calculate_metrics(outputs, targets, threshold=0.5):
    outputs = outputs.detach().cpu().numpy()
    predicted = (outputs > threshold).astype(int)
    targets = targets.detach().cpu().numpy()

    for i in range(predicted.shape[0]):
        if np.sum(predicted[i]) == 0:
            predicted[i, np.argmax(outputs[i])] = 1

    precision = precision_score(targets, predicted, average='samples')
    recall = recall_score(targets, predicted, average='samples')
    f1 = f1_score(targets, predicted, average='samples')
    accuracy = accuracy_score(targets, predicted)

    return accuracy, precision, recall, f1

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)

    if 'config' in state:
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(state['config'], file)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def main():
    args = parse_args()

    all_labels = ['alkene', 'alkyne', 'alcohols', 'amines', 'nitriles', 'aromatics', 'alkyl halides',
                  'esters', 'ketones', 'aldehydes', 'carboxylic acids', 'ether', 'acyl halides',
                  'amides', 'nitro', 'imine', 'enol', 'hydrazone', 'enamine', 'phenol']

    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    image_paths, multilabels = load_data(YAML_PATH, DATA_DIR, all_labels, label_to_index)
    image_paths, multilabels, all_labels = remove_classes_with_no_functional_groups(image_paths, multilabels, all_labels)
    
    non_empty_indices = np.where(multilabels.sum(axis=1) != 0)[0]
    image_paths = [image_paths[i] for i in non_empty_indices]
    multilabels = multilabels[non_empty_indices]

    assert np.all(multilabels.sum(axis=1) != 0), "There are still samples with no true labels."

    x_train, x_temp, y_train, y_temp = train_test_split(image_paths, multilabels, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    train_transform = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=25, p=0.5),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=(0.8,), std=(0.2,)),  # Adjust mean and std according to your dataset
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.8,), std=(0.2,)),  # Adjust mean and std according to your dataset
        ToTensorV2()
    ])

    train_dataset = CustomDataset(x_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(x_val, y_val, transform=val_transform)
    test_dataset = CustomDataset(x_test, y_test, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CustomModel(num_heads=args.num_heads, num_labels=len(all_labels)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    class_counts = np.sum(y_train, axis=0)
    total_counts = np.sum(class_counts)
    class_weights = total_counts / (len(all_labels) * class_counts)

    # Normalize the weights to have a mean of 1
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    epochs = args.epochs
    best_val_loss = float('inf')
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_trajectory = []
    val_loss_trajectory = []
    accuracy_trajectory = []
    precision_trajectory = []
    recall_trajectory = []
    f1_trajectory = []
    val_accuracy_trajectory = []

    start_epoch = 0
    resume_from_checkpoint = True
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        loss_trajectory = checkpoint['loss_trajectory']
        val_loss_trajectory = checkpoint['val_loss_trajectory']
        accuracy_trajectory = checkpoint['accuracy_trajectory']
        precision_trajectory = checkpoint['precision_trajectory']
        recall_trajectory = checkpoint['recall_trajectory']
        f1_trajectory = checkpoint['f1_trajectory']
        val_accuracy_trajectory = checkpoint['val_accuracy_trajectory']
        print(f"Resumed from checkpoint: epoch {start_epoch}, best_val_loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        current_loss = []
        current_accuracy = []
        current_precision = []
        current_recall = []
        current_f1 = []
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for inputs, targets in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = weighted_bce_loss(outputs, targets.float(), class_weights)
            loss.backward()
            optimizer.step()

            accuracy, precision, recall, f1 = calculate_metrics(outputs, targets)

            current_loss.append(loss.item())
            current_accuracy.append(accuracy)
            current_precision.append(precision)
            current_recall.append(recall)
            current_f1.append(f1)

            iterator.set_postfix(loss=torch.tensor(current_loss).mean().item(), accuracy=torch.tensor(current_accuracy).mean().item(), precision=torch.tensor(current_precision).mean().item(), recall=torch.tensor(current_recall).mean().item(), f1=torch.tensor(current_f1).mean().item())

        loss_trajectory.append(torch.tensor(current_loss).mean().item())
        accuracy_trajectory.append(torch.tensor(current_accuracy).mean().item())
        precision_trajectory.append(torch.tensor(current_precision).mean().item())
        recall_trajectory.append(torch.tensor(current_recall).mean().item())
        f1_trajectory.append(torch.tensor(current_f1).mean().item())

        model.eval()
        val_loss = 0.0
        val_accuracy = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = weighted_bce_loss(outputs, targets.float(), class_weights)
                val_loss += loss.item()

                accuracy, _, _, _ = calculate_metrics(outputs, targets)
                val_accuracy.append(accuracy)

        val_loss /= len(val_loader)
        val_accuracy = np.mean(val_accuracy)
        val_loss_trajectory.append(val_loss)
        val_accuracy_trajectory.append(val_accuracy)
        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'loss_trajectory': loss_trajectory,
            'val_loss_trajectory': val_loss_trajectory,
            'accuracy_trajectory': accuracy_trajectory,
            'precision_trajectory': precision_trajectory,
            'recall_trajectory': recall_trajectory,
            'f1_trajectory': f1_trajectory,
            'val_accuracy_trajectory': val_accuracy_trajectory
        }, is_best, checkpoint_dir)
        print(f"Saved checkpoint: epoch {epoch+1}, validation loss {val_loss:.4f}, validation accuracy {val_accuracy:.4f}")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(loss_trajectory, label='Training Loss')
    plt.plot(val_loss_trajectory, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Trajectory')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(accuracy_trajectory, label='Training Accuracy')
    plt.plot(val_accuracy_trajectory, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trajectory')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(precision_trajectory, label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision Trajectory')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(f1_trajectory, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Trajectory')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

