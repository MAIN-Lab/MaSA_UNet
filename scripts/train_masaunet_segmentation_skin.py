# train_masaunet_segmentation_skin.py
import os
import sys
import logging
import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.cuda as cuda
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Script started")

# Basic imports
try:
    logger.info("Basic imports successful")
except Exception as e:
    logger.error(f"Basic imports failed: {e}")
    sys.exit(1)

# PyTorch imports
try:
    logger.info("PyTorch imports successful")
except Exception as e:
    logger.error(f"PyTorch imports failed: {e}")
    sys.exit(1)

# Custom imports
try:
    from models import AutoencoderMaSA, MaSAUNetSegmentation
    from utils import preprocess_images, img_augmentation
    logger.info("Custom imports successful")
except Exception as e:
    logger.error(f"Custom imports failed: {e}")
    sys.exit(1)

# Remaining imports
try:
    logger.info("All imports completed")
except Exception as e:
    logger.error(f"Remaining imports failed: {e}")
    sys.exit(1)

# Set CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Paths
root_PH = "../datasets/PH2/PH2Dataset/PH2 Dataset images"
root_I16 = "../datasets/ISIC2016"
root_I17 = "../datasets/ISIC2017"
root_I18 = "../datasets/ISIC2018"
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Hyperparameters
size = 224
batch_size = 2  # Reduced for debugging
num_epochs = 100
learning_rate = 0.0001
patience = 20
num_workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Argument parser
parser = argparse.ArgumentParser(description="Train MaSA-UNet for skin lesion segmentation")
parser.add_argument('--use_masa', action='store_true', default=True, help="Use Manhattan Self-Attention (default: True)")
parser.add_argument('--use_autoencoder', action='store_true', default=True, help="Use pre-trained autoencoder (default: True)")
parser.add_argument('--dataset', type=str, default='PH2', choices=['PH2', 'ISIC2016', 'ISIC2017', 'ISIC2018'], help="Dataset to use (default: PH2)")
args = parser.parse_args()

dataset_name = args.dataset.lower()
use_masa = args.use_masa
use_autoencoder = args.use_autoencoder
logger.info(f"Options - Dataset: {dataset_name}, Use MaSA: {use_masa}, Use Autoencoder: {use_autoencoder}")

# Sequential data loading function
def load_images_sequential(image_paths, mask_paths=None):
    logger.info("Starting sequential image loading")
    images = []
    masks = [] if mask_paths else None

    for i, img_path in enumerate(image_paths):
        try:
            img = np.array(Image.open(img_path).resize((size, size)), dtype=np.float32) / 255.0
            images.append(img)
            if mask_paths:
                mask_path = mask_paths[i]
                mask = np.array(Image.open(mask_path).resize((size, size)), dtype=np.float32)
                if mask.max() > 1.0:
                    mask = mask / 255.0
                mask = mask[:, :, 0] if mask.ndim == 3 else mask
                masks.append(mask)
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            continue

    if not images:
        logger.error("No valid images loaded")
        return np.array([]), np.array([]) if mask_paths else np.array([])

    logger.info("Sequential image loading completed")
    return np.array(images), np.array(masks) if mask_paths else np.array(images)

def prepare_data(dataset_name):
    logger.info(f"Preparing data for {dataset_name}")
    
    if dataset_name == "ph2":
        imgs_path_list = sorted(glob.glob(os.path.join(root_PH, "**", "*Dermoscopic_Image", "*.bmp"), recursive=True))
        masks_path_list = sorted(glob.glob(os.path.join(root_PH, "**", "*lesion", "*.bmp"), recursive=True))
    elif dataset_name == "isic2016":
        imgs_path_list = sorted(glob.glob(os.path.join(root_I16, "ISBI2016_ISIC_Part1_Training_Data", "*.jpg")))
        masks_path_list = sorted(glob.glob(os.path.join(root_I16, "ISBI2016_ISIC_Part1_Training_GroundTruth", "*.png")))
    elif dataset_name == "isic2017":
        imgs_path_list = sorted(glob.glob(os.path.join(root_I17, "ISIC-2017_Training_Data", "*.jpg")))
        masks_path_list = sorted(glob.glob(os.path.join(root_I17, "ISIC-2017_Training_GroundTruth", "*.png")))
    elif dataset_name == "isic2018":
        imgs_path_list = sorted(glob.glob(os.path.join(root_I18, "ISIC2018_Task1-2_Training_Input", "*.jpg")))
        masks_path_list = sorted(glob.glob(os.path.join(root_I18, "ISIC2018_Task1_Training_GroundTruth", "*.png")))

    logger.info(f"Found {len(imgs_path_list)} images and {len(masks_path_list)} masks")
    if not imgs_path_list or not masks_path_list:
        raise FileNotFoundError(f"{dataset_name} dataset files not found. Check directory structure.")

    imgs_arr, masks_arr = load_images_sequential(imgs_path_list, masks_path_list)
    if imgs_arr.size == 0:
        raise ValueError(f"No valid images loaded for {dataset_name} dataset")

    X_train, X_test, y_train, y_test = train_test_split(imgs_arr, masks_arr, test_size=0.25, random_state=101)
    x_rotated, y_rotated, x_flipped, y_flipped = img_augmentation(X_train, y_train)
    X_train_full = np.concatenate([X_train, x_rotated, x_flipped])
    y_train_full = np.concatenate([y_train, y_rotated, y_flipped])
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=101)

    X_train = preprocess_images(X_train)
    X_val = preprocess_images(X_val)
    X_test = preprocess_images(X_test)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    logger.info(f"{dataset_name} - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Custom Dice and Jaccard functions
def dice_coefficient(outputs, targets, threshold=0.5, smooth=1e-6):
    outputs = (outputs.sigmoid() > threshold).float()
    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum()
    return (2. * intersection + smooth) / (union + smooth)

def jaccard_index(outputs, targets, threshold=0.5, smooth=1e-6):
    outputs = (outputs.sigmoid() > threshold).float()
    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Weighted Composed Loss
class WeightedComposedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(WeightedComposedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        dice_loss = 1 - dice_coefficient(outputs, targets)
        jaccard_loss = 1 - jaccard_index(outputs, targets)
        return self.alpha * bce_loss + self.beta * dice_loss + self.gamma * jaccard_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, filepath):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for noisy_inputs, targets in train_pbar:
            noisy_inputs, targets = noisy_inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for noisy_inputs, targets in val_pbar:
                noisy_inputs, targets = noisy_inputs.to(device), targets.to(device)
                outputs = model(noisy_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        val_loss /= len(val_loader)

        lr = scheduler(epoch, optimizer.param_groups[0]['lr'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")
        logger.info(f"GPU memory allocated: {cuda.memory_allocated(0) / 1024**2:.2f} MB")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filepath)
            logger.info(f"Saved best model with Val Loss: {val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    return model

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    dice_total, jaccard_total = 0, 0
    predictions, inputs_list, targets_list = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            dice_total += dice_coefficient(outputs, targets).item()
            jaccard_total += jaccard_index(outputs, targets).item()
            predictions.append(outputs.cpu())
            inputs_list.append(inputs.cpu())
            targets_list.append(targets.cpu())

    test_loss /= len(test_loader)
    dice = dice_total / len(test_loader)
    jaccard = jaccard_total / len(test_loader)

    predictions = torch.cat(predictions, dim=0)
    inputs = torch.cat(inputs_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    return test_loss, dice, jaccard, predictions, inputs, targets

def save_comparison_plots(inputs, predictions, targets, output_dir, num_samples=5):
    predictions = predictions.sigmoid()
    for i in range(min(num_samples, len(inputs))):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(inputs[i].permute(1, 2, 0).numpy())
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        axes[1].imshow(predictions[i][0].numpy(), cmap='gray')
        axes[1].set_title("Predicted Mask")
        axes[1].axis('off')
        axes[2].imshow(targets[i][0].numpy(), cmap='gray')
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis('off')
        plt.savefig(os.path.join(output_dir, f"seg_comparison_{i}.png"))
        plt.close()

def main():
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"Initial GPU memory allocated: {cuda.memory_allocated(0) / 1024**2:.2f} MB")

    logger.info("Loading data...")
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(dataset_name)
        logger.info(f"Data loaded - X_train: {X_train.shape}, y_train: {y_train.shape}")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    autoencoder = None
    if use_autoencoder:
        logger.info("Loading autoencoder...")
        try:
            autoencoder = AutoencoderMaSA(size, size, 3, "autoencoder_masa", use_masa=True).to(device)
            autoencoder_path = "models/skin_autoencoder.pt"
            if os.path.exists(autoencoder_path):
                autoencoder.load_state_dict(torch.load(autoencoder_path))
                autoencoder.eval()
                logger.info(f"Loaded autoencoder. GPU memory: {cuda.memory_allocated(0) / 1024**2:.2f} MB")
            else:
                logger.warning(f"Autoencoder file {autoencoder_path} not found.")
                autoencoder = None
        except Exception as e:
            logger.error(f"Failed to load autoencoder: {e}")
            autoencoder = None

    logger.info("Initializing MaSAUNetSegmentation...")
    try:
        model = MaSAUNetSegmentation(
            height=size, width=size, in_channels=3, out_channels=1, 
            autoencoder=autoencoder, use_masa=use_masa, gamma=0.9
        ).to(device)
        logger.info(f"Model initialized. GPU memory: {cuda.memory_allocated(0) / 1024**2:.2f} MB")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        sys.exit(1)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = WeightedComposedLoss(alpha=0.5, beta=0.3, gamma=0.2).to(device)

    def scheduler(epoch, lr):
        return lr if epoch < 10 else lr * np.exp(-0.1)

    filepath = f'models/{dataset_name}_masaunet_segmentation.pt'
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, filepath)

    test_loss, dice, jaccard, predictions, inputs, targets = evaluate_model(model, test_loader, criterion)

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Dice: {dice:.4f}")
    logger.info(f"Test Jaccard: {jaccard:.4f}")

    with open(f'results/{dataset_name}_masaunet_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Dice: {dice:.4f}\n")
        f.write(f"Test Jaccard: {jaccard:.4f}\n")
    logger.info(f"Metrics saved to results/{dataset_name}_masaunet_metrics.txt")

    save_comparison_plots(inputs, predictions, targets, "outputs")
    logger.info("Comparison plots saved in outputs/ folder")

if __name__ == "__main__":
    main()
