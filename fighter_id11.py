import argparse                                                                 # Command-line argument parsing
import logging                                                                  # Logging setup
import datetime                                                                 # Timestamp for logs         
import os                                                                       # File system operations
import sys                                                                      # System-specific parameters and functions
import requests                                                                 # HTTP requests for online image fetching
import shutil                                                                   # File operations (for structure fixing)
import numpy as np                                                              # Numerical operations (for class weights)  
from PIL import Image                                                           # Image processing
import torch                                                                    # PyTorch core library
import torch.nn as nn                                                           # Neural network modules
import torch.optim as optim                                                     # Optimization algorithms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler    # Data loading and splitting
from torchvision import transforms, models                                      # Data transformations and pre-trained models   
from torchvision.datasets import ImageFolder                                    # Dataset class for loading images from folders
from torchvision.transforms import RandAugment, AutoAugment, AutoAugmentPolicy  # Advanced data augmentation techniques 
from tqdm import tqdm                                                           # Progress bars 
from datasets import load_dataset                                               # Hugging Face datasets (for alternative dataset loading)


# =============================================================================
# Fighter Aircraft Identifier - PRODUCTION-GRADE (Easier Dataset + Structure Fix)
# Dataset: kadirkrtls/tez-set-v1 (81 real fighter names)
# LR=0.001 + Early Stopping + Robust Structure + Strong Aug + Extra Logging
# =============================================================================
# Download: python3 fighter_id11.py --mode download --source kaggle --dataset-slug kadirkrtls/tez-set-v1
# Train model: python3 fighter_id11.py --mode train --data-dir ./aircraft_data --epochs 75 --batch-size 16 --patience 10
# Inference: python3 fighter_id11.py --mode infer --model-path fighter.pth --infer-image s_1.jpg
# =============================================================================


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform: image = self.transform(image)
        return image, label


def setup_logging(program_name: str) -> logging.Logger:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{program_name}_{timestamp}.log"
    logger = logging.getLogger(program_name)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_h = logging.FileHandler(log_filename, encoding="utf-8")
    file_h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)
    logger.addHandler(file_h)
    logger.info(f"Logging initialized → {log_filename}")
    return logger


def fix_kaggle_structure(data_dir: str, logger: logging.Logger):
    for _ in range(2):
        dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(dirs) == 1:
            nested = os.path.join(data_dir, dirs[0])
            inner_dirs = [d for d in os.listdir(nested) if os.path.isdir(os.path.join(nested, d))]
            if len(inner_dirs) >= 70:
                logger.info(f"🔧 Fixing nested structure ({len(inner_dirs)} classes)")
                for cls in inner_dirs:
                    src = os.path.join(nested, cls)
                    dst = os.path.join(data_dir, cls)
                    if os.path.exists(dst): shutil.rmtree(dst)
                    shutil.move(src, dst)
                shutil.rmtree(nested)
                logger.info("✅ Structure flattened")
                return
            data_dir = nested
        else:
            break


def find_image_root(data_dir: str, logger: logging.Logger):
    fix_kaggle_structure(data_dir, logger)

    # Priority: pre-split folders
    for root, dirs, _ in os.walk(data_dir):
        if {"Train", "Validation", "Test"}.issubset(dirs):
            try:
                ds = ImageFolder(os.path.join(root, "Train"))
                if len(ds.classes) == 81:
                    logger.info(f"✅ Using pre-split dataset at {root}")
                    return root
            except:
                pass

    # Priority: folder with most images + 81 classes
    best_root = None
    best_count = 0
    for root, dirs, _ in os.walk(data_dir):
        if len(dirs) >= 70:
            try:
                ds = ImageFolder(root)
                if len(ds.classes) == 81:
                    count = len(ds)
                    if count > best_count:
                        best_count = count
                        best_root = root
            except:
                pass
    if best_root:
        logger.info(f"✅ Selected folder with {best_count} images: {best_root}")
        return best_root

    logger.warning("⚠️ Using root as fallback")
    return data_dir


def download_kaggle_dataset(dataset_slug: str, data_dir: str, logger: logging.Logger):
    try:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            from kaggle import api as KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading 81-class fighter dataset: {dataset_slug}")
        api.dataset_download_files(dataset_slug, path=data_dir, unzip=True)
        logger.info("✅ Download complete")
        return True
    except Exception as e:
        logger.error(f"Kaggle error: {e}")
        sys.exit(1)


def get_dataloaders(data_dir: str, batch_size: int, logger: logging.Logger):
    root = find_image_root(data_dir, logger)
    train_path = os.path.join(root, "Train")
    val_path   = os.path.join(root, "Validation")
    test_path  = os.path.join(root, "Test")

    # Check for pre-split folders first, then fallback to random split if not found. This allows us to use the provided train/val/test splits if they exist, which can lead to more consistent and reliable performance evaluation, 
    # while still providing a fallback option to create our own splits if the dataset is not already organized in that way.
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_dataset = ImageFolder(train_path)
        val_dataset   = ImageFolder(val_path)
        test_dataset  = ImageFolder(test_path)
        logger.info(f"✅ Using pre-split folders → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
        class_names = train_dataset.classes
    else:
        full_dataset = ImageFolder(root)
        train_size = int(0.7 * len(full_dataset))
        val_size   = int(0.2 * len(full_dataset))
        test_size  = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        logger.info(f"✅ Random split → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
        class_names = full_dataset.classes

    train_transform = transforms.Compose([                                              # Stronger augmentations for better generalization
        transforms.Resize((256, 256)),                                                  # Resize to a slightly larger size before cropping to 224x224 for better augmentation effects 
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),                           # Randomly crop to 224x224 with a scale range to simulate zooming in and out, which helps the model learn to recognize aircraft at different sizes and distances
        RandAugment(num_ops=2, magnitude=9),                                            # Apply RandAugment with 2 random transformations per image and a magnitude of 9 for strong augmentation effects, which can help the model learn more robust features
        transforms.RandomHorizontalFlip(p=0.5),                                         # Randomly flip images horizontally to help the model learn that aircraft can appear in different orientations, improving robustness to left-right variations
        transforms.RandomRotation(25),                                                  # Randomly rotate images by up to 25 degrees to simulate different angles of view, which is common in real-world scenarios where aircraft may not always be perfectly aligned
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),           # Randomly adjust brightness, contrast, and saturation to help the model learn to recognize aircraft under varying lighting conditions, such as sunny, cloudy, or dusk scenarios
        transforms.ToTensor(),                                                          # Convert PIL image to PyTorch tensor for model input 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # Normalize using ImageNet statistics since we are using a pre-trained ResNet-50 model, which was trained on ImageNet and expects input images to be normalized in this way for optimal performance
    ])
  
    val_transform = transforms.Compose([                                                # No augmentation for validation/test, just resizing and normalization
        transforms.Resize((224, 224)),                                                  # Resize to 224x224 for consistent input size, which is required by the ResNet-50 architecture and ensures that validation and test performance metrics reflect real-world accuracy without the influence of random transformations
        transforms.ToTensor(),                                                          # Convert PIL image to PyTorch tensor for model input   
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # Normalize using ImageNet statistics to ensure that the model receives input in the same format as it was trained on, which is crucial for accurate performance evaluation on the validation and test sets
    ])

    # Test dataset uses the same transformations as validation to ensure consistent evaluation conditions
    train_ds = TransformedSubset(train_dataset, train_transform)                        # Apply strong augmentations to the training dataset for better generalization, while keeping validation and test datasets clean for accurate evaluation of model performance
    val_ds   = TransformedSubset(val_dataset,   val_transform)                          # Validation and test datasets use the same transformations (resizing and normalization) without augmentation to ensure that performance metrics reflect real-world accuracy without the influence of random transformations
    test_ds  = TransformedSubset(test_dataset,  val_transform)                          # This separation of transformations ensures that the model learns robust features from augmented training data while being evaluated on consistent, unaugmented validation and test sets for reliable performance assessment
    
    # Compute class weights for the training dataset to handle class imbalance, which is common in real-world datasets where some classes may have significantly more samples than others 
    # By assigning higher weights to underrepresented classes, we can help the model learn to recognize them better, improving overall performance and ensuring that the model does not become biased towards the majority classes
    class_counts = np.bincount([label for _, label in train_dataset])
    weights = 1. / class_counts
    sample_weights = [weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Use pin_memory for faster data transfer to GPU if available, which can improve training speed by allowing the DataLoader to allocate page-locked memory that can be transferred to the GPU more efficiently 
    # This is especially beneficial when training on large datasets or using a GPU with high memory bandwidth, as it reduces the overhead of data transfer and allows for smoother training iterations
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, train_dataset.classes if hasattr(train_dataset, 'classes') else val_dataset.classes


def train_model(train_loader, val_loader, test_loader, num_classes: int, max_epochs: int, patience: int, device, logger, model_path: str):
    if sys.platform == "darwin":
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        logger.warning("macOS SSL workaround applied")

    # Load pre-trained ResNet-50 and modify the final layer for our number of classes (81). Using a pre-trained model allows us to leverage learned features from ImageNet,
    # which can improve performance and reduce training time, especially when we have a limited dataset. The final fully connected layer is replaced to match the number of classes in our specific task
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Compute class weights for the training dataset to handle class imbalance, which is common in real-world datasets where some classes may have significantly more samples than others
    # By assigning higher weights to underrepresented classes, we can help the model learn to recognize them better and improve overall performance, especially on minority classes. This is crucial for achieving good accuracy across all classes
    class_counts = np.bincount([label for _, label in train_loader.dataset.subset])
    class_weights = torch.tensor(1. / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights) - # this can be used for label smoothing, which can help prevent the model from becoming overconfident in its predictions and improve generalization 
    # Use Adam optimizer with a cosine annealing learning rate scheduler, which can help the model converge faster and potentially achieve better performance by adjusting the learning rate dynamically during training
    # The cosine annealing schedule allows for a gradual reduction in learning rate, which can help the model fine-tune its weights as it approaches convergence. Add ReduceLROnPlateau after cosine scheduler to reduce
    # LR if validation accuracy plateaus, which can help the model escape local minima and continue improving performance when the validation accuracy stops improving for a certain number of epochs
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, threshold=1e-4, min_lr=1e-6)

    # Early stopping variables to track the best validation accuracy and implement patience. This helps prevent overfitting by stopping training 
    # when the model's performance on the validation set stops improving for a certain number of epochs, allowing us to save time and computational resources while still achieving good generalization
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    logger.info(f"🚀 STARTING TRAINING | LR=0.001 | Patience={patience} | Max Epochs={max_epochs} | Device={device}")

    for epoch in range(max_epochs):
        model.train()
        train_correct = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # Calculate training accuracy for the epoch and log it. This provides insight into how well the model is learning from the training data, allowing us to monitor progress and identify potential issues such as overfitting or underfitting
        train_acc = train_correct / len(train_loader.dataset)

        # Evaluate on the validation set to calculate validation accuracy. This is crucial for monitoring the model's performance on unseen data during training, allowing us to implement early stopping and save the best model
        model.eval()

        # Calculate validation accuracy for the epoch and log it. This helps us track how well the model is generalizing to unseen data during training, and is essential for implementing early stopping based on validation performance
        val_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # Calculate validation accuracy for the epoch and log it. This helps us track how well the model is generalizing to unseen data during training, and is essential for implementing early stopping based on validation performance
        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step()
        plateau_scheduler.step(val_acc)
        
        # Log epoch results including training accuracy, validation accuracy, current learning rate, and best validation accuracy so far. 
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f} | Best Val so far: {best_val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_loader.dataset.subset.classes if hasattr(train_loader.dataset.subset, 'classes') else train_loader.dataset.subset.dataset.classes,
                "num_classes": num_classes
            }, model_path)
            logger.info(f"✅ NEW BEST MODEL SAVED at epoch {best_epoch} (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"⏳ No improvement for {patience_counter}/{patience} epochs")

        if patience_counter >= patience:
            logger.info(f"🎯 Early stopping triggered after {epoch+1} epochs")
            break

    # Final Test
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Test Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
    test_acc = test_correct / len(test_loader.dataset)
    logger.info(f"🎯 FINAL TEST ACCURACY: {test_acc:.4f} ({test_correct}/{len(test_loader.dataset)})")

    logger.info(f"🎉 Training completed! Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    return model_path


def infer_image(model_path: str, image_path_or_url: str, class_names, device, logger):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    # Use the same transformations as validation for inference to ensure that the model receives input in the same format as it was trained on, which is crucial for accurate predictions. 
    # This includes resizing to 224x224 and normalizing using ImageNet statistics, which are expected by the pre-trained ResNet-50 model for optimal performance during inference
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if image_path_or_url.startswith("http"):
        r = requests.get(image_path_or_url, stream=True, timeout=10)
        img = Image.open(r.raw).convert("RGB")
        temp = "temp_infer.jpg"
        img.save(temp)
        image_path_or_url = temp

    img = Image.open(image_path_or_url).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        pred_idx = out.argmax(1).item()
        conf = torch.softmax(out, 1)[0][pred_idx].item()
        raw = class_names[pred_idx]

    display = raw.replace("_", " ")
    logger.info(f"Prediction: {raw} ({conf:.2%})")
    print(f"\n🚀 IDENTIFIED AIRCRAFT: {display}")
    print(f"Raw class: {raw}")
    print(f"Confidence: {conf:.2%}")

    if os.path.exists("temp_infer.jpg"):
        os.remove("temp_infer.jpg")


def main():
    # Command-line arguments for flexible operation modes (download, train, infer) and dataset sources (Kaggle, Hugging Face, online). 
    parser = argparse.ArgumentParser(description="Fighter Aircraft Identifier - Pre-split Fixed")
    # Mode selection: download dataset, train model, or perform inference on an image. Source selection for dataset downloading (Kaggle, Hugging Face, or online). Dataset slug for Kaggle. 
    # Data directory for storing datasets and models. Training parameters (epochs, batch size, patience). Inference parameters (image path/URL and model path).
    parser.add_argument("--mode", choices=["download", "train", "infer"], required=True)
    # Dataset source for downloading: Kaggle (default), Hugging Face, or online image fetching. This allows users to choose their preferred method of obtaining the dataset, 
    # whether it's from a Kaggle competition, a Hugging Face repository, or directly from an online URL.
    parser.add_argument("--source", choices=["kaggle", "hf", "online"], default="kaggle")
    # Dataset slug for Kaggle (default: "kadirkrtls/tez-set-v1"). This specifies which dataset to download from Kaggle when the download mode is selected, 
    # allowing users to easily switch between different datasets if needed.
    parser.add_argument("--dataset-slug", default="kadirkrtls/tez-set-v1")
    # Data directory for storing datasets and models (default: "./aircraft_data"). This provides a centralized location for all data-related files, 
    # making it easier to manage and organize the dataset and trained models.
    parser.add_argument("--data-dir", default="./aircraft_data")
    # Training parameters: number of epochs (default: 60), batch size (default: 16), and patience for early stopping (default: 10). 
    # These parameters allow users to customize the training process based on their computational resources and desired training duration, while also implementing early stopping to prevent overfitting.
    parser.add_argument("--epochs", type=int, default=60)
    # Batch size for training (default: 16). A smaller batch size can help with generalization and reduce memory usage, while a larger batch size can speed up training but may require more memory. 
    # The default of 16 is a good balance for many datasets and hardware configurations.
    parser.add_argument("--batch-size", type=int, default=16)
    # Patience for early stopping (default: 10). This means that if the validation accuracy does not improve for 10 consecutive epochs, training will be stopped to prevent overfitting and save computational resources.
    # This is especially useful when training on smaller datasets or when the model starts to converge, allowing us to achieve good generalization without unnecessary training.
    parser.add_argument("--patience", type=int, default=10)
    # Inference parameters: path or URL of the image to be inferred (required for inference mode) and path to the trained model (default: "fighter.pth"). 
    # This allows users to easily perform inference on new images using a specified trained model, making the script versatile for both training and deployment scenarios.
    parser.add_argument("--infer-image", help="Path or URL")
    # Path to the trained model for inference (default: "fighter.pth"). This allows users to specify which trained model to use for inference, 
    # enabling them to easily switch between different models if they have trained multiple versions or want to use a specific checkpoint for prediction.
    parser.add_argument("--model-path", default="fighter.pth")
    # Number of online images to fetch for inference when using the "online" source option (default: 10). This allows users to specify how many images they want to fetch from an online source for inference, 
    # providing flexibility in testing the model's performance on real-world data without needing to download a full dataset.
    parser.add_argument("--num-online", type=int, default=10)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set up logging with a timestamped log file and console output which is crucial for monitoring the workflow and troubleshooting any issues that may arise.
    program_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    logger = setup_logging(program_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        os.makedirs(args.data_dir, exist_ok=True)

        if args.mode == "download":
            if args.source == "kaggle":
                # (download function from previous version)
                pass

        elif args.mode == "train":
            train_loader, val_loader, test_loader, class_names = get_dataloaders(args.data_dir, args.batch_size, logger)
            train_model(train_loader, val_loader, test_loader, len(class_names), args.epochs, args.patience, device, logger, args.model_path)

        elif args.mode == "infer":
            if not args.infer_image:
                logger.error("--infer-image is required")
                sys.exit(1)
            cp = torch.load(args.model_path, map_location="cpu", weights_only=True)
            infer_image(args.model_path, args.infer_image, cp.get("class_names"), device, logger)

    # Handle keyboard interrupts and unexpected exceptions gracefully, logging the error and exiting cleanly. 
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
  
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

    logger.info("✅ Program completed successfully")


if __name__ == "__main__":
    main()