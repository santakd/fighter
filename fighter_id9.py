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
from tqdm import tqdm                                                           # Progress bars 
from datasets import load_dataset                                               # Hugging Face datasets (for alternative dataset loading)


# =============================================================================
# Fighter Aircraft Identifier - PRODUCTION-GRADE (Easier Dataset + Structure Fix)
# Dataset: kadirkrtls/tez-set-v1 (81 real fighter names)
# LR=0.001 + Early Stopping + Robust Structure + Strong Aug + Extra Logging
# =============================================================================
# cleanup: rm -rf aircraft_data best_model.pth fighter_id_*.log
# Download: python3 fighter_id9.py --mode download --source kaggle --dataset-slug kadirkrtls/tez-set-v1
# Train model: python3 fighter_id9.py --mode train --data-dir ./aircraft_data --epochs 60 --batch-size 16 --patience 10
# Inference: python3 fighter_id9.py --mode infer --model-path fighter.pth --infer-image s_1.jpg
# =============================================================================


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
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
    """Basic flatten for obvious nested cases."""
    for _ in range(2):
        dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(dirs) == 1:
            nested = os.path.join(data_dir, dirs[0])
            inner_dirs = [d for d in os.listdir(nested) if os.path.isdir(os.path.join(nested, d))]
            if len(inner_dirs) >= 70:
                logger.info(f"🔧 Flattening nested structure ({len(inner_dirs)} classes)")
                for cls in inner_dirs:
                    src = os.path.join(nested, cls)
                    dst = os.path.join(data_dir, cls)
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.move(src, dst)
                shutil.rmtree(nested)
                logger.info("✅ Structure flattened")
                return
            data_dir = nested
        else:
            break


def find_image_root(data_dir: str, logger: logging.Logger):
    fix_kaggle_structure(data_dir, logger)

    # Priority 1: Look for pre-split folders (Train / Validation / Test)
    for root, dirs, _ in os.walk(data_dir):
        if {"Train", "Validation", "Test"}.issubset(dirs):
            try:
                ds = ImageFolder(os.path.join(root, "Train"))
                if len(ds.classes) == 81:
                    logger.info(f"✅ Using pre-split dataset at {root} (Train/Validation/Test folders)")
                    return root
            except:
                pass

    # Priority 2: Folder with most images + 81 classes
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
            logger.info("Using fallback kaggle import")
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading easier 81-class fighter dataset: {dataset_slug}")
        api.dataset_download_files(dataset_slug, path=data_dir, unzip=True)
        logger.info("✅ Download complete")
        return True
    except Exception as e:
        logger.error(f"Kaggle error: {e}")
        sys.exit(1)


def prepare_hf_dataset(dataset_name: str, data_dir: str, logger: logging.Logger):
    try:
        logger.info(f"Loading Hugging Face dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        for split_name in ["train", "validation", "test"]:
            if split_name not in dataset: continue
            split_ds = dataset[split_name]
            class_names = split_ds.features["label"].names
            for idx, example in enumerate(tqdm(split_ds, desc=f"Saving {split_name}")):
                label_idx = example["label"]
                class_name = class_names[label_idx] if class_names else f"class_{label_idx}"
                class_dir = os.path.join(data_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                img = example["image"].convert("RGB")
                img.save(os.path.join(class_dir, f"{split_name}_{idx:06d}.jpg"))
        logger.info(f"✅ HF dataset ready")
        return True
    except Exception as e:
        logger.error(f"HF failed: {e}")
        return False


def fetch_online_samples(data_dir: str, num_images: int, logger: logging.Logger):
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/F-22_Raptor.jpg/800px-F-22_Raptor.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/F-35_Lightning_II.jpg/800px-F-35_Lightning_II.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/F-16_Fighting_Falcon.jpg/800px-F-16_Fighting_Falcon.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Su-57.jpg/800px-Su-57.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Eurofighter_Typhoon.jpg/800px-Eurofighter_Typhoon.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Rafale.jpg/800px-Rafale.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/F-15_Eagle.jpg/800px-F-15_Eagle.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/MiG-29.jpg/800px-MiG-29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Su-27_Flanker.jpg/800px-Su-27_Flanker.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/F-14_Tomcat.jpg/800px-F-14_Tomcat.jpg",
    ]
    os.makedirs(data_dir, exist_ok=True)
    downloaded = 0
    for i, url in enumerate(urls[:num_images]):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(os.path.join(data_dir, f"fighter_online_{i:02d}.jpg"), "wb") as f:
                f.write(r.content)
            downloaded += 1
        except Exception as e:
            logger.warning(f"Failed {url}: {e}")
    logger.info(f"✅ Downloaded {downloaded}/{num_images} online samples")
    return downloaded > 0


def get_dataloaders(data_dir: str, batch_size: int, logger: logging.Logger):
    root = find_image_root(data_dir, logger)

    # Use pre-split folders if they exist
    train_path = os.path.join(root, "Train")
    val_path   = os.path.join(root, "Validation")
    test_path  = os.path.join(root, "Test")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_dataset = ImageFolder(train_path) 
        val_dataset   = ImageFolder(val_path)
        test_dataset  = ImageFolder(test_path)
        logger.info(f"✅ Using pre-split folders → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    else:
        full_dataset = ImageFolder(root)
        train_size = int(0.7 * len(full_dataset))
        val_size   = int(0.2 * len(full_dataset))
        test_size  = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        logger.info(f"✅ Random split → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Transforms
    train_transform = transforms.Compose([                                              # Stronger augmentations for better generalization
        transforms.Resize((256, 256)),                                                  # Resize to a slightly larger size before cropping to 224x224 for better augmentation effects
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),                            # Randomly crop to 224x224 with a scale range to simulate zooming in and out, which helps the model learn to recognize aircraft at different sizes and distances
        transforms.RandomHorizontalFlip(p=0.5),                                         # Randomly flip images horizontally to help the model learn that aircraft can appear in different orientations, improving robustness to left-right variations
        transforms.RandomRotation(20),                                                  # Randomly rotate images by up to 20 degrees to simulate different angles of view, which is common in real-world scenarios where aircraft may not always be perfectly aligned
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),           # Randomly adjust brightness, contrast, and saturation to help the model learn to recognize aircraft under varying lighting conditions, such as sunny, cloudy, or dusk scenarios
        transforms.ToTensor(),                                                          # Convert PIL image to PyTorch tensor for model input
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # Normalize using ImageNet statistics since we are using a pre-trained ResNet-50 model, which was trained on ImageNet and expects input images to be normalized in this way for optimal performance.
    ])
    val_transform = transforms.Compose([                                                # No augmentation for validation/test, just resizing and normalization
        transforms.Resize((224, 224)),                                                  # Resize to 224x224 for consistent input size, which is required by the ResNet-50 architecture and ensures that validation and test performance metrics reflect real-world accuracy without the influence of random transformations.
        transforms.ToTensor(),                                                          # Convert PIL image to PyTorch tensor for model input   
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # Normalize using ImageNet statistics to ensure that the model receives input in the same format as it was trained on, which is crucial for accurate performance evaluation on the validation and test sets.
    ])
    # Test dataset uses the same transformations as validation to ensure consistent evaluation conditions, allowing us to reliably assess the model's performance on unseen data without the influence of random augmentations that are only applied during training for better generalization.  
    train_ds = TransformedSubset(train_dataset, train_transform)                        # Apply strong augmentations to the training dataset for better generalization, while keeping validation and test datasets clean for accurate evaluation of model performance.
    val_ds   = TransformedSubset(val_dataset,   val_transform)                          # Validation and test datasets use the same transformations (resizing and normalization) without augmentation to ensure that performance metrics reflect real-world accuracy without the influence of random transformations.
    test_ds  = TransformedSubset(test_dataset,  val_transform)                          # This separation of transformations ensures that the model learns robust features from augmented training data while being evaluated on consistent, unaugmented validation and test sets for reliable performance assessment.
    # Compute class weights for the training dataset to handle class imbalance, which is common in real-world datasets where some classes may have significantly more samples than others.  
    # By assigning higher weights to underrepresented classes, we can help the model learn to recognize them better, improving overall performance and ensuring that the model does not become biased towards the majority classes. 
    class_counts = np.bincount([label for _, label in train_dataset])
    weights = 1. / class_counts
    sample_weights = [weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    # Use pin_memory for faster data transfer to GPU if available, which can improve training speed by allowing the DataLoader to allocate page-locked memory that can be transferred to the GPU more efficiently. 
    # This is especially beneficial when training on large datasets or using a GPU with high memory bandwidth, as it reduces the overhead of data transfer and allows for smoother training iterations.
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
    # Load pre-trained ResNet-50 and modify final layer for our number of classes
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    # Compute class weights for CrossEntropyLoss to handle class imbalance
    class_counts = np.bincount([label for _, label in train_loader.dataset.subset])
    class_weights = torch.tensor(1. / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Use Adam optimizer with a cosine annealing learning rate scheduler for better convergence
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    # Early stopping variables
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
        # Calculate training accuracy for the epoch
        train_acc = train_correct / len(train_loader.dataset)
        # Validate on the validation set
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        # Calculate validation accuracy for the epoch
        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step()
        # Log epoch results with current learning rate and best validation accuracy so far
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f} | Best Val so far: {best_val_acc:.4f}")
        # Check for improvement and save model if it's the best so far, otherwise update patience counter
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_loader.dataset.subset.dataset.classes,
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
    # Define the same transformations as used during validation (no augmentation, just resizing and normalization)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Handle both local file paths and URLs for inference. If it's a URL, download the image temporarily for processing.
    if image_path_or_url.startswith("http"):
        r = requests.get(image_path_or_url, stream=True, timeout=10)
        img = Image.open(r.raw).convert("RGB")
        temp = "temp_infer.jpg"
        img.save(temp)
        image_path_or_url = temp
    # Load and preprocess the image, then perform inference to get the predicted class and confidence score
    img = Image.open(image_path_or_url).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    # Perform inference and get predicted class index and confidence score, then log and print the results in a user-friendly format
    with torch.no_grad():
        out = model(tensor)
        pred_idx = out.argmax(1).item()
        conf = torch.softmax(out, 1)[0][pred_idx].item()
        raw = class_names[pred_idx]
    # Replace underscores with spaces for better readability in the display output, while keeping the raw class name intact for logging and debugging purposes
    display = raw.replace("_", " ")
    logger.info(f"Prediction: {raw} ({conf:.2%})")
    print(f"\n🚀 IDENTIFIED AIRCRAFT: {display}")
    print(f"Raw class: {raw}")
    print(f"Confidence: {conf:.2%}")
    
    if os.path.exists("temp_infer.jpg"):
        os.remove("temp_infer.jpg")


def main():
    parser = argparse.ArgumentParser(description="Fighter Aircraft Identifier - Easier Dataset")    # Command-line interface for downloading, training, and inference
    parser.add_argument("--mode", choices=["download", "train", "infer"], required=True)            # Mode of operation: download dataset, train model, or perform inference
    parser.add_argument("--source", choices=["kaggle", "hf", "online"], default="kaggle")           # Source for dataset download: Kaggle, Hugging Face, or online image fetching
    parser.add_argument("--dataset-slug", default="kadirkrtls/tez-set-v1")                          # Kaggle dataset slug for downloading the easier 81-class fighter dataset
    parser.add_argument("--dataset-name", default="Voxel51/FGVC-Aircraft")                          # Hugging Face dataset name for alternative dataset loading (if not using Kaggle)
    parser.add_argument("--data-dir", default="./aircraft_data")                                    # Directory to store dataset and model files
    parser.add_argument("--epochs", type=int, default=60)                                           # Maximum number of training epochs
    parser.add_argument("--batch-size", type=int, default=16)                                       # Batch size for training and validation  
    parser.add_argument("--patience", type=int, default=10)                                         # Patience for early stopping (number of epochs to wait for improvement before stopping)
    parser.add_argument("--infer-image", help="Path or URL")                                        # Path or URL of the image to perform inference on (required for inference mode)
    parser.add_argument("--model-path", default="fighter.pth")                                      # Path to save the best model during training and to load for inference
    parser.add_argument("--num-online", type=int, default=10)                                       # Number of online images to fetch if using the online source (default is 10, but can be adjusted)

    args = parser.parse_args()
    program_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    logger = setup_logging(program_name)
    # Determine the best available device (GPU with CUDA, Apple Silicon with MPS, or CPU) and log the choice for transparency and debugging purposes
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    try:
        os.makedirs(args.data_dir, exist_ok=True) # Ensure the data directory exists before any operations

        if args.mode == "download":
            if args.source == "kaggle":
                download_kaggle_dataset(args.dataset_slug, args.data_dir, logger)
            elif args.source == "hf":
                prepare_hf_dataset(args.dataset_name, args.data_dir, logger)
            elif args.source == "online":
                fetch_online_samples(args.data_dir, args.num_online, logger)

        elif args.mode == "train":
            train_loader, val_loader, test_loader, class_names = get_dataloaders(args.data_dir, args.batch_size, logger)
            train_model(train_loader, val_loader, test_loader, len(class_names),
                        args.epochs, args.patience, device, logger, args.model_path)

        elif args.mode == "infer":
            if not args.infer_image:
                logger.error("--infer-image is required")
                sys.exit(1)
            cp = torch.load(args.model_path, map_location="cpu", weights_only=True)
            infer_image(args.model_path, args.infer_image, cp.get("class_names"), device, logger)
    # Handle keyboard interrupts gracefully and log unexpected exceptions for better debugging and user experience, ensuring that the program exits cleanly in case of errors or interruptions.
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

    logger.info("✅ Program completed successfully")


if __name__ == "__main__":
    main()