import argparse                                                                 # Command-line argument parsing
import logging                                                                  # Logging setup
import datetime                                                                 # Timestamp for logs         
import os                                                                       # File system operations
import sys                                                                      # System-specific parameters and functions
import requests                                                                 # HTTP requests for online image fetching
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
# Fighter Aircraft Identifier - PRODUCTION-GRADE (Level 1 + Early Stopping)
# Strong Aug + Weighted Loss + Cosine Scheduler + Automatic Early Stopping
# Real fighter names (F-35, Rafale, F-22, etc.)
# Full exception handling + timestamped logging
# =============================================================================
# Usage:
# 1) Download dataset:
#    python fighter_id6.py --mode download --source kaggle
# 2) Train model:
#    python fighter_id6.py --mode train --data-dir ./aircraft_data --epochs 30 --patience 5
# 3) Inference:
#    python fighter_id6.py --mode infer --infer-image path_or_url_to_image.jpg
# =============================================================================
# Train Acc: This is the training accuracy for that epoch.
# It tells you: “Out of all the training images the model saw this epoch,
# what percentage did it classify correctly?”
#
# Val Acc: This is the validation accuracy.
# It tells you: “How well is the model doing on images it has never seen before (the validation set)?”
# This is the most important number to watch.
#
# LR: This is the current Learning Rate.
# It controls how big a step the model takes when updating its weights after each batch.
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
    """Production-grade logging with timestamped file + console."""
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


def find_image_root(data_dir: str, logger: logging.Logger):
    """Auto-detect the clean 'crop' folder."""
    crop_path = os.path.join(data_dir, "crop")
    if os.path.exists(crop_path):
        try:
            ds = ImageFolder(crop_path)
            logger.info(f"✅ Using 'crop' folder ({len(ds.classes)} real fighter classes)")
            return crop_path
        except Exception as e:
            logger.warning(f"crop folder check failed: {e}")
    try:
        ds = ImageFolder(data_dir)
        if len(ds.classes) >= 70:
            return data_dir
    except:
        pass
    logger.warning("⚠️ Using root folder as fallback")
    return data_dir


def download_kaggle_dataset(dataset_slug: str, data_dir: str, logger: logging.Logger):
    """Robust Kaggle download with full exception handling."""
    try:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            from kaggle import api as KaggleApi
            logger.info("Using fallback kaggle import")
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading clean fighter dataset: {dataset_slug}")
        api.dataset_download_files(dataset_slug, path=data_dir, unzip=True)
        logger.info("✅ Download and extraction complete")
        return True
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        logger.error("Place kaggle.json in ~/.kaggle/ with chmod 600")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        sys.exit(1)


def prepare_hf_dataset(dataset_name: str, data_dir: str, logger: logging.Logger):
    try:
        logger.info(f"Loading Hugging Face dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        for split_name in ["train", "validation", "test"]:
            if split_name not in dataset:
                continue
            split_ds = dataset[split_name]
            class_names = split_ds.features["label"].names
            for idx, example in enumerate(tqdm(split_ds, desc=f"Saving {split_name}")):
                label_idx = example["label"]
                class_name = class_names[label_idx] if class_names else f"class_{label_idx}"
                class_dir = os.path.join(data_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                img = example["image"].convert("RGB")
                img.save(os.path.join(class_dir, f"{split_name}_{idx:06d}.jpg"))
        logger.info(f"✅ HF dataset saved as ImageFolder")
        return True
    except Exception as e:
        logger.error(f"Hugging Face preparation failed: {e}")
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
            logger.warning(f"Failed to download {url}: {e}")
    logger.info(f"✅ Downloaded {downloaded}/{num_images} online samples")
    return downloaded > 0


def get_dataloaders(data_dir: str, batch_size: int, logger: logging.Logger):
    data_root = find_image_root(data_dir, logger)
    full_dataset = ImageFolder(data_root)
    logger.info(f"✅ Loaded {len(full_dataset)} images, {len(full_dataset.classes)} classes")

    torch.manual_seed(42)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_raw, val_raw, test_raw = random_split(full_dataset, [train_size, val_size, test_size])

    # Strong augmentation pipeline for training, simple normalization for val/test
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # For validation and testing, we only want to resize and normalize (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = TransformedSubset(train_raw, train_transform)
    val_ds   = TransformedSubset(val_raw,   val_transform)
    test_ds  = TransformedSubset(test_raw,  val_transform)

    # Weighted sampler for class imbalance (heavily penalizes misclassifying underrepresented classes)
    class_counts = np.bincount([label for _, label in train_raw])
    weights = 1. / class_counts
    sample_weights = [weights[label] for _, label in train_raw]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    pin_memory = torch.cuda.is_available() # Pin memory for faster GPU transfers if available
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, full_dataset.classes


def train_model(train_loader, val_loader, test_loader, num_classes: int, max_epochs: int, patience: int, device, logger, model_path: str):
    if sys.platform == "darwin":
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        logger.warning("macOS SSL workaround applied for ResNet weights")

    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Weighted loss to handle class imbalance (heavily penalizes misclassifying underrepresented classes)
    class_counts = np.bincount([label for _, label in train_loader.dataset.subset])
    class_weights = torch.tensor(1. / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    # Early stopping variables
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    logger.info(f"Starting training (max {max_epochs} epochs, early stopping patience = {patience})")

    for epoch in range(max_epochs):
        # Train phase (with progress bar)
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

        train_acc = train_correct / len(train_loader.dataset)

        # Validation phase (with progress bar)
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step()

        logger.info(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping logic - save best model and reset patience counter if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_loader.dataset.subset.dataset.classes,
                "num_classes": num_classes
            }, model_path)
            logger.info(f"✅ New best model saved at epoch {best_epoch} (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"⏳ No improvement for {patience_counter}/{patience} epochs")

        if patience_counter >= patience:
            logger.info(f"🎯 Early stopping triggered after {epoch+1} epochs (patience = {patience})")
            break

    # FINAL TEST EVALUATION (always run on best model)      
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
    # Inference transformation (resize + normalize, no augmentation)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    # Beautify fighter names 
    display = raw.replace("F16", "F-16").replace("F35", "F-35").replace("F22", "F-22")\
                 .replace("Su57", "Su-57").replace("Mig29", "MiG-29").replace("F15", "F-15")
    if display == raw and raw.startswith("F") and len(raw) <= 4:
        display = raw[:1] + "-" + raw[1:]

    logger.info(f"Prediction: {raw} ({conf:.2%})")
    print(f"\n🚀 IDENTIFIED AIRCRAFT: {display}")
    print(f"Raw class: {raw}")
    print(f"Confidence: {conf:.2%}")

    if os.path.exists("temp_infer.jpg"):
        os.remove("temp_infer.jpg")


def main():
    parser = argparse.ArgumentParser(description="Fighter Aircraft Identifier - Production Grade with Early Stopping")
    parser.add_argument("--mode", choices=["download", "train", "infer"], required=True)
    parser.add_argument("--source", choices=["kaggle", "hf", "online"], default="kaggle")
    parser.add_argument("--dataset-slug", default="a2015003713/militaryaircraftdetectiondataset")
    parser.add_argument("--dataset-name", default="Voxel51/FGVC-Aircraft")
    parser.add_argument("--data-dir", default="./aircraft_data")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs (early stopping will likely stop earlier)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without Val Acc improvement)")
    parser.add_argument("--infer-image", help="Path or URL to image for inference")
    parser.add_argument("--model-path", default="best_model.pth")
    parser.add_argument("--num-online", type=int, default=10)

    args = parser.parse_args()

    program_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    logger = setup_logging(program_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    try:
        os.makedirs(args.data_dir, exist_ok=True)

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
                logger.error("--infer-image is required for infer mode")
                sys.exit(1)
            cp = torch.load(args.model_path, map_location="cpu", weights_only=True)
            infer_image(args.model_path, args.infer_image, cp.get("class_names"), device, logger)

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

    logger.info("✅ Program completed successfully")


if __name__ == "__main__":
    main()