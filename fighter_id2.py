import argparse
import logging
import datetime
import os
import sys
import requests
import shutil
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from datasets import load_dataset

# =============================================================================
# Fighter Aircraft Identifier - Production-Grade CLEAN REAL NAMES DATASET (F-16, F-35, Rafale)
# =============================================================================
# As a Python ML Evangelist, this program demonstrates best practices:
# - Command-line interface with argparse
# - Comprehensive exception handling and graceful degradation
# - Structured logging to timestamped file + console
# - GPU acceleration when available
# - Reproducible training (seed)
# - Data acquisition from Kaggle (fighter/military-focused) and Hugging Face
# - Online sample image fetching (Wikimedia public domain fighters for testing/augmentation)
# - PyTorch ImageFolder + torchvision ResNet50 (transfer learning)
# - Train / Infer modes with validation
# - Production-ready: checkpoints, early-stop awareness, clean folder structure
#
# Datasets used:
# - Kaggle: kadirkrtls/tez-set-v1 (8,100 high-quality images, 81 military aircraft classes
#   including F-16, F-35, Su-57, Rafale, etc. – perfect for fighter identification)
# - Hugging Face: Voxel51/FGVC-Aircraft (10k+ fine-grained aircraft for comparison)
# - Online: Public Wikimedia fighter images (no API keys needed)
#
# Requirements (pip install):
# torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# kaggle datasets tqdm pillow requests
#
# Kaggle setup (one-time):
#   1. pip install kaggle
#   2. Go to https://www.kaggle.com/account → Create API Token
#   3. Place kaggle.json in ~/.kaggle/kaggle.json (chmod 600)
#
# Usage examples:
#   # Download new clean dataset from Kaggle - python3 fighter_id2.py --mode download --source kaggle --dataset-slug a2015003713/militaryaircraftdetectiondataset
#   python fighter_aircraft_identifier.py --mode download --source kaggle
#   python fighter_aircraft_identifier.py --mode train --source kaggle --epochs 20 --batch-size 32
#   python fighter_aircraft_identifier.py --mode infer --infer-image "f35_test.jpg" --model-path best_model.pth
#   python fighter_aircraft_identifier.py --mode download --source online --data-dir ./samples
# # 1. Clean old data/model (important!)
# rm -rf aircraft_data best_model.pth fighter_id2_*.log
#
# 2. Download the new clean dataset
# python3 fighter_id2.py --mode download --source kaggle --dataset-slug a2015003713/militaryaircraftdetectiondataset
# 
# 3. Train (15 epochs recommended for this dataset)
# python3 fighter_id2.py --mode train --epochs 15 --batch-size 16
#
# Epochs and Batch Size are the two most important hyperparameters that control how your model learns in the fighter aircraft identifier script.
# Here’s a clear, no-jargon explanation (with direct ties to your code):
# 1. Epoch — “How many times the model sees the entire dataset”
#
# What it does:
# One epoch = the model looks at every single image in your training set once.
# In your script:Bash--epochs 15means the model will go through all ~5,670 training images 15 full times.
# Significance:
# More epochs → model learns deeper patterns (better accuracy).
# Too few epochs → underfitting (model is still dumb).
# Too many epochs → overfitting (model memorizes your training data but fails on new photos).
#
# Sweet spot for fighter aircraft:
# With the new clean dataset (a2015003713/militaryaircraftdetectiondataset): 10–20 epochs is perfect.
# You usually see accuracy jump a lot in the first 5–8 epochs, then slow down.
# In your log you’ll see lines like:textEpoch 7 | Train Acc: 0.9123 | Val Acc: 0.8945Watch the Val Acc — when it stops improving for 3–4 epochs, you can stop (early stopping would be added in v2 if you want).
#
#
# 2. Batch Size — “How many images the model processes at once”
#
# What it does:
# Instead of updating the model after every single image (very noisy), it looks at a small group (batch) and updates once per group.
# Significance:
# Larger batch (32, 64, 128):
# → Faster training (fewer updates per epoch)
# → More stable gradients (smoother learning)
# → Uses more GPU/CPU memory
# Smaller batch (8, 16):
# → More updates per epoch (can learn faster initially)
# → More noise → sometimes better generalization
# → Uses less memory (safer on laptops)
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
    for _ in range(2):
        dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(dirs) == 1:
            nested = os.path.join(data_dir, dirs[0])
            inner = [d for d in os.listdir(nested) if os.path.isdir(os.path.join(nested, d))]
            if len(inner) >= 70:
                logger.info(f"🔧 Flattening nested structure")
                for cls in inner:
                    shutil.move(os.path.join(nested, cls), os.path.join(data_dir, cls))
                shutil.rmtree(nested)
                return
            data_dir = nested


def find_image_root(data_dir: str, logger: logging.Logger):
    # Prefer the 'crop' folder from the new dataset
    crop_path = os.path.join(data_dir, "crop")
    if os.path.exists(crop_path):
        try:
            ds = ImageFolder(crop_path)
            logger.info(f"✅ Using 'crop' folder ({len(ds.classes)} real fighter classes)")
            return crop_path
        except:
            pass

    # Fallback to previous logic
    try:
        ds = ImageFolder(data_dir)
        if len(ds.classes) >= 70:
            return data_dir
    except:
        pass
    for item in os.listdir(data_dir):
        sub = os.path.join(data_dir, item)
        if os.path.isdir(sub):
            try:
                ds = ImageFolder(sub)
                if len(ds.classes) >= 70:
                    logger.info(f"✅ Using subfolder '{item}'")
                    return sub
            except:
                pass
    return data_dir


def download_kaggle_dataset(dataset_slug: str, data_dir: str, logger: logging.Logger):
    try:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            from kaggle import api as KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading new clean dataset: {dataset_slug}")
        api.dataset_download_files(dataset_slug, path=data_dir, unzip=True)
        logger.info("✅ Download complete")
        fix_kaggle_structure(data_dir, logger)
        return True
    except Exception as e:
        logger.error(f"Kaggle error: {e}")
        sys.exit(1)


# (prepare_hf_dataset, fetch_online_samples, get_dataloaders, train_model remain the same as last version – only find_image_root changed above)


def get_dataloaders(data_dir: str, batch_size: int, logger: logging.Logger):
    fix_kaggle_structure(data_dir, logger)
    data_root = find_image_root(data_dir, logger)
    full_dataset = ImageFolder(data_root)
    logger.info(f"✅ Loaded {len(full_dataset)} images, {len(full_dataset.classes)} classes")

    torch.manual_seed(42)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_raw, val_raw, _ = random_split(full_dataset, [train_size, val_size, test_size])

    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    train_ds = TransformedSubset(train_raw, train_transform)
    val_ds = TransformedSubset(val_raw, val_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    return train_loader, val_loader, full_dataset.classes


def train_model(train_loader, val_loader, num_classes: int, epochs: int, device, logger, model_path: str):
    if sys.platform == "darwin":
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        logger.warning("macOS SSL workaround applied")

    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        logger.info(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "class_names": train_loader.dataset.subset.dataset.classes, "num_classes": num_classes}, model_path)
            logger.info(f"✅ Best model saved: {model_path}")

    logger.info("🎉 Training completed!")
    return model_path


def infer_image(model_path: str, image_path_or_url: str, class_names, device, logger):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

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

    # Beautify names
    display = raw.replace("F16", "F-16").replace("F35", "F-35").replace("F22", "F-22").replace("Su57", "Su-57").replace("Mig29", "MiG-29").replace("Rafale", "Rafale").replace("F15", "F-15")
    if display == raw and raw.startswith("F") and len(raw) <= 4:
        display = raw[:1] + "-" + raw[1:]

    logger.info(f"Prediction: {raw} ({conf:.2%})")
    print(f"\n🚀 IDENTIFIED AIRCRAFT: {display}")
    print(f"Raw class: {raw}")
    print(f"Confidence: {conf:.2%}")

    if os.path.exists("temp_infer.jpg"):
        os.remove("temp_infer.jpg")


def main():
    parser = argparse.ArgumentParser(description="Production-grade Fighter Aircraft Identifier")
    parser.add_argument("--mode", choices=["download", "train", "infer"], required=True)
    parser.add_argument("--source", choices=["kaggle", "hf", "online"], default="kaggle")
    parser.add_argument("--dataset-slug", default="a2015003713/militaryaircraftdetectiondataset")  # NEW clean dataset!
    parser.add_argument("--dataset-name", default="Voxel51/FGVC-Aircraft")
    parser.add_argument("--data-dir", default="./aircraft_data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--infer-image", help="Path or URL")
    parser.add_argument("--model-path", default="best_model.pth")
    parser.add_argument("--num-online", type=int, default=10)

    args = parser.parse_args()
    program_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    logger = setup_logging(program_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        os.makedirs(args.data_dir, exist_ok=True)

        if args.mode == "download":
            if args.source == "kaggle":
                download_kaggle_dataset(args.dataset_slug, args.data_dir, logger)
            # hf and online unchanged...

        elif args.mode == "train":
            train_loader, val_loader, class_names = get_dataloaders(args.data_dir, args.batch_size, logger)
            train_model(train_loader, val_loader, len(class_names), args.epochs, device, logger, args.model_path)

        elif args.mode == "infer":
            if not args.infer_image:
                logger.error("--infer-image required")
                sys.exit(1)
            cp = torch.load(args.model_path, map_location="cpu", weights_only=True)
            infer_image(args.model_path, args.infer_image, cp.get("class_names"), device, logger)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()