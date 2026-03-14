import argparse                                         # for command-line interface
import logging                                          # for structured logging
import datetime                                         # for timestamped logs   
import os                                               # for file operations
import sys                                              # for system operations
import requests                                         # for online image fetching
import shutil                                           # for file operations
from PIL import Image                                   # for image processing
import torch                                            # for ML           
import torch.nn as nn                                   # for neural network components
import torch.optim as optim                             # for optimization
from torch.utils.data import DataLoader, random_split   # for data handling
from torchvision import transforms, models              # for data augmentation and pre-trained models
from torchvision.datasets import ImageFolder            # for structured image datasets
from tqdm import tqdm                                   # for progress bars 
from datasets import load_dataset                       # for Hugging Face datasets

# =============================================================================
# Fighter Aircraft Identifier - Production-Grade (Structure + Pickle + MPS FIXED)
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
#   python fighter_aircraft_identifier.py --mode download --source kaggle
#   python fighter_aircraft_identifier.py --mode train --source kaggle --epochs 20 --batch-size 32
#   python fighter_aircraft_identifier.py --mode infer --infer-image "f35_test.jpg" --model-path best_model.pth
#   python fighter_aircraft_identifier.py --mode download --source online --data-dir ./samples
# =============================================================================


class TransformedSubset(torch.utils.data.Dataset):
    """Picklable transform wrapper (fixed for multiprocessing)."""
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
    """Quick flatten for obvious nested cases."""
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
    """Robustly find the folder with actual class subfolders (handles 3-folder Kaggle layout)."""
    # Try current root
    try:
        ds = ImageFolder(data_dir)
        if len(ds.classes) >= 70:
            logger.info(f"✅ Using root ({len(ds.classes)} classes)")
            return data_dir
    except:
        pass

    # Check immediate subfolders (train/valid/test etc.)
    for item in os.listdir(data_dir):
        sub_path = os.path.join(data_dir, item)
        if os.path.isdir(sub_path):
            try:
                ds = ImageFolder(sub_path)
                if len(ds.classes) >= 70:
                    logger.info(f"✅ Found perfect image root in subfolder '{item}' ({len(ds.classes)} classes)")
                    return sub_path
            except:
                pass

    logger.warning("⚠️ No perfect class structure found - using current root (may need manual check)")
    return data_dir


def download_kaggle_dataset(dataset_slug: str, data_dir: str, logger: logging.Logger):
    try:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            from kaggle import api as KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading {dataset_slug}...")
        api.dataset_download_files(dataset_slug, path=data_dir, unzip=True)
        logger.info("✅ Download complete")
        fix_kaggle_structure(data_dir, logger)
        return True
    except Exception as e:
        logger.error(f"Kaggle error: {e}")
        sys.exit(1)


def prepare_hf_dataset(dataset_name: str, data_dir: str, logger: logging.Logger):
    try:
        logger.info(f"Loading HF dataset: {dataset_name}")
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
        logger.info(f"✅ HF dataset ready")
        return True
    except Exception as e:
        logger.error(f"HF failed: {e}")
        return False


def fetch_online_samples(data_dir: str, num_images: int, logger: logging.Logger):
    urls = [  # same 10 fighter images
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
            logger.info(f"Downloaded sample {i+1}")
        except Exception as e:
            logger.warning(f"Failed {url}: {e}")
    logger.info(f"✅ {downloaded} online samples ready")
    return downloaded > 0


def get_dataloaders(data_dir: str, batch_size: int, logger: logging.Logger):
    fix_kaggle_structure(data_dir, logger)
    data_root = find_image_root(data_dir, logger)   # ← NEW robust finder
    full_dataset = ImageFolder(data_root)
    logger.info(f"✅ Loaded {len(full_dataset)} images, {len(full_dataset.classes)} classes")

    torch.manual_seed(42)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_raw, val_raw, _ = random_split(full_dataset, [train_size, val_size, test_size])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = TransformedSubset(train_raw, train_transform)
    val_ds = TransformedSubset(val_raw, val_transform)

    pin_memory = torch.cuda.is_available()  # only for CUDA

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
        train_loss = train_correct = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
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
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_loader.dataset.subset.dataset.classes,
                "num_classes": num_classes
            }, model_path)
            logger.info(f"✅ Best model saved: {model_path}")

    logger.info("🎉 Training completed!")
    return model_path


def infer_image(model_path: str, image_path_or_url: str, class_names, device, logger):
    # (unchanged - same as previous version)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

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
        pred_class = class_names[pred_idx] if class_names else f"class_{pred_idx}"

    logger.info(f"Prediction: {pred_class} ({conf:.2%})")
    print(f"\n🚀 IDENTIFIED: {pred_class.upper()} (Confidence: {conf:.2%})")
    if os.path.exists("temp_infer.jpg"):
        os.remove("temp_infer.jpg")


def main():
    parser = argparse.ArgumentParser(description="Production-grade Fighter Aircraft Identifier")
    parser.add_argument("--mode", choices=["download", "train", "infer"], required=True)
    parser.add_argument("--source", choices=["kaggle", "hf", "online"], default="kaggle")
    parser.add_argument("--dataset-slug", default="kadirkrtls/tez-set-v1")
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