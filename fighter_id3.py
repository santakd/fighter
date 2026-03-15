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
# Fighter Aircraft Identifier - REAL NAMES + EARLY STOPPING + LR SCHEDULER
# =============================================================================
# New Features
# Automatic Early Stopping: Training stops automatically if validation accuracy doesn’t improve for N epochs (default = 5).
# ReduceLROnPlateau Scheduler: Learning rate automatically drops (by 50%) when validation accuracy plateaus.
# New CLI argument: --patience 5 (you can change it).
# --epochs is now the maximum epochs (safety net).
# Beautiful logging of LR changes and early stopping reason.
# Best model is always saved.
#
# 1. Clean old stuff (recommended)
# rm -rf aircraft_data best_model.pth fighter_id_*.log
#
# 2. Download clean dataset + train with smart stopping
# python3 fighter_id3.py --mode download --source kaggle --dataset-slug a2015003713/militaryaircraftdetectiondataset
#
# python3 fighter_id3.py --mode train --epochs 30 --batch-size 16 --patience 5
# Training will now stop automatically (usually around 12–18 epochs) instead of running all 30.
# Download clean real-name dataset
# python3 fighter_id.py --mode download --source kaggle
#
# Train with smart stopping
# python3 fighter_id.py --mode train --epochs 30 --batch-size 16 --patience 5
#
# Test on a photo
# python3 fighter_id.py --mode infer --infer-image "f35_test.jpg"
# or from URL:
# python3 fighter_id.py --mode infer --infer-image "https://upload.wikimedia.org/.../F-35_Lightning_II.jpg"

# =============================================================================

'''
1. Core / Always Required
ParameterType / ChoicesDefaultSignificance & When to Change--modedownload, train, infer (required)— (must specify)The only required argument. 
Tells the program what to do:
• download → get the dataset
• train → train the model
• infer → predict on one image
Always use this first.

2. Download Mode Parameters
ParameterType / ChoicesDefaultSignificance & When to Change--sourcekaggle, hf, onlinekaggleChooses where data comes from.
• kaggle = clean fighter dataset with real names (recommended)
• hf = Hugging Face FGVC-Aircraft
• online = 10 public Wikimedia sample images
Change only if you want to test a different source.--dataset-slugstringa2015003713/militaryaircraftdetectiondataset
Kaggle dataset identifier. Controls which dataset is downloaded.
Leave as-is for real fighter names (F-35, Rafale, etc.). 
Change only if you want a different Kaggle dataset.--dataset-namestringVoxel51/FGVC-Aircraft
Hugging Face dataset name (only used when --source hf).
Ignore unless switching to HF mode.--data-dirstring./aircraft_dataFolder where images are saved/extracted.
Change if you want data stored elsewhere (e.g. /data/fighters).--num-onlineinteger10Number of sample fighter images to download when --source online.
Useful for quick testing. Increase to 20 if you want more test images.

3. Training Mode Parameters (used with --mode train)
ParameterType / ChoicesDefaultSignificance & When to Change--epochsinteger30Maximum number of epochs the model will run.
With early stopping + scheduler, training usually stops much earlier (12-18 epochs). Set higher (50+) only if you have a very large dataset or 
want to force longer training.--batch-sizeinteger16Number of images processed at once.
• Higher = faster training but uses more memory
• Lower = slower but more stable on laptops
On your Mac (MPS), 16 is the sweet spot. Try 32 if you have 32GB+ RAM.--patienceinteger5Early stopping patience (new feature).
Training stops if validation accuracy doesn't improve for this many epochs.
• 3-5 = aggressive (good for quick experiments)
• 7-10 = conservative (lets model train longer)
Default 5 is perfect for this dataset.

4. Inference Mode Parameters (used with --mode infer)
ParameterType / ChoicesDefaultSignificance & When to Change--infer-imagestring (path or URL)— (required for infer)The image you want to classify.
Can be local file (test.jpg) or web URL (https://...).
Must be provided when using --mode infer.--model-pathstringbest_model.pthPath to the trained model file.
Change only if you renamed the model or want to use a different checkpoint.

5. Other Global Parameters
ParameterType / ChoicesDefaultSignificance & When to Change--help or -h——Shows this full help message.
Use anytime: python3 fighter_id3.py --help
arly stopping + ReduceLROnPlateau (added in this version) means you can safely set --epochs 30 or even 50 — 
the program will stop automatically when it stops improving.
--batch-size 16 + MPS on your Mac gives the best speed/memory balance.
'''

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
    crop_path = os.path.join(data_dir, "crop")
    if os.path.exists(crop_path):
        try:
            ds = ImageFolder(crop_path)
            logger.info(f"✅ Using 'crop' folder ({len(ds.classes)} real fighter classes)")
            return crop_path
        except:
            pass
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
        logger.info(f"Downloading clean fighter dataset: {dataset_slug}")
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
    logger.info(f"✅ {downloaded} online samples ready")
    return downloaded > 0


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


def train_model(train_loader, val_loader, num_classes: int, epochs: int, patience: int, device, logger, model_path: str):
    if sys.platform == "darwin":
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        logger.warning("macOS SSL workaround applied")

    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    logger.info(f"Starting training (max {epochs} epochs, early stopping patience = {patience})")

    for epoch in range(epochs):
        # Train
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

        # Validate
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)

        # Scheduler & Early Stopping
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_loader.dataset.subset.dataset.classes,
                "num_classes": num_classes
            }, model_path)
            logger.info(f"✅ New best model saved (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"⏳ No improvement for {patience_counter}/{patience} epochs")

        if patience_counter >= patience:
            logger.info(f"🎯 Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
            break

    logger.info(f"🎉 Training finished! Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
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

    # Beautify fighter names
    display = raw.replace("F16", "F-16").replace("F35", "F-35").replace("F22", "F-22").replace("Su57", "Su-57").replace("Mig29", "MiG-29").replace("F15", "F-15")
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
    parser.add_argument("--dataset-slug", default="a2015003713/militaryaircraftdetectiondataset")
    parser.add_argument("--dataset-name", default="Voxel51/FGVC-Aircraft")
    parser.add_argument("--data-dir", default="./aircraft_data")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
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
            elif args.source == "hf":
                prepare_hf_dataset(args.dataset_name, args.data_dir, logger)
            elif args.source == "online":
                fetch_online_samples(args.data_dir, args.num_online, logger)

        elif args.mode == "train":
            train_loader, val_loader, class_names = get_dataloaders(args.data_dir, args.batch_size, logger)
            train_model(train_loader, val_loader, len(class_names), args.epochs, args.patience, device, logger, args.model_path)

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