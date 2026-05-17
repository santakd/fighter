# fighter ✈️

Repo for fighter plane image recognition training and inference using ResNet50 CNN

![Tejas Fighter](https://github.com/santakd/fighter/blob/main/t_4.jpg)

    
    
### ResNet50 – A Landmark in Deep Learning

ResNet50 is one of the most influential convolutional neural network architectures ever created. 
Introduced in the 2015 paper **“Deep Residual Learning for Image Recognition”** by He et al. (Microsoft Research). 
It won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) that year and became a foundational model in computer vision.
ResNet-50 is a 50-layer convolutional neural network (CNN) widely used for image recognition, featuring residual learning (skip connections) to mitigate vanishing gradient problems in deep networks.
It is a cornerstone model in computer vision, achieving ~80.8% top-1 accuracy on ImageNet-1K and commonly used for transfer learning.

![ResNet50_Arch](https://github.com/santakd/fighter/blob/main/resnet50_arch.png)


**Key innovations**:
- **Residual (skip) connections** — The core idea that allows training very deep networks (50, 101, 152 layers) without vanishing gradients. Instead of learning H(x) directly, the network learns the residual F(x) = H(x) − x, and the output is H(x) = F(x) + x.
- **Bottleneck design** — Uses 1×1 convolutions to reduce and restore dimensions, making deeper layers computationally efficient.
- Depth: 50 layers (hence “ResNet50”), with ~23 million parameters.
- Pre-trained on ImageNet (1.28 million images, 1,000 classes), it serves as an extremely strong feature extractor or backbone for transfer learning.

**Why it’s still widely used in 2026**:
- Excellent balance of accuracy, speed, and memory footprint.
- Mature, well-tested implementations in every major framework.
- Strong transfer learning performance on fine-grained tasks (birds, cars, aircraft, etc.), even though newer models (EfficientNet, ConvNeXt, ViT variants) often beat it slightly on raw accuracy.

ResNet50 remains the **“safe, reliable default”** for many production image classification and detection pipelines.

### The Fighter Classifier Program

The **fighter_id(x).py** programs are iteratively built **production-grade fine-grained image classifier** trained to recognize 81 different fighter aircraft types from the `kadirkrtls/tez-set-v1` Kaggle dataset (~8,100 images).

Refer the programs [fighter_id10.py](https://github.com/santakd/fighter/blob/main/fighter_id10.py) and [fighter_id11.py](https://github.com/santakd/fighter/blob/main/fighter_id11.py) 

**Core features & design choices**:
- **Backbone**: ResNet50 pre-trained on ImageNet (transfer learning) with the final fully-connected layer replaced to output 81 classes.
- **Data handling**: Automatically detects and uses the pre-split Train/Validation/Test folders (or falls back to random split). Fixes nested Kaggle folder structure.
- **Augmentation**: Strong training-time augmentation (RandomResizedCrop, RandAugment-style, ColorJitter, flips, rotations) to improve generalization on varied aircraft photos.
- **Loss & balancing**: Weighted CrossEntropyLoss to handle class imbalance + label smoothing option.
- **Optimizer & scheduling**: Adam with initial LR=0.001 + CosineAnnealingLR + optional ReduceLROnPlateau for fine-grained control.
- **Training tricks**: WeightedRandomSampler, early stopping (patience), best-model checkpointing, final test-set evaluation.
- **Inference**: Clean, human-readable output with fighter names beautified (e.g., “F-16 Fighting Falcon” instead of raw folder name).
- **Production qualities**: Timestamped logging, full exception handling, KeyboardInterrupt safety, argparse CLI, MPS/CPU/CUDA device support.

**Current performance** (from the latest run):
- Best validation accuracy: **46.3%** (epoch 66)
- Final test accuracy: **44.4%** (719/1620 correct)
- This is already respectable for an 81-class fine-grained task with real-world photos and no extra tricks (e.g., no ensemble, no TTA, no advanced loss like ArcFace).

**Why it’s valuable**:
This is not just a toy project — it’s a realistic pipeline for real-world fine-grained recognition (military, aviation, wildlife, product identification). The code is modular, reproducible, and easy to extend (e.g., swap backbone to EfficientNet-B4, add Mixup, enable TTA, etc.).

ResNet50 is built around the revolutionary concept of residual connections (also called skip connections or identity shortcuts), introduced in the 2015 paper Deep Residual Learning for Image Recognition by He et al. These connections are the key reason why very deep networks (50, 101, 152 layers) suddenly became trainable and even outperformed shallower networks.
The Core Problem ResNet Solved
As networks get deeper, two major issues appear:

Vanishing / exploding gradients — gradients become extremely small or large during backpropagation, making optimization nearly impossible.
Degradation problem — adding more layers sometimes makes accuracy worse on both training and test sets, even when the deeper network has enough capacity.

Surprisingly, a deeper plain network (just stacking conv layers) performs worse than a shallower one — even after careful initialization and regularization.
The Residual Learning Idea
Instead of forcing the network to learn the desired underlying mapping H(x) directly, ResNet lets it learn the residual:
```F(x) = H(x) − x
Then the actual output becomes:
H(x) = F(x) + x
This is implemented with a shortcut connection that adds the input x directly to the output of a few stacked layers (the residual block).
``` 
**Why Residual Connections Work So Well**

1. Gradient highway
During backpropagation, the gradient can flow directly through the identity path (∂L/∂x = ∂L/∂H(x) · 1 + other terms), so it never vanishes completely — even after hundreds of layers.
2. Easier optimization
The network can always learn the identity mapping (F(x) ≈ 0) if adding more layers doesn’t help. This eliminates the degradation problem — deeper networks can at least match shallower ones.
3. Feature reuse
Later layers can easily access early features via shortcuts, improving information flow and representation power.

**Summary – Intuitive Analogy**
Think of a very deep residual network like a multi-story building with emergency staircases (the shortcuts):

If one floor has a problem (bad weights), people (gradients) can still use the staircase to reach the ground safely.
You can always walk straight down the stairs (identity path) if the floors aren’t helping.
Adding more floors never makes it impossible to escape — it can only help or do no harm.

That’s why ResNet50 (and deeper ResNets) suddenly made 100+ layer networks not only possible, but state-of-the-art in 2015 — and still very strong baselines today.

---
