# 🔬 KAN-NAFNet: A Hybrid Framework for Image Super-Resolution

This project integrates two powerful components:
- **KAN (Kernel Activation Network):** A dynamic multi-activation neural module with adaptive activation function selection.
- **NAFNet:** A lightweight convolutional architecture for image enhancement and denoising.

The integration enables patch-wise image super-resolution, guided by learned activation performance and perceptual loss optimization.

---

## 🚀 Features

✅ **Patch-based super-resolution** with learned KAN upscaling  
✅ **Dynamic activation selection** from over 30 activation functions  
✅ **Fusion with NAFNet** for feature refinement and detail restoration  
✅ **Custom dataset support** with automatic HR→LR conversion  
✅ **Support for advanced loss functions:** Charbonnier & Perceptual Loss  
✅ **Activation performance tracking** for adaptive training strategies  

---

## 🧠 Architecture Overview

| Module | Purpose |
|--------|---------|
| `ActivationModule` | Provides dozens of activation functions, including Swish, Mish, SQNL, ISRLU, etc. |
| `KANLinear` | Linear layer with multiple parallel activation paths and learned weights |
| `KANNetwork` | A stack of `KANLinear` layers forming a flexible MLP |
| `KANeural` | Processes image patches and upsamples them using KAN logic |
| `NAFNet` | Lightweight image restoration network with residual and gating mechanisms |
| `KANNAFNetIntegrated` | Full hybrid model combining multiple KAN blocks + NAFNet |
| `SRDataset` | Custom dataset that dynamically creates LR images from HR originals |
| `CharbonnierLoss` | Smooth L1 loss for robustness to outliers |
| `PerceptualLoss` | Uses VGG19 features to optimize visual quality |
| `DynamicActivationRegistry` | Tracks activation performance and recommends top activations |

---

## 📦 Dependencies

```bash
pip install torch torchvision numpy pillow matplotlib tqdm h5py pyyaml
```

---

## 💡 Example Usage

```python
from model import KANNAFNetIntegrated
import torch

model = KANNAFNetIntegrated(in_channels=3)
input_tensor = torch.randn(1, 3, 64, 64)
output = model(input_tensor)
print(output.shape)  # Should output high-resolution image tensor
```

---

## 🧪 Dataset Preparation

Your high-resolution image folder should contain `.png`, `.jpg`, or `.bmp` files:

```python
train_loader, val_loader = create_dataloaders(
    hr_dir='path/to/HR', 
    batch_size=8, 
    patch_size=128,
    lr_scale=4
)
```

Each HR image will be downscaled using bicubic interpolation to simulate LR inputs.

---

## 🎯 Training Objective

You can combine multiple loss functions:

```python
loss1 = CharbonnierLoss()(output, target)
loss2 = PerceptualLoss()(output, target)
total_loss = loss1 + 0.1 * loss2
```

---

## 📈 Activation Tracking

Use the `DynamicActivationRegistry` to keep track of the most successful activation functions and dynamically select the top-performing ones during training.

```python
registry = DynamicActivationRegistry()
registry.update_metrics("swish", 0.03)
top_acts = registry.get_top_activations(n=5)
```

---

## 🧱 Applications

- Super-resolution of natural images (x2, x4 scaling)
- Denoising and artifact reduction
- Activation function analysis and dynamic architecture tuning

---

## 📌 Future Improvements

- Add support for attention-based fusion between KAN and NAFNet
- Learnable activation function replacement schedules
- Benchmarking on standard datasets like DIV2K and Set5

---


### 🔁 Super-Resolution Workflow Tree (KAN + NAFNet)

Super-Resolution Pipeline
│
├── 📥 Load High-Resolution (HR) Images
│ └── from user-specified folder (e.g., ./data/HR_Images)
│
├── 🔄 Preprocessing
│ ├── Apply random flip, rotation (for data augmentation)
│ └── Extract HR patches and downscale to get LR patches
│
├── 🧠 Model Architecture
│ ├── 🔗 KAN Blocks
│ │ └── Approximate HR pixel functions using adaptive activation networks
│ ├── 🔧 NAFNet Blocks
│ │ └── Denoise and resolve pixel congestion in upsampled images
│ └── 🔀 Combined Output
│ └── Clean, high-quality SR image
│
├── 🎯 Loss Functions
│ ├── Charbonnier Loss (pixel-wise robust loss)
│ └── Perceptual Loss (feature-level comparison via VGG)
│
├── 📚 Training Phase
│ ├── Loop over epochs and batches
│ ├── Compute losses and update model weights
│ └── Periodically update activation function usage stats
│
├── 📈 Validation
│ └── Compare predicted SR images with HR ground truth (using unseen LR inputs)
│
└── 💾 Output
├── Save trained model
├── Log losses and performance
└── Export SR validation samples

Copy
Edit


























## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

