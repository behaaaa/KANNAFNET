# Import necessary libraries
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import math
import argparse
import yaml
import logging
from tqdm import tqdm
from collections import OrderedDict
import h5py

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device to CUDA if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# --------- Activation Function Management ---------

class ActivationModule:
    @staticmethod
    def get_activation_functions():
        """
        Return a dictionary of various activation functions.
        Includes standard, advanced, and custom-defined functions.
        """
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'softplus': nn.Softplus(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softsign': nn.Softsign(),
            'hardtanh': nn.Hardtanh(),
            'tanhshrink': nn.Tanhshrink(),
            'hardshrink': nn.Hardshrink(),
            'softshrink': nn.Softshrink(),
            'swish': lambda x: x * torch.sigmoid(x),
            'mish': lambda x: x * torch.tanh(F.softplus(x)),
            'silu': nn.SiLU(),
            'prelu': nn.PReLU(),
            'hard_sigmoid': lambda x: F.relu6(x + 3) / 6,
            'hard_swish': lambda x: x * (F.relu6(x + 3) / 6),
            'sine': lambda x: torch.sin(x),
            'cosine': lambda x: torch.cos(x),
            'sinc': lambda x: torch.where(x == 0, torch.ones_like(x), torch.sin(x) / x),
            'gaussian': lambda x: torch.exp(-x**2),
            'softmax': lambda x: F.softmax(x, dim=-1),
            'log_softmax': lambda x: F.log_softmax(x, dim=-1),
            'bent_identity': lambda x: (torch.sqrt(x**2 + 1) - 1) / 2 + x,
            'arctan': lambda x: torch.atan(x),
            'sinh': lambda x: torch.sinh(x),
            'cosh': lambda x: torch.cosh(x),
            'sqnl': lambda x: torch.where(x > 2, 1.0, torch.where(x < -2, -1.0, torch.where(x > 0, 1 - (x - 2)**2/4, (x + 2)**2/4 - 1))),
            'isrlu': lambda x: torch.where(x >= 0, x, x / torch.sqrt(1 + 0.1 * x**2)),
            'isru': lambda x: torch.where(x >= 0, x, x / torch.sqrt(1 + 0.1 * x**2))
        }
        return activations

    @staticmethod
    def get_activation(name):
        """
        Retrieve activation function by name from the dictionary.
        """
        activations = ActivationModule.get_activation_functions()
        if name in activations:
            return activations[name]
        else:
            raise ValueError(f"Activation function '{name}' not found")

# --------- KAN Linear Layer ---------

class KANLinear(nn.Module):
    """
    A linear layer for Kolmogorov-Arnold Networks (KAN) with adaptable activation functions.
    """

    def __init__(self, in_features, out_features, num_activations=5, init_scale=1.0):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_activations = num_activations

        # Weight matrix for combining multiple activation functions
        self.activation_weights = nn.Parameter(torch.ones(out_features, in_features, num_activations) * (1.0 / num_activations))
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # All available activation functions
        self.activations_dict = ActivationModule.get_activation_functions()
        self.activation_names = list(self.activations_dict.keys())

        # Currently selected activation functions
        self.current_activations = random.sample(self.activation_names, num_activations)

        # Scaling factor for the output
        self.weight_scale = nn.Parameter(torch.ones(out_features, in_features) * init_scale)

        # Tracks performance of each activation function
        self.activation_performance = {name: 0.0 for name in self.activation_names}
        
        # Evaluation counter for periodic updates
        self.eval_counter = 0

    def forward(self, x):
        """
        Forward pass through the layer using a weighted combination of activation functions.
        """
        batch_size = x.size(0)
        result = torch.zeros(batch_size, self.out_features, device=x.device)

        for i, act_name in enumerate(self.current_activations):
            activation_fn = self.activations_dict[act_name]
            activated = activation_fn(x.unsqueeze(-1))  # Apply activation

            # Compute contribution from current activation
            contribution = torch.sum(
                activated * self.activation_weights[:, :, i].t().unsqueeze(0),
                dim=1
            )
            result += contribution

        # Scale output and add bias
        result = result * self.weight_scale.sum(dim=1) + self.bias
        return result

    def update_activations(self):
        """
        Periodically update the set of active activation functions based on performance.
        """
        self.eval_counter += 1
        if self.eval_counter % 10 == 0:
            # Sort activations by performance
            sorted_activations = sorted(self.activation_performance.items(), key=lambda x: x[1], reverse=True)

            # Keep top performing activations
            top_k = max(1, self.num_activations - 2)
            top_activations = [act[0] for act in sorted_activations[:top_k]]

            # Replace the rest with random choices
            remaining_activations = [act for act in self.activation_names if act not in top_activations]
            random_activations = random.sample(remaining_activations, min(self.num_activations - top_k, len(remaining_activations)))

            self.current_activations = top_activations + random_activations

            # Reset performance of newly selected random activations
            for act in self.activation_names:
                if act in random_activations:
                    self.activation_performance[act] = 0.0

    def update_activation_performance(self, activation_name, performance_delta):
        """
        Update performance score for a given activation function.
        """
        if activation_name in self.activation_performance:
            self.activation_performance[activation_name] += performance_delta

# --------- KAN Multi-layer Network ---------

class KANNetwork(nn.Module):
    """
    A multi-layer network composed of KANLinear layers.
    """

    def __init__(self, in_channels, hidden_dims, out_channels, num_activations=5):
        super(KANNetwork, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.num_activations = num_activations

        layers = []
        input_dim = in_channels

        # Construct hidden layers
        for hidden_dim in hidden_dims:
            layers.append(KANLinear(input_dim, hidden_dim, num_activations))
            input_dim = hidden_dim

        # Final output layer
        layers.append(KANLinear(input_dim, out_channels, num_activations))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the KAN network.
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten input
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def update_activations(self):
        """
        Update activations for all KANLinear layers.
        """
        for layer in self.layers:
            if isinstance(layer, KANLinear):
                layer.update_activations()

# --------- High-level Patch-wise Processing Module ---------

class KANeural(nn.Module):
    """
    A high-level model that applies a KAN network to image patches for tasks like super-resolution.
    """

    def __init__(self, patch_size, in_channels, hidden_dims=[128, 256, 512, 256, 128], out_channels=None, num_activations=5):
        super(KANeural, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels * 4

        input_features = patch_size * patch_size * in_channels
        output_features = (patch_size * 2) * (patch_size * 2) * self.out_channels

        # Auto-compute hidden dimensions if not provided
        if not hidden_dims:
            scale = (output_features / input_features) ** (1 / 3)
            hidden_dims = [
                int(input_features * scale),
                int(input_features * scale * scale),
                int(input_features * scale * scale)
            ]

        self.kan_network = KANNetwork(input_features, hidden_dims, output_features, num_activations)

    def forward(self, x):
        """
        Forward pass that splits the image into patches,
        processes each patch with the KAN network, and reconstructs the upscaled image.
        """
        batch_size, channels, height, width = x.shape

        assert height % self.patch_size == 0, f"Height must be divisible by patch_size ({self.patch_size})"
        assert width % self.patch_size == 0, f"Width must be divisible by patch_size ({self.patch_size})"

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # Extract patches from image
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                h_start = i * self.patch_size
                h_end = (i + 1) * self.patch_size
                w_start = j * self.patch_size
                w_end = (j + 1) * self.patch_size
                patch = x[:, :, h_start:h_end, w_start:w_end]
                patches.append(patch)

        upscaled_patches = []
        for patch in patches:
            # Flatten patch for KAN input
            flat_patch = patch.reshape(batch_size, -1)
            upscaled_flat = self.kan_network(flat_patch)
            upscaled_patch = upscaled_flat.reshape(batch_size, self.out_channels, self.patch_size * 2, self.patch_size * 2)
            upscaled_patches.append(upscaled_patch)

        # Initialize empty image for reconstruction
        upscaled_h = num_patches_h * self.patch_size * 2
        upscaled_w = num_patches_w * self.patch_size * 2
        upscaled_image = torch.zeros(batch_size, self.out_channels, upscaled_h, upscaled_w, device=x.device)

        # Stitch the patches together
        patch_idx = 0
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                h_start = i * self.patch_size * 2
                h_end = (i + 1) * self.patch_size * 2
                w_start = j * self.patch_size * 2
                w_end = (j + 1) * self.patch_size * 2
                upscaled_image[:, :, h_start:h_end, w_start:w_end] = upscaled_patches[patch_idx]
                patch_idx += 1

        return upscaled_image

# -------------------- Custom Layer Normalization --------------------
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors
        N, C, H, W = grad_output.size()

        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return gx, (grad_output * y).sum(dim=[0, 2, 3]), grad_output.sum(dim=[0, 2, 3]), None

# -------------------- LayerNorm2d using custom LayerNormFunction --------------------
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# -------------------- SimpleGate as used in NAFNet --------------------
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# -------------------- NAFBlock for feature transformation --------------------
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super(NAFBlock, self).__init__()
        dw_channel = c * DW_Expand
        ffn_channel = c * FFN_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1)

        self.sg = SimpleGate()

        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        x = x + y * self.gamma

        return x

# -------------------- Full NAFNet Architecture --------------------
class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=None, dec_blk_nums=None):
        super(NAFNet, self).__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        # Middle Blocks
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.output = nn.Conv2d(width, img_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_feat in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_feat
            x = decoder(x)
        x = self.output(x)
        return x

# -------------------- Final Integrated Model --------------------
class KANNAFNetIntegrated(nn.Module):
    def __init__(self,
                 in_channels=3,
                 kan_patch_size=8,
                 num_kan_blocks=4,
                 kan_hidden_dims=[128, 256, 512, 256, 128],
                 kan_num_activations=10,
                 nafnet_width=64,
                 nafnet_middle_blk_num=12,
                 nafnet_enc_blk_nums=None,
                 nafnet_dec_blk_nums=None):
        super(KANNAFNetIntegrated, self).__init__()

        self.in_channels = in_channels
        self.kan_patch_size = kan_patch_size
        self.num_kan_blocks = num_kan_blocks

        # KAN branches
        self.kan_blocks = nn.ModuleList([
            KANeural(
                patch_size=kan_patch_size,
                in_channels=in_channels,
                hidden_dims=kan_hidden_dims,
                num_activations=kan_num_activations
            ) for _ in range(num_kan_blocks)
        ])

        # NAFNet for denoising and pixel refinement
        self.nafnet = NAFNet(
            img_channel=in_channels,
            width=nafnet_width,
            middle_blk_num=nafnet_middle_blk_num,
            enc_blk_nums=nafnet_enc_blk_nums,
            dec_blk_nums=nafnet_dec_blk_nums
        )

        # Fuse KAN outputs
        self.fusion = nn.Conv2d(
            in_channels=in_channels * num_kan_blocks,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )

        # Learnable weights for each KAN block
        self.block_weights = nn.Parameter(torch.ones(num_kan_blocks) / num_kan_blocks)

        # Residual scaling
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        batch_size, _, h, w = x.shape

        # Apply each KAN block and store outputs
        kan_outputs = [kan_block(x) for kan_block in self.kan_blocks]

        # Softmax-normalized weights across blocks
        normalized_weights = F.softmax(self.block_weights, dim=0)

        # Apply weights to each KAN output
        weighted_kan_outputs = [
            kan_output * normalized_weights[i]
            for i, kan_output in enumerate(kan_outputs)
        ]

        # Concatenate and fuse
        combined_kan = torch.cat(weighted_kan_outputs, dim=1)
        fused_kan = self.fusion(combined_kan)

        # Pass through NAFNet
        denoised = self.nafnet(fused_kan)

        # Upscale input with bicubic interpolation (×4 total)
        upscaled_input = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        upscaled_input = F.interpolate(upscaled_input, scale_factor=2, mode='bicubic', align_corners=False)

        # Combine with residual connection
        output = denoised + self.residual_scale * upscaled_input

        return output
class SRDataset(Dataset):
    """Dataset class for Super Resolution training.
    """

    def __init__(self, hr_dir, lr_scale=4, patch_size=128, transform=None):
        self.hr_dir = hr_dir
        self.lr_scale = lr_scale
        self.patch_size = patch_size
        self.transform = transform

        # List all supported image files
        self.image_files = [f for f in os.listdir(hr_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        logger.info(f"Found {len(self.image_files)} images in dataset directory")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load HR image and convert to RGB
        hr_path = os.path.join(self.hr_dir, self.image_files[idx])
        hr_img = Image.open(hr_path).convert('RGB')

        # Apply transformations
        if self.transform:
            hr_img = self.transform(hr_img)
        else:
            hr_img = transforms.ToTensor()(hr_img)

        # Generate LR image using bicubic downsampling
        lr_img = F.interpolate(
            hr_img.unsqueeze(0),
            scale_factor=1 / self.lr_scale,
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=True
        ).squeeze(0)

        # Random crop if image is large enough
        _, h, w = hr_img.shape
        if h >= self.patch_size and w >= self.patch_size:
            h_start = random.randint(0, h - self.patch_size)
            w_start = random.randint(0, w - self.patch_size)

            hr_img = hr_img[:, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]

            lr_patch_size = self.patch_size // self.lr_scale
            lr_h_start = h_start // self.lr_scale
            lr_w_start = w_start // self.lr_scale
            lr_img = lr_img[:, lr_h_start:lr_h_start + lr_patch_size, lr_w_start:lr_w_start + lr_patch_size]

        return {
            'lr': lr_img,
            'hr': hr_img,
            'filename': self.image_files[idx]
        }


def create_dataloaders(hr_dir, batch_size=16, patch_size=128, lr_scale=4, num_workers=4):
    """Create training and validation DataLoaders with data augmentation."""

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor()
    ])

    full_dataset = SRDataset(hr_dir, lr_scale, patch_size, transform)

    # Split into train/val
    dataset_size = len(full_dataset)
    val_size = max(1, int(dataset_size * 0.1))
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss: A differentiable L1 loss with robustness to outliers."""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual Loss using VGG19 pre-trained features."""

    def __init__(self, weights=[0.1, 0.1, 0.2, 0.4, 0.8]):
        super(PerceptualLoss, self).__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        except:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features.eval()

        self.vgg = nn.ModuleList()
        self.weights = weights
        self.criterion = nn.L1Loss()

        # Indices for VGG slices
        slice_indices = [2, 7, 12, 21, 30]
        for i in range(len(slice_indices)):
            if i == 0:
                self.vgg.append(vgg[:slice_indices[i]])
            else:
                self.vgg.append(vgg[slice_indices[i - 1]:slice_indices[i]])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Normalize to VGG input
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)

        x = (x - mean) / std
        y = (y - mean) / std

        loss = 0
        for i, block in enumerate(self.vgg):
            x = block(x)
            y = block(y)
            loss += self.weights[i] * self.criterion(x, y)

        return loss


class DynamicActivationRegistry:
    """Registry to monitor and adapt activation function usage and performance."""

    def __init__(self, initial_activations=None):
        self.activations = ActivationModule.get_activation_functions()  # All available activations

        self.success_metrics = {name: 0.0 for name in self.activations.keys()}
        self.usage_count = {name: 0 for name in self.activations.keys()}
        self.historical_performance = {name: [] for name in self.activations.keys()}

        # Initialize active functions
        self.active_functions = set()
        if initial_activations:
            for act in initial_activations:
                if act in self.activations:
                    self.active_functions.add(act)
        else:
            defaults = ['relu', 'gelu', 'silu', 'mish', 'swish']
            for act in defaults:
                if act in self.activations:
                    self.active_functions.add(act)

    def update_metrics(self, activation_name, success_delta):
        """Update metrics for a specific activation function after usage."""
        if activation_name in self.success_metrics:
            self.success_metrics[activation_name] += success_delta
            self.usage_count[activation_name] += 1

            # Track average performance over time
            if self.usage_count[activation_name] > 0:
                normalized = self.success_metrics[activation_name] / self.usage_count[activation_name]
                self.historical_performance[activation_name].append(normalized)

    def get_top_activations(self, n=10):
        """Return top-N performing activation functions."""
        success_rates = {}
        for name in self.success_metrics:
            if self.usage_count[name] > 0:
                success_rates[name] = self.success_metrics[name] / self.usage_count[name]
            else:
                success_rates[name] = 0.0

        sorted_activations = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
        return [act[0] for act in sorted_activations[:n]]
