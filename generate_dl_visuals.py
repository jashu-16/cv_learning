"""
Generates visual demonstration figures for the PyTorch & Deep Learning Computer Vision Guide.
Saves all visual artifacts to `assets/`.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

def save_fig(filename):
    plt.tight_layout()
    plt.savefig(os.path.join("assets", filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Generated] assets/{filename}")

# -------------------------------------------------------------
# 1. 4D Tensor Memory Layout: (N, C, H, W) vs (N, H, W, C)
# -------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
sample = np.zeros((64, 64, 3), dtype=np.float32)
# Draw synthetic RGB feature pattern
sample[:32, :32, 0] = 1.0 # Red block
sample[32:, :32, 1] = 1.0 # Green block
sample[:, 32:, 2] = 1.0   # Blue stripe

axs[0].imshow(sample)
axs[0].set_title("Input Image (H=64, W=64, C=3)")
axs[0].axis("off")

# Planar channels (NCHW layout - PyTorch default)
nchw_vis = np.hstack([sample[..., 0], sample[..., 1], sample[..., 2]])
axs[1].imshow(nchw_vis, cmap="viridis")
axs[1].set_title("PyTorch NCHW Channel Planes (R | G | B)")
axs[1].axis("off")

# Memory Strides representation bar
strides_data = np.random.randn(8, 24)
axs[2].imshow(strides_data, cmap="coolwarm", aspect="auto")
axs[2].set_title("C-Contiguous 1D Stride Mapping")
axs[2].axis("off")
save_fig("dl_01_tensor_strides.png")

# -------------------------------------------------------------
# 2. ResNet Skip Connection & Gradient Highway
# -------------------------------------------------------------
x = np.linspace(-3, 3, 200)
f_x = np.tanh(x) * 1.5 # Residual transformation
h_x = f_x + x           # Skip connection identity output

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, x, 'k--', label="Identity Highway: x (Gradient = 1.0)", linewidth=2)
ax.plot(x, f_x, 'r-', label="Learned Residual: F(x)", linewidth=2)
ax.plot(x, h_x, 'g-', label="Residual Block Output: H(x) = F(x) + x", linewidth=2.5)
ax.set_title("ResNet Residual Skip Connection Dynamics: H(x) = F(x) + x", fontsize=11)
ax.set_xlabel("Input Activation x")
ax.set_ylabel("Output Activation")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")
save_fig("dl_05_resnet_skip.png")

# -------------------------------------------------------------
# 3. Vision Transformer (ViT) Patch Tokenization & Attention
# -------------------------------------------------------------
img_vit = np.zeros((192, 192), dtype=np.float32)
# Create grid pattern with synthetic attention focus
for r in range(0, 192, 48):
    for c in range(0, 192, 48):
        img_vit[r:r+48, c:c+48] = np.random.uniform(0.2, 0.8)
# Add salient object in center patches
img_vit[48:144, 48:144] += 0.5
img_vit = np.clip(img_vit, 0.0, 1.0)

# Attention map (Self-attention heat focused on center tokens)
attn_map = np.zeros((4, 4))
attn_map[1:3, 1:3] = np.array([[0.85, 0.95], [0.90, 0.88]])
attn_map[0, :] = 0.1; attn_map[3, :] = 0.1; attn_map[:, 0] = 0.1; attn_map[:, 3] = 0.1

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
axs[0].imshow(img_vit, cmap="gray")
axs[0].set_title("1. Input Image (192x192)")
# Draw 4x4 patch grid boundaries
for p in range(0, 192, 48):
    axs[0].axvline(p, color="cyan", linestyle="--", linewidth=1.5)
    axs[0].axhline(p, color="cyan", linestyle="--", linewidth=1.5)
axs[0].axis("off")

axs[1].imshow(attn_map, cmap="magma", interpolation="nearest")
axs[1].set_title("2. Patch Attention Weights (4x4 Tokens)")
for (i, j), z in np.ndenumerate(attn_map):
    axs[1].text(j, i, f"{z:.2f}", ha="center", va="center", color="white" if z < 0.5 else "black", fontsize=9)
axs[1].axis("off")

# Reconstructed Attention Overlay
import cv2
attn_upsampled = cv2.resize(attn_map.astype(np.float32), (192, 192), interpolation=cv2.INTER_CUBIC)
img_float = img_vit.astype(np.float32)
overlay = cv2.addWeighted(img_float, 0.6, attn_upsampled, 0.4, 0.0, dtype=cv2.CV_32F)
axs[2].imshow(overlay, cmap="magma")
axs[2].set_title("3. ViT Visual Attention Heatmap")
axs[2].axis("off")
save_fig("dl_06_vit_attention.png")

# -------------------------------------------------------------
# 4. Loss Functions: Cross-Entropy vs Focal Loss
# -------------------------------------------------------------
p_t = np.linspace(0.01, 1.0, 200)
ce_loss = -np.log(p_t)
focal_gamma_1 = - (1.0 - p_t)**1.0 * np.log(p_t)
focal_gamma_2 = - (1.0 - p_t)**2.0 * np.log(p_t)
focal_gamma_5 = - (1.0 - p_t)**5.0 * np.log(p_t)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(p_t, ce_loss, 'k--', label="Standard Cross-Entropy (gamma=0)", linewidth=2)
ax.plot(p_t, focal_gamma_1, 'b-', label="Focal Loss (gamma=1.0)", linewidth=2)
ax.plot(p_t, focal_gamma_2, 'r-', label="Focal Loss (gamma=2.0 - Standard RetinaNet/YOLO)", linewidth=2.5)
ax.plot(p_t, focal_gamma_5, 'g-', label="Focal Loss (gamma=5.0)", linewidth=2)
ax.set_title("Focal Loss vs Cross-Entropy: Down-weighting Easy Examples (p_t -> 1)", fontsize=11)
ax.set_xlabel("Model Probability of Ground Truth Class (p_t)")
ax.set_ylabel("Loss Magnitude")
ax.set_ylim(0, 5)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right")
save_fig("dl_08_focal_loss.png")

# -------------------------------------------------------------
# 5. U-Net Segmentation Encoder-Decoder with Skip Connections
# -------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
input_img = np.zeros((128, 128), dtype=np.float32)
cv2.circle(input_img, (64, 64), 30, 0.8, -1)
cv2.rectangle(input_img, (20, 20), (50, 50), 0.5, -1)

# Encoder feature downsampling
enc1 = cv2.resize(input_img, (64, 64))
bottleneck = cv2.resize(enc1, (16, 16))

# Segmentation mask output
gt_mask = np.zeros((128, 128), dtype=np.float32)
cv2.circle(gt_mask, (64, 64), 30, 1.0, -1)
cv2.rectangle(gt_mask, (20, 20), (50, 50), 1.0, -1)

axs[0].imshow(input_img, cmap="gray"); axs[0].set_title("1. Input (128x128)"); axs[0].axis("off")
axs[1].imshow(enc1, cmap="viridis"); axs[1].set_title("2. Enc Level 1 (64x64 + Skip)"); axs[1].axis("off")
axs[2].imshow(bottleneck, cmap="magma"); axs[2].set_title("3. Bottleneck Latent (16x16)"); axs[2].axis("off")
axs[3].imshow(gt_mask, cmap="plasma"); axs[3].set_title("4. Decoded Mask (128x128)"); axs[3].axis("off")
save_fig("dl_10_unet_segmentation.png")

# -------------------------------------------------------------
# 6. Quantization: FP32 vs INT8 Calibration Histogram
# -------------------------------------------------------------
fp32_weights = np.random.normal(0, 0.4, 10000)
# Clip outliers at [-1.0, 1.0] for INT8 symmetric quantization
int8_bins = np.linspace(-1.0, 1.0, 256)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(fp32_weights, bins=80, density=True, alpha=0.6, color="royalblue", label="FP32 Continuous Weights Distribution")
ax.axvline(-1.0, color="crimson", linestyle="--", linewidth=2, label="INT8 Lower Clamp (-128 * scale)")
ax.axvline(1.0, color="crimson", linestyle="--", linewidth=2, label="INT8 Upper Clamp (+127 * scale)")
ax.set_title("Post-Training INT8 Quantization: Dynamic Range Calibration & Clamping", fontsize=11)
ax.set_xlabel("Weight Value Range")
ax.set_ylabel("Probability Density")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right")
save_fig("dl_15_quantization.png")

print("All Deep Learning visual assets generated successfully!")
