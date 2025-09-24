import os
import re
from argparse import ArgumentParser

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur

from PIL import Image, ImageFilter, ImageDraw

def alpha_blending(rgba, background=(0, 0, 0)):
    """
    Convert RGBA image to RGB by alpha compositing with a background color.

    Parameters:
    - rgba: numpy array of shape (H, W, 4), dtype uint8
    - background: tuple RGB color in range 0-255
        White (255, 255, 255) — gives brighter results, often used for images intended for display on white backgrounds.
        Black (0, 0, 0) — darker results, can be used for transparent objects on dark backgrounds.
        Gray or neutral color (e.g., 128, 128, 128) — neutral midpoint, useful if intending to later overlay or blend further.

    Returns:
    - rgb: numpy array of shape (H, W, 3), dtype uint8
    """

    if rgba.shape[2] != 4:
        raise ValueError("Input image does not have 4 channels (RGBA)")

    # Normalize alpha channel to 0-1 range
    alpha = rgba[..., 3] / 255.0

    # Prepare output array
    rgb = np.empty_like(rgba[..., :3])

    for c in range(3):
        rgb[..., c] = rgba[..., c] * alpha + background[c] * (1 - alpha)

    return rgb.astype(np.uint8)

def save_tensor_frame_to_png(tensor, filename):
    # tensor shape: [C, H, W], values usually in [0,1] or [-1,1]

    # Move tensor to CPU, detach grad, convert to numpy
    img = tensor.clone().detach().cpu()

    # If values in [-1,1], convert to [0,1]
    if img.min() < 0:
        img = (img + 1) / 2

    # Clamp values to [0,1]
    img = img.clamp(0, 1)

    # Convert to numpy and change to HWC format
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))

    # Convert to uint8 0-255
    img = (img * 255).astype(np.uint8)

    # Convert to PIL Image and save
    pil_img = Image.fromarray(img)
    pil_img.save(filename)

def center_crop(frame, target_height, target_width):
    height, width, _ = frame.shape

    start_y = (height - target_height) // 2
    start_x = (width - target_width) // 2

    cropped_frame = frame[start_y:start_y+target_height, start_x:start_x+target_width]
    return cropped_frame

def feather(img, top=0, left=0, right=0, bottom=0, each=None):
    if each:
        top, left, right, bottom = (each for _ in range(4))

    w, h = img.size

    # Create fully opaque alpha mask
    alpha = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(alpha)

    # Draw rectangles on sides to be feathered with black (0)
    if top > 0:
        draw.rectangle([0, 0, w, top], fill=0)
    if bottom > 0:
        draw.rectangle([0, h - bottom, w, h], fill=0)
    if left > 0:
        draw.rectangle([0, 0, left, h], fill=0)
    if right > 0:
        draw.rectangle([w - right, 0, w, h], fill=0)

    # Blur the alpha mask to create feather effect on each side
    # The radius passed is max of all to get smooth gradients
    max_radius = max(top, left, right, bottom)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=max_radius))

    # Apply new alpha channel to the image
    img.putalpha(alpha)

    return img

def feather_torch(frame_tensor, top=0, left=0, right=0, bottom=0, each=None):
    if each:
        top, left, right, bottom = (each for _ in range(4))

    # Assume frame_tensor shape (N, C, H, W) with C=3 for RGB, float tensor [0..1]
    N, C, H, W = frame_tensor.shape
    assert C == 3, "Input frames must have 3 channels (RGB)"

    # Create alpha mask initialized to 1 (fully opaque) with shape (N, 1, H, W)
    alpha = torch.ones((N, 1, H, W), dtype=frame_tensor.dtype, device=frame_tensor.device)

    # Create zero masks for edges to feather
    if top > 0:
        alpha[:, :, :top, :] = 0
    if bottom > 0:
        alpha[:, :, -bottom:, :] = 0
    if left > 0:
        alpha[:, :, :, :left] = 0
    if right > 0:
        alpha[:, :, :, -right:] = 0

    # Gaussian blur alpha mask (kernel size and sigma derived from max radius)
    radius = max(top, left, right, bottom)
    if radius > 0:
        kernel_size = 2 * radius + 1
        alpha = gaussian_blur(alpha, (kernel_size, kernel_size), sigma=radius/3)

    # Concatenate alpha channel to original frame tensor (shape => N, 4, H, W)
    rgba = torch.cat((frame_tensor, alpha), dim=1)

    return rgba

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--inp", required=True, help='Input image or video')
    parser.add_argument("--suffix", default="")
    parser.add_argument("--top", default=0, type=float, help='Feather top radius')
    parser.add_argument("--left", default=0, type=float, help='Feather left radius')
    parser.add_argument("--right", default=0, type=float, help='Feather right radius')
    parser.add_argument("--bottom", default=0, type=float, help='Feather bottom radius')

    args = parser.parse_args()

    img = Image.open(args.inp).convert("RGBA")

    feathered_img = feather(img, top=args.top, left=args.left, right=args.right, bottom=args.bottom)
    if args.oup == None:
        dir_name = os.path.dirname(args.inp)
        filename = os.path.basename(args.inp)
        base, ext = os.path.splitext(filename)
        args.oup = os.path.join(dir_name, f"{base}{args.suffix}{ext}")
    feathered_img.save(args.oup)
    # feathered_img.show()
