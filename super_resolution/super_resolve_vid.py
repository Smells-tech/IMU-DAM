from __future__ import print_function
import argparse
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import imageio

from model import Net  # Your model definition file, must match loaded model

def ycbcr_to_rgb(y, cb, cr):
    out_img_cb = cb.resize(y.size, Image.BICUBIC)
    out_img_cr = cr.resize(y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

def process_frame(frame, model, device):
    img = Image.fromarray(frame).convert('YCbCr')
    y, cb, cr = img.split()
    img_to_tensor = ToTensor()
    input_tensor = img_to_tensor(y).view(1, -1, y.size[1], y.size[0]).to(device)
    with torch.no_grad():
        out = model(input_tensor)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img = ycbcr_to_rgb(out_img_y, cb, cr)
    return np.array(out_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Super Resolution with PyTorch')
    parser.add_argument('--file', '-f', type=str, required=True, help='Input video file path (mp4)')
    parser.add_argument('--model', '-m', type=str, required=True, help='Pretrained model path')
    args = parser.parse_args()

    # Load model
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.model, map_location=map_location)
    model.eval()
    model.to(map_location)

    # Prepare input/output video paths and writer
    input_path = args.file
    input_name, input_ext = os.path.splitext(input_path)
    output_path = f"{input_name}-upscaled.mp4"

    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data().get('fps', 30)  # default 30 if unknown

    # Read a frame to get size and calculate output size
    first_frame = reader.get_data(0)
    height, width = first_frame.shape[:2]
    upscaled_width, upscaled_height = width * 3, height * 3

    with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:

        print(f"Processing video: {input_path} --> {output_path}")
        for i, frame in enumerate(reader):
            upscaled_frame = process_frame(frame, model, map_location)
            writer.append_data(upscaled_frame)
            print(f"Processed frame {i+1}")

        writer.close()
        reader.close()
        print(f"Upscaled video saved to {output_path}")
