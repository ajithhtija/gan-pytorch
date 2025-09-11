#!/usr/bin/env python
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.models import *
from utils import split2, merge_image2

input_size = (256, 256, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task specified by the user
task = sys.argv[1]
epoch = sys.argv[4]

# Load the appropriate generator model and weights
if task == 'binarize':
    print(f'loading -> generator_{epoch}.pth ...')
    generator = torch.load(f"/mnt/d/DE-GAN-Pytorch/degan+vgg+spatial attention/generator_{epoch}.pth")
elif task == 'deblur':
    generator = Generator(biggest_layer=1024).to(device)
    generator = torch.load("trained_weights/deblur_weights_{epoch}.pth")
elif task == 'unwatermark':
    generator = Generator(biggest_layer=512).to(device)
    generator = torch.load("trained_weights/watermark_rem_weights_{epoch}.pth")
else:
    print("Wrong task, please specify a correct task!")
    sys.exit(1)

generator.eval()

# Load and preprocess the degraded image
deg_image_path = sys.argv[2]
deg_image = Image.open(deg_image_path).convert('L')
deg_image.save('/mnt/d/DE-GAN-Pytorch/curr_image.png')
test_image = plt.imread('/mnt/d/DE-GAN-Pytorch/curr_image.png')

# Pad the image to the nearest multiple of 256
h = ((test_image.shape[0] // 256) + 1) * 256
w = ((test_image.shape[1] // 256) + 1) * 256

test_padding = torch.ones((1, h, w), dtype=torch.float32, device=device)
test_padding[0, :test_image.shape[0], :test_image.shape[1]] = torch.tensor(test_image, dtype=torch.float32, device=device)

# Split the padded image into patches
test_image_p = split2(test_padding.unsqueeze(0), size=1, h=h, w=w)
# print('//////////////',test_image_p.shape)

predicted_list = []

# Process each patch through the generator
# for patch in test_image_p:
#     with torch.no_grad():
#         patch = patch.unsqueeze(0)  # Add batch dimension
#         predicted_patch = generator(patch)
#         predicted_list.append(predicted_patch.squeeze(0))

with torch.inference_mode():
    predicted_list = generator(test_image_p)

# Merge patches back into a single image
predicted_image = merge_image2(predicted_list, h, w)

predicted_image = predicted_image[0, :test_image.shape[0], :test_image.shape[1]]

# Postprocess for binarization if applicable
if task == 'binarize':
    bin_thresh = 0.50
    predicted_image = (predicted_image > bin_thresh).float()

# Save the final predicted image
save_path = sys.argv[3]
plt.imsave(save_path, predicted_image.cpu().numpy(), cmap='gray')
# print("Gen image stats: min =", predicted_image.min().item(), 
#       "max =", predicted_image.max().item(), 
#       "mean =", predicted_image.mean().item())