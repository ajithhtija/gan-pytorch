import os
import subprocess
import torch
import numpy as np
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 117

# Load generator and discriminator
discriminator = torch.load(f'/mnt/d/DE-GAN-Pytorch/degan+vgg+spatial attention/discriminator_{epoch}.pth')
generator = torch.load(f'/mnt/d/DE-GAN-Pytorch/degan+vgg+spatial attention/generator_{epoch}.pth')
# Set to evaluation mode
discriminator.eval()
generator.eval()
# Get list of test images
test_images = sorted(os.listdir('ajit/'))

# Run enhance.py on each image
with torch.inference_mode():
    for i in test_images:
        if not i.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue  # skip non-image files
        temp = i.split('.')[0]
        com = [
            'python', 'enhance.py', 'binarize',
            f'/mnt/d/DE-GAN-Pytorch/ajit/{i}',
            f'/mnt/d/DE-GAN-Pytorch/currResults/epoch_{epoch}_{i}',
            f'{epoch}'
        ]
        subprocess.run(com)
        # Compute PSNR
        # clean_image = Image.open(f'/mnt/d/DE-GAN-Pytorch/2013/gt/{i}').convert('L')
        # enhanced_image = Image.open(f'/mnt/d/DE-GAN-Pytorch/currResults/epoch_{epoch}_{i}')
        # psnr_value = psnr(enhanced_image, clean_image)
        print(f'Processed {i} with enhance.py')