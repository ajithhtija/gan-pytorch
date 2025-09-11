import os
# os.environ['OMP_NUM_THREADS'] = '1' 
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from utils import *
from models.models import *
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import torch.nn.functional as F
import subprocess
from torchvision.models import vgg19
from torchsummary import summary
# import easyocr
# Initialize OCR model
# ocr_model = easyocr.Reader(['en'], gpu=False)

def init_weights_he(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
# Dataset class
# class ImageDataset(Dataset):
    
#     def __init__(self, deg_path, clean_path, transform=None):
#         self.deg_images = []
#         self.clean_images = []
#         self.transform = transform

#         for i in range(len(deg_path)):
#             deg_files = sorted(os.listdir(deg_path[i]))
#             clean_files = sorted(os.listdir(clean_path[i]))
#             self.deg_images.extend([(deg_path[i], file) for file in deg_files])
#             self.clean_images.extend([(clean_path[i], file) for file in clean_files])
#     def __len__(self):
#         return len(self.deg_images)

#     def __getitem__(self, idx):
#         deg_path, deg_file = self.deg_images[idx]
#         clean_path, clean_file = self.clean_images[idx]

#         deg_image = Image.open(os.path.join(deg_path, deg_file)).convert('L')
#         clean_image = Image.open(os.path.join(clean_path, clean_file)).convert('L')

#         if self.transform:
#             deg_image = self.transform(deg_image)
#             clean_image = self.transform(clean_image)

#         # gen patches
#         deg_image, clean_image = getPatches(deg_image, clean_image, mystride=192)
        
#         return deg_image, clean_image

class PatchDataset(Dataset):
    def __init__(self, deg_paths, clean_paths, transform=None, stride=192):
        self.wm_patches = []
        self.clean_patches = []
        self.transform = transform

        for i in range(len(deg_paths)):
            deg_files = sorted(os.listdir(deg_paths[i]))
            clean_files = sorted(os.listdir(clean_paths[i]))
            for deg_file, clean_file in zip(deg_files, clean_files):
                deg_img = Image.open(os.path.join(deg_paths[i], deg_file)).convert('L')
                clean_img = Image.open(os.path.join(clean_paths[i], clean_file)).convert('L')
                
                if self.transform:
                    deg_img = self.transform(deg_img)  # shape: [1, H, W]
                    clean_img = self.transform(clean_img)

                wm_p, clean_p = getPatches(deg_img, clean_img, mystride=stride)
                self.wm_patches.extend(wm_p)
                self.clean_patches.extend(clean_p)

    def __len__(self):
        return len(self.wm_patches)

    def __getitem__(self, idx):
        return self.wm_patches[idx], self.clean_patches[idx]



def compute_total_variation_loss(img):
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2).sum()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2).sum()
    return (tv_h + tv_w)


# Training function
def train(generator, discriminator, dataloader, epochs, device, start):

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.BCELoss()
    criterion_content = nn.MSELoss()
    vgg = vgg19(pretrained=True).features[:16].to(device).eval()  # Use layers up to relu_4_1
    for param in vgg.parameters():
        param.requires_grad = False
    vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    def preprocess_vgg(img):
        img = img.repeat(1, 3, 1, 1)  # From [1,1,H,W] to [1,3,H,W]
        img = (img - vgg_mean) / vgg_std
        return img
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    for epoch in range(start + 1, epochs + 1):
        
        for deg_batch, clean_batch in tqdm(dataloader):
            # print(f"Input to Generator forward, x.shape = {deg_batch.shape}")  # deg_batch: [B, N, 1, 256, 256]
            deg_patches = deg_batch.to(device)       # [N, 1, 256, 256]
            clean_patches = clean_batch.to(device)   # [N, 1, 256, 256]
            
            ### Forward Generator ###
            gen_patches = generator(deg_patches)        # [N, 1, 256, 256]
            N = gen_patches.size(0)  # Number of patches in the batch
            ### Perceptual Loss (VGG) ###
            vgg_fake = vgg(preprocess_vgg(gen_patches))
            vgg_real = vgg(preprocess_vgg(clean_patches))
            vgg_loss = F.l1_loss(vgg_fake, vgg_real)

            ### OCR Loss (per patch, averaged) ###
            # ocr_loss_total = 0.0
            for i in range(N):
                real_np = clean_patches[i].squeeze().detach().cpu().numpy() * 255
                fake_np = gen_patches[i].squeeze().detach().cpu().numpy() * 255
                real_np = real_np.astype('uint8')
                fake_np = fake_np.astype('uint8')

                # ocr_real = ocr_model.readtext(real_np, detail=0, paragraph=False)
                # ocr_fake = ocr_model.readtext(fake_np, detail=0, paragraph=False)

                # real_text = ''.join(ocr_real) if isinstance(ocr_real, list) else str(ocr_real)
                # fake_text = ''.join(ocr_fake) if isinstance(ocr_fake, list) else str(ocr_fake)

            #     def string_similarity(s1, s2):
            #         max_len = max(len(s1), len(s2))
            #         if max_len == 0: return 1.0
            #         s1_ord = torch.tensor([ord(c) for c in s1], dtype=torch.float32)
            #         s2_ord = torch.tensor([ord(c) for c in s2[:len(s1)]], dtype=torch.float32)
            #         if s2_ord.numel() < s1_ord.numel():
            #             s2_ord = F.pad(s2_ord, (0, s1_ord.numel() - s2_ord.numel()))
            #         distance = F.l1_loss(s1_ord, s2_ord, reduction='mean')
            #         return 1.0 - distance.item() / 255.0

            #     similarity = string_similarity(fake_text, real_text)
            #     ocr_loss_total += 1.0 - similarity

            # ocr_loss = torch.tensor(ocr_loss_total / N, requires_grad=True).to(device)

            ### Train Discriminator ###
            discriminator.requires_grad_(True)
            optimizer_d.zero_grad()

            valid = torch.ones((N, 1, 16, 16), device=device)
            fake = torch.zeros((N, 1, 16, 16), device=device)

            real_input = torch.cat((clean_patches, deg_patches), dim=1)  # [N, 2, 256, 256]
            fake_input = torch.cat((gen_patches.detach(), deg_patches), dim=1)

            real_loss = criterion_pixel(discriminator(real_input), valid)
            fake_loss = criterion_pixel(discriminator(fake_input), fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            ### Train Generator ###
            optimizer_g.zero_grad()
            discriminator.requires_grad_(False)

            pred_fake = discriminator(torch.cat((gen_patches, deg_patches), dim=1))
            g_loss_gan = criterion_gan(pred_fake, valid)
            # g_loss_pixel = criterion_gan(gen_patches, clean_patches)
            g_loss_pixel = criterion_content(gen_patches, clean_patches)
            g_loss_variation = compute_total_variation_loss(gen_patches)
            # g_loss = g_loss_gan + (500 * g_loss_pixel) + (20 * vgg_loss) 
            g_loss = 0.3*g_loss_gan + (g_loss_pixel) + (vgg_loss) + (g_loss_variation)  # Adjusted weight for TV loss
            g_loss.backward()
            optimizer_g.step()
        print(f"Epoch [{epoch}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, VGG Loss: {vgg_loss.item():.4f}, TV Loss: {g_loss_variation.item():.4f}")
        print(g_loss_gan.item(), g_loss_pixel.item(), vgg_loss.item(), g_loss_variation.item())
        # for deg_images, clean_images in tqdm(dataloader):
        #     B = deg_images.size(0)
        #     for b in range(B):
        #         deg_batch = deg_images[b].to(device)    # [N, 1, 256, 256]
        #         clean_batch = clean_images[b].to(device)
        #         num_patches = deg_batch.size(0)
        #         for i in range(num_patches):
        #             deg_patch = deg_batch[i].unsqueeze(0).to(device)      # [1, 1, 256, 256]
        #             clean_patch = clean_batch[i].unsqueeze(0).to(device)  # [1, 1, 256, 256]
                    
        #            # Train discriminator
        #             discriminator.requires_grad_(True)                      
        #             # pred_real = discriminator(torch.cat((clean_patch, deg_patch), dim=1))
        #             #print(pred_real.shape)
        #             #print(clean_patch.shape,deg_patch.shape)
        #             valid = torch.ones((1, 1, 16, 16), device=device)
        #             fake = torch.zeros((1, 1, 16, 16), device=device)
        #             # valid = torch.ones_like(pred_real, device=device)
        #             # print(valid.shape)
        #             # fake = torch.zeros_like(pred_real, device=device)
        #             # print(fake.shape)
        #             optimizer_d.zero_grad()
        #             gen_patch = generator(deg_patch)
        #             vgg_fake = vgg(preprocess_vgg(gen_patch))
        #             vgg_real = vgg(preprocess_vgg(clean_patch))
        #             vgg_loss = F.l1_loss(vgg_fake, vgg_real)
        #             with torch.no_grad():
        #                 img_np = clean_patch.squeeze().detach().cpu().numpy() * 255  # [H, W] in [0,255]
        #                 img_np = img_np.astype('uint8')
        #                 img_np1 = gen_patch.squeeze().detach().cpu().numpy() * 255  # [H, W] in [0,255]
        #                 img_np1 = img_np.astype('uint8')
        #                 ocr_real = ocr_model.readtext(img_np,detail=0, paragraph=False)
        #                 ocr_fake = ocr_model.readtext(img_np1,detail=0, paragraph=False)
        #                 def string_similarity(s1, s2):
        #                     max_len = max(len(s1), len(s2))
        #                     if max_len == 0:
        #                         return 1.0  # Perfect match
        #                     s1_ord = torch.tensor([ord(c) for c in s1], dtype=torch.float32)
        #                     s2_ord = torch.tensor([ord(c) for c in s2[:len(s1)]], dtype=torch.float32)
        #                     if s2_ord.numel() < s1_ord.numel():  # pad if needed
        #                         s2_ord = F.pad(s2_ord, (0, s1_ord.numel() - s2_ord.numel()))
        #                     distance = F.l1_loss(s1_ord, s2_ord, reduction='mean')
        #                     return 1.0 - distance.item() / 255.0  # Normalize

        #                 ocr_real_text = ''.join(ocr_real) if isinstance(ocr_real, list) else str(ocr_real)
        #                 ocr_fake_text = ''.join(ocr_fake) if isinstance(ocr_fake, list) else str(ocr_fake)

        #                 similarity = string_similarity(ocr_fake_text, ocr_real_text)
        #                 ocr_loss = 1.0 - similarity  # You want to minimize loss if similarity is low
        #                 ocr_loss = torch.tensor(ocr_loss, requires_grad=True).to(device)  # convert to tensor for loss graph

        #             real_input = torch.cat((clean_patch, deg_patch), dim=1)  # [1, 2, 256, 256]
        #             fake_input = torch.cat((gen_patch.detach(), deg_patch), dim=1)
        #             real_loss = criterion_pixel(discriminator(real_input), valid)
        #             fake_loss = criterion_pixel(discriminator(fake_input), fake)
        #             d_loss = real_loss + fake_loss
        #             d_loss.backward()
        #             optimizer_d.step()

        #             # Train generator
        #             optimizer_g.zero_grad()
        #             discriminator.requires_grad_(False)
        #             # print(f"gen_patch shape: {gen_patch.shape}, deg_patch shape: {deg_patch.shape}")
        #             pred_fake = discriminator(torch.cat((gen_patch, deg_patch), dim=1))
        #             g_loss_gan = criterion_gan(pred_fake, valid)
        #             g_loss_pixel = criterion_gan(gen_patch, clean_patch)
        #             g_loss = g_loss_gan + (500 * g_loss_pixel) + (20 * vgg_loss) + (50 * ocr_loss)
        #             g_loss.backward()
        #             optimizer_g.step()

        

        if epoch % 5 == 0:
            with open('/mnt/d/DE-GAN-Pytorch/train_info.txt', 'a') as f:
                f.writelines(f"Epoch: {epoch} D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, VGG Loss: {vgg_loss.item():.4f}\n\n")
            torch.save(generator, f'/mnt/d/DE-GAN-Pytorch/trained_weights/generator_{epoch}.pth')
            torch.save(discriminator, f'/mnt/d/DE-GAN-Pytorch/trained_weights/discriminator_{epoch}.pth')
            test_images = sorted(os.listdir('2013/original/'))
            # psnr_values = []
            # print(test_images)
            with torch.inference_mode():
                for i in test_images:
                    temp = i.split('.')
                    com = ['python','enhance.py','binarize',f'2013/original/{i}',f'FResults/epoch_{epoch}_{i}', f'{epoch}']
                    subprocess.run(com)
                    #temp[0] += '.tiff'
    #                 original = cv2.imread(f'2013/gt/{temp[0]}.tiff', cv2.IMREAD_GRAYSCALE)
    #                 predicted = cv2.imread(f'Results/epoch_{epoch}_{temp[0]}.bmp', cv2.IMREAD_GRAYSCALE)
    #                 psnr_values.append(psnr(original, predicted))
    #         print(f"till {epoch} epoches : {np.mean(psnr_values)}\n")
    #         print(psnr_values)
    #         with open('/mnt/d/DE-GAN-Pytorch/PSNR_INFO.txt', 'a') as f:
    #             f.writelines(f'\nEpoch {epoch}, PSNR: {np.mean(psnr_values)}\n')

    # if epoch == epochs:
    #     return np.mean(psnr_values)


# def train(generator, discriminator, dataloader, epochs, device, start):

#     # criterion_gan = nn.MSELoss()
#     criterion_gan = nn.BCEWithLogitsLoss()
#     # criterion_pixelwise = nn.MSELoss()
#     # criterion_pixelwise = nn.L1Loss()

#     optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

#     for epoch in range(start + 1, epochs + 1):
#         for deg_images, clean_images in tqdm(dataloader):
#             deg_images = deg_images.squeeze(0).to(device)    # [N, 1, 256, 256]
#             clean_images = clean_images.squeeze(0).to(device)
#             num_patches = deg_images.size(0)
#             for i in range(num_patches):
#                 deg_patch = deg_images[i].unsqueeze(0).to(device)      # [1, 1, 256, 256]
#                 clean_patch = clean_images[i].unsqueeze(0).to(device)  # [1, 1, 256, 256]
                
#                 #deg_patch = deg_images[i].unsqueeze(0)   # [1, 1, 256, 256]
#                 #clean_patch = clean_images[i].unsqueeze(0)

#                 # Train discriminator
#                 discriminator.requires_grad_(True)
#                 valid = torch.ones((1, 1, 16, 16), device=device)
#                 fake = torch.zeros((1, 1, 16, 16), device=device)

#                 optimizer_d.zero_grad()

#                 gen_patch = generator(deg_patch)

#                 real_input = torch.cat((clean_patch, deg_patch), dim=1)  # [1, 2, 256, 256]
#                 fake_input = torch.cat((gen_patch.detach(), deg_patch), dim=1)

#                 real_loss = criterion_gan(discriminator(real_input), valid)
#                 fake_loss = criterion_gan(discriminator(fake_input), fake)
#                 d_loss = real_loss + fake_loss
#                 d_loss.backward()
#                 optimizer_d.step()

#                 # Train generator
#                 optimizer_g.zero_grad()
#                 discriminator.requires_grad_(False)

#                 pred_fake = discriminator(torch.cat((gen_patch, deg_patch), dim=1))
#                 g_loss_gan = criterion_gan(pred_fake, valid)
#                 g_loss_pixel = criterion_gan(gen_patch, clean_patch)
#                 g_loss = g_loss_gan + (500 * g_loss_pixel)
#                 g_loss.backward()
#                 optimizer_g.step()


#         print(f"Epoch [{epoch}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


#         '''
#         PREDICTION---
        
#         '''
#         if epoch%10 == 0:
#             with open('/mnt/d/DE-GAN-Pytorch/train_info.txt', 'a') as f:
#                 f.writelines(f"Epoch: {epoch} D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}\n\n")
#             torch.save(generator, f'/mnt/d/DE-GAN-Pytorch/trained_weights/generator_{epoch}.pth')
#             torch.save(discriminator, f'/mnt/d/DE-GAN-Pytorch/trained_weights/discriminator_{epoch}.pth')
#             test_images = sorted(os.listdir('2013/original/'))
#             psnr_values= []
#             with torch.inference_mode():
#                 for i in test_images:
#                     temp = i.split('.')
#                     com = ['python','enhance.py','binarize',f'2013/original/{i}',f'Results/epoch_{epoch}_{i}', f'{epoch}']
#                     subprocess.run(com)
#                     temp[0]+='.tiff'
#                     original  = cv2.imread(f'2013/gt/{temp[0]}',cv2.IMREAD_GRAYSCALE)
#                     predicted = cv2.imread(f'Results/epoch_{epoch}_{i}',cv2.IMREAD_GRAYSCALE)
#                     psnr_values.append(psnr(original,predicted))
#             print(f"till {epoch} epoches : {np.mean(psnr_values)}\n")
#             with open('/mnt/d/DE-GAN-Pytorch/PSNR_INFO.txt', 'a') as f:
#                 f.writelines(f'\nEpoch {epoch}, PSNR: {np.mean(psnr_values)}\n')

#         if epoch == epochs:
#             return np.mean(psnr_values)
#     # Initialize models and training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paths = sorted(Path('/mnt/d/DE-GAN-Pytorch/trained_weights/').iterdir(), key=os.path.getmtime, reverse = True)
if len(paths) > 0:
    #print(paths)
    paths[0] = str(paths[0]).replace("\\", '/')
    s = ""
    #print(paths[0])
    for i in paths[0]:
        if i.isdigit():
            s  = s+i
    # name = paths[0].split('/')[1]
    ep = int(s)
    if os.path.exists(paths[0]):
        discriminator = torch.load(f'/mnt/d/DE-GAN-Pytorch/trained_weights/discriminator_{ep}.pth')
        generator = torch.load(f'/mnt/d/DE-GAN-Pytorch/trained_weights/generator_{ep}.pth')
        start = ep
        print(f'loading saved model at epoch {ep}...')

else:
    with open('/mnt/d/DE-GAN-Pytorch/train_info.txt', 'w') as f:
        f.close()
    generator = Generator().to(device)
    #print(summary(generator, torch.random.rand(1, 256, 256)))
    #exit()
    generator.apply(init_weights_he)
    discriminator = Discriminator().to(device)
#   print(summary(discriminator))
    start = 0
    print('starting afresh...')

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

with open('/mnt/d/DE-GAN-Pytorch/PSNR_INFO.txt', 'w') as f:
       f.close()

# for i in range(1,n+1):
    
#     src_path_1 = f'2018/original/{i}.bmp'
#     dst_path_1 = f'data/A/{i}.bmp'
#     try:
#         deleteall('2018/Test_original/')
#     except:
#         os.makedirs('2018/Test_original/')
    
    # for j in range(i+1,n):
    #     # x = ['1.bmp', '10.bmp', '2.bmp', '3.bmp', '4.bmp', '5.bmp', '6.bmp', '7.bmp', '8.bmp', '9.bmp']
    #     x = np.zeros(shape = n)
    #     x[i], x[j] = 1, 1
        # x.remove(f'{i}.bmp')
        # x.remove(f'{j}.bmp')
        
        # try:
        #     deleteall('data/A/')
        #     deleteall('data/B/')
        #     deleteall('2018/Test_original')
        # except:
        #     'dir doesn"t exist!\n'
        
        # src_path_2 = f'2018/original/{j}.bmp'
        # dst_path_2 = f'data/A/{j}.bmp'

        # # print(src_path_1, src_path_2)
        # shutil.copy(src_path_1, dst_path_1)
        # shutil.copy(src_path_2, dst_path_2)

        # ### copy GT
        # gt_src_path_1 = f'2018/gt/{i}.bmp'
        # gt_dst_path_1 = f'data/B/{i}.bmp'

        # gt_src_path_2 = f'2018/gt/{j}.bmp'
        # gt_dst_path_2 = f'data/B/{j}.bmp'
        
        # shutil.copy(gt_src_path_1, gt_dst_path_1)
        # shutil.copy(gt_src_path_2, gt_dst_path_2)

        # deleteall('D:/DE-GAN-Pytorch/data/A/')
        # deleteall('D:/DE-GAN-Pytorch/data/B/')
        
        ##
# deg_path = ['2009/original/','2011/original/','2012/original/','2014/original/','2016/original/','2017/original/']
# clean_path = ['2009/gt/','2011/gt/','2012/gt/','2014/gt/','2016/gt/','2017/gt/']
# deg_path = ['2009/original/','2011/original/','2012/original/','2014/original/','2016/original/','2017/original/']
# clean_path = ['2009/gt/','2011/gt/','2012/gt/','2014/gt/','2016/gt/','2017/gt/']
deg_path = ['denoising-dirty-documents/noise_train/']
clean_path = ['denoising-dirty-documents/noise_gt_train/']

dataset = PatchDataset(deg_path, clean_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# print(dataset[0][0].shape)
# for i in dataloader:
#     print((i[1].shape))
#     exit()

        # dimages = sorted(os.listdir('D:/DE-GAN-Pytorch/2018/gt'))
        # oimages = sorted(os.listdir('D:/DE-GAN-Pytorch/2018/original'))
        
        # push rest to test---
        
         # idx = np.where(x == 0)[0]
         # print(idx)
        # for k in idx:
        #     src_path = f'2018/original/{k+1}.bmp'
        #     dst_path = f'2018/Test_original/{k+1}.bmp'
        #     # print(f'*****{src_path} /// {dst_path}')
        #     shutil.copy(src_path, dst_path)
        # y = [f'{i}.bmp',f'{j}.bmp']
        # for k in y:
        #     src_path = os.path.join('D:/DE-GAN-Pytorch/2018/gt', k)
        #     dst_path = os.path.join('D:/DE-GAN-Pytorch/data/B/', k)
        #     shutil.copy(src_path, dst_path)
        #     src_path = os.path.join('D:/DE-GAN-Pytorch/2018/original', k)
        #     dst_path = os.path.join('D:/DE-GAN-Pytorch/data/A/', k)
        #     shutil.copy(src_path, dst_path)
train(generator, discriminator, dataloader, start=0, epochs=250, device=device)

         
        # deleteall('D:/DE-GAN-Pytorch/trained_weights')
        # deleteall('D:/DE-GAN-Pytorch/predicted')
        # deleteall('trained_weights/')
        # deleteall('predicted/')

# with open('PSNR_INFO.txt', 'a') as f:
#     f.writelines(f'\n***** CV PSNR VALUE : {PSNR/45} *****\n\n')

# training
 
# 2009
# 2011
# 2012 
# 2014 
# 2016 
# 2017

# test 

# 2013
