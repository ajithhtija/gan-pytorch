import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from cbam import CBAM  # Ensure your CBAM module is imported properly

import torch
import torch.nn as nn
from cbam import CBAM  # make sure this has both channel + spatial attention
import torch
import torch.nn as nn # Ensure CBAM(channels) works

class Generator(nn.Module):
    def __init__(self, input_channels=1, biggest_layer=1024):
        super(Generator, self).__init__()

        def conv_block(in_channels, out_channels, dropout=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # CBAM(),  # Apply CBAM after convolution
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.encoder1 = conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = conv_block(256, biggest_layer // 2, dropout=True)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(biggest_layer // 2, biggest_layer, dropout=True)

        # Decoder
        self.upconv4 = upconv_block(biggest_layer, biggest_layer // 2)
        self.decoder4 = conv_block(biggest_layer, biggest_layer // 2)

        self.upconv3 = upconv_block(biggest_layer // 2, 256)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = upconv_block(256, 128)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = upconv_block(128, 64)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # CBAM(),  # Final CBAM before output
            nn.Conv2d(2, 1, kernel_size=1),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        out = self.decoder1(d1)

        return out


# class Generator(nn.Module):
#     def __init__(self, input_channels=1, biggest_layer=1024):
#         super(Generator, self).__init__()

#         def conv_block(in_channels, out_channels, dropout=False):
#             layers = [
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 # CBAM(out_channels),
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 # CBAM(out_channels),
#             ]
#             if dropout:
#                 layers.append(nn.Dropout(0.5))
#             return nn.Sequential(*layers)

#         def upconv_block(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#                 nn.ReLU(inplace=True),
#             )

#         self.encoder1 = conv_block(input_channels, 64)
#         self.pool1 = nn.MaxPool2d(2)
#         self.encoder2 = conv_block(64, 128)
#         self.pool2 = nn.MaxPool2d(2)

#         self.encoder3 = conv_block(128, 256)
#         self.pool3 = nn.MaxPool2d(2)

#         self.encoder4 = conv_block(256, biggest_layer // 2, dropout=True)
#         self.pool4 = nn.MaxPool2d(2)

#         self.bottleneck = conv_block(biggest_layer // 2, biggest_layer, dropout=True)

#         self.upconv4 = upconv_block(biggest_layer, biggest_layer // 2)
#         self.decoder4 = conv_block(biggest_layer, biggest_layer // 2)

#         self.upconv3 = upconv_block(biggest_layer // 2, 256)
#         self.decoder3 = conv_block(512, 256)

#         self.upconv2 = upconv_block(256, 128)
#         self.decoder2 = conv_block(256, 128)

#         self.upconv1 = upconv_block(128, 64)
#         self.decoder1 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             # CBAM(64),
#             nn.Conv2d(64, 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             # CBAM(2),
#             nn.Conv2d(2, 1, kernel_size=1),
#         )

#     def forward(self, x):
#         # print(f"Input to Generator forward, x.shape = {x.shape}")
#         e1 = self.encoder1(x)
#         p1 = self.pool1(e1)

#         e2 = self.encoder2(p1)
#         p2 = self.pool2(e2)

#         e3 = self.encoder3(p2)
#         p3 = self.pool3(e3)

#         e4 = self.encoder4(p3)
#         p4 = self.pool4(e4)

#         b = self.bottleneck(p4)

#         d4 = self.upconv4(b)
#         d4 = torch.cat((d4, e4), dim=1)
#         d4 = self.decoder4(d4)

#         d3 = self.upconv3(d4)
#         d3 = torch.cat((d3, e3), dim=1)
#         d3 = self.decoder3(d3)

#         d2 = self.upconv2(d3)
#         d2 = torch.cat((d2, e2), dim=1)
#         d2 = self.decoder2(d2)

#         d1 = self.upconv1(d2)
#         d1 = torch.cat((d1, e1), dim=1)
#         return self.decoder1(d1)

# class Generator(nn.Module):
#     def __init__(self, input_channels=1, biggest_layer=1024):
#         super(Generator, self).__init__()

#         def conv_block(in_channels, out_channels, dropout=False):
#             layers = [
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#             ]
#             if dropout:
#                 layers.append(nn.Dropout(0.5))
#             return nn.Sequential(*layers)

#         def upconv_block(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#                 nn.ReLU(inplace=True),
#             )

#         self.encoder1 = conv_block(input_channels, 64)
#         self.pool1 = nn.MaxPool2d(2)

#         self.encoder2 = conv_block(64, 128)
#         self.pool2 = nn.MaxPool2d(2)

#         self.encoder3 = conv_block(128, 256)
#         self.pool3 = nn.MaxPool2d(2)

#         self.encoder4 = conv_block(256, biggest_layer // 2, dropout=True)
#         self.pool4 = nn.MaxPool2d(2)

#         self.bottleneck = conv_block(biggest_layer // 2, biggest_layer, dropout=True)

#         self.upconv4 = upconv_block(biggest_layer, biggest_layer // 2)
#         self.decoder4 = conv_block(biggest_layer, biggest_layer // 2)

#         self.upconv3 = upconv_block(biggest_layer // 2, 256)
#         self.decoder3 = conv_block(512, 256)

#         self.upconv2 = upconv_block(256, 128)
#         self.decoder2 = conv_block(256, 128)

#         self.upconv1 = upconv_block(128, 64)
#         self.decoder1 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2, 1, kernel_size=1),
#             #nn.Sigmoid(),
#         )

#     def forward(self, x):
#         e1 = self.encoder1(x)
#         p1 = self.pool1(e1)

#         e2 = self.encoder2(p1)
#         p2 = self.pool2(e2)

#         e3 = self.encoder3(p2)
#         p3 = self.pool3(e3)

#         e4 = self.encoder4(p3)
#         p4 = self.pool4(e4)

#         b = self.bottleneck(p4)

#         d4 = self.upconv4(b)
#         # print(d4.shape, e4.shape)
#         d4 = torch.cat((d4, e4), dim=1)
        
#         d4 = self.decoder4(d4)

#         d3 = self.upconv3(d4)
#         # print(d3.shape, e3.shape)

#         d3 = torch.cat((d3, e3), dim=1)
        

#         d3 = self.decoder3(d3)

#         d2 = self.upconv2(d3)
#         # print(d2.shape, e2.shape)

#         d2 = torch.cat((d2, e2), dim=1)
#         d2 = self.decoder2(d2)

#         d1 = self.upconv1(d2)
#         # print(d1.shape, e1.shape)

#         d1 = torch.cat((d1, e1), dim=1)
#         return self.decoder1(d1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2),
            # nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.2),
            # nn.MaxPool2d(2),
            

            nn.Conv2d(256,256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.2),
            # nn.MaxPool2d(2),
            

            # nn.Conv2d(256,256, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.BatchNorm2d(256, momentum=0.8),

            nn.Conv2d(256,1,kernel_size=1,stride = 1,padding = 0),
            nn.Sigmoid(),
        )


    def forward(self, x):
       return self.model(x)
