import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_sinusoidal_embeddings
"""
Implementation of a 1D U-Net model for time series forecasting.
The model consists of an encoder and a decoder, with skip connections between corresponding encoder and decoder layers.
The encoder downsamples the input time series, while the decoder upsamples the encoded representation to generate the output.
The model also includes a time embedding layer to incorporate temporal information.
"""
class UNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, time_emb_dim=256):
        super(UNet1D, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embedding = nn.Linear(time_emb_dim, time_emb_dim)

        # Encoder
        self.enc1 = self.conv_block(in_channels + time_emb_dim, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Decoder
        self.dec1 = self.up_conv_block(512, 256, output_padding=1)
        self.dec2 = self.up_conv_block(512, 128)
        self.dec3 = self.up_conv_block(256, 64)
        self.dec4 = nn.Conv1d(128, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),  # 保持长度不变
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),  # 保持长度不变
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_conv_block(self, in_channels, out_channels, output_padding=0):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, output_padding=output_padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        t_emb = get_sinusoidal_embeddings(t, self.time_emb_dim, x.device)
        t_emb = self.time_embedding(t_emb)
        t_emb = t_emb.view(t_emb.size(0), t_emb.size(1), 1)
        t_emb = t_emb.repeat(1, 1, x.size(2))
        # print("t_emb.shape:", t_emb.shape)
        x = torch.cat([x, t_emb], dim=1)
        # print("x.shape:", x.shape)

        # Encoder
        enc1 = self.enc1(x)  # (B, 64, 100)
        enc2 = self.enc2(F.max_pool1d(enc1, 2))  # (B, 128, 50)
        enc3 = self.enc3(F.max_pool1d(enc2, 2))  # (B, 256, 25)
        enc4 = self.enc4(F.max_pool1d(enc3, 2))  # (B, 512, 12)

        # Decoder
        dec1 = self.dec1(enc4)  # (B, 256, 25)
        dec1 = torch.cat([dec1, self.crop(enc3, dec1)], dim=1)  # (B, 512, 25)
        dec2 = self.dec2(dec1)  # (B, 128, 50)
        dec2 = torch.cat([dec2, self.crop(enc2, dec2)], dim=1)  # (B, 256, 50)
        dec3 = self.dec3(dec2)  # (B, 64, 100)
        dec3 = torch.cat([dec3, self.crop(enc1, dec3)], dim=1)  # (B, 128, 100)
        dec4 = self.dec4(dec3)  # (B, 1, 100)

        return dec4

    def crop(self, enc_feature, x):
        _, _, L = x.size()
        enc_feature = F.interpolate(enc_feature, size=L, mode='linear', align_corners=True)
        return enc_feature
# Test the model, and print the output shape, given a random input tensor, and a random time tensor.
if __name__ == "__main__":
    # test the model
    B = 1  # batch size
    input_data = torch.randn(B, 2, 100)  # create a random input tensor with shape (B, 2, 100)
    t = torch.randint(0, 1000, (B,))  # create a random time tensor with shape (B,)
    print(t.shape,  input_data.shape) # print the shapes of the input tensor and the time tensor
    model = UNet1D()
    output = model(input_data, t)
    print("Output shape:", output.shape) # print the shape of the output tensor
