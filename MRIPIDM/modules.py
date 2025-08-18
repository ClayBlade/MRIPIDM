#Run Module Util

import torch
import torch.nn as nn
import torch.nn.functional as F







class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.H = size[0]
        self.W = size[1]
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.H, self.W).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.H, self.W)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            print(f"x.shape in DoubleConv from Up: {x.shape}") 
            print(f"double_conv.shape: {self.double_conv(x).shape}")
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.out_channels = out_channels

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        #x = nn.Conv2d(in_channels=x.shape[1], out_channels=self.out_channels, kernel_size=1, device = "cuda")(x) # 1x1 conv to match channels
        print(f"x.shape after 1x1 conv: {x.shape}")
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UNet(nn.Module):
  def __init__(self, H, W, c_in=3, c_out=3, time_dim=256, device="cuda"):
    super().__init__()
    self.device = device
    self.time_dim = time_dim

    self.device = device
    self.time_dim = time_dim
    self.inc = DoubleConv(c_in, 64)
    self.down1 = Down(64, 128)
    self.sa1 = SelfAttention(128, (H/2, W/2))
    self.down2 = Down(128, 256)
    self.sa2 = SelfAttention(256, (H/4, W/4))
    self.down3 = Down(256, 256)
    self.sa3 = SelfAttention(256, (H/8, W/8))  # Adjusted for 3D input

    self.bot1 = DoubleConv(256, 512)
    self.bot2 = DoubleConv(512, 512)
    self.bot3 = DoubleConv(512, 256)

    self.up1 = Up(512, 128)
    self.sa4 = SelfAttention(128, (H/4, W/4))
    self.up2 = Up(256, 64)
    self.sa5 = SelfAttention(64, (H/2, W/2))
    self.up3 = Up(128, 64)
    self.sa6 = SelfAttention(64, (H, W))
    self.outc = nn.Conv2d(64, c_out, kernel_size=1)


  def pos_encoding(self, t, channels):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
    )
    pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc


  def forward(self, x, t):
    #print(f"x.shape[2]: {x.shape[2]}") #171
    #print(f"x.shape[3]: {x.shape[3]}") #141
    #print(f"x.shape[4]: {x.shape[4]}") #3
    #print(f"x.shape[1]: {x.shape[1]}") #1
    #print(f"device: {self.device}")
    #x = pad_to_even(x)
    t = t.unsqueeze(-1).type(torch.float) #maybe another squeeze for 3D?
    t = self.pos_encoding(t, self.time_dim)
    print(f"t.shape: {t.shape}")
    print(f"input x.shape: {x.shape}")

    x1 = self.inc(x)
    print(f"\n inc: {x1.shape} \n")
    x2 = self.down1(x1, t)
    print(f"\n down1: {x2.shape} \n")
    x2 = self.sa1(x2)
    print(f"\n sa1: {x2.shape} \n")
    x3 = self.down2(x2, t)
    print(f"\n down2: {x2.shape} \n")
    x3 = self.sa2(x3)
    print(f"\n sa2: {x2.shape} \n")
    x4 = self.down3(x3, t)
    print(f"\n down3: {x2.shape} \n")
    x4 = self.sa3(x4)
    print(f"\n sa3: {x2.shape} \n")

    x4 = self.bot1(x4)
    x4 = self.bot2(x4)
    x4 = self.bot3(x4)

    x = self.up1(x4, x3, t)
    print(f"\n up1: {x.shape} \n")
    x = self.sa4(x)
    print(f"\n sa4: {x.shape} \n")
    x = self.up2(x, x2, t)
    print(f"\n up2: {x.shape} \n")
    x = self.sa5(x)
    print(f"\n sa5: {x.shape} \n")
    x = self.up3(x, x1, t)
    print(f"\n up3: {x.shape} \n")
    x = self.sa6(x)
    print(f"\n sa6: {x.shape} \n")

    return output


