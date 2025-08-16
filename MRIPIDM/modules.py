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
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)




class UNet(nn.Module):
  def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
    super().__init__()

    self.device = device
    self.time_dim = time_dim

    self.inc = DoubleConv(c_in, 64)
    self.outc = nn.Conv3d(64, c_out, kernel_size=1)


  def pos_encoding_3d(self, height, width, depth, channels, device):
    assert channels % 6 == 0, "Channels must be divisible by 6 for 3D sin/cos pairs"
    c_per_axis = channels // 3  # split channels evenly among x, y, z
    def get_pos_vec(length, c_per_axis):
      inv_freq = 1.0 / (10000 ** (torch.arange(0, c_per_axis, 2, device=device).float() / c_per_axis))
      pos = torch.arange(length, device=device).float().unsqueeze(1)
      pos_enc_a = torch.sin(pos * inv_freq)
      pos_enc_b = torch.cos(pos * inv_freq)
      return torch.cat([pos_enc_a, pos_enc_b], dim=1)
    pe_z = get_pos_vec(depth, c_per_axis)[:, None, None, :]  # shape: (D,1,1,C_axis)
    pe_y = get_pos_vec(height, c_per_axis)[None, :, None, :] # shape: (1,H,1,C_axis)
    pe_x = get_pos_vec(width, c_per_axis)[None, None, :, :]  # shape: (1,1,W,C_axis)
    # Broadcast to full volume and concatenate
    pe = torch.cat([
        pe_z.expand(depth, height, width, c_per_axis),
        pe_y.expand(depth, height, width, c_per_axis),
        pe_x.expand(depth, height, width, c_per_axis),
    ], dim=-1)
    return pe.permute(3, 0, 1, 2)  # (C,D,H,W) for CNNs

  def forward(self, x, t):
    #print(f"x.shape[2]: {x.shape[2]}") #171
    #print(f"x.shape[3]: {x.shape[3]}") #141
    #print(f"x.shape[4]: {x.shape[4]}") #3
    #print(f"x.shape[1]: {x.shape[1]}") #1
    #print(f"device: {self.device}")
    t = self.pos_encoding_3d(x.shape[2], x.shape[3], x.shape[4], 6, self.device)
    x1 = self.inc(x)
    x2 = self.down1(x1, t)


    output = self.outc(x2)
    return output


