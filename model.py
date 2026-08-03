import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DSE(nn.Module):
    def __init__(self, channel, reduction=16, lambda_init=0.5):
        super(DSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channel = max(channel // reduction, 4)

        self.fc_reduce = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc_exc = nn.Linear(mid_channel, channel, bias=False)
        self.fc_inh = nn.Linear(mid_channel, channel, bias=False)
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init), dtype=torch.float32))

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc_reduce(y)

        w_exc = F.relu(self.fc_exc(y)).pow(2)
        w_inh = F.relu(self.fc_inh(y)).pow(2)
        lambda_clamped = torch.clamp(self.lambda_param, 0.0, 1.0)

        w = w_exc - lambda_clamped * w_inh
        w = torch.sigmoid(w).view(b, c, 1, 1)
        return x * w
class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2)).transpose(-1, -2)
        return u * v


class gMLPBlock(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.channel_proj1 = nn.Linear(dim, dim * 2)
        self.sgu = SpatialGatingUnit(dim * 2, seq_len)
        self.channel_proj2 = nn.Linear(dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return x + shortcut


class MultiWindowMLP(nn.Module):
    def __init__(self, dim, window_sizes=(2, 4, 8), shift=False):

        super().__init__()
        self.dim = dim
        self.window_sizes = window_sizes
        self.shift = shift 
        self.branches = nn.ModuleList([gMLPBlock(dim, ws ** 2) for ws in window_sizes])
        self.fusion_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * len(window_sizes), len(window_sizes))
        )

    def forward(self, x):
        b, c, h, w = x.shape
        branch_outputs, active_indices = [], []

        for i, ws in enumerate(self.window_sizes):
            x_current = x
            h_ori, w_ori = h, w
            pad_h, pad_w = 0, 0
            if self.shift:
                pad_h = ws // 2
                pad_w = ws // 2
                x_pad = F.pad(x_current, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
                _, _, h_pad, w_pad = x_pad.shape
                if h_pad % ws != 0:
                    extra_h = (ws - h_pad % ws) % ws
                    x_pad = F.pad(x_pad, (0, 0, 0, extra_h), mode='reflect')
                if w_pad % ws != 0:
                    extra_w = (ws - w_pad % ws) % ws
                    x_pad = F.pad(x_pad, (0, extra_w, 0, 0), mode='reflect')

                x_current = x_pad
                h_cur, w_cur = x_current.shape[2], x_current.shape[3]
            else:
                if h % ws != 0 or w % ws != 0:
                    continue
                h_cur, w_cur = h_ori, w_ori
            windows = rearrange(x_current, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=ws, p2=ws)
            processed = self.branches[i](windows)
            merged = rearrange(processed, '(b h w) (p1 p2) c -> b c (h p1) (w p2)',
                               h=h_cur // ws, w=w_cur // ws, p1=ws, p2=ws)
            if self.shift:
                merged = merged[:, :, pad_h:pad_h + h_ori, pad_w:pad_w + w_ori]

            active_indices.append(i)
            branch_outputs.append(merged)

        if not branch_outputs:
            return x

        fusion_tensor = torch.zeros(b, c * len(self.window_sizes), h, w, device=x.device)
        for i, out in enumerate(branch_outputs):
            branch_idx = active_indices[i]
            fusion_tensor[:, branch_idx * c:(branch_idx + 1) * c] = out

        weights = self.fusion_mlp(fusion_tensor).softmax(dim=1)
        fused_output = 0
        for i, out in enumerate(branch_outputs):
            branch_idx = active_indices[i]
            fused_output += weights[:, branch_idx].view(-1, 1, 1, 1) * out

        return fused_output

class DCMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=None,
                 large_window_sizes=(4, 8),
                 small_window_sizes=(2, 4), 
                 shift=False):  
        super().__init__()
        if dim is None:
            dim = out_channels 

        self.region_mlp = MultiWindowMLP(dim, large_window_sizes, shift=shift)
        self.boundary_mlp = MultiWindowMLP(dim, small_window_sizes, shift=shift)
        self.gate_conv = nn.Conv2d(3 * dim, 1, kernel_size=1)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        shortcut = x
        _, _, h, w = x.shape
        kh, kw = min(h, 3), min(w, 3)
        kh -= 1 if kh % 2 == 0 else 0
        kw -= 1 if kw % 2 == 0 else 0
        padding = (kh // 2, kw // 2)

        x_region = F.avg_pool2d(x, (kh, kw), stride=1, padding=padding)
        x_boundary = F.max_pool2d(x, (kh, kw), stride=1, padding=padding)

        gate_input = torch.cat([x, x_region, x_boundary], dim=1)
        g_map = torch.sigmoid(self.gate_conv(gate_input))

        return shortcut + g_map * self.region_mlp(x_region) + (1 - g_map) * self.boundary_mlp(x_boundary)

class Freq_Decomposer(nn.Module):
    def __init__(self, in_channels, init_sigma=0.25):
        super(Freq_Decomposer, self).__init__()
        self.matrix_cache = {}
        self.sigma = nn.Parameter(torch.full((1, in_channels, 1, 1), init_sigma))

    def get_matrices_and_grid(self, H, W, device):
        key = f"{H}_{W}_{device}"
        if key not in self.matrix_cache:
            def create_dct(N):
                n = torch.arange(N, dtype=torch.float32, device=device).reshape((1, N))
                k = torch.arange(N, dtype=torch.float32, device=device).reshape((N, 1))
                matrix = torch.sqrt(torch.tensor(2.0 / N, device=device)) * torch.cos(
                    torch.pi * k * (2 * n + 1) / (2 * N))
                matrix[0, :] = 1.0 / torch.sqrt(torch.tensor(N, device=device))
                return matrix

            dct_h = create_dct(H)
            dct_w = create_dct(W)

            u = torch.zeros(1, device=device) if W == 1 else torch.arange(W, device=device, dtype=torch.float32) / (
                        W - 1)
            v = torch.zeros(1, device=device) if H == 1 else torch.arange(H, device=device, dtype=torch.float32) / (
                        H - 1)
            v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
            dist_sq = (u_grid ** 2 + v_grid ** 2).view(1, 1, H, W)

            self.matrix_cache[key] = (dct_h, dct_w, dist_sq)

        return self.matrix_cache[key]

    def forward(self, x):
        B, C, H, W = x.size()
        original_dtype = x.dtype

        dct_matrix_h, dct_matrix_w, dist_sq = self.get_matrices_and_grid(H, W, x.device)

        x_view = x.float().view(B * C, H, W)
        dct_x = torch.matmul(torch.matmul(dct_matrix_h, x_view), dct_matrix_w.transpose(0, 1)).view(B, C, H, W)
        safe_sigma = torch.clamp(self.sigma, min=0.01, max=1.0)
        mask = torch.exp(-dist_sq / (2 * (safe_sigma ** 2)))
        f_low = dct_x * mask
        f_high = dct_x * (1.0 - mask)

        f_low_view = f_low.view(B * C, H, W)
        low_freq_feat = torch.matmul(torch.matmul(dct_matrix_h.transpose(0, 1), f_low_view), dct_matrix_w).view(B, C, H,
                                                                                                                W)

        f_high_view = f_high.view(B * C, H, W)
        high_freq_feat = torch.matmul(torch.matmul(dct_matrix_h.transpose(0, 1), f_high_view), dct_matrix_w).view(B, C,
                                                                                                                  H, W)

        return high_freq_feat.to(original_dtype), low_freq_feat.to(original_dtype)

class LowFreqChannelAttn(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 4)
        self.proj_low = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.proj_dec = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_low, g_dec):
        b, c, _, _ = x_low.size()

        feat_low = self.proj_low(x_low)
        feat_dec = self.proj_dec(g_dec)
        consensus_feat = feat_low * feat_dec

        y = self.avg_pool(consensus_feat).view(b, c)
        weight = self.channel_excitation(y).view(b, c, 1, 1)
        return x_low * weight + x_low

class HighFreqSpatialAttn(nn.Module):

    def __init__(self, in_channels, init_base=0.001, init_penalty=0.1):
        super().__init__()
        mid_channels = in_channels // 2

        self.semantic_evaluator = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid() 
        )
        self.tau_base = nn.Parameter(torch.tensor([init_base], dtype=torch.float32))
        self.tau_penalty = nn.Parameter(torch.tensor([init_penalty], dtype=torch.float32))
        self.W_high = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        self.W_dec = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_high, g_dec):
        p_sem = self.semantic_evaluator(g_dec)
        safe_base = torch.clamp(self.tau_base, min=0.0)
        safe_penalty = torch.clamp(self.tau_penalty, min=0.0)
        tau_map = safe_base + (1.0 - p_sem) * safe_penalty
        mag = torch.abs(x_high) + 1e-8
        scale = torch.relu(mag - tau_map) / mag
        x_high_clean = torch.sign(x_high) * scale * mag
        high_proj = self.W_high(x_high_clean)
        dec_proj = self.W_dec(g_dec)

        fused_state = high_proj + dec_proj
        spatial_mask = self.psi(fused_state)
        return x_high_clean * spatial_mask + x_high


class FrequencyDecoupledSkipConnection(nn.Module):
    def __init__(self, in_enc_c, in_dec_c, init_sigma=0.05):
        super().__init__()
        mid_c = in_enc_c
        self.proj_dec = nn.Sequential(
            nn.Conv2d(in_dec_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True)
        )
        self.freq_decomposer = Freq_Decomposer(mid_c, init_sigma)
        self.low_freq_attn = LowFreqChannelAttn(mid_c)
        self.high_freq_attn = HighFreqSpatialAttn(mid_c)
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x_enc, x_dec):
        if x_enc.shape[2:] != x_dec.shape[2:]:
            x_dec = F.interpolate(x_dec, size=x_enc.shape[2:], mode='bilinear', align_corners=False)
        g_dec = self.proj_dec(x_dec)
        x_high, x_low = self.freq_decomposer(x_enc)
        x_low_refined = self.low_freq_attn(x_low, g_dec)
        x_high_refined = self.high_freq_attn(x_high, g_dec)
        out = self.fusion(x_low_refined + x_high_refined)
        return out + x_enc
class ConvBlock2D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv2D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        return self.sigmoid(self.conv1(x))


class Decoder(nn.Module):
    def __init__(self, channels=[256, 128, 64, 32], bottle_ch=512):
        super().__init__()

        # Level 4
        self.Up4 = UpConv2D(bottle_ch, channels[0])
        self.CA4 = DSE(2 * channels[0],reduction=16)
        self.ConvBlock4 = ConvBlock2D(2 * channels[0], channels[0])

        # Level 3
        self.Up3 = UpConv2D(channels[0], channels[1])
        self.CA3 = DSE(2 * channels[1], reduction=16)
        self.ConvBlock3 = ConvBlock2D(2 * channels[1], channels[1])

        # Level 2
        self.Up2 = UpConv2D(channels[1], channels[2])
        self.CA2 = DSE(2 * channels[2], reduction=16)
        self.ConvBlock2 = ConvBlock2D(2 * channels[2], channels[2])

        # Level 1
        self.Up1 = UpConv2D(channels[2], channels[3])
        self.CA1 = DSE(2 * channels[3], reduction=16)
        self.ConvBlock1 = ConvBlock2D(2 * channels[3], channels[3])


        self.SDDA4 = FrequencyDecoupledSkipConnection(in_enc_c=channels[0], in_dec_c=channels[0])
        self.SDDA3 = FrequencyDecoupledSkipConnection(in_enc_c=channels[1], in_dec_c=channels[1])
        self.SDDA2 = FrequencyDecoupledSkipConnection(in_enc_c=channels[2], in_dec_c=channels[2])
        self.SDDA1 = FrequencyDecoupledSkipConnection(in_enc_c=channels[3], in_dec_c=channels[3])

        self.SA = SpatialAttention2D()

    def forward(self, skips, bottleneck):
        # ================= Level 4 =================
        d4 = self.Up4(bottleneck)
        x4 = self.SDDA4(x_enc=skips[3], x_dec=d4)
        d4 = torch.cat((x4, d4), dim=1)
        d4_ca = self.CA4(d4) * d4
        d4_sa = self.SA(d4_ca) * d4_ca
        d4 = self.ConvBlock4(d4_sa)

        # ================= Level 3 =================
        d3 = self.Up3(d4)
        x3 = self.SDDA3(x_enc=skips[2], x_dec=d3)
        d3 = torch.cat((x3, d3), dim=1)
        d3_ca = self.CA3(d3) * d3
        d3_sa = self.SA(d3_ca) * d3_ca
        d3 = self.ConvBlock3(d3_sa)

        # ================= Level 2 =================
        d2 = self.Up2(d3)
        x2 = self.SDDA2(x_enc=skips[1], x_dec=d2)
        d2 = torch.cat((x2, d2), dim=1)
        d2_ca = self.CA2(d2) * d2
        d2_sa = self.SA(d2_ca) * d2_ca
        d2 = self.ConvBlock2(d2_sa)

        # ================= Level 1 =================
        d1 = self.Up1(d2)
        x1 = self.SDDA1(x_enc=skips[0], x_dec=d1)
        d1 = torch.cat((x1, d1), dim=1)
        d1_ca = self.CA1(d1) * d1
        d1_sa = self.SA(d1_ca) * d1_ca
        d1 = self.ConvBlock1(d1_sa)

        return d4, d3, d2, d1

class model(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, base_c=32,
                 se_reduction=16, se_lambda_init=0.5,
                 input_size=256,
                 use_shift=True):  
        super().__init__()
        ch = [base_c, base_c * 2, base_c * 4, base_c * 8]
        bottle_ch = base_c * 16
        win_h4 = input_size // 4  
        win_h8 = input_size // 8   
        win_h16 = input_size // 16 
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], 3, padding=1),
            DCMLP(ch[0], ch[0],
                  large_window_sizes=(win_h4, win_h8),
                  small_window_sizes=(win_h8, win_h16),
                  shift=use_shift)
        )
        self.down1 = nn.Conv2d(ch[0], ch[0], 2, stride=2)

        self.enc2 = DCMLP(ch[0], ch[1],
                          large_window_sizes=(win_h4//2, win_h8//2),
                          small_window_sizes=(win_h8//2, win_h16//2),
                          shift=use_shift)
        self.down2 = nn.Conv2d(ch[1], ch[1], 2, stride=2)

        self.enc3 = DCMLP(ch[1], ch[2],
                          large_window_sizes=(win_h4//4, win_h8//4),
                          small_window_sizes=(win_h8//4, win_h16//4),
                          shift=use_shift)
        self.down3 = nn.Conv2d(ch[2], ch[2], 2, stride=2)

        self.enc4 = DCMLP(ch[2], ch[3],
                          large_window_sizes=(win_h4//8, win_h8//8),
                          small_window_sizes=(win_h8//8, win_h16//8),
                          shift=use_shift)
        self.down4 = nn.Conv2d(ch[3], ch[3], 2, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch[3], bottle_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(bottle_ch), nn.LeakyReLU(inplace=True),
            nn.Conv2d(bottle_ch, bottle_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(bottle_ch), nn.LeakyReLU(inplace=True),
            DSE(channel=bottle_ch, reduction=se_reduction, lambda_init=se_lambda_init)
        )

        self.decoder = Decoder(channels=[ch[3], ch[2], ch[1], ch[0]], bottle_ch=bottle_ch)

        self.out_head_main = nn.Conv2d(ch[0], out_channels, 1)
        self.out_head_aux1 = nn.Conv2d(ch[1], out_channels, 1)
        self.out_head_aux2 = nn.Conv2d(ch[2], out_channels, 1)
        self.out_head_aux3 = nn.Conv2d(ch[3], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        bottle = self.bottleneck(self.down4(e4))
        d4, d3, d2, d1 = self.decoder([e1, e2, e3, e4], bottle)
        out_main = self.out_head_main(d1)
        out_aux1 = self.out_head_aux1(d2)
        out_aux2 = self.out_head_aux2(d3)
        out_aux3 = self.out_head_aux3(d4)
        return out_main, out_aux1, out_aux2, out_aux3
