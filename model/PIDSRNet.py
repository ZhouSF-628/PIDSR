import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FeedForward_P(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_P, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.con1x1 =  nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

    def forward(self, x, p):
        x = self.project_in(x)
        p = self.con1x1(p)
        x = F.gelu(x) * p
        x = self.project_out(x)
        return x

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
    
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class TransformerBlock_P(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_P, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_P(dim, ffn_expansion_factor, bias)

    def forward(self, x, p):
        x = x + self.attn(self.norm1(x))
        p = p + self.attn(self.norm1(p))
        x = x + self.ffn(self.norm2(x), self.norm2(p))

        return x

# Feature Extractor Module
class ImgFeatureExtractorModule(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(ImgFeatureExtractorModule, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class PolarFeatureExtractorModule(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(PolarFeatureExtractorModule, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.proj(x)

        return x
        

class Refinement(nn.Module):
    def __init__(self, dim):
        super(Refinement, self).__init__()

        self.refinement = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            )

    def forward(self, x):
        return self.refinement(x)

## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class PIDSRNet(nn.Module):
    def __init__(self, inp_channels=12, out_channels=12, dim=48, num_blocks=[4, 6, 6], heads=[1, 2, 4], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(PIDSRNet, self).__init__()

        # Feature Extractor
        self.img_feature_extractor = ImgFeatureExtractorModule(inp_channels, dim)
        self.polar_feature_extractor = PolarFeatureExtractorModule(6, dim)

        #------------------------ Encoder ------------------------#
        # Stage 1
        self.encoder1 = nn.Sequential(
            *[TransformerBlock_P(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type) 
             for _ in range(num_blocks[0])]
        )
        self.down1 = Downsample(dim)
        
        # Stage 2
        self.encoder2 = nn.Sequential(
            *[TransformerBlock_P(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
             for _ in range(num_blocks[1])]
        )
        self.down2 = Downsample(dim*2)\
        
        
        #------------------------ Bottleneck ------------------------#
        self.bottleneck = nn.Sequential(
            *[TransformerBlock_P(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
             for _ in range(num_blocks[2])]
        )

        #------------------------ Decoder for Demosaic ------------------------#
        # Stage 3 -> 2
        self.de_up3 = Upsample(dim*4)
        self.de_reduce_chan3 = nn.Conv2d(dim*4, dim*2, 1, bias=bias)
        self.de_decoder2 = nn.Sequential(
            *[TransformerBlock_P(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
             for _ in range(num_blocks[1])]
        )
        
        # Stage 2 -> 1
        self.de_up2 = Upsample(dim*2)
        self.de_reduce_chan2 = nn.Conv2d(dim*2, dim, 1, bias=bias)
        self.de_decoder1 = nn.Sequential(
            *[TransformerBlock_P(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
             for _ in range(num_blocks[0])]
        )

        #------------------------ Decoder for SuperResolution ------------------------#
        # Stage 3 -> 2
        self.sr_up3 = Upsample(dim*4)
        self.sr_reduce_chan3 = nn.Conv2d(dim*4, dim*2, 1, bias=bias)
        self.sr_decoder2 = nn.Sequential(
            *[TransformerBlock_P(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
             for _ in range(num_blocks[1])]
        )

        # Stage 2 -> 1
        self.sr_up2 = Upsample(dim*2)
        self.sr_decoder1 = nn.Sequential(
            *[TransformerBlock_P(dim*2, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(num_blocks[0])]
        )

        self.con1x1 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)

        #------------------------ Outputs ------------------------#
        self.mid_refine = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
             for _ in range(2)]
        )
        self.mid_output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

        self.sr_upsample = Upsample(dim*2)
        self.sr_refine = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(2)]
        )
        self.sr_output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def calculate_polar(self, img):
        I0, I45, I90, I135 = img[:,0:3], img[:,3:6], img[:,6:9], img[:,9:12]
        S1 = (I0 - I90 + 1) / 2
        S2 = (I45 - I135 + 1) / 2
        return torch.cat([S1, S2], 1)

    def forward(self, inp_img):
        polar = self.calculate_polar(inp_img)
        polar_feat = self.polar_feature_extractor(polar)                    # (B, C, H, W)

        #------------------------ Encoder ------------------------#
        img_feat = self.img_feature_extractor(inp_img)                      # (B, C, H, W)

        # Encoder 1
        x1 = self.encoder1[0](img_feat, polar_feat)                         # (B, C, H, W)
        for block in self.encoder1[1:]:
            x1 = block(x1, polar_feat)
        x1_down = self.down1(x1)                                            # (B, 2C, H/2, W/2)  
        p1_down = self.down1(polar_feat)                                    # (B, 2C, H/2, W/2)
        
        # Encoder 2
        x2 = self.encoder2[0](x1_down, p1_down)                             # (B, 2C, H/2, W/2)
        for block in self.encoder2[1:]:
            x2 = block(x2, p1_down)
        x2_down = self.down2(x2)                                            # (B, 4C, H/4, W/4)
        p2_down = self.down2(p1_down)                                       # (B, 4C, H/4, W/4)
        
        #------------------------ Bottleneck ------------------------#
        x3 = self.bottleneck[0](x2_down, p2_down)                           # (B, 4C, H/4, W/4)
        for block in self.bottleneck[1:]:
            x3 = block(x3, p2_down)

        #------------------------ Decoder for Demosaic ------------------------#
        # Stage 3 -> 2
        de_x3_up = self.de_up3(x3)                                          # (B, 2C, H/2, W/2)
        de_x3_up = self.de_reduce_chan3(torch.cat([de_x3_up, x2], 1))       # (B, 2C, H/2, W/2)
        de_x2_dec = self.de_decoder2[0](de_x3_up, p1_down)                  # (B, 2C, H/2, W/2)
        for block in self.de_decoder2[1:]:
            de_x2_dec = block(de_x2_dec, p1_down)

        # Stage 2 -> 1
        de_x2_up = self.de_up2(de_x2_dec)                                   # (B, C, H, W)
        de_x1_dec = self.de_decoder1[0](de_x2_up, polar_feat)               # (B, C, H, W)
        for block in self.de_decoder1[1:]:
            de_x1_dec = block(de_x1_dec, polar_feat)

        mid_out = self.mid_output(self.mid_refine(de_x1_dec)) + inp_img

        #------------------------ Decoder for SuperResolution ------------------------#
        # Stage 3 -> 2
        sr_x3_up = self.sr_up3(x3)                                          # (B, 2C, H/2, W/2)
        sr_x3_up = self.sr_reduce_chan3(torch.cat([sr_x3_up, x2], 1))       # (B, 2C, H/2, W/2)
        sr_x2_dec = self.sr_decoder2[0](sr_x3_up, p1_down)                  # (B, 2C, H/2, W/2)
        for block in self.sr_decoder2[1:]:
            sr_x2_dec = block(sr_x2_dec, p1_down)

        # Stage 2 -> 1
        sr_x2_up = self.sr_up2(sr_x2_dec)                                  # (B, C, H, W)
        sr_x2_up = torch.cat([sr_x2_up, de_x1_dec], 1)                     # (B, 2C, H, W)
        polar_feat = self.con1x1(polar_feat)
        sr_x1_dec = self.sr_decoder1[0](sr_x2_up, polar_feat)              # (B, 2C, H, W)
        for block in self.sr_decoder1[1:]:
            sr_x1_dec = block(sr_x1_dec, polar_feat)
  
        sr_out = self.sr_output(self.sr_refine(self.sr_upsample(sr_x1_dec))) + F.interpolate(inp_img, scale_factor=2, mode='bilinear')
        
        return mid_out, sr_out


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = PIDSRNet().to('cuda')
    inp = torch.randn(1, 12, 128, 128).to('cuda')
    out, out1  = model(inp)
    print(out.size(), out1.size())
