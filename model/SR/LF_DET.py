'''
@article{cong2023lfdet,
  title={Exploiting Spatial and Angular Correlations With Deep Efficient Transformers for Light Field Image Super-Resolution},
  author={Cong, Ruixuan and Sheng, Hao and Yang, Da and Cui, Zhenglong and Chen, Rongshan},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        args.channels = 64
        args.ang_num_heads = 4
        args.spa_num_heads = 4
        args.ang_mlp_ratio = 4
        args.spa_mlp_ratio = 4
        args.depth = 4
        args.ang_sr_ratio = 1
        args.spa_sr_ratio = 2
        args.spa_trans_num = 2
        args.attn_drop_rate = 0.0
        args.drop_rate = 0.0
        args.drop_path_rate = 0.1
        args.patch_size = 32

        self.channels = args.channels
        self.angRes = args.angRes_in
        self.factor = args.scale_factor
        self.patch_size = args.patch_size
        self.num_ang = self.angRes * self.angRes
        self.num_spa = self.patch_size * self.patch_size

        # step1: Local Feature Extraction
        self.conv_init0 = nn.Sequential(nn.Conv2d(1, self.channels, kernel_size=3, padding=1, dilation=1, bias=False))
        self.conv_init_spa = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # step2: Spatial-Angular Global Feature Extraction
        self.blocks = nn.ModuleList()
        iter_path = args.spa_trans_num
        total_depth = args.depth * iter_path
        dpr = [x.item() for x in torch.linspace(0, args.drop_path_rate, total_depth)]
        for i in range(0, args.depth):
            dpr_iter = dpr[i * iter_path:(i + 1) * iter_path]
            self.blocks.append(
                MixTransfomerBlock(self.channels, args.ang_num_heads, args.spa_num_heads, args.ang_mlp_ratio,
                                   args.spa_mlp_ratio, args.spa_trans_num, ang_sr_ratio=args.ang_sr_ratio,
                                   spa_sr_ratio=args.spa_sr_ratio,
                                   qkv_bias=True, qk_scale=None, drop=args.drop_rate, attn_drop=args.attn_drop_rate,
                                   drop_path=dpr_iter, act_layer=nn.GELU, norm_layer=nn.LayerNorm))

        # step3: Hierarchical Feature Aggregation & Upsampling
        self.mla = MLA(self.channels)
        self.upsampling = nn.Sequential(
            nn.Conv2d(self.channels * args.depth // 2, self.channels * self.factor ** 2, kernel_size=1, padding=0,
                      dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, lr, data_info=None):
        batch, _, _, _ = lr.size()
        lr_upscale = interpolate(lr, self.angRes, scale_factor=self.factor, mode='bicubic')
        lr = rearrange(lr, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        lr = rearrange(lr, 'b c a h w -> (b a) c h w')
        # step1: Local Feature Extraction
        init_feature = self.conv_init0(lr)
        init_feature = self.conv_init_spa(init_feature) + init_feature

        tokens = rearrange(init_feature.flatten(2), 'b c hw -> b hw c')
        feature = rearrange(tokens, 'b (h w) c -> b c h w', h=self.patch_size, w=self.patch_size)
        hierarchical_feature = []
        # step2: Spatial-Angular Global Feature Extraction
        for (i, blk) in enumerate(self.blocks):
            feature = blk(feature, self.angRes, self.patch_size)
            hierarchical_feature.append(feature)

        # step3: Hierarchical Feature Aggregation & Upsampling
        fusion_feature = self.mla(hierarchical_feature[0], hierarchical_feature[1], hierarchical_feature[2],
                                  hierarchical_feature[3])
        buffer = self.upsampling(fusion_feature)
        buffer = rearrange(buffer, '(b a1 a2) c h w -> b (a1 a2) c h w', a1=self.angRes, a2=self.angRes)
        final_buffer = rearrange(buffer, 'b (a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        out = final_buffer + lr_upscale
        return out


class DWConv(nn.Module):
    def __init__(self, dim=64):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):     
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))) 
        x = self.dwconv(x)         
        x = x.flatten(2).transpose(1, 2)       
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)  
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)     
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)   
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   
        
        if self.sr_ratio > 1:   
            x_ = x.permute(0, 2, 1).reshape(B, C, int(math.sqrt(N)), int(math.sqrt(N)))    
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_)                             
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:   
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]         
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#NOTE: MixTransfomerBlock is spatial-angular separable transformer encoder
class MixTransfomerBlock(nn.Module):
    def __init__(self,embed_dim,ang_num_heads,spa_num_heads,ang_mlp_ratio,spa_mlp_ratio,spa_trans_num,ang_sr_ratio=1,spa_sr_ratio=1,
                qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # K cascaded spatial transformers
        self.spa_Transformer_Blocks = nn.ModuleList()
        self.spa_trans_num = spa_trans_num
        for i in range(spa_trans_num):
            self.spa_Transformer_Blocks.append(Block(dim=embed_dim,num_heads=spa_num_heads,mlp_ratio=spa_mlp_ratio,qkv_bias=qkv_bias,drop=drop,
                            attn_drop=attn_drop,drop_path=drop_path[i],sr_ratio=spa_sr_ratio,act_layer=act_layer,norm_layer=norm_layer))
        # Three parallal branches angular transform
        self.ang_Transformer_Blocks = nn.ModuleList()
        for i in range(3):
            self.ang_Transformer_Blocks.append(Block(dim=embed_dim,num_heads=ang_num_heads,mlp_ratio=ang_mlp_ratio,qkv_bias=qkv_bias,drop=drop,
                            attn_drop=attn_drop,drop_path=drop_path[-1],sr_ratio = ang_sr_ratio, act_layer=act_layer,norm_layer=norm_layer))
        # Spatial Attention Fusion
        self.cal = nn.Conv2d(embed_dim*3, 3, kernel_size=1, stride=1)
        
    def forward(self, feature, angRes, patch_size):
        feature_tokens = rearrange(feature, 'b c h w-> b (h w) c')
        # K cascaded spatial transformers
        spa_tokens = feature_tokens
        for i in range(self.spa_trans_num):
            spa_tokens = self.spa_Transformer_Blocks[i](spa_tokens)     
        
        feature = rearrange(spa_tokens, 'b (h w) c -> b c h w ', h=patch_size, w=patch_size)
        feature = rearrange(feature, '(b a1 a2) c h w -> b c (h a1) (w a2)', a1=angRes, a2=angRes)
        [B, C, AH, AW] = feature.size()
        
        # Three parallal branches angular transform
        features = []
        for num_windows in range(3):   # num_windows is M
            ang_window_size = (num_windows+1)*angRes
            ang_stride = ang_window_size
            if (AH - ang_window_size) % ang_stride != 0:       
                # Partial areas around right and bottom need overlapping. 
                rest = AH - ang_window_size - (AH - ang_window_size)//ang_stride*ang_stride
                final_feature = torch.zeros_like(feature)
                total_times = torch.zeros_like(torch.empty(1,C,AH,AW)).cuda() 
                windows1 = torch.nn.functional.unfold(feature[:,:,:AH-rest,:AW-rest],(ang_window_size, ang_window_size),padding=0,stride=ang_stride)    
                windows2 = torch.nn.functional.unfold(feature[:,:,AH-ang_window_size:AH,:AW-rest],(ang_window_size, ang_window_size),padding=0,stride=ang_stride) 
                windows3 = torch.nn.functional.unfold(feature[:,:,:AH-rest,AW-ang_window_size:AW],(ang_window_size, ang_window_size),padding=0,stride=ang_stride) 
                windows4 = feature[:,:,AH-ang_window_size:AH,AW-ang_window_size:AW].unsqueeze(1)
                windows1 = rearrange(windows1,'b (c k1 k2) l -> b l c k1 k2',c=C,k1=ang_window_size,k2=ang_window_size)  
                num_windows1 = windows1.size()[1]
                windows2 = rearrange(windows2,'b (c k1 k2) l -> b l c k1 k2',c=C,k1=ang_window_size,k2=ang_window_size)  
                num_windows2 = windows2.size()[1]
                windows3 = rearrange(windows3,'b (c k1 k2) l -> b l c k1 k2',c=C,k1=ang_window_size,k2=ang_window_size)  
                num_windows3 = windows3.size()[1]
                num_windows4 = 1
                total_windows = torch.cat((windows1,windows2,windows3,windows4),1)
                
                ang_features = self.ang_Transformer_Blocks[num_windows](rearrange(total_windows,'b l c k1 k2 -> (b l) (k1 k2) c'))  
                ang_features = rearrange(ang_features,'(b l) (k1 k2) c -> b l c k1 k2 ',b=B,k1=ang_window_size,k2=ang_window_size)
                
                windows1 = rearrange(ang_features[:,0:num_windows1,:,:,:],'b l c k1 k2 -> b (c k1 k2) l')
                windows2 = rearrange(ang_features[:,num_windows1:num_windows1+num_windows2,:,:,:],'b l c k1 k2 -> b (c k1 k2) l')
                windows3 = rearrange(ang_features[:,num_windows1+num_windows2:num_windows1+num_windows2+num_windows3,:,:,:],'b l c k1 k2 -> b (c k1 k2) l')
                windows4 = ang_features[:,num_windows1+num_windows2+num_windows3:,:,:,:].squeeze(1)
                final_feature[:,:,:AH-rest,:AW-rest] += torch.nn.functional.fold(windows1, (AH-rest, AW-rest), (ang_window_size, ang_window_size), padding=0,stride=ang_stride)  
                final_feature[:,:,AH-ang_window_size:AH,:AW-rest] += torch.nn.functional.fold(windows2, (ang_window_size, AW-rest), (ang_window_size, ang_window_size), padding=0,stride=ang_stride) 
                final_feature[:,:,:AH-rest,AW-ang_window_size:AW] += torch.nn.functional.fold(windows3, (AH-rest, ang_window_size), (ang_window_size, ang_window_size), padding=0,stride=ang_stride) 
                final_feature[:,:,AH-ang_window_size:AH,AW-ang_window_size:AW] += windows4
                
                # count overlapping area
                time1 = torch.ones_like(torch.empty(1,C,AH-rest,AW-rest)).cuda()
                time1 = torch.nn.functional.unfold(time1,(ang_window_size, ang_window_size),padding=0,stride=ang_stride)
                total_times[:,:,:AH-rest,:AW-rest] += torch.nn.functional.fold(time1, (AH-rest, AW-rest), (ang_window_size, ang_window_size), padding=0,stride=ang_stride)
                time2 = torch.ones_like(torch.empty(1,C,ang_window_size,AW-rest)).cuda()
                time2 = torch.nn.functional.unfold(time2,(ang_window_size, ang_window_size),padding=0,stride=ang_stride)
                total_times[:,:,AH-ang_window_size:AH,:AW-rest] += torch.nn.functional.fold(time2, (ang_window_size,AW-rest), (ang_window_size, ang_window_size), padding=0,stride=ang_stride)
                time3 = torch.ones_like(torch.empty(1,C,AH-rest,ang_window_size)).cuda()
                time3 = torch.nn.functional.unfold(time3,(ang_window_size, ang_window_size),padding=0,stride=ang_stride)
                total_times[:,:,:AH-rest,AW-ang_window_size:AW] += torch.nn.functional.fold(time3, (AH-rest, ang_window_size), (ang_window_size, ang_window_size), padding=0,stride=ang_stride)
                total_times[:,:,AH-ang_window_size:AH,AW-ang_window_size:AW] += 1
                features.append(final_feature/total_times)
            else:
                windows = torch.nn.functional.unfold(feature,(ang_window_size, ang_window_size),padding=0,stride=ang_stride)     
                windows = rearrange(windows,'b (c k1 k2) l -> (b l) c k1 k2',c=C,k1=ang_window_size,k2=ang_window_size)   
                ang_features = self.ang_Transformer_Blocks[num_windows](rearrange(windows,'b c k1 k2 -> b (k1 k2) c')) 
                ang_features = rearrange(ang_features,'(b l) (k1 k2) c -> b (c k1 k2) l',b=B,k1=ang_window_size,k2=ang_window_size)
                final_feature = torch.nn.functional.fold(ang_features, (AH, AW), (ang_window_size, ang_window_size), padding=0,stride=ang_stride) 
                time = torch.ones_like(torch.empty(1,C,AH,AW)).cuda()
                time = torch.nn.functional.unfold(time,(ang_window_size, ang_window_size),padding=0,stride=ang_stride)
                times = torch.nn.functional.fold(time, (AH, AW), (ang_window_size, ang_window_size), padding=0,stride=ang_stride)
                features.append(final_feature/times)
        
        # Spatial Attention Fusion        
        attn = self.cal(torch.cat(features,1))    
        attn = attn.softmax(dim=1)    
        final_feature = (attn.unsqueeze(2) * torch.stack(features,1)).sum(dim=1)       
        final_feature = rearrange(final_feature,'b c (h a1) (w a2) -> b c (a1 a2) h w',a1=angRes,a2=angRes,h=patch_size,w=patch_size)
        final_feature = rearrange(final_feature,'b c a h w -> (b a) c h w')
        return final_feature


# NOTE: MLA is Hierarchical Feature Aggregation Module
# If the number of transformer encoder (i.e., N) is changed, this model needs to be modified manually.
# Here take N = 4 as example.
class MLA(nn.Module):          
    def __init__(self, channel):
        super(MLA, self).__init__()
        self.conv_1_fuse = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2_fuse = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3_fuse = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_4_fuse = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv_1_fine = nn.Sequential(nn.Conv2d(channel, channel//2, 3, padding=1, bias=False),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(channel//2, channel//2, 3, padding=1, bias=False),
                                   nn.LeakyReLU(0.1, inplace=True))
        self.conv_2_fine = nn.Sequential(nn.Conv2d(channel, channel//2, 3, padding=1, bias=False),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(channel//2, channel//2, 3, padding=1, bias=False),
                                   nn.LeakyReLU(0.1, inplace=True))
        self.conv_3_fine = nn.Sequential(nn.Conv2d(channel, channel//2, 3, padding=1, bias=False),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(channel//2, channel//2, 3, padding=1, bias=False),
                                   nn.LeakyReLU(0.1, inplace=True))
        self.conv_4_fine = nn.Sequential(nn.Conv2d(channel, channel//2, 3, padding=1, bias=False),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(channel//2, channel//2, 3, padding=1, bias=False),
                                   nn.LeakyReLU(0.1, inplace=True))

    def forward(self, feature1,feature2,feature3,feature4):   
        feature4_fuse = feature4
        feature3_fuse = feature3 + feature4_fuse
        feature2_fuse = feature2 + feature3_fuse
        feature1_fuse = feature1 + feature2_fuse
        
        feature4_fuse = self.lrelu(self.conv_4_fuse(feature4_fuse))
        feature3_fuse = self.lrelu(self.conv_3_fuse(feature3_fuse))
        feature2_fuse = self.lrelu(self.conv_2_fuse(feature2_fuse))
        feature1_fuse = self.lrelu(self.conv_1_fuse(feature1_fuse))
        
        feature4_fine = self.conv_4_fine(feature4_fuse)
        feature3_fine = self.conv_3_fine(feature3_fuse)
        feature2_fine = self.conv_2_fine(feature2_fuse)
        feature1_fine = self.conv_1_fine(feature1_fuse)
        
        fuse_feature = torch.cat([feature1_fine, feature2_fine, feature3_fine, feature4_fine], dim=1)  
        
        return fuse_feature              

        
def interpolate(x, angRes, scale_factor, mode):    
    [B, _, H, W] = x.size()
    h = H // angRes         
    w = W // angRes     
    x_upscale = x.view(B, 1, angRes, h, angRes, w) 
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)  
    x_upscale = F.interpolate(x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False)  
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)           
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor, W * scale_factor) 

    return x_upscale


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
        

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, data_info=None):
        loss = self.criterion_Loss(SR, HR)

        return loss


def weights_init(m):
    pass

