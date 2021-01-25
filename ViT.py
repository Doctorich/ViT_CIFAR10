# Implementing ViT in towards data science

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

img = Image.open('./data/hymenoptera_data/train/ants/0013035.jpg')
fig = plt.figure()
plt.imshow(img)
# plt.show()

# resize to imagenet size
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])
x = transform(img)
fig_x = x.numpy()
fig_x = np.transpose(fig_x, (1, 2, 0))
plt.imshow(fig_x)
# plt.show()

x = x.unsqueeze(0)  # add batch dim
print(x.shape)

# patch_size = 16
# patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size,
#                     s2=patch_size)
# print(patches.shape)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768,
                 img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size)) # size of (1,1,768)
        # Position embedding
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)
                                                   ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) # create it for each batch
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

print(PatchEmbedding()(x).shape)

# Multi-head attention - query, key, value >> attention matrix using q and k
 # use it to attend v
# We can use nn.MultiHeadAttention or implement our own
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # self.keys = nn.Linear(emb_size, emb_size)
        # self.queries = nn.Linear(emb_size, emb_size)
        # self.values = nn.Linear(emb_size, emb_size)
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)  # dropout = 0 means no dropout
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split k, q and v in num_heads, x.shape = [1, 197, 512]
        # keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_head)
        # queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        # values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> (qkv) b h n d', h=self.num_heads,
                        qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min  # minimum value of float32 (-inf)
            energy.mask_fill(~mask, fill_value)  # ??, ~ is a complementary operation

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis (scale v with att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out

patches_embedded = PatchEmbedding()(x)
print(MultiHeadAttention()(patches_embedded).shape)

# Residuals
 # create a nice wrapper
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwards):
        res = x
        x = self.fn(x, **kwards)
        x += res
        return x

# MLP - upsamples by a factor of expansion
class FeedForwardBlock(nn.Sequential):  # subclass nn.Sequential to avoid writing def forward
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),  # activation func used in GPT or BERT: curve in positive area, not monotonic, add probabilitic concept
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# Finally, transformer encoder block
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
                )
            ))

patches_embedded = PatchEmbedding()(x)
print(TransformerEncoderBlock()(patches_embedded).shape)

 # we can use built-in multi-head attention but it will expect 3 inputs: q, k, v >> need to subclass it
# Transformer - Transformer blocks of the number L
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])  # super().__init__  runs __init__ method of superclass

# fully connected layer after mean over the whole sequence
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

# create the final ViT architecture
class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 n_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

print(ViT()(x).shape)
summary(ViT(), (3, 224, 224), device='cpu')







