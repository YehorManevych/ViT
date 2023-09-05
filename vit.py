import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d, d_hidden, dropout):
        super().__init__()
        self.lin1 = nn.Linear(d, d_hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_hidden, d)
    
    def forward(self, z:torch.Tensor) -> torch.Tensor:
        return self.lin2(self.dropout(self.act(self.lin1(z))))

class MSA(nn.Module):
    def __init__(self, heads, d):
        super().__init__()
        self.heads = heads
        assert d % heads == 0, "The latent vector size d must be divisible by the number of heads"
        dh = int(d/heads)
        self.dh = dh
        self.qkv = nn.Parameter(torch.randn(heads * dh * 3, d)) 
        self.qkv_bias = nn.Parameter(torch.randn(heads * dh * 3))

        self.w0 = nn.Parameter(torch.randn((heads * dh, d)))
        self.w0_bias = nn.Parameter(torch.randn((heads * dh)))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch, n = x.shape[0], x.shape[1]
        #[BATCH, N, HEADS*DH*3]
        qkv = x@self.qkv.transpose(0,1)+self.qkv_bias.reshape(1,1,-1)
        # [BATCH, N, 3, HEADS, DH]
        qkv_split = qkv.reshape(batch, n, 3, self.heads, self.dh)
        # [BATCH, 3, HEADS, N, DH]
        qkv_by_head = qkv_split.permute(0, 2, 3, 1, 4)
        # [BATCH, HEADS, N, DH] each
        q, k, v = (tens.squeeze(1) for tens in qkv_by_head.split(1, dim=1))
        qkT = q@k.transpose(-1,-2) / (self.dh**0.5)
        # [BATCH, HEADS, N, N]
        A = torch.softmax(qkT, dim=-1)
        # [BATCH, N, HEADS*DH]
        Av = (A@v).permute(0,2,1,3).reshape(batch, n, self.heads*self.dh)
        return Av@self.w0.T + self.w0_bias.reshape(1,1,-1)

class EncoderBlock(nn.Module):
    def __init__(self, d, heads, dmlp, dropout, norm_eps): 
        super().__init__()
        self.ln_1 = nn.LayerNorm(d, norm_eps)
        self.msa = MSA(heads, d)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(d, norm_eps)
        self.mlp = MLP(d, dmlp, dropout)

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        x = self.ln_1(z)
        x = self.msa(x)
        x = self.dropout(x) + z
        y = self.mlp(self.ln_2(x))
        return y+x

class Encoder(nn.Module):
    def __init__(self, n_embeds, d, heads, dmlp, n_layers, dropout, norm_eps):
        super().__init__()
        self.pos_embeddings = nn.Parameter(torch.randn(1, n_embeds, d))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(*[EncoderBlock(d, heads, dmlp, dropout, norm_eps) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d, norm_eps)
        
    def forward(self, z:torch.Tensor) -> torch.Tensor:
        return self.ln(self.layers(self.dropout(z+self.pos_embeddings)))

class ViT(nn.Module):
    def __init__(self, d, image_w, patch, heads, dmlp, layers, n_classes, dropout, norm_eps):
        super().__init__()
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=d, kernel_size=patch, stride=patch)
        self.flatten = nn.Flatten(start_dim=-2)
        self.class_token = nn.Parameter(torch.randn(d))
        assert image_w % patch == 0, "Image size must be divisible by the patch size"
        n = int((image_w/patch)**2)
        self.encoder = Encoder(n+1, d, heads, dmlp, layers, dropout, norm_eps)
        self.head = nn.Linear(in_features=d, out_features=n_classes)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        z = self.flatten(self.conv_proj(x)).permute(0, 2, 1)
        expanded_class_token = self.class_token.reshape(1,1, z.shape[-1])
        z = torch.concatenate((expanded_class_token.repeat_interleave(z.shape[0], dim=0), z), dim=-2)
        return self.head(self.encoder(z)[:, 0])
