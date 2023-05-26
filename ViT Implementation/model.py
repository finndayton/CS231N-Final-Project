import torch
from torch import nn

from utils import image_to_patches


# Code is adapted from: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Credit goes to Brian Pulfer

class ViT(nn.Module):

    def __init__(self, chw = (3, 32, 32), n_patches = 8, n_blocks=1, hidden_dim = 16, n_heads=2, n_classes=10):
        super(ViT, self).__init__()
        self.chw = chw
        self.n_patches = n_patches
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"

        # Linear embedding step
        input_dim = chw[0] * chw[1] * chw[2] // n_patches ** 2
        self.linear_embedding = nn.Linear(input_dim, hidden_dim)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(n_patches ** 2 + 1, hidden_dim))

        # Transformer encoder blocks 
        self.transformer_blocks = nn.Sequential(*[ViTBlock(hidden_dim, n_heads) for _ in range(n_blocks)])

        # MLP. Linear then softmax
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, n_classes),
            nn.Softmax(dim=-1), # softmax over the last element 
        )

    
    # images size = [N, C, H, W]
    def forward(self, images):

        # convert the images to patches
        patches = image_to_patches(images, self.n_patches) # (N, C, H, W) -> (N, n_patches^2, C * patch_size^2)

        # get the linear embedding
        tokens = self.linear_embedding(patches) # (N, n_patches^2, hidden_dim)

        # add the class token 
        # the purpose of this is to go from (N, n_patches^2, C * patch_size^2) -> (N, n_patches^2 + 1, C * patch_size^2)
        # we need to add a (1, 1, C * patch_size^2) tensor to the beginning of the patches tensor
        tokens = torch.cat((self.class_token.repeat(patches.size(0), 1, 1), tokens), dim=1) # (N, n_patches^2 + 1, hidden_dim)

        # add in the positional embedding
        tokens += self.positional_embedding.repeat(patches.size(0), 1, 1) # (N, n_patches^2 + 1, hidden_dim)

        # pass through the transformer blocks
        out = self.transformer_blocks(tokens) # (N, n_patches^2 + 1, hidden_dim)

        # take the class token out
        class_tokens = out[:, 0, :] # (N, hidden_dim)

        # pass through the MLP
        out = self.mlp(class_tokens) # (N, n_classes)

        return out


# The entire transformer encoder. We can stack these arbitrarily many times
class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x)) # residual connection 
        out = out + self.mlp(self.norm2(out)) # residual connection
        return out
    
# Multi-head self-attention module. 
class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


    def forward_vectorized(self, sequences):
        N, seq_length, token_dim = sequences.shape
        qkv = self.qkv_mapping(sequences)  # Shape: (N, seq_length, 3 * d)
        qkv = qkv.view(N, seq_length, self.n_heads, 3 * self.d_head)  # Shape: (N, seq_length, n_heads, 3 * d_head)
        qkv = qkv.permute(0, 2, 1, 3)  # Transpose for attention computation: (N, n_heads, seq_length, 3 * d_head)
        
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)  # Split last dimension into q, k, v
        q /= self.d_head ** 0.5  # Scale q

        attention = self.softmax(q @ k.transpose(-2, -1))  # Compute scaled dot-product attention
        out = attention @ v  # Apply attention weights to v
        out = out.permute(0, 2, 1, 3).contiguous()  # Back to original shape: (N, seq_length, n_heads, d_head)
        out = out.view(N, seq_length, -1)  # Concatenate heads: (N, seq_length, d)
        
        return out