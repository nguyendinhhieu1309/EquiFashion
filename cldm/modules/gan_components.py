import torch
import torch.nn as nn

# Generator for latent space
class LatentGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, features_g=64, context_dim=1024):
        super(LatentGenerator, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, features_g * 4),
            nn.SiLU(),
            nn.Linear(features_g * 4, features_g * 4),
        )
        
        self.context_projection = nn.Linear(context_dim, features_g * 4)

        # in_channels * 2 for z_t and u, features_g * 8 for tiled time and context embeddings
        self.net = nn.Sequential(
            # Initial layer
            nn.Conv2d(in_channels * 2 + features_g * 8, features_g, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Downsampling
            self._block(features_g, features_g * 2, 4, 2, 1),
            self._block(features_g * 2, features_g * 4, 4, 2, 1),

            # Upsampling
            self._upsample_block(features_g * 4, features_g * 2, 4, 2, 1), # No need to add time/context emb here
            self._upsample_block(features_g * 2, features_g, 4, 2, 1),

            # Output layer
            nn.ConvTranspose2d(features_g, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def _upsample_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, z_t, u, c, t):
        # z_t: noisy latent (batch_size, channels, H, W)
        # u: stochastic variation (batch_size, channels, H, W)
        # c: multimodal conditioning; shape can be:
        #    - (batch_size, context_dim) or
        #    - (batch_size, seq_len, context_dim) as in cross-attention.
        # t: timestep (batch_size,)

        # If c comes from cross-attention (B, L, D), pool over tokens.
        if c.dim() == 3:
            # simple mean pooling over sequence dimension
            c = c.mean(dim=1)

        # Embed timestep and context
        t_emb_input = t.float().unsqueeze(1)  # (batch_size, 1)
        t_emb = self.time_embed(t_emb_input)  # (batch_size, features_g*4)
        c_emb = self.context_projection(c)    # (batch_size, features_g*4)

        # Reshape t_emb and c_emb to match spatial dimensions for concatenation
        t_emb_reshaped = t_emb.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        c_emb_reshaped = c_emb.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Tile t_emb and c_emb to match spatial dimensions of z_t and u
        t_emb_tiled = t_emb_reshaped.expand(-1, -1, z_t.shape[2], z_t.shape[3])
        c_emb_tiled = c_emb_reshaped.expand(-1, -1, z_t.shape[2], z_t.shape[3])

        # Concatenate all inputs along channel dimension
        x = torch.cat([z_t, u, t_emb_tiled, c_emb_tiled], dim=1)
        
        return self.net(x)

# Discriminator for latent space
class LatentDiscriminator(nn.Module):
    def __init__(self, in_channels, features_d=64, context_dim=1024):
        super(LatentDiscriminator, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, features_d * 4),
            nn.SiLU(),
            nn.Linear(features_d * 4, features_d * 4),
        )
        
        self.context_projection = nn.Linear(context_dim, features_d * 4)

        # in_channels for z, features_d * 8 for tiled time and context embeddings
        self.net = nn.Sequential(
            # Initial layer
            nn.Conv2d(in_channels + features_d * 8, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Downsampling
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),

            # Output layer
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, z, c, t):
        # z: latent vector (batch_size, channels, H, W)
        # c: multimodal conditioning; shape can be:
        #    - (batch_size, context_dim) or
        #    - (batch_size, seq_len, context_dim) as in cross-attention.
        # t: timestep (batch_size,)

        # If c comes from cross-attention (B, L, D), pool over tokens.
        if c.dim() == 3:
            c = c.mean(dim=1)

        # Embed timestep and context
        t_emb_input = t.float().unsqueeze(1)  # (batch_size, 1)
        t_emb = self.time_embed(t_emb_input)  # (batch_size, features_d*4)
        c_emb = self.context_projection(c)    # (batch_size, features_d*4)

        # Reshape t_emb and c_emb to match spatial dimensions for concatenation
        t_emb_reshaped = t_emb.unsqueeze(-1).unsqueeze(-1)
        c_emb_reshaped = c_emb.unsqueeze(-1).unsqueeze(-1)

        # Tile t_emb and c_emb to match spatial dimensions of z
        t_emb_tiled = t_emb_reshaped.expand(-1, -1, z.shape[2], z.shape[3])
        c_emb_tiled = c_emb_reshaped.expand(-1, -1, z.shape[2], z.shape[3])

        x = torch.cat([z, t_emb_tiled, c_emb_tiled], dim=1)
        
        return self.net(x)
