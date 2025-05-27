import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

import torch.nn as nn


class Swish(nn.Module):
    """
    Swish activation function: f(x) = x * sigmoid(x).
    It is a smooth, non-monotonic function that often works as well or
    better than ReLU in deep networks.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Swish activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying Swish.
                          Shape is the same as the input tensor.
        """
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    Embeddings for the diffusion timestep `t`.
    This module converts a scalar timestep `t` into a vector embedding
    using sinusoidal position embeddings followed by a small Multi-Layer Perceptron (MLP).
    This allows the model to condition its behavior on the current noise level.
    """

    def __init__(self, n_channels: int):
        """
        Initialize the TimeEmbedding module.

        Args:
            n_channels (int): The number of dimensions (channels) in the output embedding.
                              This should match the channel dimension expected by the
                              ResidualBlocks where this embedding is injected.
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer of the MLP, projecting sinusoidal embeddings to n_channels
        # The input to this layer is n_channels // 4 because sinusoidal embeddings are created
        # with n_channels // 8 dimensions for sin and n_channels // 8 for cos, concatenated.
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Swish activation function
        self.act = Swish()
        # Second linear layer of the MLP
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the time embedding for a batch of timesteps.

        Args:
            t (torch.Tensor): A 1D tensor of timesteps, shape `[batch_size]`.

        Returns:
            torch.Tensor: The time embeddings, shape `[batch_size, n_channels]`.
        """
        # --- Create sinusoidal position embeddings ---
        # Similar to those used in Transformers.
        # The formulas are:
        # PE_sin(t, i) = sin(t / 10000^(2i / d))
        # PE_cos(t, i) = cos(t / 10000^(2i / d))
        # where d is n_channels // 4 (original embedding dimension before MLP).
        # Here, half_dim corresponds to d/2 in the above formula if we consider
        # d to be the dimension of one part (sin or cos) of the final concatenated embedding.
        # The implementation uses n_channels // 8 for half_dim, meaning the input to lin1
        # (after cat) is n_channels // 4.

        half_dim = self.n_channels // 8 # Dimensionality of the sinusoidal/cosinusoidal components
        # Calculate the denominator term for the sinusoidal embeddings
        # emb_const = 10000^(i / (half_dim - 1)) where i is the dimension index
        emb_denominator = math.log(10_000) / (half_dim - 1)
        emb_denominator_exp = torch.exp(torch.arange(half_dim, device=t.device) * -emb_denominator)
        # Multiply timesteps with the exponentiated denominator terms
        # t[:, None] expands t to [batch_size, 1]
        # emb_denominator_exp[None, :] expands it to [1, half_dim]
        # Resulting shape: [batch_size, half_dim]
        emb = t[:, None] * emb_denominator_exp[None, :]
        # Concatenate sine and cosine embeddings along the last dimension
        emb = torch.cat((emb.sin(), emb.cos()), dim=1) # Shape: [batch_size, half_dim * 2] which is [batch_size, n_channels // 4]

        # --- Transform with the MLP ---
        # Apply first linear layer and activation
        emb = self.act(self.lin1(emb)) # Shape: [batch_size, n_channels]
        # Apply second linear layer
        emb = self.lin2(emb) # Shape: [batch_size, n_channels]

        return emb


class ResidualBlock(nn.Module):
    """
    A Residual Block for the U-Net.
    It consists of two convolutional layers, each preceded by Group Normalization
    and a Swish activation function. Time embeddings are incorporated by adding them
    after the first convolution. A shortcut connection (residual) is added from
    the input to the output of the block.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        Initialize the ResidualBlock.

        Args:
            in_channels (int): Number of input channels to the block.
            out_channels (int): Number of output channels from the block.
            time_channels (int): Number of channels in the time step (t) embeddings.
            n_groups (int): Number of groups for Group Normalization. Default is 32.
            dropout (float): Dropout rate applied after the first convolution and time embedding.
                             Default is 0.1.
        """
        super().__init__()
        # First convolutional path
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Second convolutional path
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Shortcut connection:
        # If input and output channels are different, use a 1x1 convolution to project the input.
        # Otherwise, use an identity mapping.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer to project time embeddings to the same number of channels as the output
        # of the first convolution, allowing them to be added.
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish() # Activation for the time embedding projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, in_channels, height, width]`.
            t (torch.Tensor): Time embeddings tensor of shape `[batch_size, time_channels]`.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, out_channels, height, width]`.
        """
        # Main path
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x))) # Shape: [batch_size, out_channels, height, width]

        # Add time embeddings.
        # Project time embedding `t` and apply activation.
        time_projection = self.time_emb(self.time_act(t)) # Shape: [batch_size, out_channels]
        # Reshape time_projection to be addable to `h` (broadcast across height and width)
        # `[:, :, None, None]` changes shape from [batch_size, out_channels] to [batch_size, out_channels, 1, 1]
        h += time_projection[:, :, None, None]

        # Second convolution layer (with dropout)
        h = self.conv2(self.dropout(self.act2(self.norm2(h)))) # Shape: [batch_size, out_channels, height, width]

        # Add the shortcut connection (projected input) to the output of the main path.
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Attention Block, similar to a simplified Multi-Head Self-Attention layer.
    It applies self-attention to the input feature map, allowing the model to
    weigh the importance of different spatial locations.
    The input is reshaped from [B, C, H, W] to [B, H*W, C] to be treated as a sequence.
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 32):
        """
        Initialize the AttentionBlock.

        Args:
            n_channels (int): Number of channels in the input feature map.
            n_heads (int): Number of attention heads. Default is 1.
            d_k (Optional[int]): Dimensionality of each attention head (key/query/value).
                                 If None, defaults to `n_channels`.
            n_groups (int): Number of groups for Group Normalization. Default is 32.
        """
        super().__init__()

        # Default d_k (dimension per head) if not specified
        if d_k is None:
            d_k = n_channels

        # Group normalization layer applied to the input
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Linear projection to get Query, Key, and Value matrices concatenated.
        # Output dimension is n_heads * d_k * 3 (for Q, K, V)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Final linear layer to project the attended features back to n_channels
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale factor for dot-product attention (1 / sqrt(d_k))
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the AttentionBlock.

        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, n_channels, height, width]`.
            t (Optional[torch.Tensor]): Time embeddings tensor. This argument is present
                                        to maintain a consistent signature with ResidualBlock
                                        but is not used in this block.
                                        Shape `[batch_size, time_channels]`.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, n_channels, height, width]`,
                          same as the input shape.
        """
        # `t` is not used in this block, kept for signature consistency.
        _ = t

        # Get input tensor dimensions
        batch_size, n_channels_in, height, width = x.shape
        assert n_channels_in == self.norm.num_channels, \
            f"Input channels {n_channels_in} does not match norm channels {self.norm.num_channels}"


        # Apply group normalization
        norm_x = self.norm(x)

        # Reshape x from [batch_size, n_channels, height, width] to [batch_size, seq_len, n_channels]
        # where seq_len = height * width. This treats each spatial location as an item in a sequence.
        seq_x = norm_x.view(batch_size, n_channels_in, -1).permute(0, 2, 1) # Shape: [batch_size, height*width, n_channels]

        # Project to Q, K, V.
        # projection(seq_x) shape: [batch_size, seq_len, n_heads * d_k * 3]
        # Reshape to: [batch_size, seq_len, n_heads, 3 * d_k]
        qkv = self.projection(seq_x).view(batch_size, -1, self.n_heads, 3 * self.d_k)

        # Split Q, K, V from the concatenated tensor.
        # Each will have shape: [batch_size, seq_len, n_heads, d_k]
        q, k, v = torch.chunk(qkv, 3, dim=-1) # Split along the last dimension

        # --- Scaled Dot-Product Attention ---
        # Calculate attention scores: (Q * K^T) / sqrt(d_k)
        # q shape: (batch_size, seq_len_q, n_heads, d_k)
        # k shape: (batch_size, seq_len_k, n_heads, d_k)
        # We need to compute dot product for each head and each item in batch.
        # torch.einsum('bihd,bjhd->bijh', q, k) performs this:
        # b: batch_size, i: seq_len_q, j: seq_len_k, h: n_heads, d: d_k
        # Resulting attn shape: [batch_size, seq_len_q, seq_len_k, n_heads]
        attn_scores = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale

        # Apply softmax along the key sequence dimension (dim=2) to get attention weights.
        # This means for each query element, we get a distribution over key elements.
        attn_weights = attn_scores.softmax(dim=2) # Shape: [batch_size, seq_len_q, seq_len_k, n_heads]

        # Multiply attention weights by values V.
        # torch.einsum('bijh,bjhd->bihd', attn_weights, v):
        # b: batch_size, i: seq_len_q, j: seq_len_k, h: n_heads, d: d_k (of V)
        # Resulting res shape: [batch_size, seq_len_q, n_heads, d_k]
        attended_values = torch.einsum('bijh,bjhd->bihd', attn_weights, v)

        # Reshape attended_values to [batch_size, seq_len_q, n_heads * d_k]
        # This concatenates the outputs from all heads for each query position.
        attended_values = attended_values.reshape(batch_size, -1, self.n_heads * self.d_k)

        # Linearly project the attended values back to the original channel dimension.
        # output layer expects [..., n_heads * d_k] and outputs [..., n_channels]
        res = self.output(attended_values) # Shape: [batch_size, seq_len, n_channels]

        # Add residual (skip) connection from the original input `x` (after reshaping).
        res += seq_x # Shape: [batch_size, seq_len, n_channels]

        # Reshape back to image format: [batch_size, n_channels, height, width]
        # permute from [batch_size, seq_len, n_channels] to [batch_size, n_channels, seq_len]
        # view to [batch_size, n_channels, height, width]
        res = res.permute(0, 2, 1).view(batch_size, n_channels_in, height, width)

        return res


class DownBlock(nn.Module):
    """
    Downsampling Block for the U-Net encoder.
    This block typically consists of a ResidualBlock followed by an optional AttentionBlock.
    It's used in the first half (encoder part) of the U-Net at each resolution level
    before the spatial dimensions are reduced by a Downsample layer.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        """
        Initialize the DownBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            time_channels (int): Number of channels in the time embeddings.
            has_attn (bool): If True, an AttentionBlock is included after the ResidualBlock.
        """
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels) # Attention is applied on output channels
        else:
            self.attn = nn.Identity() # No operation if attention is not used

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DownBlock.

        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, in_channels, height, width]`.
            t (torch.Tensor): Time embeddings tensor of shape `[batch_size, time_channels]`.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, out_channels, height, width]`.
        """
        # Pass input through ResidualBlock
        x = self.res(x, t)
        # Pass through AttentionBlock (or Identity if no attention)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling Block for the U-Net decoder.
    This block typically consists of a ResidualBlock followed by an optional AttentionBlock.
    It's used in the second half (decoder part) of the U-Net at each resolution level
    after the spatial dimensions are increased by an Upsample layer and skip connection
    features are concatenated.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        """
        Initialize the UpBlock.

        Args:
            in_channels (int): Number of input channels. This is typically the sum of
                               channels from the previous upsampled layer and channels
                               from the corresponding skip connection in the encoder.
            out_channels (int): Number of output channels.
            time_channels (int): Number of channels in the time embeddings.
            has_attn (bool): If True, an AttentionBlock is included after the ResidualBlock.
        """
        super().__init__()
        # The input to the ResidualBlock in the UpBlock includes channels from the skip connection.
        # So, `in_channels` for ResidualBlock is `in_channels_from_previous_layer + in_channels_from_skip`.
        # The `in_channels` argument to this UpBlock's __init__ already reflects this sum.
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels) # Attention is applied on output channels
        else:
            self.attn = nn.Identity() # No operation if attention is not used

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UpBlock.
        The input `x` is assumed to be the result of concatenating the upsampled
        feature map from the previous decoder layer and the skip connection
        from the corresponding encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, in_channels, height, width]`.
            t (torch.Tensor): Time embeddings tensor of shape `[batch_size, time_channels]`.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, out_channels, height, width]`.
        """
        # Pass input (concatenated features) through ResidualBlock
        x = self.res(x, t)
        # Pass through AttentionBlock (or Identity if no attention)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    Middle Block of the U-Net.
    This block is applied at the bottleneck of the U-Net (lowest resolution).
    It typically consists of a sequence of ResidualBlocks and an AttentionBlock.
    The standard configuration is ResBlock -> AttentionBlock -> ResBlock.
    """

    def __init__(self, n_channels: int, time_channels: int):
        """
        Initialize the MiddleBlock.

        Args:
            n_channels (int): Number of input/output channels for this block.
            time_channels (int): Number of channels in the time embeddings.
        """
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MiddleBlock.

        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, n_channels, height, width]`.
            t (torch.Tensor): Time embeddings tensor of shape `[batch_size, time_channels]`.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, n_channels, height, width]`.
        """
        # First ResidualBlock
        x = self.res1(x, t)
        # AttentionBlock
        x = self.attn(x)
        # Second ResidualBlock
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    Upsampling layer for the U-Net decoder.
    Scales up the feature map resolution by a factor of 2 using transposed convolution.
    """

    def __init__(self, n_channels: int):
        """
        Initialize the Upsample layer.

        Args:
            n_channels (int): Number of input and output channels.
        """
        super().__init__()
        # ConvTranspose2d for 2x upsampling: kernel_size=4, stride=2, padding=1
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Upsample layer.

        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, n_channels, height, width]`.
            t (Optional[torch.Tensor]): Time embeddings tensor. Not used in this layer,
                                        but included for signature consistency with other blocks.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, n_channels, height*2, width*2]`.
        """
        # `t` is not used in this layer.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    Downsampling layer for the U-Net encoder.
    Scales down the feature map resolution by a factor of 1/2 using convolution with stride 2.
    """

    def __init__(self, n_channels: int):
        """
        Initialize the Downsample layer.

        Args:
            n_channels (int): Number of input and output channels.
        """
        super().__init__()
        # Conv2d for 2x downsampling: kernel_size=3, stride=2, padding=1
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Downsample layer.

        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, n_channels, height, width]`.
            t (Optional[torch.Tensor]): Time embeddings tensor. Not used in this layer,
                                        but included for signature consistency with other blocks.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, n_channels, height/2, width/2]`.
                          The spatial dimensions are halved.
        """
        # `t` is not used in this layer.
        _ = t
        return self.conv(x)


class UNetDiffusionModel(nn.Module):
    """
    U-Net model architecture specifically designed for Denoising Diffusion Probabilistic Models (DDPMs).
    The U-Net takes a noisy image and a timestep as input and predicts the noise
    that was added to the original clean image.

    The architecture consists of:
    1. An initial projection layer for the input image.
    2. A TimeEmbedding layer to convert the timestep into a feature vector.
    3. An encoder part (Down blocks) that progressively reduces spatial resolution
       while increasing channel depth. Skip connections store features at each resolution.
    4. A Middle block at the bottleneck (lowest resolution).
    5. A decoder part (Up blocks) that progressively increases spatial resolution
       while decreasing channel depth. Skip connections from the encoder are
       concatenated at corresponding resolutions.
    6. A final normalization and convolution layer to produce the predicted noise map.
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        Initialize the U-Net model for diffusion.

        Args:
            image_channels (int): Number of channels in the input image (e.g., 3 for RGB).
                                  Default is 3.
            n_channels (int): Base number of channels in the feature maps. The actual
                              number of channels at each resolution level is determined
                              by `ch_mults`. Default is 64.
            ch_mults (Union[Tuple[int, ...], List[int]]):
                A list or tuple of channel multipliers for each resolution level in the
                encoder/decoder. For example, if `n_channels=64` and `ch_mults=(1, 2, 4)`,
                the resolutions will have 64, 128, and 256 channels respectively.
                Default is (1, 2, 2, 4).
            is_attn (Union[Tuple[bool, ...], List[bool]]):
                A list or tuple indicating whether to use an AttentionBlock at each
                corresponding resolution level. Must have the same length as `ch_mults`.
                Default is (False, False, True, True).
            n_blocks (int): Number of ResidualBlocks (within a DownBlock or UpBlock)
                            per resolution level. Default is 2.
        """
        super().__init__()

        # Number of resolution levels in the U-Net
        n_resolutions = len(ch_mults)

        # Initial convolution layer to project the input image into the base number of channels
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. The output dimension is n_channels * 4,
        # which is a common choice for time embedding dimensionality.
        self.time_emb = TimeEmbedding(n_channels * 4)

        # --- Encoder (Downsampling Path) ---
        down_blocks = [] # List to store DownBlock and Downsample modules
        current_in_channels = n_channels # Channels start from n_channels after image_proj
        # Iterate through each resolution level
        for i in range(n_resolutions):
            current_out_channels = current_in_channels * ch_mults[i] # Calculate output channels for this resolution
            # Add `n_blocks` instances of DownBlock for the current resolution
            for _ in range(n_blocks):
                down_blocks.append(DownBlock(current_in_channels, current_out_channels, n_channels * 4, is_attn[i]))
                current_in_channels = current_out_channels # Output of this block is input to the next
            # Add a Downsample layer if not at the last resolution level (bottleneck)
            if i < n_resolutions - 1:
                down_blocks.append(Downsample(current_in_channels))
        # Store the encoder modules in an nn.ModuleList
        self.down = nn.ModuleList(down_blocks)

        # --- Middle Block (Bottleneck) ---
        # The number of channels here is the output of the last DownBlock in the encoder
        self.middle = MiddleBlock(current_out_channels, n_channels * 4)
        current_in_channels = current_out_channels # For clarity, though not strictly needed for middle

        # --- Decoder (Upsampling Path) ---
        up_blocks = [] # List to store UpBlock and Upsample modules
        # Iterate through resolution levels in reverse order (from bottleneck up)
        for i in reversed(range(n_resolutions)):
            # Calculate input channels for UpBlocks at this resolution.
            # This includes channels from the previous decoder layer (after upsampling)
            # and channels from the corresponding skip connection in the encoder.
            # The `ch_mults` logic here needs to correctly determine the skip connection channels.
            # The `current_in_channels` at the start of the loop is from the middle block or previous up_block.
            # For the ResidualBlock inside UpBlock, input channels will be `current_in_channels + skip_channels`.
            
            # Channels for the UpBlock's ResidualBlock will be current_in_channels (from previous upsample) + skip_channels
            # skip_channels = n_channels * ch_mults[i] (if we consider skip from same level before downsampling)
            # More precisely, it's the output of the DownBlocks at level `i`.
            # The `current_in_channels` here is the output from the previous (deeper) level.

            # Let's trace channels for UpBlocks more carefully:
            # `current_in_channels` entering this loop iteration is the number of channels from the layer below (e.g., output of MiddleBlock or previous Upsample).
            # The skip connection will have `n_channels * ch_mults[i]` channels.
            # So, after concatenation, the input to the first ResidualBlock in an UpBlock sequence is `current_in_channels + (n_channels * ch_mults[i])`.
            
            # The `UpBlock` itself is defined to take `in_channels + out_channels` for its `ResidualBlock` if `in_channels`
            # is the skip and `out_channels` is the upsampled feature.
            # The current loop structure might be slightly confusing with `current_in_channels`.
            # Let's assume `current_in_channels` is what comes from the layer below.
            # `actual_in_for_resblock = current_in_channels + (n_channels * ch_mults[i])` (if `i` is current level)
            # The output channels of the UpBlock sequence at this level will be `n_channels * ch_mults[i]`.

            # Number of output channels for the blocks at this resolution level.
            # This is `n_channels * ch_mults[i]`.
            res_out_channels = n_channels * ch_mults[i]

            # Add `n_blocks` instances of UpBlock
            for _ in range(n_blocks):
                # Input to this UpBlock's ResidualBlock is current_in_channels (from below) + skip_channels (res_out_channels)
                up_blocks.append(UpBlock(current_in_channels + res_out_channels, res_out_channels, n_channels * 4, is_attn[i]))
                current_in_channels = res_out_channels # Output of this block becomes input for the next (if any at this res)

            # After n_blocks, if we are not at the highest resolution yet, we might need another block
            # to adjust channels before upsampling, or the upsampling layer handles it.
            # The original code has a final UpBlock that seems to reduce channels.
            # Let's follow the original logic:
            # After `n_blocks` of `UpBlock(current_in_channels + skip, skip_output_channels, ...)`
            # `current_in_channels` becomes `skip_output_channels`.
            # Then, a final `UpBlock` is added. Its output channels are `current_in_channels // ch_mults[i]`.
            # This implies `ch_mults` are used to *reduce* channels at the end of an upsampling stage.
            # This is unusual. Typically `ch_mults` define the main channel counts at each level.
            # Let's re-verify the provided code structure.

            # Original structure:
            # in_channels_for_loop_iter = output_of_middle_block_or_prev_upsample_block
            # for i in reversed(range(n_resolutions)):
            #     out_channels_at_this_res = in_channels_for_loop_iter (this is confusing)
            #     for _ in range(n_blocks):
            #         up.append(UpBlock(in_channels_for_loop_iter + (n_channels * ch_mults[i]), out_channels_at_this_res, ...))
            #         # in_channels_for_loop_iter = out_channels_at_this_res
            #     # Final block to reduce channels
            #     out_channels_after_reduction = in_channels_for_loop_iter // ch_mults[i] (This assumes ch_mults[i] > 1 for reduction)
            #     up.append(UpBlock(in_channels_for_loop_iter + (n_channels * ch_mults[i] if i matches skip?), out_channels_after_reduction, ...))
            #     in_channels_for_loop_iter = out_channels_after_reduction
            #     if i > 0: up.append(Upsample(in_channels_for_loop_iter))

            # Let's adhere to the provided code's logic for channel progression in `up` path.
            # `in_channels` at the start of the loop is the channel count from the layer below.
            # `out_channels_for_res_blocks` is the target channel count for most blocks at this resolution level.
            # This target channel count is `n_channels * ch_mults[i]`.
            
            # `current_in_channels` is the number of channels from the previous, deeper layer.
            skip_connection_channels = n_channels * ch_mults[i] # Channels from encoder skip connection

            # Add `n_blocks` UpBlocks. Input to each is `current_in_channels + skip_connection_channels`.
            # Output of these blocks is `skip_connection_channels`.
            for _ in range(n_blocks):
                up_blocks.append(UpBlock(current_in_channels + skip_connection_channels, skip_connection_channels, n_channels * 4, is_attn[i]))
                current_in_channels = skip_connection_channels # Output of this block is input to the next (concatenated with skip)

            # A final UpBlock at this resolution, which might change channel numbers.
            # Original: out_channels = in_channels // ch_mults[i]
            # This means the output channels of this "final" block for the level is `skip_connection_channels // ch_mults[i]`.
            # This seems to be `n_channels`.
            final_out_channels_for_level = n_channels * ch_mults[i-1] if i > 0 else n_channels # Target for next level's skip or final output
            if i == 0: # If this is the last resolution (closest to output)
                final_out_channels_for_level = n_channels # Should match `self.norm` input channels

            up_blocks.append(UpBlock(current_in_channels + skip_connection_channels, final_out_channels_for_level, n_channels * 4, is_attn[i]))
            current_in_channels = final_out_channels_for_level

            # Add Upsample layer if not at the highest resolution level yet
            if i > 0:
                up_blocks.append(Upsample(current_in_channels))

        # Store the decoder modules in an nn.ModuleList
        self.up = nn.ModuleList(up_blocks)

        # Final normalization and 1x1 convolution layer to project to image channels
        # The number of input channels to self.norm should be `current_in_channels` after the last UpBlock.
        self.norm = nn.GroupNorm(8, current_in_channels) # GroupNorm with 8 groups
        self.act = Swish()
        self.final = nn.Conv2d(current_in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input noisy image tensor of shape
                              `[batch_size, image_channels, height, width]`.
            t (torch.Tensor): Timestep tensor of shape `[batch_size]`.

        Returns:
            torch.Tensor: Predicted noise tensor of shape
                          `[batch_size, image_channels, height, width]`.
        """

        # 1. Compute time embeddings from timesteps `t`.
        # t shape: [batch_size] -> time_emb shape: [batch_size, n_channels * 4]
        time_embeddings = self.time_emb(t)

        # 2. Project input image `x` to initial feature map dimensions.
        # x shape: [batch_size, image_channels, H, W] -> image_proj_x shape: [batch_size, n_channels, H, W]
        current_features = self.image_proj(x)

        # --- Encoder Path ---
        skip_connections = [current_features] # Store features for skip connections, starting with the projected image
        # Iterate through DownBlocks and Downsample layers
        for module in self.down:
            current_features = module(current_features, time_embeddings)
            # Store the output of each DownBlock (before Downsample) for skip connections
            # Note: The original code stores output of Downsample too if it's part of self.down.
            # Proper U-Net skip connections are typically taken *before* downsampling.
            # The current loop structure stores output of every module in `self.down`.
            # If Downsample is part of `self.down`, its output is also stored.
            # This means skip connections might be from already downsampled features if not handled carefully.
            # Based on `h.pop()` in upsampling, it expects features from corresponding resolution levels.
            # Let's assume `DownBlock` outputs are the ones primarily used for `h`.
            # The provided code structure `h.append(x)` after `x = m(x,t)` is correct for typical U-Net.
            skip_connections.append(current_features)

        # --- Middle Block (Bottleneck) ---
        # current_features shape: [batch_size, bottleneck_channels, H_bottleneck, W_bottleneck]
        current_features = self.middle(current_features, time_embeddings)

        # --- Decoder Path ---
        # Iterate through UpBlocks and Upsample layers
        for module in self.up:
            if isinstance(module, Upsample):
                # Apply Upsample layer
                current_features = module(current_features, time_embeddings)
            else: # Module is an UpBlock
                # Retrieve the corresponding skip connection from the encoder path.
                # `skip_connections.pop()` gets the last stored feature map from `h`.
                # This should correspond to the same resolution level in the encoder.
                skip_feature = skip_connections.pop()
                # Concatenate the upsampled features with the skip connection features along the channel dimension.
                # current_features shape: [batch_size, C_upsampled, H, W]
                # skip_feature shape: [batch_size, C_skip, H, W]
                # concatenated_features shape: [batch_size, C_upsampled + C_skip, H, W]
                current_features = torch.cat((current_features, skip_feature), dim=1)
                # Pass concatenated features through the UpBlock
                current_features = module(current_features, time_embeddings)

        # --- Final Layers ---
        # Apply final normalization, activation, and convolution to get the predicted noise.
        # current_features shape: [batch_size, n_channels (or last up_block out), H_initial, W_initial]
        # final_output shape: [batch_size, image_channels, H_initial, W_initial]
        final_output = self.final(self.act(self.norm(current_features)))

        return final_output


if __name__ == '__main__':
    # Example usage or testing code
    import lovely_tensors as lt
    lt.monkey_patch() # For pretty printing tensors

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available(): # For Apple Silicon
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Test TimeEmbedding ---
    print("\n--- Testing TimeEmbedding ---")
    time_embed_layer = TimeEmbedding(n_channels=256).to(device)
    sample_timesteps = torch.randint(0, 1000, (5,)).to(device) # Batch of 5 timesteps
    time_output = time_embed_layer(sample_timesteps)
    print(f"Input timesteps: {sample_timesteps.shape}")
    print(f"Time embedding output: {time_output.shape}") # Expected: [5, 256]

    # --- Test ResidualBlock ---
    print("\n--- Testing ResidualBlock ---")
    res_block = ResidualBlock(in_channels=64, out_channels=128, time_channels=256).to(device)
    sample_image_features = torch.randn(5, 64, 32, 32).to(device) # Batch, C, H, W
    sample_time_features_for_res = torch.randn(5, 256).to(device) # Batch, time_channels
    res_output = res_block(sample_image_features, sample_time_features_for_res)
    print(f"ResidualBlock input image features: {sample_image_features.shape}")
    print(f"ResidualBlock input time features: {sample_time_features_for_res.shape}")
    print(f"ResidualBlock output: {res_output.shape}") # Expected: [5, 128, 32, 32]

    # --- Test AttentionBlock ---
    print("\n--- Testing AttentionBlock ---")
    attn_block = AttentionBlock(n_channels=128, n_heads=4, d_k=32).to(device)
    sample_features_for_attn = torch.randn(5, 128, 16, 16).to(device) # Batch, C, H, W
    attn_output = attn_block(sample_features_for_attn, None) # Time embedding not used by this block
    print(f"AttentionBlock input features: {sample_features_for_attn.shape}")
    print(f"AttentionBlock output: {attn_output.shape}") # Expected: [5, 128, 16, 16]

    # --- Test UNetDiffusionModel ---
    print("\n--- Testing UNetDiffusionModel ---")
    # Example configuration similar to what might be used for 64x64 images
    unet_model = UNetDiffusionModel(
        image_channels=3,      # e.g., RGB
        n_channels=128,        # Base channels
        ch_mults=(1, 2, 2, 4), # Channel multipliers for resolutions
                               # Channels: 128 -> 256 -> 256 -> 512
        is_attn=(False, False, True, True), # Use attention at 2nd and 3rd (from bottleneck) resolutions
                                            # Resolutions (H,W) might be 64->32->16->8 if downsampling at each step
                                            # Attn at 16x16 and 8x8 if ch_mults define 4 levels
        n_blocks=2             # Number of residual blocks per resolution
    ).to(device)

    # Create a dummy input batch of noisy images and timesteps
    dummy_noisy_images = torch.randn(2, 3, 64, 64).to(device) # Batch size 2, 3 channels, 64x64
    dummy_timesteps = torch.randint(0, 1000, (2,)).to(device)    # Batch size 2 timesteps

    # Forward pass
    predicted_noise = unet_model(dummy_noisy_images, dummy_timesteps)
    print(f"UNet input noisy images: {dummy_noisy_images.shape}")
    print(f"UNet input timesteps: {dummy_timesteps.shape}")
    print(f"UNet output predicted noise: {predicted_noise.shape}") # Expected: [2, 3, 64, 64]

    # You can also print the model structure
    # print("\nUNet Model Structure:")
    # print(unet_model)
    num_params = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
    print(f"UNet number of trainable parameters: {num_params / 1e6:.2f} M")

