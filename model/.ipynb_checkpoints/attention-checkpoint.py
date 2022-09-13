import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


class MlpBlock(nn.Module):
    
    mlp_dim:int = 2048
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    dropout_rate: float = 0.1
    deterministic: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Applies Transformer MlpBlock module."""
        
        actual_out_dim = (inputs.shape[-1])
        x = nn.Dense(
            self.mlp_dim,            
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(
                inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=self.deterministic)
        
        output = nn.Dense(
            actual_out_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(
                x)
        output = nn.Dropout(rate=self.dropout_rate)(
            output, deterministic=self.deterministic)
        return output 
    
class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    """
    num_heads:int = 1
    qkv_dim:int = 512
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    dropout_rate: float = 0.1
    deterministic: bool = False
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self,
               inputs,
               encoder_mask=None):
    

        # Attention block.
        assert inputs.ndim == 3
        
        x = nn.LayerNorm()(inputs)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.attention_dropout_rate,
            deterministic=self.deterministic)(x, encoder_mask)

        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=self.deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm()(x)
        y = MlpBlock()(y)

        return x + y    

class Encoder(nn.Module):
    num_layers:int = 6
    @nn.compact
    def __call__(self,
                   inputs,
                   inputs_positions=None,
                   encoder_mask=None):
        """Applies Transformer model on the inputs.

        Args:
          inputs_positions: input subsequence positions for packed examples.
          encoder_mask: decoder self-attention mask.

        Returns:
          output of a transformer encoder.
        """
        last_shape = inputs.shape[-1]
        x = inputs.astype('float32')
        x = nn.Dense(512)(x)
        

        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(name=f'encoderblock_{lyr}')(x, encoder_mask)

        encoded = nn.LayerNorm(name='encoder_norm')(x)
        encoded = nn.Dense(last_shape, name='last_dense')(encoded)
        
#         encoded = encoded.reshape(encoded.shape[0], -1)
#         encoded = nn.Dense(512, name='hidden_layer')(encoded)
        
#         encoded = nn.Dense(10, name='last_dense')(encoded)
        return encoded
    