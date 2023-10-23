"""Implements models for training and inference."""

from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

PRNGKey = Any

class MLP(nn.Module):
    """Implements the MLP model.
    
    Attributes:
        hidden_dims: number of hidden units.
    """
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Implements the forward pass."""
        for _, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
        return x


class Encoder(nn.Module):
    """Implements the VAE encoder.
    
    Attributes:
        hidden_dims: number of hidden units.
    """
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Implements the forward pass."""
        return MLP(self.hidden_dims)(x)


class Decoder(nn.Module):
    """Implements the VAE decoder.
    
    Attributes:
        hidden_dims: number of hidden units.
    """
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Implements the forward pass."""
        return MLP(self.hidden_dims)(x)


class VAE(nn.Module):
    """Implements the VAE model.
    
    Attributes:
        encoder_dims: number of units in the encoder.
        decoder_dims: number of units in the decoder.
        N: size of latent dimension.
        K: number of latent particles.
    """
    encoder_dims: Sequence[int]
    decoder_dims: Sequence[int]
    N: int
    K: int

    @nn.compact
    def __call__(self, key: PRNGKey, x: jnp.ndarray,
                 sampler: Any, tau: float) -> Tuple[jnp.ndarray,
                                                    jnp.ndarray, jnp.ndarray]:
        """Implements the forward pass."""
        cats = Encoder(self.encoder_dims)(x)
        cats = jnp.reshape(cats, (-1, self.K))
        q_cats = jax.nn.softmax(cats)
        log_q_cats = jnp.log(q_cats+1e-20)
        sample = jnp.reshape(sampler(key, cats, tau), (-1,self.N*self.K))
        x_hat = Decoder(self.decoder_dims)(sample)
        return q_cats, log_q_cats, x_hat
