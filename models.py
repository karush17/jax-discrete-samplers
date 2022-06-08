import os
import random
from typing import Any, Sequence, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn

PRNGKey = Any

class MLP(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
        return x
    
class Encoder(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return MLP(self.hidden_dims)(x)

class Decoder(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return MLP(self.hidden_dims)(x)

class VAE(nn.Module):
    encoder_dims: Sequence[int]
    decoder_dims: Sequence[int]
    N: int
    K: int

    @nn.compact
    def __call__(self, key, x, sampler, tau):
        cats = Encoder(self.encoder_dims)(x)
        cats = jnp.reshape(cats, (-1, self.K))
        q_cats = jax.nn.softmax(cats)
        log_q_cats = jnp.log(q_cats+1e-20)
        sample = jnp.reshape(sampler(key, cats, tau), (-1,self.N*self.K))
        x_hat = Decoder(self.decoder_dims)(sample)
        return q_cats, log_q_cats, x_hat


