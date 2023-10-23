"""Implements discrete samplers."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
PRNGKey = Any
K = 10

def identity(key: PRNGKey, x: jnp.ndarray, tau: float) -> jnp.ndarray:
    """Implements the identity sampler."""
    return x

def sample_gumbel(key: PRNGKey, shape: Tuple[int, int],
                  eps: float =1e-20) -> jnp.ndarray:
    """Implements the gumbel sampler."""
    U = jax.random.uniform(key, shape, minval=0, maxval=1)
    return -jnp.log(-jnp.log(U + eps) + eps)

def gumbel_softmax_sample(key: PRNGKey,
                          logits: jnp.ndarray,
                          temperature: float) -> jnp.ndarray:
    """Implements the softmax gumble sampler.""" 
    y = logits + sample_gumbel(key, logits.shape)
    return jax.nn.softmax( y / temperature)

def gumbel_max_sample(key: PRNGKey, logits: jnp.ndarray,
                      temperature: float) -> jnp.ndarray:
    """Implements the gumble max sampler."""
    y = logits + sample_gumbel(key, logits.shape)
    return y

def gumbel_max(key: PRNGKey, logits: jnp.ndarray,
               temperature: float) -> jnp.ndarray:
    """Implements the gumbel max sampler."""
    y = gumbel_max_sample(key, logits, temperature)
    y_arg = jnp.equal(y, jnp.max(y, 1, keepdims=True))
    return y_arg

def straight_through(key: PRNGKey, logits: jnp.ndarray,
                     temperature: float) -> jnp.ndarray:
    """Implements the straight through sampler."""
    y = gumbel_max_sample(key, logits, temperature)
    y_hard = jnp.equal(y, jnp.max(y, 1, keepdims=True))
    y = stop_gradient(y_hard - y) + y
    return y

def gumbel_softmax(key: PRNGKey, logits: jnp.ndarray,
                   temperature: float) -> jnp.ndarray:
    """Implements the gumbel softmax sampler."""
    y = gumbel_softmax_sample(key, logits, temperature)
    return y

def straight_through_gumbel_softmax(key: PRNGKey,
                                    logits: jnp.ndarray,
                                    temperature: float) -> jnp.ndarray:
    """Implements the straight through gumbel softmax sampler."""
    y = gumbel_softmax_sample(key, logits, temperature)
    y_hard = jnp.equal(y, jnp.max(y, 1, keepdims=True))
    y = stop_gradient(y_hard - y) + y
    return y

def conditional_gumbel(key: PRNGKey, logits: jnp.ndarray,
                       d: int) -> jnp.ndarray:
    """Implements the conditional gumbel sampler."""
    E = tfd.Exponential(jnp.ones(logits.shape)).sample([K], seed=key)
    Ei = jnp.sum(d*E, axis=-1, keepdims=True)
    Z = jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)
    adjusted = (d*(-jnp.log(Ei) + jnp.log(Z)) +
                (1-d)*jnp.log(E/jnp.exp(logits) + Ei/Z))
    return adjusted - logits

def gumbel_rao_monte_carlo(key: PRNGKey, logits: jnp.ndarray,
                           temperature: float) -> jnp.ndarray:
    """Implements the gumbel rao monte carlo sampler."""
    d = jnp.equal(logits, jnp.max(logits, 1, keepdims=True))
    adjusted = logits + stop_gradient(conditional_gumbel(key, logits, d))
    surrogate = jnp.mean(jax.nn.softmax(adjusted / temperature, axis=-1), axis=0)
    y = stop_gradient(d - surrogate) + surrogate
    return y

def sample_rao(key: PRNGKey, shape: Tuple[int, int],
               eps: float = 1e-20) -> jnp.ndarray:
    """Implements the rao sampler."""
    U = jax.random.uniform(key, shape, minval=0, maxval=1)
    return -jnp.log(-jnp.log(U + eps) + eps)

def gumbel_rao_monte_carlo_alternate(key: PRNGKey, logits: jnp.ndarray,
                                     temperature: float) -> jnp.ndarray:
    """Implements the gumbel rao monte carlo sampler."""
    logits = jnp.repeat(jnp.expand_dims(logits, axis=0), K, axis=0)
    y = logits + sample_rao(key, logits.shape)
    y = jax.nn.softmax(y / temperature)
    y = jnp.mean(y, axis=0)
    y_hard = jnp.equal(y, jnp.max(y, 1, keepdims=True))
    y = stop_gradient(y_hard - y) + y
    return y
