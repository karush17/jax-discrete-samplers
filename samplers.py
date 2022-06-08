import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

K = 10

def identity(key, x, tau):
    return x

def sample_gumbel(key, shape, eps=1e-20):
    U = jax.random.uniform(key, shape, minval=0, maxval=1)
    return -jnp.log(-jnp.log(U + eps) + eps)

def gumbel_softmax_sample(key, logits, temperature): 
    y = logits + sample_gumbel(key, logits.shape)
    return jax.nn.softmax( y / temperature)

def gumbel_max_sample(key, logits, temperature):
    y = logits + sample_gumbel(key, logits.shape)
    return y

def gumbel_max(key, logits, temperature):
  y = gumbel_max_sample(key, logits, temperature)
  y_arg = jnp.equal(y, jnp.max(y, 1, keepdims=True))
  return y_arg

def straight_through(key, logits, temperature):
  y = gumbel_max_sample(key, logits, temperature)
  y_hard = jnp.equal(y, jnp.max(y, 1, keepdims=True))
  y = stop_gradient(y_hard - y) + y
  return y

def gumbel_softmax(key, logits, temperature):
  y = gumbel_softmax_sample(key, logits, temperature)
  return y

def straight_through_gumbel_softmax(key, logits, temperature):
  y = gumbel_softmax_sample(key, logits, temperature)
  k = logits.shape[-1]
  y_hard = jnp.equal(y, jnp.max(y, 1, keepdims=True))
  y = stop_gradient(y_hard - y) + y
  return y

# https://github.com/nshepperd/gumbel-rao-pytorch/blob/master/gumbel_rao.py
def conditional_gumbel(key, logits, d):
  E = tfd.Exponential(jnp.ones(logits.shape)).sample([K], seed=key)
  Ei = jnp.sum(d*E, axis=-1, keepdims=True)
  Z = jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)
  adjusted = (d*(-jnp.log(Ei) + jnp.log(Z)) + (1-d)*jnp.log(E/jnp.exp(logits) + Ei/Z))
  return adjusted - logits

def gumbel_rao_monte_carlo(key, logits, temperature):
  d = jnp.equal(logits, jnp.max(logits, 1, keepdims=True))
  adjusted = logits + stop_gradient(conditional_gumbel(key, logits, d))
  surrogate = jnp.mean(jax.nn.softmax(adjusted / temperature, axis=-1), axis=0)
  y = stop_gradient(d - surrogate) + surrogate
  return y

def sample_rao(key, shape, eps=1e-20):
    U = jax.random.uniform(key, shape, minval=0, maxval=1)
    return -jnp.log(-jnp.log(U + eps) + eps)

def gumbel_rao_monte_carlo_alternate(key, logits, temperature):
  logits = jnp.repeat(jnp.expand_dims(logits, axis=0), K, axis=0)
  y = logits + sample_rao(key, logits.shape)
  y = jax.nn.softmax(y / temperature)
  y = jnp.mean(y, axis=0)
  y_hard = jnp.equal(y, jnp.max(y, 1, keepdims=True))
  y = stop_gradient(y_hard - y) + y
  return y
