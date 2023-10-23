"""Implements the main training protocol."""

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import optax
import functools
import samplers

from tensorflow.keras.datasets import mnist
from typing import Any, Tuple, Dict
from absl import app, flags
from models import VAE
from common import Model, PRNGKey
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions

FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 1000, 'training steps')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_integer('batch_size', 100, 'training batch size')
flags.DEFINE_integer('N', 10, 'number of classes')
flags.DEFINE_integer('K', 30, 'number of categoricals')
flags.DEFINE_integer('seed', 32, 'random seed')
flags.DEFINE_float('max_temp', 1, 'maximum temperature')
flags.DEFINE_float('min_temp', 0.5, 'minimum temperature')
flags.DEFINE_float('anneal', 0.00003, 'temperature anneal rate')
flags.DEFINE_string('sampler', 'identity', 'sampler')

@functools.partial(jax.jit, static_argnums=(0,4,5,6))
def _update(sampler: Any, key: PRNGKey, model: Any,
            x: jnp.ndarray, K: int, N: int,
            tau: float) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Updates the model.
    
    Args:
        sampler: sampler function for obtaining latent samples.
        key: JAX random key.
        model: VAE model for training.
        x: input observations to the VAE encoder.
        K: number of latent particles.
        N: number of latent dimensions.
        tau: temperature parameter.
    
    Returns:
        rng: new JAX random key.
        new_model: updated model.
        info: logging info.
    """
    rng, key = jax.random.split(key)
    def loss_fn(params):
        q_cats, log_q_cats, x_hat = model.apply_fn({'params': params}, key, x, sampler, tau)
        p_x = tfd.Bernoulli(logits=x_hat)
        kl_temp = q_cats*(log_q_cats - jnp.log(1.0/K))
        kl_temp = jnp.reshape(kl_temp, (-1, N, K))
        kl = jnp.sum(kl_temp, axis=(1,2))
        ll = -jnp.sum(p_x.log_prob(x), axis=1)
        elbo = ll + kl
        loss = jnp.mean(elbo)
        return loss, {'loss': loss, 'll': jnp.mean(ll), 'kl': jnp.mean(kl)}

    new_model, info = model.apply_gradient(loss_fn)
    return rng, new_model, info

def main(_) -> None:
    """Main function for execution."""
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, key = jax.random.split(rng, 2)
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = np.array(x_train, np.float32)
    x_train = x_train.reshape([-1, 784]) / 255.
    data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    data = data.repeat().shuffle(5000).batch(FLAGS.batch_size).prefetch(1)
    np_x = x_train[:FLAGS.batch_size]
    tau = FLAGS.max_temp
    sampler = getattr(samplers, FLAGS.sampler)
    model_def = VAE([512, 256, FLAGS.K*FLAGS.N],
                    [FLAGS.K*FLAGS.N, 256, 512, 784], FLAGS.N, FLAGS.K)
    model = Model.create(model_def, inputs=[key, rng, np_x, sampler, tau],
                         tx=optax.adam(learning_rate=FLAGS.lr))
    rand_x = np.zeros((100,784))
    rand_x[range(100), np.random.choice(784,100)] = 1
    for i in range(1,FLAGS.epochs+1):
        for step, (np_x, _) in enumerate(data.take(6000), 1):
            rng, new_model, info = _update(sampler, rng, model,
                                           jnp.array(np_x),
                                           FLAGS.N, FLAGS.K, tau)
            model = new_model
            if step % 1000 == 0:
                tau = np.maximum(tau*np.exp(-FLAGS.anneal*step),FLAGS.min_temp)
            print(step, 'Epoch: ', i, '/', FLAGS.epochs, '| ELBO: ', info['loss'])

    _, _, x_hat = model.apply_fn({'params': model.params}, rng, rand_x, sampler, tau)
    x_hat = tfd.Bernoulli(logits=x_hat).mean()
    x_hat = x_hat.reshape((10,10,28,28))
    x_hat = np.concatenate(np.split(x_hat, 10, axis=0), axis=3)
    x_hat = np.concatenate(np.split(x_hat, 10, axis=1), axis=2)
    x_hat = np.squeeze(x_hat)
    plt.figure()
    plt.imshow(x_hat, cmap=plt.cm.inferno, interpolation='none')
    plt.title('Generated Images')
    plt.savefig('results.png')


if __name__=="__main__":
    app.run(main)
