"""Implements the common train state model."""

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

def default_init(scale: Optional[float] = jnp.sqrt(2)) -> jnp.ndarray:
    """Initializes the model parameters."""
    return nn.initializers.orthogonal(scale)

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]


@flax.struct.dataclass
class Model:
    """Implements the base model for training.
    
    Attributes:
        step: training step.
        apply_fn: forward pass of the model.
        params: stores trainable parameters of the model.
        tx: gradient transformation to the optimizer.
        opt_state: optimizer state following gradient update.
    """
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        """Creates the model."""
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        """Exectues the forward pass of the model."""
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Callable[[Params], Any],
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:
        """Applies gradient to model parameters."""
        grad_fn = jax.grad(loss_fn, has_aux=has_aux)
        if has_aux:
            grads, aux = grad_fn(self.params)
        else:
            grads = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save(self, save_path: str):
        """Saves model parameters to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        """Loads model parameters from disk."""
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
