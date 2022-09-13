import jax.numpy as jnp
import jax
from functools import partial


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@partial(jax.jit, static_argnames=['k'])
def top_k(logits, y,k):
    top_k = jax.lax.top_k(logits, k)[1]
    ts = jnp.argmax(y, axis=1)
    correct = 0
    for i in range(ts.shape[0]):
        b = (jnp.where(top_k[i,:] == ts[i], jnp.ones((top_k[i,:].shape)), 0)).sum()
        correct += b
    correct /= ts.shape[0]
    return correct