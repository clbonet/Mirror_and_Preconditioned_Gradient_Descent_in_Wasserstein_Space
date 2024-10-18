import jax
import jax.numpy as jnp
import numpy as np

from copy import deepcopy
from tqdm import trange

from .mirror_maps import EuclideanMirror

@jax.jit
def get_potential(x_proj, target_proj):
    n_projs = x_proj.shape[0]
    percentiles = jnp.linspace(0, 1, 100)

    quantiles_x = jnp.percentile(x_proj, percentiles*100, axis=1).T
    cdf_x = jax.vmap(lambda x, q: jnp.interp(x, q, percentiles))(x_proj, quantiles_x)
    
    quantiles_target = jnp.percentile(target_proj, percentiles*100, axis=1).T
    x_transported = jax.vmap(lambda x, q_tgt: jnp.interp(x, percentiles, q_tgt))(cdf_x, quantiles_target)

    return x_proj - x_transported


def target_grad_sw(x, sampler_src, n_projs=50):
    n, d = x.shape
    
    master_key, key_samples, key_projs = jax.random.split(sampler_src.key, 3)
    sampler_src.key = master_key
    target = sampler_src.generate_samples(key_samples, n)

    v = jax.random.normal(key_projs, (n_projs, d))
    v = v / np.linalg.norm(v[:,None], axis=-1)

    target_proj = (target@v.T).T
    x_proj = (x@v.T).T

    d_potential = get_potential(x_proj, target_proj)
    nabla_proj = v[:, None, :]
    nabla_SW = jnp.mean(d_potential[:, :, None] * nabla_proj, axis=0)

    return nabla_SW
    