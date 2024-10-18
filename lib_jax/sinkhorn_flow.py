import jax

import jax.numpy as jnp
import numpy as np

from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from ott.geometry.costs import SqEuclidean
from ott.geometry.pointcloud import PointCloud


def target_grad_sinkhorn_div(x, sampler_src, cost_fn=SqEuclidean()):
    n, d = x.shape

    master_key, key_samples = jax.random.split(sampler_src.key, 2)
    sampler_src.key = master_key
    target = sampler_src.generate_samples(key_samples, n)

    loss = lambda y: sinkhorn_divergence(PointCloud, y, target,
                                         relative_epsilon=True, static_b=True,
                                         cost_fn=cost_fn).divergence
    nabla = jax.grad(loss)(x)

    return nabla
