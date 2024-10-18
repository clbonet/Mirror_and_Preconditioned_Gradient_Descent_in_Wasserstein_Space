import jax

import jax.numpy as jnp
import numpy as np

from copy import deepcopy
from tqdm import trange

from .mirror_maps import EuclideanMirror


def mirror_langevin(key, grad_target, mirror=EuclideanMirror(), n_epochs=100, lr=1, n_particles=500, d=2,
                    x0=None, bar=True):
    """
        Mirror Langevin Algorithm [1]

        Inputs:
        - key: jax rng
        - grad_target: function returning the gradient of the log potential of the target
        - mirror: Object from the class Mirror Map
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - x0: initial particles (default None, randomly generated)

        Outputs:
        - List of particles at each step
        - List of gradients

        [1] Hsieh, Y. P., Kavis, A., Rolland, P., & Cevher, V. (2018). Mirrored langevin dynamics. Advances in Neural Information Processing Systems, 31.
    """
    keys = jax.random.split(key, n_epochs+1)

    if x0 is None:
        xk = jax.random.normal(keys[-1], (n_particles, d))
    else:
        xk = deepcopy(x0)
        n_particles, d = x0.shape

    L_loss = []
    L_particles = [xk]
    L_gradients = []

    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)
    
    for e in pbar:
        grad_x = grad_target(xk)
        cov = mirror.hessian_phi(xk)        
        cov_12 = jnp.linalg.cholesky(cov)

        noise = jax.random.normal(keys[e], (n_particles, d))
        scaled_noise = jnp.einsum("nij, nj -> ni", cov_12, noise)
                
        zk = mirror.grad_phi(xk) - lr * grad_x + np.sqrt(2*lr) * scaled_noise
        xk = mirror.grad_phi_star(zk)
        
        L_particles.append(xk)
        L_gradients.append(grad_x)
        
    return L_particles, L_gradients
