import jax
import jaxopt

import jax.numpy as jnp
import numpy as np

from copy import deepcopy
from tqdm import trange
# from jax_tqdm import scan_tqdm
from functools import partial

from .mirror_maps import EuclideanMirror


@partial(jax.jit, static_argnums=[0,1,5])
def gradient_descent(target_grad, n_epochs=100, lr=1, n_particles=500, d=2,
                     preconditioner=lambda x: x, x0=None, scale_grad=1.):
    """
        (Preconditioned) gradient descent

        Inputs:
        - target_grad: function returning the (Wasserstein) gradient of the target
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - preconditioner: function returning the preconditioner on the gradient
        - x0: initial particles (default None, randomly generated)

        Outputs:
        - List of particles at each step
        - List of gradients
    """

    # @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk = carry
        grad_x = target_grad(xk)
        xk = xk - lr * preconditioner(scale_grad * grad_x)
        return xk, (xk, grad_x)

    # Initial state
    if x0 is None:
        x0 = np.random.randn(n_particles, d)

    # Use `lax.scan` to loop over epochs
    xk, L = jax.lax.scan(step, x0, jnp.arange(n_epochs))

    L_particles, L_gradients = L
    return jnp.insert(L_particles, 0, x0, axis=0), L_gradients


@partial(jax.jit, static_argnums=[0,1,2])
def mirror_descent(target_grad, mirror=EuclideanMirror(), n_epochs=100, lr=1,
                   n_particles=500, d=2, x0=None):
    """
        Mirror descent

        Inputs:
        - target_grad: function returning the (Wasserstein) gradient of the target
        - mirror: Object from the class Mirror Map
        - n_epochs: number of epochs
        - lr: step size
        - n_particles: number of particles
        - d: dimension
        - x0: initial particles (default None, randomly generated)

        Outputs:
        - List of particles at each step
        - List of gradients
    """

    # @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk = carry
        grad_x = target_grad(xk)
        zk = mirror.grad_phi(xk) - lr * grad_x
        xk = mirror.grad_phi_star(zk)
        return xk, (xk, grad_x)

    # Initial state
    if x0 is None:
        x0 = np.random.randn(n_particles, d)

    # Use `lax.scan` to loop over epochs
    xk, L = jax.lax.scan(step, x0, jnp.arange(n_epochs))

    L_particles, L_gradients = L
    return jnp.insert(L_particles, 0, x0, axis=0), L_gradients


def fwd_solver(f, z_init):
    """
        From https://implicit-layers-tutorial.org/implicit_functions/
    """

    def cond_fun(carry):
        z_prev, z = carry
        return jnp.linalg.norm(z_prev - z) > 1e-5

    def body_fun(carry):
        _, z = carry
        return z, f(z)

    init_carry = (z_init, f(z_init))
    # _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
    _, z_star = jaxopt.loop.while_loop(cond_fun, body_fun, init_carry,
                                       maxiter=100, jit=True)
    return z_star


def newton_solver(f, z_init, eps=0):
    """
        Newton solver
    """
    n, d = z_init.shape

    grad_f = jax.jit(jax.jacobian(f))
    Id = eps * jnp.eye(n*d, n*d)

    def g(z):
        grad = grad_f(z).reshape(n*d, n*d) + Id
        fz = f(z).reshape((n*d,))
        newton_step = jnp.linalg.solve(grad, fz).reshape(n, d)
        return z - newton_step

    return fwd_solver(g, z_init)


@partial(jax.jit, static_argnums=[0,1,2])
def mirror_descent_implicit(target_grad, mirror=EuclideanMirror(), n_epochs=100, lr=1,
                            n_particles=500, d=2, x0=None, eps=1e-8):
    """
        Solve the Mirror Descent in an implicit way (i.e. without inverting
        \nabla V or \nabla_W\phi(T#\mu)\circ T)

        Inputs:
        - target_grad: Wasserstein gradient of the target
        - mirror: Mirror map
        - n_epochs
        - lr
        - n_particles: if x0 is None
        - d: if x0 is None
        - x0: Initial particles, array of size(n_particles, d)
        - eps: regularization inverse in Newton

        Outputs:
        - List of particles at each step
        - List of gradients
    """    
    # @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk = carry
        grad_x = target_grad(xk)

        @jax.jit
        def f(y):
            return mirror.grad_phi(y) - mirror.grad_phi(xk) + lr * grad_x

        y0 = np.random.randn(n_particles, d)
        xk = newton_solver(f, y0, eps=eps)
        
        jax.clear_caches()

        return xk, (xk, grad_x)

    # Initial state
    if x0 is None:
        x0 = np.random.randn(n_particles, d)
    
    n_particles, d = x0.shape

    # Use `lax.scan` to loop over epochs
    xk, L = jax.lax.scan(step, x0, jnp.arange(n_epochs))

    L_particles, L_gradients = L
    return jnp.insert(L_particles, 0, x0, axis=0), L_gradients



def gradient_descent_v0(target_grad, n_epochs=100, lr=1, n_particles=500, d=2,
                        preconditioner=lambda x: x, x0=None, bar=True, scale_grad=1.):
    if x0 is None:
        xk = np.random.randn(n_particles, d)
    else:
        xk = deepcopy(x0)

    L_particles = [xk]
    L_gradients = []

    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)

    for e in pbar:
        grad_x = target_grad(xk)
        xk = xk - lr * preconditioner(scale_grad * grad_x)

        L_particles.append(xk)
        L_gradients.append(grad_x)

    return L_particles, L_gradients


def mirror_descent_v0(target_grad, mirror=EuclideanMirror(), n_epochs=100, lr=1,
                      n_particles=500, d=2, x0=None, bar=True):
    if x0 is None:
        xk = np.random.randn(n_particles, d)
    else:
        xk = deepcopy(x0)

    L_particles = [xk]
    L_gradients = []

    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)

    for e in pbar:
        grad_x = target_grad(xk)

        zk = mirror.grad_phi(xk) - lr * grad_x
        xk = mirror.grad_phi_star(zk)

        # L_loss.append(y.item())
        L_particles.append(xk)
        L_gradients.append(grad_x)

    return L_particles, L_gradients


def mirror_descent_implicit_v0(target_grad, mirror, n_epochs=100, lr=1,
                               n_particles=500, d=2, x0=None, bar=True, eps=1e-8):
    """
        Solve the Mirror Descent in an implicit way (i.e. without inverting
        \nabla V or \nabla_W\phi(T#\mu)\circ T)

        Inputs:
        - target_grad: Wasserstein gradient of the target
        - mirror: Mirror map
        - n_epochs
        - lr
        - n_particles: if x0 is None
        - d: if x0 is None
        - x0: Initial particles, array of size(n_particles, d)
        - bar: plot bar trange
        - eps: regularization inverse in Newton
    """
    if x0 is None:
        xk = np.random.randn(n_particles, d)
    else:
        n_particles, d = x0.shape
        xk = deepcopy(x0)

    L_particles = [x0]
    L_gradients = []

    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)

    for e in pbar:
        grad_x = target_grad(xk)

        @jax.jit
        def f(y):
            return mirror.grad_phi(y) - mirror.grad_phi(xk) + lr * grad_x

        y0 = np.random.randn(n_particles, d)
        xk = newton_solver(f, y0, eps=eps)

        if jnp.any(jnp.isnan(xk)):
            print("NaN")
            break

        # L_loss.append(y.item())
        L_particles.append(xk)
        L_gradients.append(grad_x)

        jax.clear_caches()

    return L_particles, L_gradients
