import jax
import jax.numpy as jnp


class LinearMirror():
    def __init__(self, P):
        self.P = P
        self.P_ = jnp.linalg.inv(P)

    def forward(self, x):
        return jnp.norm(x, dim=-1)**2

    def grad_phi(self, x):
        return x@self.P_.T

    def grad_phi_star(self, y):
        return y@self.P.T


class InteractionMirror():
    def __init__(self, W):
        @jax.jit
        def sum_W(x):
            return jnp.sum(W(x))

        self.W = W
        grad_W = jax.jit(jax.grad(sum_W))

        @jax.jit
        def target_grad(x):
            C = x[None] - x[:,None]
            return jnp.mean(grad_W(C), axis=0)
        self.grad = target_grad

    def forward(self, x):
        return jnp.mean(self.W(x[:,None] - x[None]))

    def grad_phi(self, x):
        return self.grad(x)
