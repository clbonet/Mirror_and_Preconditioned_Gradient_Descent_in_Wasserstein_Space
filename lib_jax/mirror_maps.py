import numpy as np
import jax.numpy as jnp


class MirrorMap():
    def __init__(self):
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError
        
    def grad_phi(self, x):
        raise NotImplementedError

    def grad_phi_star(self, y):
        raise NotImplementedError

        
class EuclideanMirror(MirrorMap):
    def __init__(self):
        pass
    
    def forward(self, x):
        return jnp.norm(x, dim=-1)**2

    def grad_phi(self, x):
        return x
    
    def grad_phi_star(self, y):
        return y
    
    
class BallMirror(MirrorMap):
    def __init__(self, R, gamma=1):
        self.R = R
        self.gamma = gamma
        
    def forward(self, x):
        norm_x2 = jnp.norm(x, axis=-1, keepdims=True)**2
        return -self.gamma * jnp.log(self.R-norm_x2)
    
    def grad_phi(self, x):
        norm_x2 = jnp.norm(x, axis=-1, keepdims=True)**2
        return 2 * self.gamma * x / (self.R - norm_x2)
    
    def grad_phi_star(self, y):
        norm_y2 = jnp.norm(y, axis=-1, keepdims=True)**2
        return self.R * y / (jnp.sqrt(self.R * norm_y2 + self.gamma**2) + self.gamma)


class EntropicMirror(MirrorMap):
    def __init__(self):
        pass

    def forward(self, x):
        """
            Return \sum_i x_i \log(x_i) + (1-\sum_i x_i) \log(1-\sum_i x_i)
        """
        sum_x = jnp.sum(x, dim=-1, keepdim=True)
        return jnp.sum(x * jnp.log(x), axis=-1, keepdims=True) + (1-sum_x) * jnp.log(1-sum_x)

    def grad_phi(self, x):
        """
            Return \nabla \phi(x), where [\nabla\phi(x)]_i = \log(x_i) - \log(1-\sum_j x_j)
        """
        #assert not np.any(x<=0) and not np.any(jnp.sum(x, axis=-1)>=1)
        return jnp.log(x) - jnp.log(1 - jnp.sum(x, axis=-1, keepdims=True))

    def grad_phi_star(self, y):
        """
            Return \nabla\phi^*(y) where [\nabla\phi^*(y)]_i = e^{y_i} / (1+\sum_j e^{y_j})
        """
        return jnp.exp(y) / (1 + jnp.sum(jnp.exp(y), axis=-1, keepdims=True))
    
    def hessian_phi(self, x):
        """
            Return Hess\phi(x) = diag(1/x) + (1/(1-\sum_j x_k))J
        """
        n, d = x.shape
        sum_x = jnp.sum(x, axis=-1, keepdims=True)

        diag = np.zeros((n, d, d),dtype=x.dtype)
        diag.reshape(-1,d**2)[...,::d+1] = 1/x
        
        return diag + np.ones((n, d, d)) / (1-sum_x)[:,:,None]


class BoxMirror(MirrorMap):
    def __init__(self):
        pass
        
    def forward(self, x):
        """
            Return -\sum_i (\log(1-x_i) + \log(1+x_i))
        """
        return jnp.sum(-jnp.log(1-x)-jnp.log(1+x), axis=-1, keepdims=True)
    
    def grad_phi(self, x):
        """
            Return \nabla\phi(x) where [\nabla \phi(x)]_i = 1/(1-x_i) - 1/(1+x_i)
        """
        return (1/(1-x) - 1/(1+x))
    
    def grad_phi_star(self, y):
        """
            Return \nabla\phi^*(y) where [\nabla\phi^*(y)]_i = (-1 + \sqrt{1+y_i^2})/y_i
        """
        return (-1 + jnp.sqrt(1+y**2)) / y
        