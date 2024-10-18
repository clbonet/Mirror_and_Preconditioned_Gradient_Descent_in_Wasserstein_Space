import jax
import jax.numpy as jnp


## kernels
@jax.jit
def gaussian_kernel(x, y, h=1):
    return jnp.exp(-jnp.sum(jnp.square(x-y), axis=-1)/ (2*h))

@jax.jit
def imq_kernel(x, y, c=1):
    return 1 / jnp.sqrt(c + jnp.sum(jnp.square(x-y), axis=-1))

@jax.jit
def laplace_kernel(x, y, h):
    return jnp.exp(-jnp.sum(jnp.abs(x-y), axis=-1)/ h)
    
def riesz_kernel(x, y, r=1):
    return -jnp.linalg.norm(x-y, axis=-1)**r
    
def mmd(x, y, kernel):
    vmapped_kernel = jax.vmap(kernel, in_axes=(0, None))
    pairwise_kernel = jax.vmap(vmapped_kernel, in_axes=(None, 0))
    
    ## Unbiased estimate
    Kxx = pairwise_kernel(x, x)
    Kyy = pairwise_kernel(y, y)
    Kxy = pairwise_kernel(x, y)

    n = x.shape[0]
    cpt1 = (jnp.sum(Kxx)-jnp.sum(jnp.diag(Kxx)))/(n-1) ## remove diag terms
    cpt2 = (jnp.sum(Kyy)-jnp.sum(jnp.diag(Kyy)))/(n-1)
    cpt3 = jnp.sum(Kxy)/n

    return (cpt1+cpt2-2*cpt3)/n    
    

## Code gradient MMD

def sum_kernel(x, ys, kernel):
    z = jax.vmap(lambda y: kernel(x, y))(ys)
    return jnp.sum(z)

def sum_kernel_batch(xs, ys, kernel):
    f = lambda x: sum_kernel(x, ys, kernel)
    z = jax.vmap(f)(xs)
    grad_z = jax.grad(f)(xs)
    return z, grad_z


def target_grad_mmd(x, sampler_src, kernel, noise_injection=0):
    """
        Return the Wasserstein gradient of the MMD, see [1].
    
        Inputs:
        - x: (n, d)
        - sampler_src: object from sampler_from_data
        - kernel: function k(x,y)
        - noise_injection: levef of noise to add, cf [1]

        [1] Arbel M, Korba A, Salim A, Gretton A. Maximum mean discrepancy gradient flow. Advances in Neural Information Processing Systems. 2019;32.
    """
    n, d = x.shape
    
    master_key, key_samples, key_noise = jax.random.split(sampler_src.key, 3)
    sampler_src.key = master_key
    target = sampler_src.generate_samples(key_samples, n)

    noise = jax.random.normal(key_noise, (n, d))

    _, grad_x = sum_kernel_batch(x + noise_injection * noise, x, kernel)
    _, grad_tgt = sum_kernel_batch(x + noise_injection * noise, target, kernel)
    nabla_mmd = (grad_x - grad_tgt) / n

    return nabla_mmd


## Sliced MMD Riesz kernel
def sliced_factor(d):
    '''
        Compute the scaling factor of sliced MMD

        From https://github.com/johertrich/sliced_MMD_flows/blob/main/utils/utils.py
    '''
    k=(d-1)//2
    fac=1.
    if (d-1)%2==0:
        for j in range(1,k+1):
            fac=2*fac*j/(2*j-1)
    else:
        for j in range(1,k+1):
            fac=fac*(2*j+1)/(2*j)
        fac=fac*jnp.pi/2
    return fac

@jax.jit
def derivative_1d_interaction(x):
    """
        Inputs:
        - x of size (n,)

        (use with jax.vmap for data of size (n_projs, n))
    """
    n = len(x)
    s = jnp.argsort(x)
    
    hs = (n - (2*jnp.arange(n) - 1)) / n**2
    return hs[jnp.argsort(s)]

@jax.jit
def derivative_1d_potential(x, y):
    """
        Inputs: 
        - x of size (n,)
        - y of size (m,)

        (use with jax.vmap for data of size (n_projs, n))
    """
    n = len(x)
    m = len(y)
    
    s = jnp.argsort(jnp.concatenate([x, y]))

    h = jnp.where(s>=n, 1, 0)
    h = (2 * jnp.cumsum(h) - m) / (n * m)

    inv_inds = jnp.argsort(s)
    return h[inv_inds][:n]


def target_grad_mmd_riesz1(x, sampler_src, cst_sliced_factor, n_projs=50):
    """
        Return the gradient of the Sliced MMD with Riesz kernel, see [1].
    
        Inputs:
        - x: (n, d)
        - sampler_src: object from sampler_from_data
        - cst_sliced_factor: constant sliced_factor(d)
        - n_projs: number of slices

        [1] Hertrich J, Wald C, Altekrüger F, Hagemann P. Generative sliced MMD flows with Riesz kernels. arXiv preprint arXiv:2305.11463. 2023.
    """
    n, d = x.shape
    
    master_key, key_samples, key_projs = jax.random.split(sampler_src.key, 3)
    sampler_src.key = master_key
    target = sampler_src.generate_samples(key_samples, n)

    v = jax.random.normal(key_projs, (n_projs, d))
    v = v / jnp.linalg.norm(v[:,None], axis=-1)

    target_proj = (target@v.T).T
    x_proj = (x@v.T).T

    grad_inter = jax.vmap(derivative_1d_interaction)(x_proj)
    grad_potential = jax.vmap(derivative_1d_potential)(x_proj, target_proj)
    nabla_F1 = grad_potential + grad_inter

    nabla_proj = v[:, None, :]

    nabla_mmd = jnp.mean(nabla_F1[:,:,None] * nabla_proj, axis=0)

    return nabla_mmd * cst_sliced_factor
