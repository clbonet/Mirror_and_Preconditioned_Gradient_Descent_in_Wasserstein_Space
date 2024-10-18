import jax.numpy as jnp


def fill_diagonal(arr, val):
    # fill the diagonal of arr with `val` (no JAX function for that yet
    # https://github.com/google/jax/issues/2680
    assert arr.ndim >= 2
    i, j = jnp.diag_indices(min(arr.shape[-2:]))
    return arr.at[..., i, j].set(val)


def ent(X):
    """
        Kozachenko-Leonenko estimator of entropy.
    
        From https://colab.research.google.com/drive/1B-0kUVPdjdqrdObp6sMJtx7chmuvzIbp?usp=sharing#scrollTo=UBN-q8sl6nO1
        
        [1] Delattre, S. and Fournier, N., 2017. On the Kozachenko–Leonenko entropy estimator. Journal of Statistical Planning and Inference, 185, pp.69-93.
    """
    N,D = X.shape

    # Compute pairwise squared distances
    dist_sq = jnp.sum((X[:, jnp.newaxis, :] - X[jnp.newaxis, :, :]) ** 2, axis=-1)

    # Set the diagonal to a large number so it doesn't affect the min calculation
    dist_sq = fill_diagonal(dist_sq, jnp.inf)

    # Find the minimum distance for each point
    min_dist = jnp.sqrt(jnp.min(dist_sq, axis=1))

    # Kozachenko-Leonenko estimator of the entropy (up to irrelevant const)
    return jnp.mean( jnp.log((N-1) * min_dist**D))
