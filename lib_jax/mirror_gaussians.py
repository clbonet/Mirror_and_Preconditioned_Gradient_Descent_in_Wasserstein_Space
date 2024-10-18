import numpy as np

from tqdm import trange
from copy import deepcopy
from scipy.linalg import sqrtm


def forward_backward(n_epochs, lr, d, m_opt, Sigma_opt_inv,
                     preconditioner_cov=None, m0=None, Sigma0=None,
                     bar=True):
    """
        We suppose every covariance matrices commute

        Inputs:
        - n_epochs
        - lr
        - d
        - m_opt: objective mean
        - Sigma_opt_inv: inverse of the objective covariance matrix
        - preconditioner_cov: preconditioning matrix, i.e. mirror map x^T preconditioner_cov^{-1} x / 2.
    """
    if m0 is None:
        mk = np.random.randn(1, d)
        Sigmak = np.eye(d)
    else:
        mk = deepcopy(m0)
        Sigmak = deepcopy(Sigma0)

    if preconditioner_cov is None:
        preconditioner_cov = np.eye(d)

    L_means = [mk]
    L_covs = [Sigmak]

    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)

    for e in pbar:
        C = preconditioner_cov @ Sigma_opt_inv
        m_k12 = (np.eye(d) - lr * C) @ mk + lr * (C @ m_opt)

        M = np.eye(d) - lr * C
        Sigma_k12 = M @ Sigmak @ M

        mk = m_k12

        sqrt_matrix = np.real(sqrtm(Sigma_k12 @ (4 * lr * preconditioner_cov + Sigma_k12)))
        Sigmak = (Sigma_k12 + 2*lr*preconditioner_cov + sqrt_matrix) / 2

        L_means.append(m_k12)
        L_covs.append(Sigma_k12)
        L_covs.append(Sigmak)

    return L_means, L_covs


def mirror_descent_kl(n_epochs, lr, d, m_opt, Sigma_opt_inv,
                      preconditioner_cov=None, m0=None, Sigma0=None,
                      bar=True):
    """
        We suppose every covariance matrices are diagonal

        Inputs:
        - n_epochs
        - lr
        - d
        - m_opt: objective mean
        - Sigma_opt_inv: inverse of the objective covariance matrix
        - preconditioner_cov: preconditioning matrix
    """
    if m0 is None:
        mk = np.random.randn(1, d)
        Sigmak = np.eye(d)
    else:
        mk = deepcopy(m0)
        Sigmak = deepcopy(Sigma0)

    if preconditioner_cov is None:
        preconditioner_cov = np.eye(d)

    L_means = [mk]
    L_covs = [Sigmak]

    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)

    for e in pbar:
        E = Sigma_opt_inv * preconditioner_cov
        mk = (np.eye(d) - lr * E) @ mk + lr * E @ m_opt

        Sigmak_inv = np.linalg.inv(Sigmak) # np.diag(1/np.diag(Sigmak))
        preconditioner_cov_squared = preconditioner_cov @ preconditioner_cov
        M = (np.eye(d) - lr * E)
        
        cpt1 = M @ Sigmak @ M
        cpt2 = 2 * lr * preconditioner_cov
        cpt3 =  2 * lr * (1-lr) * E @ preconditioner_cov
        cpt4 = (1-lr)**2 * preconditioner_cov_squared @ Sigmak_inv

        C = cpt1 + cpt2 + cpt3 + cpt4
        sqrt_matrix = np.real(sqrtm(C @ C - 4 * preconditioner_cov_squared))
        Sigmak = (C + sqrt_matrix) / 2

        L_means.append(mk)
        L_covs.append(Sigmak)

    return L_means, L_covs


def mirror_descent_entropy(n_epochs, lr, d, Sigma_opt_inv, Sigma0=None, bar=True):
    """
        Inputs:
        - n_epochs
        - lr
        - d
        - Sigma_opt_inv: inverse of the objective covariance matrix
    """
    if Sigma0 is None:
        Sigmak = np.eye(d)
    else:
        Sigmak = deepcopy(Sigma0)

    L_covs = [Sigmak]

    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)

    for e in pbar:
        Sigmak_inv = np.linalg.inv(Sigmak)
        C = ((1-lr) * Sigmak_inv + lr * Sigma_opt_inv)
        Sigma_k1_inv = C.T @ Sigmak @ C
        Sigmak = np.linalg.inv(Sigma_k1_inv)

        L_covs.append(Sigmak)

    return L_covs


def mirror_descent_w(n_epochs, lr, d, m_opt, Sigma_opt_inv, preconditioner_cov=None, m0=None, Sigma0=None, bar=True):
    """
        Inputs:
        - n_epochs
        - lr
        - d
        - m_opt: objective mean
        - Sigma_opt_inv: inverse of the objective covariance matrix
        - preconditioner_cov: preconditioning matrix
    """
    if m0 is None:
        mk = np.random.randn(1, d)
        Sigmak = np.eye(d)
    else:
        mk = deepcopy(m0)
        Sigmak = deepcopy(Sigma0)

    if preconditioner_cov is None:
        preconditioner_cov = np.eye(d)

    L_means = [mk]
    L_covs = [Sigmak]

    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)

    E = preconditioner_cov @ Sigma_opt_inv


    for e in pbar:
        mk = (np.eye(d) - lr * E) @ mk + lr * E @ m_opt
        
        Sigmak_inv = np.linalg.inv(Sigmak) #np.diag(1/np.diag(Sigmak))
        C = np.eye(d) - lr * E + lr * preconditioner_cov@Sigmak_inv
        Sigmak = C.T @ Sigmak @ C

        L_covs.append(Sigmak)
        L_means.append(mk)

    return L_means, L_covs
