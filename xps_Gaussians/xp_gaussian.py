import argparse
import jax
import sys

import numpy as np

from tqdm import trange
from scipy.stats import ortho_group

sys.path.append("../")
from lib_jax.mirror_gaussians import forward_backward, mirror_descent_kl, mirror_descent_entropy, mirror_descent_w, gaussian_svgd


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=20, help="Number of try")
parser.add_argument("--target", type=str, default="diag", help="Which type of target")
parser.add_argument("--method", type=str, default="nem", help="Which type of method")
parser.add_argument("--n_epochs", type=int, default=1500, help="Number of epochs")
parser.add_argument("--d", type=int, default=10, help="Dimension")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
args = parser.parse_args()


def kl_divergence(mu1, mu2, sigma_1, sigma_2):

    sigma_2_inv = np.linalg.inv(sigma_2)

    kl = 0.5 * (np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1))
                - sigma_1.shape[0] 
                + np.trace(np.matmul(sigma_2_inv, sigma_1))
                + np.matmul(np.matmul((mu2 - mu1).reshape(1,-1),
                                            sigma_2_inv),
                               (mu2 - mu1).reshape(-1,1)))

    return kl


def gen_matrix_with_eigs(eigs):
    """
    Generates a symmetric matrix with eigenvalues `eigs`.
    """
    dim = len(eigs)
    x = ortho_group.rvs(dim)
    return x.T @ np.diag(eigs) @ x


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    keys = jax.random.split(rng, args.ntry)

    np.random.seed(0)

    L_kl = np.zeros((args.ntry, args.n_epochs))

    m0 = np.zeros(args.d)
    s0 = np.eye(args.d)

    mu_opt = np.zeros(args.d)

    pbar = trange(args.ntry)
    for i in pbar:
        key = keys[i]
        if args.target == "diag":
            Sigma_opt = np.diag(np.random.rand(args.d) * 50)
            Sigma_opt_inv = np.linalg.inv(Sigma_opt)
        elif args.target == "non_diag":
            alpha = 0.01
            beta = 1
            Sigma_opt = gen_matrix_with_eigs(
                np.geomspace(1/beta, 1/alpha, args.d)
            )
            Sigma_opt_inv = np.linalg.inv(Sigma_opt)

        preconditioner = Sigma_opt

        if args.method == "nem": ## Bregman potential: negative entropy
            covs = mirror_descent_entropy(args.n_epochs, args.lr, args.d, 
                                          Sigma_opt_inv, Sigma0=s0, bar=False)
        elif args.method == "fb": ## Bregman potential: Forward-Backward
            _, covs = forward_backward(args.n_epochs, args.lr, args.d, 
                                       mu_opt, Sigma_opt_inv, m0=m0, 
                                       Sigma0=s0, bar=False)
            covs = covs[::2]
        elif args.method == "pfb": ## Bregman potential: Preconditioned Forward-Backward
            _, covs = forward_backward(args.n_epochs, args.lr, args.d, mu_opt, 
                                       Sigma_opt_inv, preconditioner, m0=m0, 
                                       Sigma0=s0, bar=False)
            covs = covs[::2]
        elif args.method == "wgd": ## Wasserstein Gradient Descent
            _, covs = mirror_descent_w(args.n_epochs, args.lr, args.d, 
                                       mu_opt, Sigma_opt_inv, m0=m0, 
                                       Sigma0=s0, bar=False)
        elif args.method == "pwgd": ## Preconditioned Wasserstein GD
            _, covs = mirror_descent_w(args.n_epochs, args.lr, args.d, 
                                       mu_opt, Sigma_opt_inv, preconditioner, 
                                       m0=m0, Sigma0=s0, bar=False)
        elif args.method == "klm": ## Bregman potential: KL
            _, covs = mirror_descent_kl(args.n_epochs, args.lr, args.d, 
                                        mu_opt, Sigma_opt_inv, m0=m0, 
                                        Sigma0=s0, bar=False)
        elif args.method == "pklm": ## Bregman potential: KL with preconditioned V
            _, covs = mirror_descent_kl(args.n_epochs, args.lr, args.d, 
                                        mu_opt, Sigma_opt_inv, preconditioner, 
                                        m0=m0, Sigma0=s0, bar=False)
            
        for k in range(args.n_epochs):    
            kl = kl_divergence(mu_opt, mu_opt, covs[k], Sigma_opt) 
            L_kl[i, k] = kl[0,0]

        np.savetxt("./results/KL_method_"+args.method+"_target"+args.target+"_d"+str(args.d)+"_lr"+str(args.lr), L_kl)
