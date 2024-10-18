# Mirror and Preconditioned Gradient Descent in Wasserstein Space

This repository constains the code to reproduce the experiments of the paper [Mirror and Preconditioned Gradient Descent in Wasserstein Space](https://arxiv.org/abs/2406.08938). We propose in this paper to minimize functionals on the space of probability measures through the Mirror Descent and Preconditioned Gradient Descent schemes.

## Abstract

As the problem of minimizing functionals on the Wasserstein space encompasses many applications in machine learning, different optimization algorithms on $\mathbb{R}^d$ have received their counterpart analog on the Wasserstein space. We focus here on lifting two explicit algorithms: mirror descent and preconditioned gradient descent. These algorithms have been introduced to better capture the geometry of the function to minimize and are provably convergent under appropriate (namely relative) smoothness and convexity conditions. Adapting these notions to the Wasserstein space, we prove guarantees of convergence of some Wasserstein-gradient-based discrete-time schemes for new pairings of objective functionals and regularizers. The difficulty here is to carefully select along which curves the functionals should be smooth and convex. We illustrate the advantages of adapting the geometry induced by the regularizer on ill-conditioned optimization tasks, and showcase the improvement of choosing different discrepancies and geometries in a computational biology task of aligning single-cells.

## Citation

```
@inproceedings{bonet2024mirror,
    title={Mirror and Preconditioned Gradient Descent in Wasserstein Space},
    author={Clément Bonet and Théo Uscidda and Adam David and Pierre-Cyril Aubin-Frankowski and Anna Korba},
    year={2024},
    booktitle={Thirty-eight Conference on Neural Information Processing Systems}
}
```

## Experiments

- Figure 1 can be reproduced by running the notebook "MD_mirror_interaction.ipynb" in the folder xps_interaction
- Figure 2 can be reproduced by first running "xps.sh" in the folder xps_Gaussians, and then by running the notebook "Results_Gaussian.ipynb"
- The experiment on the simplex of Appendix G is available in the notebook "MD - Dirichlet Posterior.ipynb".

## Requirements

- jax, jaxopt
- [ott](https://github.com/ott-jax/ott)
- [python-ternary](https://github.com/marcharper/python-ternary) package for the Dirichlet experiment

## Credits

- Some code of https://implicit-layers-tutorial.org/implicit_functions/ was used for the Newton solver in jax.
- For the gradient of the MMD Riesz kernel, a part of the code was adapted from the [sliced_MMD_flows](https://github.com/johertrich/sliced_MMD_flows/tree/main) repository.
