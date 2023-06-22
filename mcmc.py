"""
mcmc.py
Use MCMC methods to estimate Gaussian process parameters that
recover expected log-normal distribution of some data.

Copyright(C) 2023 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Trey Wenger - June 2023
"""

import numpy as np
import pymc as pm
import matplotlib.pyplot as plt


def main(fname):
    """
    Determine optimal Gaussian process parameters to recover the expected
    log-normal distribution of some data. Along the way, generate some
    plots:
        data.pdf - log10 data
        trace.pdf - posterior sample chains
        model.pdf - MAP Gaussian process model
        model_sigma.pdf - MAP GP sqrt(covariance). This is like the confidence
                          of the GP.
        residual.pdf - (log10 data - model) residuals
        distribution.pdf - histogram of log10 data and residuals

    Also saves model and residuals to disk:
        model.npy - model
        model_sigma.npy - model sqrt(covariance)
        residual.npy - residuals

    N.B. Ideally we would run inference on the GP prior and draw samples of
    the model from the posterior distribution. The posterior sampling is
    quite slow for the large dataset, though, so we estimate the GP
    hyperparameters from the maximum a posteriori (MAP) fit, but also run inference
    to demonstrate that the MAP is a good point estimate for the posterior
    distribution.

    Inputs:
        fname :: string
            Data file containing 2-D numpy array

    Returns: Nothing
    """
    # load the data
    data = np.load(fname)
    xaxis = np.arange(data.shape[0])
    yaxis = np.arange(data.shape[1])

    # plot the data
    fig, ax = plt.subplots()
    cax = ax.imshow(
        np.log10(data.T),
        origin="lower",
        extent=[xaxis.min(), xaxis.max(), yaxis.min(), yaxis.max()],
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(cax)
    cbar.set_label(r"log$_{10}$ M0")
    fig.tight_layout()
    fig.savefig("data.pdf", bbox_inches="tight")
    plt.close(fig)

    # flatten the data
    xgrid, ygrid = np.meshgrid(xaxis, yaxis, indexing="ij")
    X = np.stack((xgrid.flatten(), ygrid.flatten()), axis=-1)
    y = data.flatten()

    # keep only non-nan data
    good = ~np.isnan(y)
    y = y[good]
    X = X[good]

    # log10 data
    log10_y = np.log10(y)

    # number of inducing points
    n_induce = 100

    # define the model
    with pm.Model() as model:
        # Set prior on GP hyperparameters
        eta = pm.HalfNormal("eta", sigma=20.0)
        ell = pm.HalfNormal("ell", sigma=20.0)

        # spatial covariance function
        cov_func = eta**2.0 * pm.gp.cov.ExpQuad(2, ell)

        # inducing points
        Xu = pm.gp.util.kmeans_inducing_points(n_induce, X)

        # specify the GP with zero mean
        gp = pm.gp.MarginalApprox(cov_func=cov_func, approx="FITC")

        # GP prior
        sigma = pm.HalfNormal("sigma", sigma=20.0)
        _ = gp.marginal_likelihood("log10_y", X=X, Xu=Xu, y=log10_y, sigma=sigma)

    # find maximum a posteriori
    with model:
        max_posterior = pm.find_MAP()
        print("MAP estimate:")
        print(f"eta = {max_posterior['eta']:.3f}")
        print(f"ell = {max_posterior['ell']:.3f}")
        print(f"sigma = {max_posterior['sigma']:.3f}")
        print()

    # predict at MAP
    with model:
        mu, var = gp.predict(X, point=max_posterior, diag=True)

    # fill data with GP model at MAP
    y_model = np.zeros_like(data)
    y_model[X[:, 0], X[:, 1]] = mu

    y_model_sigma = np.zeros_like(data)
    y_model_sigma[X[:, 0], X[:, 1]] = np.sqrt(var)

    # hopefully you have a fast CPU
    with model:
        trace = pm.sample(draws=500, tune=500, chains=4, cores=1)
    print("Inference statistics:")
    print(pm.summary(trace, var_names=["eta", "ell", "sigma"]))

    # plot trace
    axes = pm.plot_trace(trace)
    fig = axes.ravel()[0].figure
    fig.savefig("trace.pdf", bbox_inches="tight")

    """
    # conditional distribution over new positions (which are just
    # the original positions in this case)
    with model:
        fcond = gp.conditional("fcond", Xnew=X)

    # posterior sampling is REALLY slow
    with model:
        posterior = pm.sample_posterior_predictive(
            trace.sel(draw=slice(None, None, 10)), var_names=["fcond"]
        )

    y_model_ppc = np.zeros_like(data)
    y_model_ppc[X[:, 0], X[:, 1]] = posterior.posterior_predictive.fcond.sel(chain=0, draw=0)

    y_model_sigma_ppc = np.zeros_like(data)
    y_model_sigma_ppc[X[:, 0], X[:, 1]] = posterior.posterior_predictive.fcond.sel(chain=0, draw=0)
    """

    # plot MAP model
    fig, ax = plt.subplots()
    cax = ax.imshow(
        y_model.T,
        origin="lower",
        extent=[xaxis.min(), xaxis.max(), yaxis.min(), yaxis.max()],
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(cax)
    cbar.set_label(r"log$_{10}$ M0 Model")
    fig.tight_layout()
    fig.savefig("model.pdf", bbox_inches="tight")
    plt.close(fig)

    # plot MAP model error
    fig, ax = plt.subplots()
    cax = ax.imshow(
        y_model_sigma.T,
        origin="lower",
        extent=[xaxis.min(), xaxis.max(), yaxis.min(), yaxis.max()],
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(cax)
    cbar.set_label(r"log$_{10}$ M0 Model Error")
    fig.tight_layout()
    fig.savefig("model_sigma.pdf", bbox_inches="tight")
    plt.close(fig)

    # residual
    res = np.log10(data) - y_model

    # plot residual
    fig, ax = plt.subplots()
    cax = ax.imshow(
        res.T,
        origin="lower",
        extent=[xaxis.min(), xaxis.max(), yaxis.min(), yaxis.max()],
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(cax)
    cbar.set_label(r"log$_{10}$ M0 Residual")
    fig.tight_layout()
    fig.savefig("residual.pdf", bbox_inches="tight")
    plt.close(fig)

    # histogram of data and residuals
    bins = np.linspace(-1.0, 2.0, 100)
    fig, ax = plt.subplots()
    ax.hist(
        np.log10(data).flatten(),
        bins=bins,
        color="k",
        alpha=0.5,
        label="M0",
        density=True,
    )
    ax.hist(
        res.flatten(),
        bins=bins,
        color="r",
        alpha=0.5,
        label="M0 Residual",
        density=True,
    )
    ax.legend(loc="best")
    ax.set_xlabel(r"log$_{10}$ M0")
    ax.set_ylabel("Probability Density")
    fig.tight_layout()
    fig.savefig("distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    np.save("model.npy", y_model)
    np.save("model_sigma.npy", y_model_sigma)
    np.save("residual.npy", res)


if __name__ == "__main__":
    main("c1_M0.npy")
