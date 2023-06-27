"""
mcmc_mixture.py
Use MCMC methods to estimate mixture model fit to data.

conda create -n pymc -c conda-forge pymc
conda activate pymc
conda install -c conda-forge astropy corner


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

from astropy.io import fits
import numpy as np
import pymc as pm
import pytensor as pt
import matplotlib.pyplot as plt
import corner


def main(fname):
    """
    Determine optimal mixture model parameters to recover the expected
    log-normal distribution of some data. Along the way, generate some
    plots:
        data.pdf - log10 data
        trace.pdf - posterior sample chains

    Inputs:
        fname :: string
            FITS image containing data

    Returns: Nothing
    """
    # load the data
    with fits.open(fname) as hdulist:
        data = 5.5 * np.nansum(hdulist[0].data, axis=0)

    data = data[130:150, :]
    xaxis = np.arange(data.shape[0])
    yaxis = np.arange(data.shape[1])

    # normalize data
    y = ((data - np.mean(data)) / np.std(data)).flatten()

    # plot the data
    fig, ax = plt.subplots()
    cax = ax.imshow(
        y.reshape(data.shape).T,
        origin="lower",
        extent=[xaxis.min(), xaxis.max(), yaxis.min(), yaxis.max()],
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(cax)
    cbar.set_label(r"Scaled M0")
    fig.tight_layout()
    fig.savefig("data.pdf", bbox_inches="tight")
    plt.close(fig)

    # define the model
    with pm.Model() as model:
        # prior on noise
        noise_mu = pm.Normal("noise_mu", mu=0.0, sigma=1.0)
        noise_sigma = pm.HalfNormal("noise_sigma", sigma=0.5)
        noise = pm.Normal.dist(mu=noise_mu, sigma=noise_sigma)

        # prior on signal (log-normal distribution)
        signal_mu = pm.Normal("signal_mu", mu=0.0, sigma=0.5)
        signal_sigma = pm.HalfNormal("signal_sigma", sigma=1.0)
        signal = pm.LogNormal.dist(mu=signal_mu, sigma=signal_sigma)

        # mixture weights
        w = pm.Dirichlet("w", a=np.ones(2))

        # mixture
        _ = pm.Mixture("m0", w=w, comp_dists=[noise, signal], observed=y)

    # generate posterior samples
    with model:
        prior = pm.sample_prior_predictive(samples=100)

    # prior predictive checks
    fig, ax = plt.subplots()
    bins = np.linspace(-3.0, 4.0, 100)
    for chain in prior.prior_predictive.chain:
        for draw in prior.prior_predictive.draw:
            ax.hist(
                prior.prior_predictive["m0"].sel(chain=chain, draw=draw),
                bins=bins,
                color="k",
                alpha=0.1,
                density=True,
                histtype="step",
            )
    ax.hist(
        y.flatten(),
        bins=bins,
        color="r",
        linewidth=2.0,
        density=True,
        histtype="step",
    )
    ax.set_xlabel("M0")
    ax.set_ylabel("Probability Density")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig("prior_dist.pdf", bbox_inches="tight")
    plt.close(fig)

    # sample posterior
    with model:
        trace = pm.sample()
    pm.summary(trace)

    # plot trace
    axes = pm.plot_trace(
        trace, var_names=["noise_mu", "noise_sigma", "signal_mu", "signal_sigma"]
    )
    fig = axes.ravel()[0].figure
    fig.savefig("trace.pdf", bbox_inches="tight")
    plt.close(fig)

    # posterior predictive
    with model:
        posterior = pm.sample_posterior_predictive(
            trace.sel(draw=slice(None, None, 20))
        )

    # posterior predictive checks
    fig, ax = plt.subplots()
    bins = np.linspace(-3.0, 4.0, 100)
    for chain in posterior.posterior_predictive.chain:
        for draw in posterior.posterior_predictive.draw:
            ax.hist(
                posterior.posterior_predictive["m0"].sel(chain=chain, draw=draw),
                bins=bins,
                color="k",
                alpha=0.1,
                density=True,
                histtype="step",
            )
    ax.hist(
        y.flatten(),
        bins=bins,
        color="r",
        linewidth=2.0,
        density=True,
        histtype="step",
    )
    ax.set_xlabel("M0")
    ax.set_ylabel("Probability Density")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig("posterior_dist.pdf", bbox_inches="tight")
    plt.close(fig)

    # evaluate likelihood that data belong to each component individually
    p_noise = trace.posterior["w"].sel(w_dim_0=0).mean().data * np.exp(
        pm.logp(
            pm.Normal.dist(
                mu=trace.posterior["noise_mu"].mean().data,
                sigma=trace.posterior["noise_sigma"].mean().data,
            ),
            y,
        ).eval()
    )
    p_signal = trace.posterior["w"].sel(w_dim_0=1).mean().data * np.exp(
        pm.logp(
            pm.LogNormal.dist(
                mu=trace.posterior["signal_mu"].mean().data,
                sigma=trace.posterior["signal_sigma"].mean().data,
            ),
            y,
        ).eval()
    )
    total = p_noise + p_signal
    p_noise /= total
    p_signal /= total

    # plot distributions
    fig, ax = plt.subplots()
    bins = np.linspace(-3.0, 4.0, 100)
    ax.hist(
        y[p_signal > 0.99],
        bins=bins,
        color="k",
        histtype="step",
        label="Signal",
        linewidth=2,
    )
    ax.hist(
        y[p_signal < 0.99],
        bins=bins,
        color="r",
        histtype="step",
        label="Noise",
        linewidth=2,
    )
    ax.hist(y, bins=bins, color="g", histtype="step", label="Data")
    ax.set_xlabel("M0")
    ax.set_ylabel("Number")
    ax.set_yscale("log")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # plot signal distribution
    fig, ax = plt.subplots()
    bins = np.linspace(0.0, 0.5, 50)
    ax.hist(
        np.log10(y[p_signal > 0.99]),
        bins=bins,
        color="k",
        histtype="step",
        label="Signal",
        linewidth=2,
    )
    ax.hist(np.log10(y), bins=bins, color="r", histtype="step", label="Data")
    ax.set_xlabel(r"log$_{10}$ M0")
    ax.set_ylabel("Number")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("signal.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main("C1.fits")
