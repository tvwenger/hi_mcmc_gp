# hi_mcmc_gp
Use Gaussian processes to model HI distribution in a cloud

## Installation
```bash
conda create -n mcmc -c conda-forge pymc
conda activate mcmc
```

## Usage
Hopefully you have the data file `c1_M0.npy`. Then all you have to do is
```bash
python mcmc.py
```