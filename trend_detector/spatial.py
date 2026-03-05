import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pymc as pm
import pymc_extras as pmx
import xarray as xr
import numpy as np
import arviz as az
import os,glob,sys
from sklearn.preprocessing import StandardScaler
import pytensor.tensor as pt


def build_spatial_GEV_trend_model(model, data):
    ps = [1/10, 1/25, 1/100]
    y=data.values
    x_mean = np.average(y,axis=1)
    x_sd   = np.std(y,axis=1)
    years = np.arange(y.shape[1])
    t_std = (years - years.mean()) / years.std()  # standardized time

    coords = {"time": data.time.values,\
             "region":data.region.values}

    with model:
        for k, v in coords.items():
            model.add_coord(k, v)

        # Priors
        alpha = pm.Normal("alpha", mu=x_mean, sigma=2 * x_sd,dims="region")
        beta  = pm.Normal("beta",  mu=0.0, sigma=x_sd,dims="region")
        mu = pm.Deterministic(
            "mu",
            alpha[:, None] + beta[:, None] * t_std[None, :],
            dims=("region", "time")
            )
        
        log_sigma = pm.Normal(
            "log_sigma",
            mu=np.log(x_sd),
            sigma=0.5,
            dims="region"
        )
        
        sigma = pm.Deterministic(
            "sigma",
            pm.math.exp(log_sigma),
            dims="region"
        )
        
        xi = pm.TruncatedNormal(
            "xi",
            mu=0.0,
            sigma=0.05,
            lower=-0.2,
            upper=0.2,
            dims="region"
        )
        
        pmx.GenExtreme(
            "observed",
            mu=mu,
            sigma=sigma[:, None],
            xi=xi[:, None],
            observed=y,
            dims=("region", "time")
        )

        # # Return levels
        # zp = {}
        # for p in ps:
        #     zp[str(int(1/p))] = pm.Deterministic(
        #         f"z_{int(1/p)}",
        #         mu - sigma / xi * (1 - (-np.log(1 - p)) ** (-xi)),
        #         dims=("time")
        #     )

        return {"alpha": alpha, "beta": beta, "sigma": sigma, "xi": xi} #| zp


def build_spatial_GEV_switchpoint_model(model, data, lower=None, upper=None,use_Gumbel=False):
    """
    Build a GEV switchpoint model for data with dims ("region", "time").
    Assumes:
        - A single shared switchpoint across all regions
        - Region-specific mu_early, mu_late, xi_early, xi_late
        - Shared sigma across all regions

    Parameters:
        model : pm.Model
        data : xarray.DataArray with dims ("region", "time")
        lower, upper : optional bounds for switchpoint
    """
    y = data.values              # shape: (n_region, n_time)
    years = data.time.values     # shape: (n_time,)
    n_region, n_time = y.shape

    x_mean = float(np.mean(y))
    x_sd   = float(np.std(y))

    if lower is None:
        lower = years.min()
    if upper is None:
        upper = years.max()
    coords={"region":data.region.values,\
           "time":years}
    with model:
        model.add_coord("region",data.region.values)
        model.add_coord("time",years)
        # Shared switchpoint
        switchpoint = pm.DiscreteUniform("switchpoint", lower=lower, upper=upper)

        # Shared sigma
        log_sigma = pm.Normal("log_sigma", mu=np.log(x_sd), sigma=0.5)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        # Region-specific parameters
        mu_early = pm.Normal("mu_early", mu=x_mean, sigma=2 * x_sd, dims=("region"))#shape=n_region)
        mu_late  = pm.Normal("mu_late",  mu=x_mean, sigma=2 * x_sd, dims=("region"))#shape=n_region)

    

        # Allocate regimes: shape (n_region, n_time)
        # broadcast over regions
        years_shared = np.broadcast_to(years, (n_region, n_time))
        mu = pm.math.switch(years_shared <= switchpoint, mu_early[:, None], mu_late[:, None])
        
        if not use_Gumbel:
            xi_early = pm.TruncatedNormal("xi_early", mu=0.0, sigma=0.05, lower=-0.2, upper=0.2, dims=("region"))#shape=n_region)
            xi_late  = pm.TruncatedNormal("xi_late",  mu=0.0, sigma=0.05, lower=-0.2, upper=0.2, dims=("region"))#shape=n_region)
            xi = pm.math.switch(years_shared <= switchpoint, xi_early[:, None], xi_late[:, None])
            
        else:
            xi_early=0.
            xi_late = 0.0
            xi=0.0
        # Observed GEV
        pmx.GenExtreme("observed", mu=mu, sigma=sigma, xi=xi, observed=y)
        
  
        return {
            "switchpoint": switchpoint,
            "mu_early": mu_early, "mu_late": mu_late,
            "sigma": sigma,
            "xi_early": xi_early, "xi_late": xi_late
        }
     
            
