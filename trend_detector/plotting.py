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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import regionmask
import cartopy.crs as ccrs



def plot_GEV_trend_with_predictive(data, trace, var_mu="mu", var_sigma="sigma", var_xi="xi", n_ppc=1000):
    """
    Plot RX1day data with:
    - Posterior mean and 95% credible interval for μ(t)
    - Posterior predictive intervals for RX1day (full GEV)

    Parameters
    ----------
    data : pd.Series
        Observed annual maxima, indexed by time.
    trace : arviz.InferenceData
        Posterior trace from the trend model.
    var_mu : str
        Name of the mu variable in the trace.
    var_sigma : str
        Name of the sigma variable in the trace.
    var_xi : str
        Name of the xi variable in the trace.
    n_ppc : int
        Number of posterior predictive samples to draw.
    """
    # Extract posterior samples
    mu_samples = trace.posterior[var_mu].stack(draws=("chain","draw")).values  # shape: (time, n_samples)
    sigma_samples = trace.posterior[var_sigma].stack(draws=("chain","draw")).values  # shape: n_samples
    xi_samples = trace.posterior[var_xi].stack(draws=("chain","draw")).values  # shape: n_samples

    n_samples = mu_samples.shape[1]
    idx = np.random.choice(n_samples, size=min(n_ppc, n_samples), replace=False)

    # Posterior mean and 95% CI for mu
    mu_mean = mu_samples.mean(axis=1)
    mu_lower = np.percentile(mu_samples, 2.5, axis=1)
    mu_upper = np.percentile(mu_samples, 97.5, axis=1)

    # Posterior predictive samples
    y_ppc = np.zeros((len(data), len(idx)))
    for i, s in enumerate(idx):
        # GEV: y = mu - sigma/xi * (1 - (-log(U))**(-xi)), U~Uniform(0,1)
        U = np.random.uniform(size=len(data))
        xi = xi_samples[s]
        sigma = sigma_samples[s]
        mu = mu_samples[:, s]
        if np.abs(xi) < 1e-6:
            y_ppc[:, i] = mu - sigma * np.log(-np.log(U))  # Gumbel limit
        else:
            y_ppc[:, i] = mu - sigma/xi * (1 - (-np.log(U))**(-xi))

    ppc_lower = np.percentile(y_ppc, 2.5, axis=1)
    ppc_upper = np.percentile(y_ppc, 97.5, axis=1)
    ppc_median = np.median(y_ppc, axis=1)

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(data.time.values, data.values, 'o', color='black', label='Observed RX1day')
    plt.plot(data.time.values, mu_mean, '-', color='red', label='Posterior mean μ(t)')
    plt.fill_between(data.time.values, mu_lower, mu_upper, color='red', alpha=0.3, label='95% CI μ(t)')
    plt.plot(data.time.values, ppc_median, '--', color='blue', label='Posterior predictive median')
    plt.fill_between(data.time.values, ppc_lower, ppc_upper, color='blue', alpha=0.2, label='95% predictive interval')
    plt.xlabel("Year")
    plt.ylabel("RX1day (mm)")
    plt.title("RX1day with Posterior μ(t) and Predictive Intervals")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_ar6_land(X, cmap="viridis", projection=ccrs.Robinson(), grid_res=1.0):
    """
    Plot an xarray DataArray X with dimension 'region' on a map using AR6 land regions.
    
    Parameters:
        X : xarray.DataArray
            Must have dimension 'region' containing AR6 land region abbreviations.
        cmap : str
            Colormap for the plot.
        projection : cartopy.crs
            Cartopy projection for plotting.
        grid_res : float
            Resolution of lat/lon grid in degrees (higher = slower, lower = coarser).
    """
    if "region" not in X.dims:
        raise ValueError("X must have a dimension called 'region'.")

    ar6 = regionmask.defined_regions.ar6.land

    # Create lat/lon grid
    lon = np.arange(-180, 180 + grid_res, grid_res)
    lat = np.arange(-90, 90 + grid_res, grid_res)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Create AR6 region mask on this grid
    mask = ar6.mask(lon2d, lat2d)

    # Initialize empty grid
    X_map = np.full(mask.shape, np.nan, dtype=float)

    # Fill grid with X values
    for region_abbr, value in zip(X.region.values, X.values):
        rnum = ar6.map_keys([region_abbr])[0]  # AR6 region number
        X_map[mask == rnum] = value

    # Wrap as DataArray
    X_map_da = xr.DataArray(
        X_map,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"]
    )

    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=projection)
    
    X_map_da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        cbar_kwargs={"label": X.name or "Value"}
    )

    ax.coastlines()
    ax.set_global()
    ax.set_title(f"AR6 Land Regions: {X.name or ''}")
    return fig
    #plt.show()


def plot_ar6_ocean(X, cmap="viridis", projection=ccrs.Robinson(), grid_res=1.0):
    """
    Plot an xarray DataArray X with dimension 'region' on a map using AR6 land regions.
    
    Parameters:
        X : xarray.DataArray
            Must have dimension 'region' containing AR6 land region abbreviations.
        cmap : str
            Colormap for the plot.
        projection : cartopy.crs
            Cartopy projection for plotting.
        grid_res : float
            Resolution of lat/lon grid in degrees (higher = slower, lower = coarser).
    """
    if "region" not in X.dims:
        raise ValueError("X must have a dimension called 'region'.")

    ar6 = regionmask.defined_regions.ar6.ocean

    # Create lat/lon grid
    lon = np.arange(-180, 180 + grid_res, grid_res)
    lat = np.arange(-90, 90 + grid_res, grid_res)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Create AR6 region mask on this grid
    mask = ar6.mask(lon2d, lat2d)

    # Initialize empty grid
    X_map = np.full(mask.shape, np.nan, dtype=float)

    # Fill grid with X values
    for region_abbr, value in zip(X.region.values, X.values):
        rnum = ar6.map_keys([region_abbr])[0]  # AR6 region number
        X_map[mask == rnum] = value

    # Wrap as DataArray
    X_map_da = xr.DataArray(
        X_map,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"]
    )

    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=projection)
    
    X_map_da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        cbar_kwargs={"label": X.name or "Value"}
    )

    ax.coastlines()
    ax.set_global()
    ax.set_title(f"AR6 Ocean Regions: {X.name or ''}")

    plt.show()
