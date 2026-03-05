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

# Replace with  local data directory for github push

from .io import get_data_path
data_dir = get_data_path("ar6_regions")
REALM_DICT = {}

for realm in ["land", "ocean"]:
    realm_path = data_dir /realm  # pathlib Path object
    filenames = os.listdir(realm_path)
    
    regions = np.unique(
        sorted([f.split("_")[-1].split(".nc")[0] for f in filenames])
    )
    
    for region in regions:
        REALM_DICT[region] = realm
# data_dir = "/Users/kmarvel/Documents/PMM/DATA/ar6_regions/"
# realm="land"
# REALM_DICT={}
# land_regions= np.unique(sorted([x.split("_")[-1].split(".nc")[0] for x in os.listdir(data_dir+realm)]))
# for region in land_regions:
#     REALM_DICT[region]="land"

# realm="ocean"
# ocean_regions= np.unique(sorted([x.split("_")[-1].split(".nc")[0] for x in os.listdir(data_dir+realm)]))
# for region in ocean_regions:
#     REALM_DICT[region]="ocean"

import xarray as xr
from trend_detector.io import get_data_path

def get_regional_data(region, index):
    """
    Load a regional timeseries dataset.

    Parameters
    ----------
    region : str
        Region abbreviation (must be a key in REALM_DICT)
    index : str
        Variable name inside the netCDF file

    Returns
    -------
    xr.DataArray
        DataArray with "year" renamed to "time"
    """
    # Project-relative data directory
    data_dir = get_data_path("ar6_regions")         # Path object
    realm_dir = data_dir / REALM_DICT[region]

    # Pattern for matching files
    pattern = f"*ts_{region}.nc"
    files = sorted(realm_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found for region {region} in {realm_dir}")

    # Open the first matching dataset
    dataset = xr.open_dataset(files[0])
    
    # Extract the requested variable and rename 'year' -> 'time'
    data = getattr(dataset, index).rename({"year": "time"})
    
    # Optional: load into memory to avoid issues with context manager
    data = data.load()

    # Close the dataset
    dataset.close()

    return data

# def get_regional_data(region,index):
#     data_dir=get_data_path("")
#     realm_dir = data_dir / REALM_DICT[region]   # e.g., root/DATA/land
#     pattern = f"*ts_{region}.nc"

#     # use glob via Path.glob
#     regional_mean = sorted(realm_dir.glob(pattern))
#     with xr.open_dataset(regional_mean[0]) as dataset:
#         data=getattr(dataset,index).rename({"year":"time"})
#     return data
    
def poisson_physical_priors(y):
    """
    Construct physically meaningful priors for Poisson precipitation extremes.
    """
    mean_y = float(np.mean(y))
    std_y  = float(np.std(y))

    # Coefficient of variation (used to scale trend uncertainty)
    cv = std_y / mean_y if mean_y > 0 else 0.5

    # Log-rate prior
    log_rate_mu = np.log(mean_y)
    log_rate_sd = np.sqrt(np.log(1 + cv**2))  # lognormal approximation

    # Trend prior: fractional change over standardized time
    trend_sd = 0.5 * cv  # conservative, physically motivated

    return log_rate_mu, log_rate_sd, trend_sd

def gamma_physical_priors(y):
    """
    Physically motivated priors for Gamma-distributed precipitation variables.
    """
    mean_y = float(np.mean(y))
    std_y  = float(np.std(y))

    cv = std_y / mean_y if mean_y > 0 else 0.5

    mu_sd = 2.0 * std_y
    log_sigma_mu = np.log(cv)
    log_sigma_sd = 0.5

    trend_sd = 0.5 * cv

    return mean_y, mu_sd, log_sigma_mu, log_sigma_sd, trend_sd

## Build a stationary Poisson model 
def build_poisson_model_notrend(model,data):
    y=data.values
    years = np.arange(0, len(y))
    coords={"time":data.time.values}
    with model:
        for k,v in coords.items():
            model.add_coord(k,v)
        #observations=pm.Data("observations",y)
        # Priors for log-linear Poisson regression
        alpha = pm.Normal("alpha", mu=np.log(y.mean()), sigma=2)
        #beta = pm.Normal("beta", mu=0, sigma=1)  
        
        # Log-linear model
        mu = alpha #+ beta * years
        rate=pm.math.exp(mu)
        #rate = pm.Deterministic("rate",pm.math.exp(mu))
        #Interpretation of beta: percent change per year
        #trend=pm.Deterministic("trend",(pm.math.exp(beta)-1)*100)
        
        # Poisson likelihood
        observed = pm.Poisson("observed", mu=rate, observed=y,dims=("time"))
    
        
        return {"alpha":alpha
                }
    

## Build a nonstationary Poisson model with linear trend
def build_poisson_model_trend(model, data):
    y = data.values
    years = np.arange(len(y))

    # Standardize time (zero mean, unit variance)
    t_std = (years - years.mean()) / years.std()

    coords = {"time": data.time.values}

    with model:
        for k, v in coords.items():
            model.add_coord(k, v)

        # Physically motivated priors
        log_rate_mu, log_rate_sd, trend_sd = poisson_physical_priors(y)

        # Intercept: log mean precipitation rate
        alpha = pm.Normal(
            "alpha",
            mu=log_rate_mu,
            sigma=log_rate_sd,
        )

        # Trend: fractional change over standardized time
        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=trend_sd,
        )

        # Log-linear Poisson rate
        log_rate = alpha + beta * t_std
        rate = pm.Deterministic(
            "rate",
            pm.math.exp(log_rate),
            dims="time",
        )

        # Percent change per +1 SD of time
        trend = pm.Deterministic(
            "trend",
            (pm.math.exp(beta) - 1) * 100
        )

        # Likelihood
        observed = pm.Poisson(
            "observed",
            mu=rate,
            observed=y,
            dims="time",
        )

        return {
            "alpha": alpha,
            "beta": beta,
        }

 # Build a Poisson changepoint model with different trends in each 
def build_poisson_switchpoint_trend_model(
    model,
    data,
    lower=None,
    upper=None,
):
    y = data.values
    n = len(y)

    # Calendar years from the data
    years_cal = data.time.values

    # Convert calendar bounds → index bounds
    if lower is None:
        lower_idx = 0
    else:
        lower_idx = int(np.searchsorted(years_cal, lower, side="left"))

    if upper is None:
        upper_idx = n - 1
    else:
        upper_idx = int(np.searchsorted(years_cal, upper, side="right") - 1)

    # Safety
    lower_idx = max(0, lower_idx)
    upper_idx = min(n - 1, upper_idx)

    if lower_idx >= upper_idx:
        raise ValueError("Invalid switchpoint bounds after year→index conversion.")

    # Time index and standardized time
    t = np.arange(n)
    t_std = (t - t.mean()) / t.std()

    coords = {"time": years_cal}

    with model:
        for k, v in coords.items():
            model.add_coord(k, v)

        # Physical priors
        log_rate_mu, log_rate_sd, trend_sd = poisson_physical_priors(y)

        # ---- SWITCHPOINT (INDEX SPACE) ----
        switchpoint = pm.DiscreteUniform(
            "switchpoint",
            lower=lower_idx,
            upper=upper_idx,
        )

        # ---- EARLY SEGMENT ----
        alpha_early = pm.Normal(
            "alpha_early",
            mu=log_rate_mu,
            sigma=log_rate_sd,
        )
        beta_early = pm.Normal(
            "beta_early",
            mu=0.0,
            sigma=trend_sd,
        )

        # ---- LATE SEGMENT ----
        alpha_late = pm.Normal(
            "alpha_late",
            mu=log_rate_mu,
            sigma=log_rate_sd,
        )
        beta_late = pm.Normal(
            "beta_late",
            mu=0.0,
            sigma=trend_sd,
        )

        # Log-rates
        mu_early = alpha_early + beta_early * t_std
        mu_late  = alpha_late  + beta_late  * t_std

        mu = pm.math.switch(
            t <= switchpoint,
            mu_early,
            mu_late,
        )

        rate = pm.Deterministic(
            "rate",
            pm.math.exp(mu),
            dims="time",
        )

        # Interpretable trends
        pm.Deterministic("trend_early", (pm.math.exp(beta_early) - 1) * 100)
        pm.Deterministic("trend_late",  (pm.math.exp(beta_late)  - 1) * 100)

        pm.Poisson(
            "observed",
            mu=rate,
            observed=y,
            dims=("time",),
        )

    return {
        "switchpoint": switchpoint,
        "alpha_early": alpha_early,
        "beta_early": beta_early,
        "alpha_late": alpha_late,
        "beta_late": beta_late,
        "rate": rate,
    }

    
# Switchpoint, no trends
def build_poisson_model_switchpoint(
    model,
    data,
    lower=None,
    upper=None,
):
    y = data.values
    n = len(y)

    years_cal = data.time.values
    t = np.arange(n)

    # ---- Convert calendar-year bounds → index bounds ----
    if lower is None:
        lower_idx = 0
    else:
        lower_idx = int(np.searchsorted(years_cal, lower, side="left"))

    if upper is None:
        upper_idx = n - 1
    else:
        upper_idx = int(np.searchsorted(years_cal, upper, side="right") - 1)

    lower_idx = max(0, lower_idx)
    upper_idx = min(n - 1, upper_idx)

    if lower_idx >= upper_idx:
        raise ValueError("Invalid switchpoint bounds after year→index conversion.")

    coords = {"time": years_cal}

    with model:
        for k, v in coords.items():
            model.add_coord(k, v)

        # ---- Physical priors ----
        log_rate_mu, log_rate_sd, _ = poisson_physical_priors(y)

        # ---- Switchpoint (index space) ----
        switchpoint = pm.DiscreteUniform(
            "switchpoint",
            lower=lower_idx,
            upper=upper_idx,
        )

        # ---- Segment log-rates (NO TREND) ----
        early_log_rate = pm.Normal(
            "early_log_rate",
            mu=log_rate_mu,
            sigma=log_rate_sd,
        )

        late_log_rate = pm.Normal(
            "late_log_rate",
            mu=log_rate_mu,
            sigma=log_rate_sd,
        )

        # Select log-rate by segment
        mu = pm.math.switch(
            t <= switchpoint,
            early_log_rate,
            late_log_rate,
        )

        rate = pm.Deterministic(
            "rate",
            pm.math.exp(mu),
            dims="time",
        )

        # Optional interpretability
        
        pm.Deterministic("early_rate", pm.math.exp(early_log_rate))
        pm.Deterministic("late_rate",  pm.math.exp(late_log_rate))
        years_tensor = pt.as_tensor_variable(years_cal)
        pm.Deterministic("switchpoint_year", years_tensor[switchpoint])
        #pm.Deterministic("switchpoint_year", years_cal[switchpoint])

        # ---- Likelihood ----
        pm.Poisson(
            "observed",
            mu=rate,
            observed=y,
            dims=("time",),
        )

    return {
        "switchpoint": switchpoint,
        "early_log_rate": early_log_rate,
        "late_log_rate": late_log_rate,
        "rate": rate,
    }



# -------------------------------
# 1️ GEV model with linear trend
# -------------------------------
def build_GEV_model_trend(model, data):
    ps = [1/10, 1/25, 1/100]
    y = data.values
    x_mean = float(np.mean(y))
    x_sd   = float(np.std(y))
    years = np.arange(len(y))
    t_std = (years - years.mean()) / years.std()  # standardized time

    coords = {"time": data.time.values}

    with model:
        for k, v in coords.items():
            model.add_coord(k, v)

        # Priors
        alpha = pm.Normal("alpha", mu=x_mean, sigma=2 * x_sd)
        beta  = pm.Normal("beta",  mu=0.0, sigma=x_sd)
        mu = pm.Deterministic("mu", alpha + beta * t_std, dims="time")

        log_sigma = pm.Normal("log_sigma", mu=np.log(x_sd), sigma=0.5)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        xi = pm.TruncatedNormal("xi", mu=0.0, sigma=0.05, lower=-0.2, upper=0.2)

        # Likelihood
        pmx.GenExtreme("observed", mu=mu, sigma=sigma, xi=xi, observed=y)

        # Return levels
        zp = {}
        for p in ps:
            zp[str(int(1/p))] = pm.Deterministic(
                f"z_{int(1/p)}",
                mu - sigma / xi * (1 - (-np.log(1 - p)) ** (-xi)),
                dims=("time")
            )

        return {"alpha": alpha, "beta": beta, "sigma": sigma, "xi": xi} | zp

# -------------------------------
# 2️ GEV model without trend
# -------------------------------
def build_GEV_model_notrend(model, data):
    ps = [1/10, 1/25, 1/100]
    y = data.values
    x_mean = float(np.mean(y))
    x_sd   = float(np.std(y))
    coords = {"time": data.time.values}

    with model:
        for k, v in coords.items():
            model.add_coord(k, v)

        mu = pm.Normal("mu", mu=x_mean, sigma=2 * x_sd)

        log_sigma = pm.Normal("log_sigma", mu=np.log(x_sd), sigma=0.5)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        xi = pm.TruncatedNormal("xi", mu=0.0, sigma=0.05, lower=-0.2, upper=0.2)

        pmx.GenExtreme("observed", mu=mu, sigma=sigma, xi=xi, observed=y)

        zp = {}
        for p in ps:
            zp[str(int(1/p))] = pm.Deterministic(
                f"z_{int(1/p)}",
                mu - sigma / xi * (1 - (-np.log(1 - p)) ** (-xi))
            )

        return {"mu": mu, "sigma": sigma, "xi": xi} | zp

# -------------------------------
# 3️ GEV switchpoint model
# -------------------------------
def build_GEV_model_switchpoint(model, data, lower=None, upper=None):
    y = data.values
    years = data.time.values
    x_mean = float(np.mean(y))
    x_sd   = float(np.std(y))

    if lower is None:
        lower = years.min()
    if upper is None:
        upper = years.max()

    with model:
        # Discrete switchpoint
        switchpoint = pm.DiscreteUniform("switchpoint", lower=lower, upper=upper)

        # Shared log_sigma to reduce divergences
        log_sigma = pm.Normal("log_sigma", mu=np.log(x_sd), sigma=0.5)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        # Early regime μ and ξ
        mu_early = pm.Normal("mu_early", mu=x_mean, sigma=2 * x_sd)
        xi_early = pm.TruncatedNormal("xi_early", mu=0.0, sigma=0.05, lower=-0.2, upper=0.2)

        # Late regime μ and ξ
        mu_late = pm.Normal("mu_late", mu=x_mean, sigma=2 * x_sd)
        xi_late = pm.TruncatedNormal("xi_late", mu=0.0, sigma=0.05, lower=-0.2, upper=0.2)

        # Allocate regimes
        mu = pm.math.switch(years <= switchpoint, mu_early, mu_late)
        xi = pm.math.switch(years <= switchpoint, xi_early, xi_late)

        pmx.GenExtreme("observed", mu=mu, sigma=sigma, xi=xi, observed=y)

        return {
            "switchpoint": switchpoint,
            "mu_early": mu_early, "mu_late": mu_late,
            "sigma": sigma,
            "xi_early": xi_early, "xi_late": xi_late
        }


def build_GEV_switchpoint_trend_model(model, data, lower=None, upper=None):
    """
    GEV model with a discrete switchpoint. Before and after the switchpoint, the location parameter
    can have independent linear trends. The scale is shared across all time, xi can differ.
    """
    ps = [1/10, 1/25, 1/100]
    y = data.values
    years = np.arange(len(y))
    t_std = (years - years.mean()) / years.std()  # standardized time
    x_mean = float(np.mean(y))
    x_sd   = float(np.std(y))

    coords = {"time": data.time.values}
    
    if lower is None:
        lower = years.min()
    if upper is None:
        upper = years.max()
    
    with model:
        # Add coordinates
        for k, v in coords.items():
            model.add_coord(k, v)
        
        # Discrete switchpoint
        switchpoint = pm.DiscreteUniform("switchpoint", lower=lower, upper=upper)
        
        # Shared scale
        log_sigma = pm.Normal("log_sigma", mu=np.log(x_sd), sigma=0.5)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        
        # --- Priors for early segment ---
        alpha_early = pm.Normal("alpha_early", mu=x_mean, sigma=2*x_sd)
        beta_early  = pm.Normal("beta_early", mu=0.0, sigma=x_sd)
        xi_early    = pm.TruncatedNormal("xi_early", mu=0.0, sigma=0.05, lower=-0.2, upper=0.2)
        
        # --- Priors for late segment ---
        alpha_late = pm.Normal("alpha_late", mu=x_mean, sigma=2*x_sd)
        beta_late  = pm.Normal("beta_late", mu=0.0, sigma=x_sd)
        xi_late    = pm.TruncatedNormal("xi_late", mu=0.0, sigma=0.05, lower=-0.2, upper=0.2)
        
        # Allocate trends according to switchpoint
        mu_early_array = alpha_early + beta_early * t_std
        mu_late_array  = alpha_late  + beta_late  * t_std
        
        mu = pm.math.switch(years <= switchpoint, mu_early_array, mu_late_array)
        xi = pm.math.switch(years <= switchpoint, xi_early, xi_late)
        
        # Likelihood
        pmx.GenExtreme("observed", mu=mu, sigma=sigma, xi=xi, observed=y)
        
        # Return levels
        zp = {}
        for p in ps:
            zp[str(int(1/p))] = pm.Deterministic(
                f"z_{int(1/p)}",
                mu - sigma / xi * (1 - (-np.log(1 - p)) ** (-xi)),
                dims=("time")
            )
        
        return {
            "switchpoint": switchpoint,
            "alpha_early": alpha_early, "beta_early": beta_early,
            "alpha_late": alpha_late, "beta_late": beta_late,
            "sigma": sigma,
            "xi_early": xi_early, "xi_late": xi_late
        } | zp



# GAMMA distributions
def build_gamma_model_notrend(model, data):
    y = data.values
    coords = {"time": data.time.values}

    mean_y, mu_sd, log_sigma_mu, log_sigma_sd, _ = gamma_physical_priors(y)

    with model:
        for k, v in coords.items():
            model.add_coord(k, v)

        mu = pm.Normal(
            "mu",
            mu=mean_y,
            sigma=mu_sd
        )

        log_sigma = pm.Normal(
            "log_sigma",
            mu=log_sigma_mu,
            sigma=log_sigma_sd
        )

        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        pm.Gamma(
            "observed",
            mu=mu,
            sigma=sigma,
            observed=y,
            dims=("time",)
        )

    return {"mu": mu, "sigma": sigma}


def build_gamma_model_trend(model, data):
    y = data.values
    years = np.arange(len(y))
    t_std = (years - years.mean()) / years.std()

    coords = {"time": data.time.values}

    mean_y, mu_sd, log_sigma_mu, log_sigma_sd, trend_sd = gamma_physical_priors(y)

    with model:
        for k, v in coords.items():
            model.add_coord(k, v)

        alpha = pm.Normal(
            "alpha",
            mu=mean_y,
            sigma=mu_sd
        )

        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=trend_sd
        )

        mu = pm.Deterministic(
            "mu",
            alpha + beta * t_std,
            dims=("time",)
        )

        log_sigma = pm.Normal(
            "log_sigma",
            mu=log_sigma_mu,
            sigma=log_sigma_sd
        )

        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        pm.Gamma(
            "observed",
            mu=mu,
            sigma=sigma,
            observed=y,
            dims=("time",)
        )

        trend = pm.Deterministic(
            "trend_percent",
            beta / alpha * 100
        )

    return {"alpha": alpha, "beta": beta, "sigma": sigma}

def build_gamma_model_switchpoint(model, data, lower=None, upper=None):
    y = data.values
    years = data.time.values

    mean_y, mu_sd, log_sigma_mu, log_sigma_sd, _ = gamma_physical_priors(y)

    if lower is None:
        lower = years.min()
    if upper is None:
        upper = years.max()

    with model:
        switchpoint = pm.DiscreteUniform(
            "switchpoint",
            lower=lower,
            upper=upper
        )

        log_sigma = pm.Normal(
            "log_sigma",
            mu=log_sigma_mu,
            sigma=log_sigma_sd
        )
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))

        mu_early = pm.Normal(
            "mu_early",
            mu=mean_y,
            sigma=mu_sd
        )

        mu_late = pm.Normal(
            "mu_late",
            mu=mean_y,
            sigma=mu_sd
        )

        mu = pm.math.switch(
            years <= switchpoint,
            mu_early,
            mu_late
        )

        pm.Gamma(
            "observed",
            mu=mu,
            sigma=sigma,
            observed=y
        )

    return {
        "switchpoint": switchpoint,
        "mu_early": mu_early,
        "mu_late": mu_late,
        "sigma": sigma,
    }

class AR6ModelFactory:
    """
    Factory class for building AR6 precipitation extreme models and traces
    based on region and index.
    """

    # Map indices to distribution families
    INDEX_DISTRIBUTION_MAP = {
        "RX1day": "gev",
        "RX5day": "gev",
        "R50": "poisson",   
        "AnnualMean":"gamma"
    }

    # Map (distribution, model_type) → builder function
    MODEL_BUILDERS = {
        ("gev", "standard"): build_GEV_model_notrend,
        ("gev", "trend"): build_GEV_model_trend,
        ("gev", "switchpoint"): build_GEV_model_switchpoint,

        ("poisson", "standard"): build_poisson_model_notrend,
        ("poisson", "trend"): build_poisson_model_trend,
        ("poisson", "switchpoint"): build_poisson_model_switchpoint,

        ("gamma", "standard"): build_gamma_model_notrend,
        ("gamma", "trend"): build_gamma_model_trend,
        ("gamma", "switchpoint"): build_gamma_model_switchpoint,
    }

    def __init__(self, region: str, index: str):
        self.region = region
        self.index = index
        self.models = {}
        self.traces = {}
        self.posterior_predictive = {}

        if index not in self.INDEX_DISTRIBUTION_MAP:
            raise ValueError(
                f"Unknown index '{index}'. "
                f"Valid options: {list(self.INDEX_DISTRIBUTION_MAP.keys())}"
            )

        self.distribution = self.INDEX_DISTRIBUTION_MAP[index]

    def get_data(self):
        """Retrieve regional xarray DataArray."""
        return get_regional_data(self.region, self.index)

    def build(self, model_type: str = "standard", **kwargs):
        """
        Build and return a PyMC model.

        Parameters
        ----------
        model_type : {"standard", "trend", "switchpoint"}
        **kwargs : passed through to the ar6models build function
        """
        key = (self.distribution, model_type)

        if key not in self.MODEL_BUILDERS:
            raise ValueError(
                f"No model available for distribution='{self.distribution}', "
                f"model_type='{model_type}'"
            )

        build_fn = self.MODEL_BUILDERS[key]
        data = self.get_data()

        with pm.Model() as model:
            params = build_fn(model,data, **kwargs)

        return model

    def build_and_sample(
        self,
        model_type: str = "standard",
        rebuild: bool = False,
        resample: bool = False,
        sample_kwargs: dict | None = None,
        **build_kwargs,
    ):
        """
        Build (if needed) and sample (if needed) a PyMC model.
    
        Parameters
        ----------
        model_type : str
            {"standard", "trend", "switchpoint"}
        resample : bool, default False
                Force resampling even if a trace already exists.
            sample_kwargs : dict, optional
                Keyword arguments passed to pm.sample().
            **build_kwargs
                Passed to the model builder (e.g. switchpoint bounds).
        
            Returns
            -------
            model : pm.Model
            trace : arviz.InferenceData
            """

        if sample_kwargs is None:
            sample_kwargs = {}
    
        key = model_type#(self.distribution, model_type)
    
        # -----------------------------
        # Build model if needed
        # -----------------------------
        if (key not in self.models) or rebuild:
            print(f"Building {model_type} model")
            model = self.build(model_type=model_type, **build_kwargs)
            self.models[key] = model
        else:
            model = self.models[key]
    
        # -----------------------------
        # Sample if needed
        # -----------------------------
        if (key not in self.traces) or resample:
            print(f"Sampling {model_type} model")
            with model:
                trace = pm.sample(**sample_kwargs)
                ll = pm.compute_log_likelihood(trace,progressbar=False)
    
            self.traces[key] = trace
        else:
            trace = self.traces[key]
    
        return model, trace

    def compare(
        self,
        model_types: list[str] | None = None,
        ic: str = "loo",
        **compare_kwargs,
    ):
        """
        Compare sampled models using ArviZ.

        Parameters
        ----------
        model_types : list of str, optional
            Model types to compare (e.g. ["standard", "trend", "switchpoint"]).
            If None, compare all available traces.
        ic : str, default "loo"
            Information criterion ("loo" or "waic").
        **compare_kwargs
            Passed to az.compare().

        Returns
        -------
        pandas.DataFrame
            ArviZ comparison table.
        """
        if not self.traces:
            raise RuntimeError("No traces available to compare.")

        # Select traces
        if model_types is None:
            traces = self.traces
        else:
            traces = {
                k: v for k, v in self.traces.items()
                if k in model_types
            }

        if len(traces) < 2:
            raise ValueError("Need at least two models to compare.")

        return az.compare(traces, ic=ic, **compare_kwargs)


    
