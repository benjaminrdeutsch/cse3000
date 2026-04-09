import pandas as pd

POWER_PER_CORE_KW = 0.05  # 50W per CPU core


def compute_energy_kwh(jobs_df, power_per_core_kw=POWER_PER_CORE_KW):
    """
    Adds a column 'energy_kwh' to the jobs dataframe.
    """
    df = jobs_df.copy()

    # df = df.loc[:, ~df.columns.duplicated()]
    df["energy_kwh"] = df["cpu_cores"] * df["runtime_hours"] * power_per_core_kw
    

    # if isinstance(power_per_core_kw, pd.DataFrame):
    #     # Grabs the specific 'cpu_cores' column value based on the assignment hint
    #     if "cpu_cores" in power_per_core_kw.columns:
    #         kw_val = power_per_core_kw["cpu_cores"].iloc[0]
    #     else:
    #         kw_val = power_per_core_kw.iloc[0, 0]
    # elif isinstance(power_per_core_kw, dict):
    #     kw_val = power_per_core_kw.get("cpu_cores", 0.05)
    # else:
    #     kw_val = power_per_core_kw
        
    # # Now we multiply by a clean, 1D float
    # df["energy_kwh"] = df["cpu_cores"] * df["runtime_hours"] * float(kw_val)

    return df


def compute_emissions(jobs_df, carbon_intensity):
    """
    Given a jobs dataframe with 'energy_kwh',
    returns total emissions (kg CO2) for the region.
    """
    if "energy_kwh" not in jobs_df.columns:
        raise ValueError("energy_kwh column missing")

    total_energy = jobs_df["energy_kwh"].sum()
    return total_energy * carbon_intensity


def emissions_by_region(jobs_df, carbon_intensity_dict):
    """
    Returns a dictionary mapping region -> total emissions.
    """
    results = {}
    for region, intensity in carbon_intensity_dict.items():
        results[region] = compute_emissions(jobs_df, intensity)
    return results


def compute_total_runtime(jobs_df, max_cores):
    """
    Approximate total runtime given parallelism limit. 
    Requires dictionary for max_cores_per_region with region as key.
    """
    # Sort jobs by cores descending
    df = jobs_df.sort_values("cpu_cores", ascending=False).copy()
    remaining_jobs = df.to_dict("records")
    
    total_time = 0
    while remaining_jobs:
        # Assign jobs to available cores
        available_cores = max_cores
        batch_time = 0
        next_remaining = []
        for job in remaining_jobs:
            if job["cpu_cores"] <= available_cores:
                batch_time = max(batch_time, job["runtime_hours"])
                available_cores -= job["cpu_cores"]
            else:
                next_remaining.append(job)
        total_time += batch_time
        remaining_jobs = next_remaining
    return total_time



