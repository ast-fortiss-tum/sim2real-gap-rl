from datetime import timedelta
from environments.smartgrid_env import SmartGrid_Nonlinear

def get_new_soc_env(degree, env_name):
    params_battery = {
        "rho": 0.1,
        "p_lim": 2.0,
        "etac": 0.6,    # nominal charging efficiency
        "etad": 0.7,    # nominal discharging efficiency
        "etas": degree  # modified self-discharge efficiency
    }
    env_instance = SmartGrid_Nonlinear(
        rl=True,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start="27.11.2016",
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        params_battery=params_battery
    )
    env_instance.env._max_episode_steps = 24
    return env_instance.env

def get_new_charge_env(degree, env_name):
    params_battery = {
        "rho": 0.1,
        "p_lim": 2.0,
        "etac": degree, # modified charging efficiency
        "etad": 0.7,    
        "etas": 0.8     
    }
    env_instance = SmartGrid_Nonlinear(
        rl=True,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start="27.11.2016",
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        params_battery=params_battery
    )
    env_instance.env._max_episode_steps = 24
    return env_instance.env

def get_new_discharge_env(degree, env_name):
    params_battery = {
        "rho": 0.1,
        "p_lim": 2.0,
        "etac": 0.6,    
        "etad": degree, # modified discharging efficiency
        "etas": 0.8     
    }
    env_instance = SmartGrid_Nonlinear(
        rl=True,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start="27.11.2016",
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        params_battery=params_battery
    )
    env_instance.env._max_episode_steps = 24
    return env_instance.env

def get_new_all_eff_env(degree, env_name):
    params_battery = {
        "rho": 0.1,
        "p_lim": 2.0,
        "etac": degree,
        "etad": degree,
        "etas": degree
    }
    env_instance = SmartGrid_Nonlinear(
        rl=True,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start="27.11.2016",
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        params_battery=params_battery
    )
    env_instance.env._max_episode_steps = 24
    return env_instance.env

def get_new_limited_capacity_env(nominal_capacity, nominal_p_lim, env_name):
    """
    Creates an environment with a reduced capacity.
    The capacity is set to 10% of nominal_capacity while p_lim remains nominal.
    """
    limited_capacity = nominal_capacity * 0.1
    params_battery = {
        "rho": 0.1,
        "p_lim": nominal_p_lim,
        "etac": 0.6,
        "etad": 0.7,
        "etas": 0.8
    }
    env_instance = SmartGrid_Nonlinear(
        rl=True,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start="27.11.2016",
        capacity=limited_capacity,
        data_path="./data/1-LV-rural2--1-sw",
        params_battery=params_battery
    )
    env_instance.env._max_episode_steps = 24
    return env_instance.env

def get_new_limited_plim_env(nominal_capacity, nominal_p_lim, env_name):
    """
    Creates an environment with a reduced power limit.
    The p_lim is set to 10% of nominal_p_lim while capacity remains nominal.
    """
    limited_p_lim = nominal_p_lim * 0.1
    params_battery = {
        "rho": 0.1,
        "p_lim": limited_p_lim,
        "etac": 0.6,
        "etad": 0.7,
        "etas": 0.8
    }
    env_instance = SmartGrid_Nonlinear(
        rl=True,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start="27.11.2016",
        capacity=nominal_capacity,
        data_path="./data/1-LV-rural2--1-sw",
        params_battery=params_battery
    )
    env_instance.env._max_episode_steps = 24
    return env_instance.env
