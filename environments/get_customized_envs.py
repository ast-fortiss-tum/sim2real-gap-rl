from datetime import timedelta
from environments.smartgrid_env import SmartGrid_Linear, SmartGrid_Nonlinear, SmartGrid_TwoHouses

def get_simple_linear_env(seed, rl=True, fixed_start="27.11.2016"):
    """
    Factory function to create a linear SmartGrid environment.
    Returns a gym.Env instance (ControlEnv) configured via SmartGrid_Linear.
    """
    env_instance = SmartGrid_Linear(
        rl=rl,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start=fixed_start,
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        seed=seed,
        params_battery={"rho": 0.1, "p_lim": 1.5}
    )
    # Set maximum episode steps.
    env_instance.setup_system()
    env_instance.setup_runner_trainer(rl=rl)
    env_instance.env._max_episode_steps = 24
    return env_instance

def get_new_soc_env(degree, seed, rl=True, fixed_start="27.11.2016"):
    params_battery = {
        "rho": 0.1,
        "p_lim": 1.5,
        "etac": 1.0,    # nominal charging efficiency
        "etad": 1.0,    # nominal discharging efficiency
        "etas": degree  # modified self-discharge efficiency
    }
    env_instance = SmartGrid_Nonlinear(
        rl=rl,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start=fixed_start,
        capacity=1,
        data_path="./data/1-LV-rural2--1-sw",
        seed=seed,
        params_battery=params_battery
    )
    env_instance.setup_system()
    env_instance.setup_runner_trainer(rl=rl)
    env_instance.env._max_episode_steps = 24
    print(env_instance.env)
    return env_instance

def get_new_charge_env(degree, seed, rl=True, fixed_start="27.11.2016"):
    params_battery = {
        "rho": 0.1,
        "p_lim": 1.5,
        "etac": degree, # modified charging efficiency
        "etad": 1.0,    
        "etas": 1.0     
    }
    env_instance = SmartGrid_Nonlinear(
        rl=rl,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start=fixed_start,
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        seed=seed,
        params_battery=params_battery
    )
    env_instance.setup_system()
    env_instance.setup_runner_trainer(rl=rl)
    env_instance.env._max_episode_steps = 24
    return env_instance

def get_new_discharge_env(degree, seed, rl=True, fixed_start="27.11.2016"):
    params_battery = {
        "rho": 0.1,
        "p_lim": 1.5,
        "etac": 1.0,    
        "etad": degree, # modified discharging efficiency
        "etas": 1.0 
    }
    env_instance = SmartGrid_Nonlinear(
        rl=rl,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start=fixed_start,
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        seed=seed,
        params_battery=params_battery
    )
    env_instance.setup_system()
    env_instance.setup_runner_trainer(rl=rl)
    env_instance.env._max_episode_steps = 24
    return env_instance

def get_new_all_eff_env(degree, seed, rl=True, fixed_start="27.11.2016"):
    params_battery = {
        "rho": 0.1,
        "p_lim": 1.5,
        "etac": degree,
        "etad": degree,
        "etas": degree
    }
    env_instance = SmartGrid_Nonlinear(
        rl=rl,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start=fixed_start,
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        seed=seed,
        params_battery=params_battery
    )
    env_instance.setup_system()
    env_instance.setup_runner_trainer(rl=rl)
    env_instance.env._max_episode_steps = 24
    return env_instance

def get_new_limited_capacity_env(nominal_capacity, nominal_p_lim, seed, rl=True, fixed_start="27.11.2016"):
    """
    Creates an environment with a reduced capacity.
    The capacity is set to 10% of nominal_capacity while p_lim remains nominal.
    """
    params_battery = {
        "rho": 0.1,
        "p_lim": nominal_p_lim,
        "etac": 1.0,
        "etad": 1.0,
        "etas": 1.0
    }
    env_instance = SmartGrid_Nonlinear(
        rl=rl,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start=fixed_start,
        capacity=nominal_capacity,
        data_path="./data/1-LV-rural2--1-sw",
        seed=seed,
        params_battery=params_battery
    )
    env_instance.setup_system()
    env_instance.setup_runner_trainer(rl=rl)
    env_instance.env._max_episode_steps = 24
    return env_instance

def get_new_limited_plim_env(nominal_capacity, nominal_p_lim, seed, rl=True, fixed_start="27.11.2016"):
    """
    Creates an environment with a reduced power limit.
    The p_lim is set to 10% of nominal_p_lim while capacity remains nominal.
    """
    params_battery = {
        "rho": 0.1,
        "p_lim": nominal_p_lim,
        "etac": 1.0,
        "etad": 1.0,
        "etas": 1.0
    }
    env_instance = SmartGrid_Nonlinear(
        rl=rl,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start=fixed_start,
        capacity=nominal_capacity,
        data_path="./data/1-LV-rural2--1-sw",
        seed=seed,
        params_battery=params_battery
    )
    env_instance.setup_system()
    env_instance.setup_runner_trainer(rl=rl)
    env_instance.env._max_episode_steps = 24
    return env_instance

def get_twoHouses_env(damaged_battery, seed, rl=True, fixed_start="27.11.2016"):
    # Define battery parameters.
    params_battery = {
        "rho": 0.1,    # wear cost per kWh
        "p_lim": 1,     # power limit
        "etac": 1.0,   # charging efficiency
        "etad": 1.0,   # discharging efficiency
        "etas": 1.0    # self-discharge rate
    }
    
    # Create the SmartGrid_TwoHouses instance.
    env_instance = SmartGrid_TwoHouses(
        params_battery=params_battery,
        fixed_start=fixed_start,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        capacity=1,
        seed=seed,
        battery2_damaged=damaged_battery  # Set True to simulate a damaged battery for household 2.
    )
    
    # Explicitly set up the system and runner/trainer.
    env_instance.setup_system()
    env_instance.setup_runner_trainer(rl=rl)
    env_instance.env._max_episode_steps = 24
    
    print("SmartGrid_TwoHouses grid has been set up successfully.")

    return env_instance
