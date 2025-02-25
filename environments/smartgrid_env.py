from datetime import timedelta
from environments.SmartGrid import SmartGrid_Linear, SmartGrid_Nonlinear

# -----------------------------------------------------------------------------
# Factory functions to create the environments.
# -----------------------------------------------------------------------------

def make_smartgrid_linear():
    """
    Factory function to create a linear SmartGrid environment.
    Returns a gym.Env instance (ControlEnv) configured via SmartGrid_Linear.
    """
    env_instance = SmartGrid_Linear(
        rl=True,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start="27.11.2016",
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        params_battery={"rho": 0.1, "p_lim": 2.0}
    )
    # Set maximum episode steps.
    env_instance.env._max_episode_steps = 24
    return env_instance.env

def make_smartgrid_nonlinear():
    """
    Factory function to create a nonlinear SmartGrid environment.
    Returns a gym.Env instance (ControlEnv) configured via SmartGrid_Nonlinear.
    """
    env_instance = SmartGrid_Nonlinear(
        rl=True,
        policy_path=None,
        horizon=timedelta(hours=24),
        frequency=timedelta(minutes=60),
        fixed_start="27.11.2016",
        capacity=3,
        data_path="./data/1-LV-rural2--1-sw",
        params_battery={"rho": 0.1, "p_lim": 2.0, "etac": 0.6, "etad": 0.7, "etas": 0.8}
    )
    env_instance.env._max_episode_steps = 24
    return env_instance.env

