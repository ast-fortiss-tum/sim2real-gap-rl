#!/usr/bin/env python3
import pathlib
import wandb
import matplotlib.pyplot as plt
from functools import partial
from datetime import datetime, timedelta

from models.sac import ContSAC
from models.darc import DARC
from utils import ZFilter
from environments.get_customized_envs import get_simple_linear_env
from architectures.gaussian_policy import ContGaussianPolicy
from architectures.value_networks import ContTwinQNet

from commonpower.modelling import ModelHistory
from commonpower.core import System, Node, Bus
from commonpower.models.busses import *
from commonpower.models.components import *
from commonpower.models.powerflow import *
from commonpower.control.controllers import RLControllerSB3, OptimalController, RLControllerSAC_Customized, RLBaseController
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer
from commonpower.control.safety_layer.penalties import DistanceDependingPenalty
from commonpower.control.runners import SingleAgentTrainer, DeploymentRunner
from commonpower.control.wrappers import SingleAgentWrapper
from commonpower.control.logging.loggers import TensorboardLogger
from commonpower.control.configs.algorithms import SB3MetaConfig, SB3PPOConfig, ContSACConfig
from commonpower.data_forecasting import CSVDataSource, LookBackForecaster, PerfectKnowledgeForecaster, DataProvider, ConstantDataSource
from commonpower.utils.helpers import get_adjusted_cost
from commonpower.utils.param_initialization import RangeInitializer
from stable_baselines3 import PPO, SAC
import tensorboard

# Configuration for simulation
n_hours = 24
horizon = timedelta(hours=n_hours)
frequency = timedelta(minutes=60)
fixed_start = datetime(2016, 11, 27)

# Path to data profiles
current_path = pathlib.Path().absolute()
data_path = (current_path / 'data' / '1-LV-rural2--1-sw').resolve()

# Create data sources
d11 = CSVDataSource(
    data_path / 'LoadProfile.csv',
    delimiter=";",
    datetime_format="%d.%m.%Y %H:%M",
    rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis", "G2-A_pload": "psi"},
    auto_drop=True,
    resample=timedelta(minutes=60)
)

ds1 = ConstantDataSource({
            "psis": 0.08,  # price units for kWh exported to the external grid
            "psib": 0.34   # price units for kWh imported from the external grid
            },
            date_range=d11.get_date_range(),
            frequency=timedelta(minutes=60)
        )
ds2 = CSVDataSource(
    data_path / 'LoadProfile.csv',
    delimiter=";",
    datetime_format="%d.%m.%Y %H:%M",
    rename_dict={"time": "t", "H0-A_pload": "p", "H0-A_qload": "q"},
    auto_drop=True,
    resample=timedelta(minutes=60)
)
ds3 = CSVDataSource(
    data_path / 'RESProfile.csv',
    delimiter=";",
    datetime_format="%d.%m.%Y %H:%M",
    rename_dict={"time": "t", "PV3": "p"},
    auto_drop=True,
    resample=timedelta(minutes=60)
).apply_to_column("p", lambda x: -x)

# Create DataProviders
dp1 = DataProvider(ds1, LookBackForecaster(frequency=frequency, horizon=horizon))
dp2 = DataProvider(ds2, LookBackForecaster(frequency=frequency, horizon=horizon))
dp3 = DataProvider(ds3, PerfectKnowledgeForecaster(frequency=frequency, horizon=horizon))

# Define nodes
n1 = RTPricedBus("Household").add_data_provider(dp1)
m1 = ExternalGrid("ExternalGrid")

# Define components
capacity = 3  # kWh
e1 = ESSLinear("ESS1", {
    'rho': 0.1,
    'p': (-1.5, 1.5),
    'q': (0, 0),
    'soc': (0.2 * capacity, 0.8 * capacity),
    "soc_init": RangeInitializer(0.2 * capacity, 0.8 * capacity)
})
r1 = RenewableGen("PV1").add_data_provider(dp3)
d1 = Load("Load1").add_data_provider(dp2)

# Build the system
sys = System(power_flow_model=PowerBalanceModel()).add_node(n1).add_node(m1)
n1.add_node(d1).add_node(e1).add_node(r1)
sys.pprint()
print(sys.controllers)

# ----------------------------
# Train the RL agent using RLControllerSB3
# ----------------------------
agent1 = RLControllerSB3(
    name='agent1', 
    safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=0.001)),
)
agent2 = RLBaseController(
    name="BasedRL",
    safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=0.001)),
)

# specify a seed for the random number generator used during training (It is common to train with ~5 different
# random seeds when you are, for example, testing a new safeguarding approach. For this notebook, one seed is enough.
# It will improve reproducibility of results.)
training_seed = 42

# set up configuration for the PPO algorithm
# we use pydantic classes for configuration of algorithms

# ----------------------------
# Deployment using RLControllerSB3 (original RL agent)
# ----------------------------
eval_seed = 5
oc_model_history = ModelHistory([sys])
oc_deployer = DeploymentRunner(
    sys=sys, 
    global_controller=OptimalController('global'), 
    forecast_horizon=horizon,
    control_horizon=horizon,
    history=oc_model_history,
    seed=eval_seed
)

oc_deployer.run(n_steps=24, fixed_start=fixed_start)
# we retrieve logs for the system cost
oc_power_import_cost = oc_model_history.get_history_for_element(m1, name='cost') # cost for buying electricity
oc_dispatch_cost = oc_model_history.get_history_for_element(n1, name='cost') # cost for operating the components in the household
oc_total_cost = [(oc_power_import_cost[t][0], oc_power_import_cost[t][1] + oc_dispatch_cost[t][1]) for t in range(len(oc_power_import_cost))]
oc_soc = oc_model_history.get_history_for_element(e1, name="soc") # state of charge

# plotting the cost of RL agent and optimal controller

plt.plot(range(len(oc_total_cost)), [x[1] for x in oc_total_cost], label="Cost optimal control")
plt.xticks(rotation=45)
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.title("Comparison of household cost for RL and optimal controller")
plt.tight_layout()
plt.legend()
#plt.show()

# ----------------------------
# New setting: Deployment using Customized SAC Controller
# ----------------------------
# Here we use our RLControllerSAC_Customized to deploy a SAC-based agent.
# Replace YOUR_SAC_MODEL_INSTANCE with your actual SAC model instance (e.g., a trained ContSAC or DARC).

policy_config = {
    "input_dim": [126],
    "architecture": [
        {"name": "linear1", "size": 256},
        {"name": "linear2", "size": 256},
        {"name": "split1", "sizes": [1, 1]}
    ],
    "hidden_activation": "relu",
    "output_activation": "none"
}
value_config = {
    "input_dim": [126 + 1],
    "architecture": [
        {"name": "linear1", "size": 256},
        {"name": "linear2", "size": 256},
        {"name": "linear2", "size": 1}
    ],
    "hidden_activation": "relu",
    "output_activation": "none"
}
running_state = ZFilter((126,), clip=20)

alg_config_custom = ContSACConfig(
    policy_config=policy_config,
    value_config=value_config,
    action_range=(-1.5, 1.5),
    device="cpu",
    load_path="/home/cubos98/Desktop/MA/DARAIL/saved_weights/DARC__20250325_041917_lr0.0008_noise0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids/DARC__20250325_041917_lr0.0008_noise0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids",
    running_mean=ZFilter((126,), clip=20),
    policy=ContGaussianPolicy(policy_config, (-1.5, 1.5)),
    twin_q=ContTwinQNet(value_config),
    target_twin_q=ContTwinQNet(value_config),
    algorithm = None,
    seed = 42
)
agent3 = RLControllerSAC_Customized(
    name='Custom',
    safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=0.001)),
    #sac_model=ContSAC(policy_config, value_config, env, "cpu"),
    #load_path="DARC__20250325_041917_lr0.0008_noise0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids/DARC__20250325_041917_lr0.0008_noise0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids"
)

alg_config = SB3MetaConfig(
    total_steps=50*int(horizon.total_seconds() // 3600),
    seed=training_seed,
    algorithm=PPO,
    penalty_factor=0.001,
    algorithm_config=SB3PPOConfig(
        n_steps=int(horizon.total_seconds() // 3600),
        learning_rate=0.0008,
        batch_size=12,
    )  # default hyperparameters for PPO
)

rl_runner = SingleAgentTrainer(
    sys=sys, 
    policy=ContGaussianPolicy(policy_config, (-1.5, 1.5)),
    global_controller=agent3, 
    wrapper=SingleAgentWrapper, 
    alg_config=alg_config_custom, 
    forecast_horizon=horizon,
    control_horizon=horizon,
    save_path="tryOut", 
    seed=alg_config.seed
)

rl_runner.fixed_start = datetime.strptime("27.11.2016", "%d.%m.%Y")
rl_runner.prepare_run()
#print(runner.env.action_space)
#print(runner.env.observation_space)

env = rl_runner.env

soc_dim = 1

agent3 = RLControllerSAC_Customized(
    name='Custom',
    safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=0.001)),
    #sac_model=ContSAC(policy_config, value_config, env, "cpu"),
    #load_path="DARC__20250325_041917_lr0.0008_noise0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids/DARC__20250325_041917_lr0.0008_noise0_seed42_lin_src1_varietyv_degree0.5_Smart_Grids"
)
# Save the customized controller (if not already saved) and then load it.
#customized_controller.save(save_path="saved_weights/my_custom_model_folder")
#customized_controller.load(env, device="cpu")

custom_model_history = ModelHistory([sys])
custom_deployer = DeploymentRunner(
    sys=sys,
    global_controller=agent3,
    alg_config=alg_config_custom,
    wrapper=SingleAgentWrapper,
    forecast_horizon=horizon,
    control_horizon=horizon,
    history=custom_model_history,
    seed=eval_seed
)
custom_deployer.run(n_steps=24, fixed_start=fixed_start)
custom_power_import_cost = custom_model_history.get_history_for_element(m1, name='cost')
custom_dispatch_cost = custom_model_history.get_history_for_element(n1, name='cost')
custom_total_cost = [
    (custom_power_import_cost[t][0], custom_power_import_cost[t][1] + custom_dispatch_cost[t][1])
    for t in range(len(custom_power_import_cost))
]
custom_soc = custom_model_history.get_history_for_element(e1, name="soc")

# ----------------------------
# Plot cost comparison between original RL agent and Customized SAC Controller
# ----------------------------
plt.figure()
plt.plot(
    range(len(custom_total_cost)), [x[1] for x in custom_total_cost],
    label="Cost RLControllerSB3", marker="o"
)
plt.plot(
    range(len(custom_total_cost)), [x[1] for x in custom_total_cost],
    label="Cost Customized Controller", marker="s"
)
plt.xticks(
    ticks=range(len(custom_power_import_cost)), 
    labels=[x[0] for x in custom_power_import_cost], rotation=45
)
plt.xlabel("Timestamp")
plt.ylabel("Cost")
plt.title("Comparison of Household Cost")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Plot SOC (State of Charge) comparison
# ----------------------------
plt.figure()
plt.plot(
    range(len(custom_soc)), [x[1] for x in custom_soc],
    label="SOC RLControllerSB3", marker="o"
)
plt.plot(
    range(len(custom_soc)), [x[1] for x in custom_soc],
    label="SOC Customized Controller", marker="s"
)
plt.xticks(
    ticks=range(len(rl_soc)),
    labels=[x[0] for x in rl_soc], rotation=45
)
plt.xlabel("Timestamp")
plt.ylabel("State of Charge (SOC)")
plt.title("Comparison of Battery SOC")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Compare daily cost
# ----------------------------
cost_day_rl = sum(get_adjusted_cost(custom_model_history, sys))
cost_day_custom = sum(get_adjusted_cost(custom_model_history, sys))
print(f"Daily cost:\n RLControllerSB3: {cost_day_rl}\n Customized Controller: {cost_day_custom}")
