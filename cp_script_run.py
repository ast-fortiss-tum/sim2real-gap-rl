import pathlib
import wandb
import matplotlib.pyplot as plt
from functools import partial
from commonpower.modelling import ModelHistory
from commonpower.core import System, Node, Bus
from commonpower.models.busses import *
from commonpower.models.components import *
from commonpower.models.powerflow import *
from commonpower.control.controllers import RLControllerSB3, OptimalController
from commonpower.control.safety_layer.safety_layers import ActionProjectionSafetyLayer
from commonpower.control.safety_layer.penalties import *
from commonpower.control.runners import SingleAgentTrainer, DeploymentRunner
from commonpower.control.wrappers import SingleAgentWrapper
from commonpower.control.logging.loggers import *
from commonpower.control.configs.algorithms import *
from commonpower.data_forecasting import *
from commonpower.utils.helpers import get_adjusted_cost
from commonpower.utils.param_initialization import *
from stable_baselines3 import PPO, SAC
import tensorboard
import time

n_hours = 24
horizon = timedelta(hours=n_hours)
frequency = timedelta(minutes=60)
fixed_start = datetime(2016, 11, 27)

# path to data profiles
current_path = pathlib.Path().absolute()
data_path = current_path / 'data' / '1-LV-rural2--1-sw'
data_path = data_path.resolve()

ds1 = CSVDataSource(data_path  / 'LoadProfile.csv',
            delimiter=";", 
            datetime_format="%d.%m.%Y %H:%M", 
            rename_dict={"time": "t", "H0-A_pload": "p", "H0-A_qload": "q"},
            auto_drop=True, 
            resample=timedelta(minutes=60))

ds2 = CSVDataSource(data_path / 'LoadProfile.csv',
            delimiter=";", 
            datetime_format="%d.%m.%Y %H:%M", 
            rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis", "G2-A_pload": "psi"},
            auto_drop=True, 
            resample=timedelta(minutes=60))

ds3 = CSVDataSource(data_path / 'RESProfile.csv', 
        delimiter=";", 
        datetime_format="%d.%m.%Y %H:%M", 
        rename_dict={"time": "t", "PV3": "p"},
        auto_drop=True, 
        resample=timedelta(minutes=60)).apply_to_column("p", lambda x: -x)

dp1 = DataProvider(ds1, LookBackForecaster(frequency=frequency, horizon=horizon))
dp2 = DataProvider(ds2, LookBackForecaster(frequency=frequency, horizon=horizon))
dp3 = DataProvider(ds3, PerfectKnowledgeForecaster(frequency=frequency, horizon=horizon))

# nodes
n1 = Bus("MultiFamilyHouse", {
    'p': (-50, 50),
    'q': (-50, 50),
    'v': (0.95, 1.05),
    'd': (-15, 15)
})

# trading unit with price data for buying and selling electricity (to reduce problem complexity, we assume that
# prices for selling and buying are the same --> TradingLinear)
m1 = TradingBusLinear("Trading1", {
    'p': (-50, 50),
    'q': (-50, 50)
}).add_data_provider(dp2)

# components
# energy storage sytem
capacity = 3  #kWh
e1 = ESSLinear("ESS1", {
    'rho': 0.1, 
    'p': (-1.5, 1.5), 
    'q': (0, 0), 
    'soc': (0.2 * capacity, 0.8 * capacity), 
    "soc_init": RangeInitializer(0.2 * capacity, 0.8 * capacity)
})

# photovoltaic with generation data
r1 = RenewableGen("PV1").add_data_provider(dp3)

# static load with data source
d1 = Load("Load1").add_data_provider(dp1)

# we first have to add the nodes to the system 
# and then add components to the node in order to obtain a tree-like structure
sys = System(power_flow_model=PowerBalanceModel()).add_node(n1).add_node(m1)

# add components to nodes
n1.add_node(d1).add_node(e1).add_node(r1)

# show system structure: 
sys.pprint()

print(sys.controllers)

agent1 = RLControllerSB3(
    name='agent1', 
    safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=0.001)),
)

# specify a seed for the random number generator used during training (It is common to train with ~5 different
# random seeds when you are, for example, testing a new safeguarding approach. For this notebook, one seed is enough.
# It will improve reproducibility of results.)
training_seed = 42

# set up configuration for the PPO algorithm
# we use pydantic classes for configuration of algorithms
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

# set up logger
log_dir = './test_run/'
logger = TensorboardLogger(log_dir='./test_run/')
# You can also use Weights&Biases to monitor training. If you uncomment the next line, make sure to exchange the 
# "entity_name" parameter!
# logger = WandBLogger(log_dir='./test_run/', entity_name="srl4ps", project_name="commonpower", alg_config=alg_config, callback=WandBSafetyCallback)


# specify the path where the model should be saved
model_path = "./saved_models/my_model_original"

print(sys.controllers)


runner = SingleAgentTrainer(
    sys=sys, 
    global_controller=agent1, 
    wrapper=SingleAgentWrapper, 
    alg_config=alg_config, 
    forecast_horizon=horizon,
    control_horizon=horizon,
    logger=logger,
    save_path=model_path, 
    seed=alg_config.seed
)

runner.fixed_start = datetime.strptime("27.11.2016", "%d.%m.%Y")
runner.prepare_run()
#print(runner.env.action_space)
#print(runner.env.observation_space)
runner.run(fixed_start=fixed_start)

# Just for demonstration purposes, we show here how to load a pre-trained policy
# However, in the present case this would not be necessary, since "agent1" has saved the policy after training

# First, we need to create a new agent and pass the pretrained_policy_path from which to load the neural network 
# params. Adding n1 to this agent will create a warning since we are overwriting agent1, which is desired in this case
"""
agent2 = RLControllerSB3(
    name="pretrained_agent", 
    safety_layer=ActionProjectionSafetyLayer(penalty=DistanceDependingPenalty(penalty_factor=alg_config.penalty_factor)),
    pretrained_policy_path=model_path
)"""

# The deployment runner has to be instantiated with the same arguments used during training
# The runner will automatically recognize that it has to load the policy for agent2
# To ensure proper comparison of the trained RL agent with an optimal controller, we use the same seed for both
eval_seed = 5

rl_model_history = ModelHistory([sys])
rl_deployer = DeploymentRunner(
    sys=sys, 
    global_controller=agent1,  
    alg_config=alg_config,
    wrapper=SingleAgentWrapper,
    forecast_horizon=horizon,
    control_horizon=horizon,
    history=rl_model_history,
    seed=eval_seed
)
# Finally, we can simulate the system with the trained controller for the given day
rl_deployer.run(n_steps=24, fixed_start=fixed_start)

# let us extract some logs for comparison with an optimal controller
# We want to compare the cost of the household over the curse of the day. 
rl_power_import_cost = rl_model_history.get_history_for_element(m1, name='cost') # cost for buying electricity
rl_dispatch_cost = rl_model_history.get_history_for_element(n1, name='cost') # cost for operating the components in the household
rl_total_cost = [(rl_power_import_cost[t][0], rl_power_import_cost[t][1] + rl_dispatch_cost[t][1]) for t in range(len(rl_power_import_cost))]
rl_soc = rl_model_history.get_history_for_element(e1, name="soc") # state of charge of the battery

# We can use the same system but we have to set up a new runner. 
# This time, the global controller will take over the control of the household
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
print(oc_deployer.controllers)

time.sleep(20)
# we retrieve logs for the system cost
oc_power_import_cost = oc_model_history.get_history_for_element(m1, name='cost') # cost for buying electricity
oc_dispatch_cost = oc_model_history.get_history_for_element(n1, name='cost') # cost for operating the components in the household
oc_total_cost = [(oc_power_import_cost[t][0], oc_power_import_cost[t][1] + oc_dispatch_cost[t][1]) for t in range(len(oc_power_import_cost))]
oc_soc = oc_model_history.get_history_for_element(e1, name="soc") # state of charge

# plotting the cost of RL agent and optimal controller
plt.plot(range(len(rl_total_cost)), [x[1] for x in rl_total_cost], label="Cost RL")
plt.plot(range(len(oc_total_cost)), [x[1] for x in oc_total_cost], label="Cost optimal control")
plt.xticks(ticks=range(len(rl_power_import_cost)), labels=[x[0] for x in rl_power_import_cost])
plt.xticks(rotation=45)
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.title("Comparison of household cost for RL and optimal controller")
plt.tight_layout()
plt.legend()
plt.show()

# plotting the state of charge of the batteries
plt.plot(range(len(rl_soc)), [x[1] for x in rl_soc], label="SOC RL")
plt.plot(range(len(oc_soc)), [x[1] for x in oc_soc], label="SOC optimal control")
plt.xticks(ticks=range(len(rl_soc)), labels=[x[0] for x in rl_soc])
plt.xticks(rotation=45)
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.title("Comparison of battery state of charge (SOC) for RL and optimal controller")
plt.tight_layout()
plt.legend()
plt.show()

# We compare controllers by tracking the realized cost until the last timestep. 
# The cost of the last timestep is the accumulated cost of the projected horizon. 
# Since the projection is computed by the system's "internal" solver, which is by definition optimal wrt. to the system's cost function, this represents the "best case" cost (subject to the forecaster).
# This makes sure that costs realized in the future, e.g. by discharing batteries, is considered in the comparison.
cost_day_rl = sum(get_adjusted_cost(rl_model_history, sys))
cost_day_oc = sum(get_adjusted_cost(oc_model_history, sys))
print(f"The daily cost \n a) with the RL controller: {cost_day_rl} \n b) with the optimal controller: {cost_day_oc}")

