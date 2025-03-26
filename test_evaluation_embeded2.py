#!/usr/bin/env python3
import pathlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import necessary modules from commonpower framework and others
from commonpower.modelling import ModelHistory
from commonpower.core import System
from commonpower.models.busses import RTPricedBus, ExternalGrid
from commonpower.models.components import ESSLinear, RenewableGen, Load
from commonpower.models.powerflow import PowerBalanceModel
from commonpower.control.controllers import OptimalController
from commonpower.control.runners import DeploymentRunner
from commonpower.data_forecasting import CSVDataSource, LookBackForecaster, PerfectKnowledgeForecaster, DataProvider, ConstantDataSource
from commonpower.utils.param_initialization import RangeInitializer
from commonpower.utils.helpers import get_adjusted_cost

# ------------------------------------------------------------------------------
# Helper functions for the Optimal Controller
# ------------------------------------------------------------------------------
def run_optimal_control(sys, m1, n1, e1, fixed_start, horizon, eval_seed):
    """Run deployment using the Optimal Controller."""
    oc_history = ModelHistory([sys])
    oc_deployer = DeploymentRunner(
        sys=sys,
        global_controller=OptimalController('global'),
        forecast_horizon=horizon,
        control_horizon=horizon,
        history=oc_history,
        seed=eval_seed
    )
    oc_deployer.run(n_steps=24, fixed_start=fixed_start)
    return oc_history

def extract_cost_soc(history, m1, n1, e1):
    """Extract the total cost and SOC history from a ModelHistory instance."""
    power_import_cost = history.get_history_for_element(m1, name='cost')
    dispatch_cost = history.get_history_for_element(n1, name='cost')
    total_cost = [
        (power_import_cost[t][0], power_import_cost[t][1] + dispatch_cost[t][1])
        for t in range(len(power_import_cost))
    ]
    soc = history.get_history_for_element(e1, name="soc")
    return total_cost, soc

def plot_optimal_results(oc_cost, oc_soc):
    """Plot the cost and SOC results for the Optimal Controller."""
    # Cost plot
    plt.figure()
    plt.plot(range(len(oc_cost)), [x[1] for x in oc_cost], label="Optimal Control Cost", marker="o")
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("Cost")
    plt.title("Optimal Controller Household Cost")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # SOC plot
    plt.figure()
    plt.plot(range(len(oc_soc)), [x[1] for x in oc_soc], label="Optimal Control SOC", marker="o")
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("State of Charge (SOC)")
    plt.title("Optimal Controller Battery SOC")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# Main script for Optimal Controller Deployment
# ------------------------------------------------------------------------------
def main():
    # Simulation configuration
    n_hours = 24
    horizon = timedelta(hours=n_hours)
    fixed_start = datetime(2016, 11, 27)
    eval_seed = 5  # evaluation seed

    # Path to data profiles
    current_path = pathlib.Path().absolute()
    data_path = (current_path / 'data' / '1-LV-rural2--1-sw').resolve()

    # ----------------------------
    # Data Sources and Providers
    # ----------------------------
    d11 = CSVDataSource(
        data_path / 'LoadProfile.csv',
        delimiter=";",
        datetime_format="%d.%m.%Y %H:%M",
        rename_dict={"time": "t", "G1-B_pload": "psib", "G1-C_pload": "psis", "G2-A_pload": "psi"},
        auto_drop=True,
        resample=timedelta(minutes=60)
    )

    ds1 = ConstantDataSource(
        {"psis": 0.08, "psib": 0.34},
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

    dp1 = DataProvider(ds1, PerfectKnowledgeForecaster(frequency=timedelta(minutes=60), horizon=horizon))
    dp2 = DataProvider(ds2, PerfectKnowledgeForecaster(frequency=timedelta(minutes=60), horizon=horizon))
    dp3 = DataProvider(ds3, PerfectKnowledgeForecaster(frequency=timedelta(minutes=60), horizon=horizon))

    # ----------------------------
    # System Definition: Nodes, Components, and System Build
    # ----------------------------
    n1 = RTPricedBus("Household").add_data_provider(dp1)
    m1 = ExternalGrid("ExternalGrid")

    capacity = 3  # kWh
    e1 = ESSLinear("ESS1", {
        'rho': 0.1,
        'p': (-1.5, 1.5),
        'q': (0, 0),
        'soc': (0.2 * capacity, 0.8 * capacity),
        "soc_init": RangeInitializer(0.2 * capacity, 0.8 * capacity)
    })
    d1 = Load("Load1").add_data_provider(dp2)
    r1 = RenewableGen("PV1").add_data_provider(dp3)

    sys = System(power_flow_model=PowerBalanceModel()).add_node(n1).add_node(m1)
    n1.add_node(d1).add_node(e1).add_node(r1)
    sys.pprint()
    print("Controllers:", sys.controllers)

    # ----------------------------
    # Run Optimal Control Deployment
    # ----------------------------
    oc_history = run_optimal_control(sys, m1, n1, e1, fixed_start, horizon, eval_seed)

    # Extract and plot results
    oc_total_cost, oc_soc = extract_cost_soc(oc_history, m1, n1, e1)
    plot_optimal_results(oc_total_cost, oc_soc)

    # Daily cost printout
    daily_cost_oc = sum(get_adjusted_cost(oc_history, sys))
    print("Daily cost for Optimal Control:")
    print(f" Optimal Control: {daily_cost_oc}")

if __name__ == "__main__":
    main()
