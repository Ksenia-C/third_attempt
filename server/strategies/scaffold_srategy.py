from flwr.server.strategy import Strategy
import numpy as np
from flwr.server.strategy.aggregate import aggregate
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl

class ScaffoldStrategy(Strategy):
    def __init__(self, eta_global=1, eta_local=0.001, initial_parameters=None, fraction_fit=1, min_fit_clients=2,
                 on_fit_config_fn=None, fit_metrics_aggregation_fn=None, evaluate_metrics_aggregation_fn=None):
        super().__init__()
        self.eta_global = eta_global
        self.eta_local = eta_local
        self.c_global = None
        self.x_gloval = None

        self.initial_parameters = initial_parameters
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def initialize_parameters(self, client_manager):
        # Return initial parameters (e.g., from a saved model, or None for random init)
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        """Select clients for training and return configuration."""
        # Sample clients
        try:
            self.N_clients = len(client_manager)
            sample_size = max(int(self.fraction_fit * len(client_manager)), self.min_fit_clients)
            clients = client_manager.sample(sample_size)

            # Create fit instructions for each client
            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            config["round"] = server_round
            config["eta_local"] = self.eta_local
            fit_ins = []

            if parameters is not None:
                if self.c_global is None:
                    self.x_gloval = parameters_to_ndarrays(parameters)
                    self.c_global = [np.zeros(el.shape) for el in self.x_gloval]
                parameters = ndarrays_to_parameters(parameters_to_ndarrays(parameters) + self.c_global)
            for client in clients:
                fit_ins.append((client, fl.common.FitIns(parameters, config)))
            return fit_ins
        except Exception as ex:
            print("ERORRE", ex)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate received model updates."""
        if not results:
            return None, {}

        # Simple FedAvg: average parameters weighted by number of examples

        weight_paramers = []
        c_update = []
        for _, fit_res in results:
            result_concat = parameters_to_ndarrays(fit_res.parameters);
            weight_paramers.append((result_concat[:len(result_concat) // 2], fit_res.num_examples))
            c_update.append((result_concat[len(result_concat) // 2:], fit_res.num_examples))

        # TODO ksenia here: aggregate make tunable and mine instead of weighted
        aggregated_parameters = aggregate(weight_paramers)
        aggregated_c_update = aggregate(c_update)
        
        # Collect metrics (e.g., loss from each client)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            print("WARNING", "No fit_metrics_aggregation_fn provided")

        for x_i, agg_i in zip(self.x_gloval, aggregated_parameters):
            x_i += self.eta_global * agg_i
        for i in range(len(self.c_global)):
            self.c_global[i] += (len(c_update) / self.N_clients) * aggregated_c_update[i]
        return ndarrays_to_parameters(self.x_gloval), metrics_aggregated

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure client-side evaluation."""
        # In this example, evaluate on all available clients
        clients = client_manager.all()
        config = {"round": server_round}
        eval_ins = []
        for client in clients.values():
            eval_ins.append((client, fl.common.EvaluateIns(parameters, config)))
        return eval_ins

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results."""
        # For simplicity, just return the average loss
        loss_aggregated = sum(r.loss for  _, r in results) / len(results)
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            print("WARNING", "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def evaluate(self, server_round, parameters):
        """Server-side evaluation (optional)."""
        # Return loss and metrics if you have a server-side test set
        return None
