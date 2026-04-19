from flwr.server.strategy import Strategy
import math
import json
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

def calc_weight_importance(global_model, results, server_round, saved_directory, aggregate_foo):
    global_norm = None
    global_norm_1 = None
    if global_model is None:
        global_norm = np.linalg.norm(global_model)
        global_norm_1 = global_model / global_norm 

    # Simple FedAvg: average parameters weighted by number of examples

    weight_paramers = []
    local_models_dists_previous = []
    local_models_dists_new = []
    for _, fit_res in results:
        result_concat = parameters_to_ndarrays(fit_res.parameters);
        weight_paramers.append((result_concat, fit_res.num_examples))
        
        if global_norm is not None:
            local_norm = np.linalg.norm(result_concat)
            knorm = local_norm
            result_concat_normed = result_concat / local_norm
            angle = 1 - np.sum(global_norm_1 * result_concat_normed)
            local_models_dists_previous.append({'angle': angle, 'knorm': knorm})


    aggregated_parameters = aggregate_foo(weight_paramers)
    new_global_norm = np.linalg.norm(aggregated_parameters)
    new_global_params_with_norm_1 = aggregated_parameters / new_global_norm
    for _, fit_res in results:
        result_concat = parameters_to_ndarrays(fit_res.parameters);
        
        if global_norm is not None:
            local_norm = np.linalg.norm(result_concat)
            knorm = local_norm
            result_concat_normed = result_concat / local_norm
            angle = 1 - np.sum(new_global_params_with_norm_1 * result_concat_normed)
            local_models_dists_new.append({'angle': angle, 'knorm': knorm})


    if global_norm is not None:
        distance_s = []
        # global_norm new_global_norm

        for client_id, (previous_distance, new_distance) in enumerate(zip(local_models_dists_previous, local_models_dists_new)):
            eps = 1e-6
            distance_for_id = previous_distance['angle'] - new_distance['angle']
            if distance_for_id >= 0:
                distance_for_id /= abs(new_global_norm - new_distance['knorm'])
            else:
                distance_for_id *= new_distance['knorm']

            distance_s.append((client_id, distance_for_id))
        
        distance_s = np.array(distance_s)
        mask_pos = distance_s[:, 1] > 0
        positive_inputs = {distance_s[i, 0]: distance_s[i, 1] for i in np.where(mask_pos)[0]}
        mask_pos = distance_s[:, 1] < 0
        negative_inputs = {distance_s[i, 0]: distance_s[i, 1] for i in np.where(mask_pos)[0]}

        if saved_directory is not None:
            with open(saved_directory / f'round_{server_round}', 'w') as file_to_write:
                weights_json = {
                    'diff_with_previous': local_models_dists_previous,
                    'diff_with_new': local_models_dists_new,
                    'positive_effect_clients_ind': positive_inputs,
                    'negative_effect_clients_ind': negative_inputs
                }
                json.dump(weights_json, file_to_write)
    return aggregated_parameters

class FedAvgStrategy(Strategy):
    def __init__(self, initial_parameters=None, fraction_fit=1, min_fit_clients=2,
                 on_fit_config_fn=None, fit_metrics_aggregation_fn=None, evaluate_metrics_aggregation_fn=None,
                 saved_directory=None):
        super().__init__()
        self.global_model = None

        self.initial_parameters = initial_parameters
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        self.saved_directory = saved_directory

    def initialize_parameters(self, client_manager):
        # Return initial parameters (e.g., from a saved model, or None for random init)
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        self.global_model = initial_parameters
        return initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        """Select clients for training and return configuration."""
        # Sample clients
        try:
            sample_size = max(int(self.fraction_fit * len(client_manager)), self.min_fit_clients)
            clients = client_manager.sample(sample_size)

            # Create fit instructions for each client
            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            config["round"] = server_round
            fit_ins = []

            if parameters is not None:
                parameters = parameters_to_ndarrays(parameters)
            self.global_model = parameters
            for client in clients:
                fit_ins.append((client, fl.common.FitIns(parameters, config)))
            return fit_ins
        except Exception as ex:
            print("ERORRE", ex)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate received model updates."""
        if not results:
            return None, {}

        aggregated_parameters = calc_weight_importance(self.global_model, results, server_round, self.saved_directory, aggregate)

        # Collect metrics (e.g., loss from each client)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            print("WARNING", "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(aggregated_parameters), metrics_aggregated

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
