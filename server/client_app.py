# credentials: https://apxml.com/courses/federated-learning/chapter-6-federated-learning-system-design/practice-fl-simulation-framework

import pickle
import copy
import json
import numpy as np
import ray
import torch
import flwr
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr_datasets.visualization import plot_label_distributions
from datasets import Dataset
from pathlib import Path
from models import FLAlgorithm

import torchvision.models as models
import torch.nn as nn

from task import load_partition, get_partiotioner, train, test, scaffold_train

def last_layers_of_densenet(name):
    return 'classifier' in name or 'norm5' in name or 'denseblock4' in name # or re.match('.*denseblock4.denselayer1[1-9].*', name)


class ClientModificationBase:
    def __init__(self, method):
        self.method = method

class DensenetFLClient(flwr.client.NumPyClient):
    def __init__(self, client_id, num_clients, partition, run_id, saver_directory, number_classes, label_col, random_state, augemntation_pipeline=None):
        self.client_id = client_id
        self.num_clients = num_clients
        # self.model = models.densenet121(weights=None)
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, number_classes)
        self.saver_directory = saver_directory / ("client_" + str(client_id))
        self.saver_directory.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.class_names=range(number_classes)

        # for param in self.model.parameters():
        #     param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     if last_layers_of_densenet(name):
        #         param.requires_grad = True
        self.x_train, self.x_test = load_partition(partition, self.client_id, self.num_clients, run_id, label_col, augemntation_pipeline, self.saver_directory, random_state)


        # for Scaffold and for other they don't disturb
        self.c_local = [torch.from_numpy(np.zeros(param.shape)) for param in self.get_parameters({})]
        self.c_global = [torch.from_numpy(np.zeros(param.shape)) for param in self.get_parameters({})]
        

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters, save_to_disk=False):
        if parameters is None:
            return
        with torch.no_grad():
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
        if save_to_disk:
            with open(self.saver_directory / "weights.pkl", "wb") as file:
                pickle.dump(self.model, file)
        
        
    def scaffold_set_parameters(self, parameters):
        if parameters is None:
            return
        self.parameters, new_global_c = parameters[:len(parameters) // 2], parameters[len(parameters) // 2:]
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), self.parameters):
                param.data=torch.from_numpy(new_param).to(param.device)
            self.c_global = [torch.from_numpy(global_c_comp) for global_c_comp in  new_global_c]
            assert [el.shape for el in self.c_local] == [el.shape for el in self.c_global], "at set parameters"
            self.parameters = [torch.from_numpy(el) for el in self.parameters]

    def fit(self, parameters, config):
        local_epochs = config.get("local_epochs", 1)
        current_round = config.get("round", 0)
        client_modification = FLAlgorithm(config.get("client_modification", str(FLAlgorithm.FED_AVG)))
        client_modification = ClientModificationBase(client_modification)
        if client_modification.method == FLAlgorithm.FED_PROX:
            client_modification.global_params = copy.deepcopy(self.model).parameters()
            client_modification.proximal_mu = config["proximal_mu"]
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if client_modification.method == FLAlgorithm.SCAFFOLD:
            eta_local = config["eta_local"]
            self.scaffold_set_parameters(parameters)
            history, K_calculate = scaffold_train(
                self.model,
                self.c_local,
                self.c_global,
                eta_local,
                self.x_train,
                self.x_test,
                local_epochs, 0.001, device, 
                self.saver_directory / ('round' + str(current_round)),
                client_modification,
                self.class_names
            )
            params_local = [torch.from_numpy(el) for el in self.get_parameters(config={})]
            c_plus = [
                c_local_i - c_global_i + (1.0 / K_calculate / eta_local) * (params_global_i - params_local_i)
                for c_local_i, c_global_i, params_global_i, params_local_i in zip(
                    self.c_local, self.c_global, self.parameters, params_local
                )
            ] 
            assert [el.shape for el in c_plus] == [el.shape for el in self.c_global], "at scaffold end local"
            returning_params = []
            for local_i, par_i in zip(params_local, self.parameters):
                returning_params.append(local_i - par_i)
            for c_i, loc_i in zip(c_plus, self.c_local):
                returning_params.append(c_i - loc_i)

            assert [el.shape for el in returning_params[:len(returning_params) // 2]] == [el.shape for el in returning_params[len(returning_params) // 2:]], "at scaffold end local rgith before"

            self.c_local = c_plus

        else:
            self.set_parameters(parameters)
            history = train(
                self.model, 
                self.x_train,
                self.x_test,
                local_epochs, 0.001, device, 
                self.saver_directory / ('round' + str(current_round)),
                client_modification,
                self.class_names
            )
            returning_params = self.get_parameters(config={})
        # Return updated weights, number of training examples, and optional metrics
        results = {
            "loss": history,
        }
        with open(self.saver_directory / "train_losses.txt", "a") as file:
            print(f"client {self.client_id}: {history}", file=file)

        return returning_params, len(self.x_train.dataset), results

    def evaluate(self, parameters, config):
        # Update model with received parameters
        save_model = config.get("save_model", False)
        if self.client_id != 0:
            save_model = False
        self.set_parameters(parameters, save_model)
        # Evaluate the model on local test data
        current_round = config.get("round", 0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss, metrics = test(self.model, self.x_test, device, self.saver_directory / ('round' + str(current_round)), self.class_names)
        # Return loss, number of evaluation examples, and metrics
        with open(self.saver_directory / "test_losses.txt", "a") as file:
            print(f"client {self.client_id}: {loss}", file=file)
        return loss, len(self.x_test.dataset), metrics

# Function to instantiate clients based on ID
def client_fn(context: Context, num_total_clients, run_id, saver_directory, number_classes, label_col, augemntation_pipeline, random_state_inits) -> flwr.client.Client:
    client_id = int(context.node_config["partition-id"])
    current_ns = ray.get_runtime_context().namespace
    partitioner_actor = ray.get_actor("partitioner_actor")

    
    return DensenetFLClient(client_id=client_id, num_clients=num_total_clients, partition=ray.get(partitioner_actor.get_partiotion.remote(client_id)), run_id=run_id, saver_directory=saver_directory, number_classes=number_classes, label_col=label_col, random_state=random_state_inits.get_random_state(client_id),  augemntation_pipeline=augemntation_pipeline).to_client()


def client_standlone_run(run_id, epochs_nums, saver_directory, number_classes, label_col, augemntation_pipeline, random_state_inits):
    partitioner_actor = ray.get_actor("partitioner_actor")
    all_data=ray.get(partitioner_actor.get_partiotion.remote(0))
    base_model = DensenetFLClient(0, 1, all_data, run_id, saver_directory, number_classes, label_col, augemntation_pipeline, random_state_inits.get_random_state(0))

    base_model.fit(None, {'local_epochs': epochs_nums})
    loss, len_x_test, metrics = base_model.evaluate(None, None)
    return {'loss': loss, 'metrics': metrics}
