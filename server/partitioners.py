import division_helpers as div_help
from flwr_datasets.partitioner import Partitioner
from typing import Optional, List, Dict
import numpy as np

from datasets import Dataset

class BasePartitioner():
    def __init__(self, num_partitions: int, sized_distribution: str = 'uni', random_state=42, sized_dist_params: Dict = {}):
        self._num_partitions = num_partitions
        self.sized_distribution = sized_distribution
        self.random_state = random_state
        self.custom_dataset: list[div_help.pd.DataFrame] = []
        self.sized_dist_params = sized_dist_params
    
    @property
    def num_partitions(self) -> int:
        return self._num_partitions
    
    
    def get_partitions_if_needed(self):
        if len(self.custom_dataset) > 0:
            return
        
        sizes = div_help.get_sizes_for_clients(self._num_partitions, self.dataset.shape[0], div_help.get_distribution(self.sized_distribution, self.sized_dist_params), self.random_state)
        self.custom_dataset = div_help.random_distribution(sizes, self.dataset)
        
    
    def load_partition(self, partition_id: int) -> List[Dataset]:
        assert partition_id >= 0 and partition_id < self.num_partitions, "Out of Range"
        self.get_partitions_if_needed()
        return self.custom_dataset[partition_id]


class RandomPartitioner(BasePartitioner):
    def __init__(self, num_partitions: int, sized_distribution: str = 'norm', random_state=42, sized_dist_params: Dict = {}):
        super().__init__(num_partitions, sized_distribution, random_state, sized_dist_params)
    
    def get_partitions_if_needed(self):
        if len(self.custom_dataset) > 0:
            return
        
        sizes = div_help.get_sizes_for_clients(self._num_partitions, self.dataset.shape[0], div_help.get_distribution(self.sized_distribution, self.sized_dist_params), self.random_state)
        self.custom_dataset = div_help.random_distribution(sizes, self.dataset)


class RareOnRarePartitioner(BasePartitioner):
    def __init__(self, num_partitions: int, sized_distribution: str = 'uni', random_state=42, rare_clients=10, rare_data_count=2, fillage_percent=0.9, sized_dist_params: Dict = {}):
        super().__init__(num_partitions, sized_distribution, random_state, sized_dist_params)
        self.rare_clients = rare_clients
        self.rare_data_count = rare_data_count
        self.fillage_percent = fillage_percent

    
    def get_partitions_if_needed(self):
        print("VIERAN get partition")
        if len(self.custom_dataset) > 0:
            return
        
        sizes = div_help.get_sizes_for_clients(self._num_partitions, self.dataset.shape[0], div_help.get_distribution(self.sized_distribution, self.sized_dist_params), self.random_state)
        self.custom_dataset = div_help.rare_on_rare_distribution(sizes, self.dataset, rare_clients=self.rare_clients, rare_data_count=self.rare_data_count, fillage_percent=self.fillage_percent)


class RareOnOftenPartitioner(BasePartitioner):
    def __init__(self, num_partitions: int, sized_distribution: str = 'uni', random_state=42, often_clients=10, rare_data_count=2, fillage_percent=0.1, sized_dist_params: Dict = {}):
        super().__init__(num_partitions, sized_distribution, random_state, sized_dist_params)
        self.often_clients = often_clients
        self.rare_data_count = rare_data_count
        self.fillage_percent = fillage_percent

    
    def get_partitions_if_needed(self):
        if len(self.custom_dataset) > 0:
            return
        
        sizes = div_help.get_sizes_for_clients(self._num_partitions, self.dataset.shape[0], div_help.get_distribution(self.sized_distribution, self.sized_dist_params), self.random_state)
        self.custom_dataset = div_help.rare_on_often_distribution(sizes, self.dataset, often_clients=self.often_clients, rare_data_count=self.rare_data_count, fillage_percent=self.fillage_percent)


class OftenOnOftenPartitioner(BasePartitioner):
    def __init__(self, num_partitions: int, sized_distribution: str = 'uni', random_state=42, often_clients=10, often_data_cnt=2, fillage_percent=0.9, sized_dist_params: Dict = {}):
        super().__init__(num_partitions, sized_distribution, random_state, sized_dist_params)
        self.often_clients = often_clients
        self.often_data_cnt = often_data_cnt
        self.fillage_percent = fillage_percent

    
    def get_partitions_if_needed(self):
        if len(self.custom_dataset) > 0:
            return
        
        sizes = div_help.get_sizes_for_clients(self._num_partitions, self.dataset.shape[0], div_help.get_distribution(self.sized_distribution, self.sized_dist_params), self.random_state)
        self.custom_dataset = div_help.often_on_often_distribution(sizes, self.dataset, often_clients=self.often_clients, often_data_cnt=self.often_data_cnt, fillage_percent=self.fillage_percent)


class OftenEverywherePartitioner(BasePartitioner):
    def __init__(self, num_partitions: int, sized_distribution: str = 'uni', random_state=42, often_clients=10, often_data_cnt=2,  constant_on_often=200, percentage_on_others=0.2, sized_dist_params: Dict = {}):
        super().__init__(num_partitions, sized_distribution, random_state, sized_dist_params)
        self.often_clients = often_clients
        self.often_data_cnt = often_data_cnt
        self.constant_on_often = constant_on_often
        self.percentage_on_others = percentage_on_others

    
    def get_partitions_if_needed(self):
        if len(self.custom_dataset) > 0:
            return
        
        sizes = div_help.get_sizes_for_clients(self._num_partitions, self.dataset.shape[0], div_help.get_distribution(self.sized_distribution, self.sized_dist_params), self.random_state)
        self.custom_dataset = div_help.often_everywhere_distribution(sizes, self.dataset, often_clients=self.often_clients, often_data_cnt=self.often_data_cnt, constant_on_often=self.constant_on_often, percentage_on_others=self.percentage_on_others)
