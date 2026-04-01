# TODO: move on to DAtaset completely, not pandas (make legacy ot even remove)

import scipy.stats as stats
import numpy as np
import pandas as pd


from datasets import Dataset
from datasets import concatenate_datasets


def get_distribution(name: str, config):
    if name == "norm":
        return stats.norm(loc=config.get('loc', 25), scale=config.get('scale', 1))
    if name == 'exp':
        return stats.expon(scale=config.get('scale', 2))
    if name == 'uni':
        return stats.uniform(loc=config.get('loc', 0), scale=config.get('scale', 1))
    if name == 'pow':
        return stats.powerlaw(a = config.get('a', 1.5))
    raise RuntimeError("unimplemented [get_distribution]")


def get_sizes_for_clients(client_num, dataset_size, distribution, randome_state):
    random_size = distribution.rvs(size=client_num, random_state=randome_state)

    for i in range(len(random_size)):
        random_size[i] = max(random_size[i], 0)
    if np.all(random_size == 0):
        random_size[0] = 1
    random_sum = random_size.sum()
    random_size = np.floor(random_size / random_sum * dataset_size).astype(int)
    for i in range(len(random_size)):
        if random_size[i] == 0:
            for j in range(len(random_size)):
                if i == j:
                    continue 
                if random_size[j] > 2:
                    random_size[j] -= 1
                    random_size[i] += 1
                    break
    random_sum = random_size.sum()
    leftover = dataset_size - random_sum
    client_ind = 0
    for i in range(leftover):
        random_size[client_ind] += 1
        client_ind += 1
    return random_size

def make_clients_sorted(client_datasizes, up=False):
    return sorted(enumerate(client_datasizes), key = lambda x: x[1], reverse = up)

from collections import Counter

def get_rare_labels(dataset, label_column, image_column, rare_data_count):
    if isinstance(dataset, pd.DataFrame):
        return dataset.groupby([label_column]).count().sort_values(by=[image_column]).index[:rare_data_count]
    else:
        labels = dataset[label_column]
        label_counts = Counter(labels)
        
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1])
        return [label for label, _ in sorted_labels[:rare_data_count]] 
def get_often_labels(dataset, label_column, image_column, rare_data_count):
    if isinstance(dataset, pd.DataFrame):
        return dataset.groupby([label_column]).count().sort_values(by=[image_column]).index[-rare_data_count:]
    else:
        labels = dataset[label_column]
        label_counts = Counter(labels)
        
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1])
        return [label for label, _ in sorted_labels[-rare_data_count:]]
def get_data_split(dataset, label_column, labels):
    if isinstance(dataset, pd.DataFrame):
        rare_data = dataset[dataset[label_column].isin(labels)]
        rare_data = rare_data.sample(rare_data.shape[0])
        common_data = dataset[~dataset[label_column].isin(labels)]
        common_data = common_data.sample(common_data.shape[0])

    else: 
        rare_set = set(labels)
        rare_data = dataset.filter(lambda example: example[label_column] in rare_set)
        rare_data = rare_data.shuffle(seed=None)
        common_data = dataset.filter(lambda example: example[label_column] not in rare_set)
        common_data = common_data.shuffle(seed=None)
    return rare_data, common_data

def get_rare_data_and_common(dataset, label_column, image_column, rare_data_count):
    rare_labels = get_rare_labels(dataset, label_column, image_column, rare_data_count)
    return get_data_split(dataset, label_column, rare_labels)

def get_often_data_and_common(dataset, label_column, image_column, rare_data_count):
    often_labels = get_often_labels(dataset, label_column, image_column, rare_data_count)
    return get_data_split(dataset, label_column, often_labels)


def iloc_(dataset, start_, end_):
    if end_ is None and start_ is not None:
        if isinstance(dataset, pd.DataFrame):
            return dataset.iloc[start_:]
        else:
            if start_ >= len(dataset):
                return Dataset.from_dict(
                    {col: [] for col in dataset.column_names},
                    features=dataset.features
                )
            else:
                return dataset.select(range(start_, len(dataset)))
    if start_ is None and end_ is not None:
        if isinstance(dataset, pd.DataFrame):
            return dataset.iloc[:end_]
        else:
            return dataset.select(range(0, end_))
    if start_ is None and end_ is None:
        if isinstance(dataset, pd.DataFrame):
            return dataset.iloc[:]
        else:
            return dataset.select(range(0, len(dataset)))
    if isinstance(dataset, pd.DataFrame):
        return dataset.iloc[start_:end_]
    else:
        return dataset.select(range(start_, end_))

def u_concat(datalist):
    if isinstance(datalist[0], pd.DataFrame):
        return pd.concat(datalist)
    else:
        print(datalist[1])
        return concatenate_datasets(datalist)



def get_remain_data(first_data, first_data_ind, second_data, second_data_ind):
    remain_data = u_concat([iloc_(first_data, first_data_ind, None), iloc_(second_data, second_data_ind, None)])
    
    if isinstance(remain_data, pd.DataFrame):
        remain_data = remain_data.sample(remain_data.shape[0])
    else:
        remain_data = remain_data.shuffle(seed=None)
    return remain_data

def get_data_fill_from_lake(lake_data, other_data, fillage_formula, sorted_client, result, client_ind, lake_data_ind, other_data_ind):
    if lake_data_ind >= lake_data.shape[0]:
        new_lake_data_ind = lake_data_ind
    else:
        new_lake_data_ind = min(lake_data.shape[0], min(lake_data_ind + sorted_client[client_ind][1], int(lake_data_ind + fillage_formula(0))))
        result[client_ind] = u_concat([result[client_ind], iloc_(lake_data, lake_data_ind, new_lake_data_ind)])
    common_data_fill = sorted_client[client_ind][1] - (new_lake_data_ind - lake_data_ind)
    result[client_ind] = u_concat([result[client_ind], iloc_(other_data, other_data_ind, other_data_ind + common_data_fill)])
    
    other_data_ind += common_data_fill
    lake_data_ind = new_lake_data_ind
    return lake_data_ind, other_data_ind

def pretty_result(result, sorted_clients):
    return list(map(lambda x: x[0], sorted(zip(result, sorted_clients), key=lambda x: x[1][0])))

def random_distribution(client_datasizes, dataset):
    if isinstance(dataset, pd.DataFrame):
        randomized = dataset.sample(dataset.shape[0])
    else:
        randomized = dataset.shuffle(seed=None)
    result = []
    start_ind = 0
    for client_datasize in client_datasizes:
        result.append(iloc_(randomized, start_ind, start_ind + client_datasize))
        start_ind = start_ind + client_datasize
    return result




def rare_on_rare_distribution(client_datasizes, dataset, rare_clients=10, rare_data_count=2, label_column='label', image_column='image', fillage_percent=0.9):
    rare_clients = min(rare_clients, len(client_datasizes))
    sorted_clients = make_clients_sorted(client_datasizes)
    rare_data, common_data = get_rare_data_and_common(dataset, label_column, image_column, rare_data_count)

    if isinstance(dataset, pd.DataFrame):
        result = [pd.DataFrame([], columns=dataset.columns)] * len(sorted_clients)
    else: 
        column_names = dataset.column_names
        features = dataset.features

        result = [
            Dataset.from_dict({col: [] for col in column_names}, features=features)
            for _ in range(len(sorted_clients))
        ]
    rare_data_ind = 0
    common_data_ind = 0
    for client_ind in range(rare_clients):
        rare_data_ind, common_data_ind = get_data_fill_from_lake(rare_data, common_data, lambda _: fillage_percent * sorted_clients[client_ind][1], sorted_clients, result, client_ind, rare_data_ind, common_data_ind)
    remain_data = get_remain_data(common_data, common_data_ind, rare_data, rare_data_ind)
    remain_data_ind = 0
    for client_ind in range(rare_clients, len(sorted_clients)):
        result[client_ind] = u_concat([result[client_ind], iloc_(remain_data, remain_data_ind, remain_data_ind + sorted_clients[client_ind][1])])
        remain_data_ind += sorted_clients[client_ind][1]
    
    return pretty_result(result, sorted_clients)
    
def rare_on_often_distribution(client_datasizes, dataset, often_clients=10, rare_data_count=2, label_column='label', image_column='image', fillage_percent=0.1):
    often_clients = min(often_clients, len(client_datasizes))
    sorted_clients = make_clients_sorted(client_datasizes, up=True)
    rare_data, common_data = get_rare_data_and_common(dataset, label_column, image_column, rare_data_count)

    if isinstance(dataset, pd.DataFrame):
        result = [pd.DataFrame([], columns=dataset.columns)] * len(sorted_clients)
    else: 
        column_names = dataset.column_names
        features = dataset.features

        result = [
            Dataset.from_dict({col: [] for col in column_names}, features=features)
            for _ in range(len(sorted_clients))
        ]
    rare_data_ind = 0
    common_data_ind = 0
    for client_ind in range(often_clients):
        rare_data_ind, common_data_ind = get_data_fill_from_lake(rare_data, common_data, lambda _: fillage_percent * rare_data.shape[0], sorted_clients, result, client_ind, rare_data_ind, common_data_ind)
    
    remain_data = get_remain_data(common_data, common_data_ind, rare_data, rare_data_ind)
    remain_data_ind = 0
    for client_ind in range(often_clients, len(sorted_clients)):
        result[client_ind] = u_concat([result[client_ind], iloc_(remain_data, remain_data_ind, remain_data_ind + sorted_clients[client_ind][1])]) 
        remain_data_ind += sorted_clients[client_ind][1]
    
    return pretty_result(result, sorted_clients)
    
def often_on_often_distribution(client_datasizes, dataset, often_clients=10, often_data_cnt=2, label_column='label', image_column='image', fillage_percent=0.9):
    sorted_clients = make_clients_sorted(client_datasizes, up=True)
    often_data, common_data = get_often_data_and_common(dataset, label_column, image_column, often_data_cnt)

    result = [pd.DataFrame([], columns=dataset.columns)] * len(sorted_clients)
    often_data_ind = 0
    common_data_ind = 0
    for client_ind in range(often_clients):
        # often_data_ind, common_data_ind = get_data_fill_from_lake(often_data, common_data, lambda _: fillage_percent * often_data.shape[0] / often_clients, sorted_clients, result, client_ind, often_data_ind, common_data_ind)
        often_data_ind, common_data_ind = get_data_fill_from_lake(often_data, common_data, lambda _: fillage_percent * sorted_clients[client_ind][1], sorted_clients, result, client_ind, often_data_ind, common_data_ind)
    
    remain_data = get_remain_data(common_data, common_data_ind, often_data, often_data_ind)
    remain_data_ind = 0
    for client_ind in range(often_clients, len(sorted_clients)):
        result[client_ind] = u_concat([result[client_ind], iloc_(remain_data, remain_data_ind, remain_data_ind + sorted_clients[client_ind][1])])
        remain_data_ind += sorted_clients[client_ind][1]
    
    return pretty_result(result, sorted_clients)

def often_everywhere_distribution(client_datasizes, dataset, often_data_cnt=2, often_clients = 10, label_column='label', image_column='image', constant_on_often=200, percentage_on_others=0.2):
    sorted_clients = make_clients_sorted(client_datasizes, up=True)
    often_data, common_data = get_often_data_and_common(dataset, label_column, image_column, often_data_cnt)

    result = [pd.DataFrame([], columns=dataset.columns)] * len(sorted_clients)
    often_data_ind = 0
    common_data_ind = 0
    for client_ind in range(often_clients):
        often_data_ind, common_data_ind = get_data_fill_from_lake(often_data, common_data, lambda _: constant_on_often, sorted_clients, result, client_ind, often_data_ind, common_data_ind)

    for client_ind in range(often_clients, len(sorted_clients)):
        often_data_ind, common_data_ind = get_data_fill_from_lake(often_data, common_data, lambda _: percentage_on_others * sorted_clients[client_ind][1], sorted_clients, result, client_ind, often_data_ind, common_data_ind)
    
    result = list(map(lambda x: x[0], sorted(zip(result, sorted_clients), key=lambda x: x[1][0])))
    return result
