import scipy.stats as stats
import numpy as np
import pandas as pd


norm_dist = stats.norm(loc=25, scale=1)
exp_dist = stats.expon(scale=2)
uniform_dist = stats.uniform(loc=0, scale=1)
powerlaw_dist = stats.powerlaw(a = 1.5)


def get_distribution(name: str):
    if name == "norm":
        return norm_dist
    if name == 'exp':
        return exp_dist
    if name == 'uni':
        return uniform_dist
    if name == 'pow':
        return powerlaw_dist
    raise RuntimeError("unimplemented [get_distribution]")


def get_sizes_for_clients(client_num, dataset_size, distribution):
    random_size = distribution.rvs(size=client_num, random_state=42)
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

def get_rare_labels(dataset, label_column, image_column, rare_data_count):
    return dataset.groupby([label_column]).count().sort_values(by=[image_column]).index[:rare_data_count]
def get_often_labels(dataset, label_column, image_column, rare_data_count):
    return dataset.groupby([label_column]).count().sort_values(by=[image_column]).index[-rare_data_count:]
def get_data_split(dataset, label_column, labels):
    rare_data = dataset[dataset[label_column].isin(labels)]
    rare_data = rare_data.sample(rare_data.shape[0])
    common_data = dataset[~dataset[label_column].isin(labels)]
    common_data = common_data.sample(common_data.shape[0])
    return rare_data, common_data

def get_rare_data_and_common(dataset, label_column, image_column, rare_data_count):
    rare_labels = get_rare_labels(dataset, label_column, image_column, rare_data_count)
    return get_data_split(dataset, label_column, rare_labels)

def get_often_data_and_common(dataset, label_column, image_column, rare_data_count):
    often_labels = get_often_labels(dataset, label_column, image_column, rare_data_count)
    return get_data_split(dataset, label_column, often_labels)


def get_remain_data(first_data, first_data_ind, second_data, second_data_ind):
    remain_data = pd.concat([first_data.iloc[first_data_ind:], second_data[second_data_ind:]])
    remain_data = remain_data.sample(remain_data.shape[0])
    return remain_data

def get_data_fill_from_lake(lake_data, other_data, fillage_formula, sorted_client, result, client_ind, lake_data_ind, other_data_ind):
    if lake_data_ind >= lake_data.shape[0]:
        new_lake_data_ind = lake_data_ind
    else:
        new_lake_data_ind = min(lake_data.shape[0], min(lake_data_ind + sorted_client[client_ind][1], int(lake_data_ind + fillage_formula(0))))
        result[client_ind] = pd.concat([result[client_ind], lake_data.iloc[lake_data_ind:new_lake_data_ind]])
    print(sorted_client, client_ind)
    common_data_fill = sorted_client[client_ind][1] - (new_lake_data_ind - lake_data_ind)
    result[client_ind] = pd.concat([result[client_ind], other_data.iloc[other_data_ind:other_data_ind + common_data_fill]])
    
    other_data_ind += common_data_fill
    lake_data_ind = new_lake_data_ind
    return lake_data_ind, other_data_ind

def pretty_result(result, sorted_clients):
    return list(map(lambda x: x[0], sorted(zip(result, sorted_clients), key=lambda x: x[1][0])))



def random_distribution(client_datasizes, dataset):
    randomized = dataset.sample(dataset.shape[0])
    result = []
    start_ind = 0
    for client_datasize in client_datasizes:
        result.append(randomized.iloc[start_ind:start_ind + client_datasize])
        start_ind = start_ind + client_datasize
    return result

def rare_on_rare_distribution(client_datasizes, dataset, rare_clients=10, rare_data_count=2, label_column='label', image_column='image', fillage_percent=0.9):
    rare_clients = min(rare_clients, len(client_datasizes))
    sorted_clients = make_clients_sorted(client_datasizes)
    rare_data, common_data = get_rare_data_and_common(dataset, label_column, image_column, rare_data_count)

    result = [pd.DataFrame([], columns=dataset.columns)] * len(sorted_clients)
    rare_data_ind = 0
    common_data_ind = 0
    for client_ind in range(rare_clients):
        rare_data_ind, common_data_ind = get_data_fill_from_lake(rare_data, common_data, lambda _: fillage_percent * sorted_clients[client_ind][1], sorted_clients, result, client_ind, rare_data_ind, common_data_ind)
        print(rare_data_ind)
    remain_data = get_remain_data(common_data, common_data_ind, rare_data, rare_data_ind)
    remain_data_ind = 0
    for client_ind in range(rare_clients, len(sorted_clients)):
        result[client_ind] = pd.concat([result[client_ind], remain_data.iloc[remain_data_ind:remain_data_ind + sorted_clients[client_ind][1]]])
        remain_data_ind += sorted_clients[client_ind][1]
    
    return pretty_result(result, sorted_clients)
    
def rare_on_often_distribution(client_datasizes, dataset, often_clients=10, rare_data_count=2, label_column='label', image_column='image', fillage_percent=0.1):
    often_clients = min(often_clients, len(client_datasizes))
    sorted_clients = make_clients_sorted(client_datasizes, up=True)
    rare_data, common_data = get_rare_data_and_common(dataset, label_column, image_column, rare_data_count)

    result = [pd.DataFrame([], columns=dataset.columns)] * len(sorted_clients)
    rare_data_ind = 0
    common_data_ind = 0
    for client_ind in range(often_clients):
        rare_data_ind, common_data_ind = get_data_fill_from_lake(rare_data, common_data, lambda _: fillage_percent * rare_data.shape[0], sorted_clients, result, client_ind, rare_data_ind, common_data_ind)
    
    remain_data = get_remain_data(common_data, common_data_ind, rare_data, rare_data_ind)
    remain_data_ind = 0
    for client_ind in range(often_clients, len(sorted_clients)):
        result[client_ind] = pd.concat([result[client_ind], remain_data.iloc[remain_data_ind:remain_data_ind + sorted_clients[client_ind][1]]]) 
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
        result[client_ind] = pd.concat([result[client_ind], remain_data.iloc[remain_data_ind:remain_data_ind + sorted_clients[client_ind][1]]])
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