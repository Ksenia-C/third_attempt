import asyncio
from datetime import datetime, timedelta
import random
import ray
from fastapi import FastAPI, Depends, HTTPException, status
from models import FLAlgorithm, Distribution, RunStatus
from sql_models import SimulationRuns, RunProgress, RunStats, Base
import json
import uuid
import flwr
from server.strategies.scaffold_srategy import ScaffoldStrategy
from drawings import *
from flwr.simulation import run_simulation
from pathlib import Path
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, AsyncGenerator, List
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from client_app import client_fn, client_standlone_run
import task as task_module

import albumentations as alb


from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Select, create_engine, desc, func,true,  Column, Integer, String, Boolean, Text, Float, ForeignKey, Table, Date, Enum
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy.orm import Session
RAY_NAMESPACE = "flower_simulation"

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        # Create all tables / Создать все таблицы
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    print("✅ Database tables created / Таблицы БД созданы")

    ray.init(address="auto", namespace=RAY_NAMESPACE)
#     partitioner_actor = task_module.PartitionerActor.options(name="partitioner_actor",namespace=RAY_NAMESPACE, lifetime="detached"
# ).remote()


    yield  # Application runs / Приложение работает

    # Shutdown / Остановка
    await engine.dispose()
    # ray.kill(partitioner_actor)
    ray.shutdown()

app = FastAPI(
    title="Run Simulation API",
    description="run fl simulation with FastAPI",
    lifespan=lifespan
)

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/fl_distributions_tries.db"

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()




class SimulationRun(BaseModel):
    fl_algorithm: FLAlgorithm = Field(default=FLAlgorithm.FED_AVG, description="FL algorithm to run within the simulation")
    data_distribution: Distribution = Field(default=Distribution.RANDOM_UNIFORM, description="distribution of the training data among clients")
    algorithm_params: Dict[str, Any] = {}
    distribution_params: Dict[str, Any] = {}
    saver_directory: Path = None
    augemntation_need: bool = False
    clients: Dict[str, Any] = {}


class SimulationManyRun(BaseModel):
    saver_directory: Path = None

import torchvision.models as models
import torch
import numpy as np

# https://www.kaggle.com/code/snehilsanyal/federated-learning-tutorial-part-1-with-flower?scriptVersionId=210040083
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    result = {}
    if len(metrics) == 0:
        print("WARNING - no data from evaluation to aggregate")
        return result
    
    evil_gather = {}
    class_set = set()
    for metric_name in metrics[0][1].keys():
        # Aggregate and return custom metric (weighted average)
        if metric_name.startswith("evil"):
            evil_gather[metric_name] = evil_gather.get(metric_name, 0) +  np.sum(np.array([m[metric_name] for _, m in metrics]))
            class_set.add(metric_name.split('_')[1])
            continue

        accuracies = [num_examples * m[metric_name] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        result[metric_name] = sum(accuracies) / sum(examples)

    weighted_f1 = []
    all_class_size = 0
    for one_class in class_set:
        all_class_size += evil_gather['evil_' +str(one_class) + '_all_for_recall']
    for one_class in class_set:
        if evil_gather['evil_' +str(one_class) + '_all_for_precision'] == 0 or evil_gather['evil_' +str(one_class) + '_all_for_recall'] == 0 or evil_gather['evil_' +str(one_class) + '_TP'] == 0:
            weighted_f1.append((0, 0))
            continue
        evil_precision = evil_gather['evil_' +str(one_class) + '_TP'] / evil_gather['evil_' +str(one_class) + '_all_for_precision']
        evil_recall =evil_gather['evil_' +str(one_class) + '_TP'] / evil_gather['evil_' +str(one_class) + '_all_for_recall']
        evil_f1 = 2 * evil_precision *evil_recall / (evil_precision  + evil_recall)   
        weighted_f1.append((evil_gather['evil_' +str(one_class) + '_all_for_recall'] / all_class_size, evil_f1))

    result['evil_global_weighted_f1'] = sum([x[0] * x[1] for x in weighted_f1]).item()
    return result

async def on_success(history, simulation_time, run_id):
    async with AsyncSession(engine) as db:
        current_task_sql = await db.execute(Select(RunProgress).where(RunProgress.run_id == run_id))
        current_task_stats_sql = await db.execute(Select(RunStats).where(RunStats.run_id == run_id))
        
        current_task = current_task_sql.scalar_one_or_none()
        current_task_stats = current_task_stats_sql.scalar_one_or_none()
        current_task.status = RunStatus.COMPLETED
        current_task.result = str(history)
        current_task_stats.simulation_time_ms = simulation_time
        await db.commit()
from flwr.common import ndarrays_to_parameters

async def on_error(run_id):
    async with AsyncSession(engine) as db:
        current_task_sql = await db.execute(Select(RunProgress).where(RunProgress.run_id == run_id))
        current_task = current_task_sql.scalar_one_or_none()
        current_task.status = RunStatus.ERROR
        await db.commit()


class ExtraMods:
    def __init__(self):
        self.augmentation = None
    def set_augmentation(self):
        self.augmentation = alb.Compose([
            alb.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-170, 170),
                p=1),
            alb.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0,
                p=0.9
            )
            ])
    def string(self):
        return  f"{self.augmentation}"
    

class RandomStateInitialization:
    def __init__(self, clients_id_ranges):
        self.random_state = {}
        for client_id in clients_id_ranges:
            self.random_state[client_id] = random.randrange(400000)
    def get_random_state(self, client_id):
        return self.random_state[client_id]

def run_simulation_impl(fl_algorithm: FLAlgorithm, data_distribution: Distribution, algorithm_params: Dict, distribution_params: Dict, run_id: str, saver_directory: str, extra_mods : ExtraMods, clients_params: Dict):
    # label_col = 'coarse_label'
    # TODO: move it to some other class
    label_col = 'label'
    augemntation_pipeline = extra_mods.augmentation
    
    num_clients = int(clients_params['count']) if fl_algorithm != FLAlgorithm.STANDALONE else 1
    saver_directory = Path(saver_directory) / run_id
    saver_directory.mkdir(parents=True)
    with open(saver_directory / "simulation_params.txt", "w") as file:
        print(f"fl_algorithm {fl_algorithm}\n data_distribution {data_distribution}\n algorithm_params {algorithm_params}\n distribution_params {distribution_params} extra_mods {extra_mods.string()} clients {clients_params}", file=file)

    partitioner_actor = ray.get_actor("partitioner_actor")
    creation_ref = partitioner_actor.create_partitioner.remote(data_distribution.value, distribution_params, num_clients, run_id, saver_directory, label_col)
    class_number = ray.get(creation_ref)

    random_state_inits = RandomStateInitialization(range(num_clients))


    iid_metrics = partitioner_actor.get_disbalance_metric.remote(class_number, random_state_inits)
    plot_std_data(ray.get(iid_metrics), save_dir=saver_directory)
    

    def get_config_fn():
        def fit_config(server_round: int):
            # Pass round-specific configuration to clients
            config = {
                "run_id": run_id,
                "local_epochs": algorithm_params.get('local_epochs', 15),
                "batch_size": 32,
                'round': server_round,
                'client_modification': fl_algorithm.value
           }
            return config
        return fit_config


    try:
        if fl_algorithm == FLAlgorithm.STANDALONE:
            time_start = datetime.now()
            history = client_standlone_run(run_id, algorithm_params.get('local_epochs', 15), saver_directory, class_number, label_col, augemntation_pipeline, random_state_inits)
            simulation_time = (datetime.now() - time_start) / timedelta(milliseconds=1)
            asyncio.run(on_success(history, simulation_time, run_id))
            return
        if fl_algorithm == FLAlgorithm.FED_AVG:
            strategy = flwr.server.strategy.FedAvg(
                on_fit_config_fn=get_config_fn(),
                on_evaluate_config_fn=get_config_fn(),
                evaluate_metrics_aggregation_fn=weighted_average,
                )
        elif fl_algorithm == FLAlgorithm.FED_PROX:
            strategy = flwr.server.strategy.FedProx(
                on_fit_config_fn=get_config_fn(),
                on_evaluate_config_fn=get_config_fn(),
                evaluate_metrics_aggregation_fn=weighted_average,
                proximal_mu=algorithm_params['proximal_mu'])
        elif fl_algorithm == FLAlgorithm.SCAFFOLD:
            strategy = ScaffoldStrategy(
                on_fit_config_fn=get_config_fn(),
                evaluate_metrics_aggregation_fn=weighted_average)
        else:
            raise RuntimeError("strategy is unimplemented")
        
        
        time_start = datetime.now()

        history = flwr.simulation.start_simulation(
            client_fn=lambda context: client_fn(context, num_clients, run_id, saver_directory, class_number, label_col, augemntation_pipeline, random_state_inits),
            num_clients=num_clients,
            config=flwr.server.ServerConfig(num_rounds=algorithm_params.get('global_rounds_num', 20)),
            strategy=strategy,
            client_resources={"num_cpus": 3, "num_gpus": 0.0},
            ray_init_args={"address": "auto", "namespace": RAY_NAMESPACE})
        simulation_time = (datetime.now() - time_start) / timedelta(milliseconds=1)
        data = parse_history(str(history))
        print(data)
        save_parsed_data(data, saver_directory)
        plot_server_metrics(data, saver_directory)

        client_losses = read_client_losses(saver_directory)
        if client_losses:
            plot_client_losses(client_losses, saver_directory)
        else:
            print("No client loss files found.")
        asyncio.run(on_success(history, simulation_time, run_id))
        
    except Exception as exp:
        print("ERROR", exp)
        asyncio.run(on_error(run_id))

def run_many_exps_from_path(path_to_config_folder: Path):
    with open(path_to_config_folder / 'config.json', 'r') as file:
        conf_exps = json.load(file)
    for exp_name, exp_config in conf_exps:
        with open(path_to_config_folder / 'exp_saviour.txt', 'r') as saviour_file:
            if exp_name in saviour_file.readlines():
                continue
        run_id = exp_name + str(uuid.uuid4())
        extra_mods = ExtraMods()
        if exp_config['augemntation_need'] == 'true':
            extra_mods.set_augmentation()

        run_simulation_impl(exp_config['fl_algorithm'], exp_config['data_distribution'], exp_config['algorithm_params'], exp_config['distribution_params'], run_id, path_to_config_folder, extra_mods)

        with open(path_to_config_folder / 'exp_saviour.txt', 'a') as saviour_file:
            saviour_file.write(exp_name + '\n')

class ForwardResponse(BaseModel):
    run_id: str = Field(...)

class StatusResponse(BaseModel):
    is_finished: bool  = Field(False)
    is_error: bool = Field(False)
    result: str |None = Field(...)

class StatsResponse(BaseModel):
    mean_simulation_time_ms: float | None = Field(None, )
    simulation_time_50_ms: float | None = Field(None)
    simulation_time_95_ms: float | None = Field(None)
    simulation_time_99_ms: float | None = Field(None)
    most_often_data_distribution: List[Distribution] | None = Field(None)
    

class HistoryResponse(SimulationRun):
    run_id: str = Field(...)
    run_time: datetime = Field(...)

@app.post(
    "/forward",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new run of simulation",
    description="Start a new simulation with results that can be gotten by /check"
)
async def forward(
    user_data: SimulationRun,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> ForwardResponse:
    run_id = str(uuid.uuid4())
    task_request = SimulationRuns(
        run_id = run_id,
        fl_algorithm = user_data.fl_algorithm,
        distribution = user_data.data_distribution,
        distribution_params = json.dumps(user_data.distribution_params)
    )
    current_task = RunProgress(
        run_id = task_request.run_id,
        status = RunStatus.IN_PROGRESS,
        result = None,
        related_run=task_request
    )
    run_stats = RunStats(
        run_id = task_request.run_id,
        related_run=task_request
    )

    extra_mods = ExtraMods()
    if user_data.augemntation_need:
        extra_mods.set_augmentation()

    task_request.progress = current_task
    task_request.stats = run_stats
    db.add(current_task)
    db.add(run_stats)
    await db.commit()
    coro = asyncio.to_thread(run_simulation_impl,
            user_data.fl_algorithm,
            user_data.data_distribution,
            user_data.algorithm_params,
            user_data.distribution_params,
            run_id, 
            user_data.saver_directory,
            extra_mods,
            user_data.clients
        )
    task = asyncio.create_task(coro)
    return ForwardResponse(run_id = run_id)

@app.post(
    "/forward_distribution",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new run of simulation",
    description="Start a new simulation with results that can be gotten by /check"
)
async def forward_distribution(
    user_data: SimulationRun,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> ForwardResponse:
    # TODO: make it as a common code without anything hardcoded
    fl_algorithm = user_data.fl_algorithm
    saver_directory = Path(user_data.saver_directory) / "view_data_distribution_ahead"

    num_clients = int(user_data.clients['count']) if fl_algorithm != FLAlgorithm.STANDALONE else 1
    saver_directory.mkdir(parents=True, exist_ok=True)

    label_col = 'label'

    partitioner_actor = ray.get_actor("partitioner_actor")
    creation_ref = partitioner_actor.create_partitioner.remote(user_data.data_distribution.value, user_data.distribution_params, num_clients, "", saver_directory, label_col)
    class_numbers = ray.get(creation_ref)
    iid_metrics = partitioner_actor.get_disbalance_metric.remote(class_numbers, RandomStateInitialization(range(num_clients)))
    plot_std_data(ray.get(iid_metrics), save_dir=saver_directory)
    return ForwardResponse(run_id = "")

@app.post(
    "/forward_long_run",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new runs of simulation",
    description="Start a new simulation with results that can be gotten by /check"
)
async def forward(
    user_data: SimulationManyRun,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> ForwardResponse:

    coro = asyncio.to_thread(run_many_exps_from_path, user_data.saver_directory)
    task = asyncio.create_task(coro)
    return ForwardResponse(run_id = 'run_id')



@app.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Check if the run of simulation finished with something",
    description="after forward - completed with results or error"
)
async def status_result(
    run_id: str,
    db: AsyncSession = Depends(get_db)
) -> StatusResponse:
    current_task_sql = await db.execute(
        Select(RunProgress).where(RunProgress.run_id == run_id)
    )
    current_task = current_task_sql.scalar_one_or_none()
    result = StatusResponse(result='')
    if current_task is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No task found"
        )

    if current_task.status != RunStatus.IN_PROGRESS:
        result.is_finished = True 
    else:
        result.is_finished = False 
    
    if current_task.status == RunStatus.ERROR:
        result.is_error = True 
    else:
        result.is_error = False 
    
    if current_task.status == RunStatus.COMPLETED:
        result.result = current_task.result
    await db.commit()
    return result

@app.get(
    "/stats",
    status_code=status.HTTP_200_OK,
    summary="Get statistics of a run_id",
    description="Statistics: simulation time in milliseconds"
)
async def stats_result(
    over_last: int = 100,
    db: AsyncSession = Depends(get_db)
) -> StatsResponse:
    current_tasks_stats_sql = await db.execute(
        Select(RunStats).order_by(RunStats.run_time.desc()).where(RunStats.simulation_time_ms is not None).limit(over_last)
    )
    tasks_stats = current_tasks_stats_sql.scalars().all()
    times = []
    for stat in tasks_stats:
        if stat.simulation_time_ms is not None:
            times.append(stat.simulation_time_ms)
    times.sort()
    if len(times) == 0:
        await db.commit()
        return StatsResponse()
    result = StatsResponse(
        mean_simulation_time_ms=sum(times)/len(times),
        simulation_time_50_ms = times[len(times)//2],
        simulation_time_95_ms =times[int(len(times) * 0.95)],
        simulation_time_99_ms =times[int(len(times) * 0.99)],
        most_often_data_distribution = None
    )

    tasks_requests_sql = await db.execute(Select(SimulationRuns.distribution, 
        func.count(SimulationRuns.distribution).label('count_')).group_by(SimulationRuns.distribution).limit(over_last).order_by(desc('count_')
    ))
    tasks_requests = tasks_requests_sql.tuples().all()
    most_popular_algo = []
    for alg, count_ in tasks_requests:
        if len(most_popular_algo) == 0 or most_popular_algo[-1][1] == count_:
            most_popular_algo.append((alg, count_))
        else:
            break
    await db.commit()
    result.most_often_data_distribution = list(zip(*most_popular_algo))[0]
    return result

@app.get(
    "/history",
    status_code=status.HTTP_200_OK,
    summary="Get history with offset and limit",
    description="after forward - what was asked at all + run_id to see results"
)
async def history_result(
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
) -> List[HistoryResponse]:
    tasks_sql = await db.execute(
        Select(SimulationRuns).order_by(SimulationRuns.run_time.desc()).limit(limit).offset(offset)
    )
    tasks = tasks_sql.fetchall()
    await db.commit()
    result = [
        HistoryResponse(
            fl_algorithm=task[0].fl_algorithm, 
            data_distribution = task[0].distribution,
            distribution_params = json.loads(task[0].distribution_params),
            run_id = task[0].run_id,
            run_time = task[0].run_time
        )
        for task in tasks
    ]
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
