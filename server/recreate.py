import ray
import task as task_module
ray.init(address="auto", namespace="flower_simulation")

try:
    actor = ray.get_actor("partitioner_actor")
    ray.kill(actor)
    print("Old actor killed.")
except ValueError as exc:
    print("Actor not found – nothing to kill.", exc)

partitioner_actor = task_module.PartitionerActor.options(name="partitioner_actor",namespace="flower_simulation", lifetime="detached").remote()
print("and created")