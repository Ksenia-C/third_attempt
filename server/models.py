from enum import Enum as PyEnum

class FLAlgorithm(PyEnum):
    STANDALONE = "standalone"
    FED_AVG = "fed_avg"
    FED_PROX = "fed_prox"
    SCAFFOLD = "scaffold"

class Distribution(PyEnum):
    RANDOM_UNIFORM = "random_uniform"
    RARE_ON_RARE = "rare_on_rare"
    RARE_ON_OFTEN = "rare_on_often"
    OFTEN_ON_OFTEN = "often_on_often"
    OFTEN_EVERYWHERE = "often_everywhere"

class RunStatus(PyEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
