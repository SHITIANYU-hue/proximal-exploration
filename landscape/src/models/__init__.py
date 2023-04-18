from src.models.counter_selection import (
    EnsembleInference,
    EnsembleInferenceBCE,
)

module_collection = {
    "EnsembleInference": EnsembleInference,
    "EnsembleInferenceBCE": EnsembleInferenceBCE,
}


def get_module(name):
    return module_collection[name]
