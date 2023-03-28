from src.models.counter_selection import (
    EnsembleBinaryClassificationModel,
    EnsembleRegressionModel,
)

module_collection = {
    "EnsembleBinaryClassificationModel": EnsembleBinaryClassificationModel,
    "EnsembleRegressionModel": EnsembleRegressionModel,
}


def get_module(name):
    return module_collection[name]