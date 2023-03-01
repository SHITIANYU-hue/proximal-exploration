import numpy as np


ensemble_rules = {
    'mean': lambda x: np.mean(x, axis=0),
    'lcb': lambda x: np.mean(x, axis=0) - np.std(x, axis=0),
    'ucb': lambda x: np.mean(x, axis=0) + np.std(x, axis=0)
}

class Ensemble:
    def __init__(self, models, ensemble_rule):
        self.models = models
        self.ensemble_func = ensemble_rules[ensemble_rule]
        self.cost = 0
    def train(self, sequences, labels):
        for model in self.models:
            loss=model.train(sequences, labels)
            return loss

    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]
        self.cost +=len(sequences)
        return self.ensemble_func([model.get_fitness(sequences) for model in self.models])
    
    def get_uncertainty(self,sequences):
        fitness=[]
        for model in self.models:
            fitness.append(model.get_fitness(sequences))
        fitness_std=np.std(fitness,axis=0)

        return fitness_std