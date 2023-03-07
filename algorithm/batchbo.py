import random
import numpy as np
from scipy.stats import norm

from . import register_algorithm
from utils.seq_utils import hamming_distance, random_mutation

@register_algorithm("batchbo")
class ProximalExploration: 
    """
        batchbo
    """
    
    def __init__(self, args, model, alphabet, starting_sequence):
        method = "UCB"
        name = f"BO_method={method}"
        self.method = method
        self.model = model
        self.alphabet = alphabet
        self.wt_sequence = starting_sequence
        self.num_queries_per_round = args.num_queries_per_round
        self.num_model_queries_per_round = args.num_model_queries_per_round
        self.batch_size = args.batch_size
        self.num_random_mutations = args.num_random_mutations
        self.frontier_neighbor_size = args.frontier_neighbor_size
    
    def propose_sequences(self, measured_sequences, score_max):
        # Input:  - measured_sequences: pandas.DataFrame
        #           - 'sequence':       [sequence_length]
        #           - 'true_score':     float
        # Output: - query_batch:        [num_queries, sequence_length]
        #         - model_scores:       [num_queries]
        
        query_batch = self._propose_sequences(measured_sequences,score_max)
        model_scores = np.concatenate([
            self.model.get_fitness(query_batch[i:i+self.batch_size])
            for i in range(0, len(query_batch), self.batch_size)
        ])
        return query_batch, model_scores


    def pick_action(self, candidate_pool,score_max):
        """Pick action."""
        states_to_screen = []
        states_to_screen = []
        method_pred = []
        # local search for all satisfied seq candidate pool
        # not enough do global search
        states_to_screen = candidate_pool
        ensemble_preds = self.model.get_fitness(states_to_screen) 
        uncertainty_pred=self.model.get_uncertainty(states_to_screen)
        max_pred = max(ensemble_preds)
        mean_pred = np.mean(self.model.get_fitness(candidate_pool))
        std_pre = np.std(self.model.get_fitness(candidate_pool))
        risk=np.mean(std_pre)
        best_fitness_obs = score_max ## this is the best fitness observed from last round
        best_fitness = best_fitness_obs
        if self.method == "EI":
            method_pred = self.EI(ensemble_preds,uncertainty_pred,best_fitness)##https://machinelearningmastery.com/what-is-bayesian-optimization/ 
        if self.method == "KG":
            for i in range(len(ensemble_preds)):
                kg = self.calculate_knowledge_gradient(ensemble_preds[i], uncertainty_pred[i], best_fitness, num_fantasies=128)
                method_pred.append(kg)
        if self.method == "UCB_risk":
            method_pred = self.UCB_risk(ensemble_preds, uncertainty_pred,risk)
        if self.method == "UCB":
            method_pred = self.UCB(ensemble_preds, uncertainty_pred)            
        action_ind = np.argpartition(method_pred, -self.num_queries_per_round)[-self.num_queries_per_round:]
        action_ind = action_ind.tolist()
        new_state_string = np.asarray(states_to_screen)[action_ind]
        # self.state = string_to_one_hot(new_state_string, self.alphabet)
        # new_state = self.state
        reward = np.mean(ensemble_preds[action_ind])
        # if new_state_string not in all_measured_seqs:
        #     self.best_fitness = max(self.best_fitness, reward)
        #     self.memory.store(state.ravel(), action, reward, new_state.ravel())
        return  new_state_string, reward



    @staticmethod
    def EI(mu, std, best):
        """Compute expected improvement."""
        # print('vals',vals)
        # return np.mean([max(val - self.best_fitness, 0) for val in vals])
        return norm.cdf((mu - best) / (std+1E-9))

    @staticmethod
    def UCB_risk(vals,std_pre, risk):
        """Upper confidence bound."""
        discount = 0.5
        return vals + discount * std_pre / (0.5+risk)

    @staticmethod
    def UCB(vals,std_pre):
        """Upper confidence bound."""
        discount = 0.5
        return vals + discount * std_pre 

        
    @staticmethod
    def calculate_knowledge_gradient(mean, std, current_best, num_fantasies):
        # Sample fantasized functions
        f = np.random.normal(mean, std, size=(num_fantasies,1))
        f_best = np.max(f, axis=1)
        
        # Compute mean and std of maximum value from fantasized functions
        f_best_mean = np.mean(f_best)
        f_best_std = np.std(f_best, ddof=1)
        
        # Compute knowledge gradient
        kg = (f_best_mean - current_best) * norm.cdf((f_best_mean - current_best) / f_best_std) + \
            f_best_std * norm.pdf((f_best_mean - current_best) / f_best_std)
        
        return kg




    def _propose_sequences(self, measured_sequences,score_max):
        measured_sequence_set = set(measured_sequences['sequence'])
        
        # Generate random mutations in the first round.
        if len(measured_sequence_set)==1:
            query_batch = []
            while len(query_batch) < self.num_queries_per_round:
                random_mutant = random_mutation(self.wt_sequence, self.alphabet, self.num_random_mutations)
                if random_mutant not in measured_sequence_set:
                    query_batch.append(random_mutant)
                    measured_sequence_set.add(random_mutant)
            return query_batch
        
        # Arrange measured sequences by the distance to the wild type.
        measured_sequence_dict = {}
        for _, data in measured_sequences.iterrows():
            distance_to_wt = hamming_distance(data['sequence'], self.wt_sequence)
            if distance_to_wt not in measured_sequence_dict.keys():
                measured_sequence_dict[distance_to_wt] = []
            measured_sequence_dict[distance_to_wt].append(data)
        
        # Highlight measured sequences near the proximal frontier.
        frontier_neighbors, frontier_height = [], -np.inf
        for distance_to_wt in sorted(measured_sequence_dict.keys()):
            data_list = measured_sequence_dict[distance_to_wt]
            data_list.sort(reverse=True, key=lambda x:x['true_score'])
            for data in data_list[:self.frontier_neighbor_size]:
                if data['true_score'] > frontier_height:
                    frontier_neighbors.append(data)
            frontier_height = max(frontier_height, data_list[0]['true_score'])

        # Construct the candiate pool by randomly mutating the sequences. (line 2 of Algorithm 2 in the paper)
        # An implementation heuristics: only mutating sequences near the proximal frontier.
        candidate_pool = []
        while len(candidate_pool) < self.num_model_queries_per_round:
            candidate_sequence = random_mutation(random.choice(frontier_neighbors)['sequence'], self.alphabet, self.num_random_mutations)
            if candidate_sequence not in measured_sequence_set:
                candidate_pool.append(candidate_sequence)
                measured_sequence_set.add(candidate_sequence)
    


        new_state_string, _ = self.pick_action(
            candidate_pool, score_max
        ) 

        return new_state_string
