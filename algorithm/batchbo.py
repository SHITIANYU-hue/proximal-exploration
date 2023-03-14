import random
import numpy as np
from scipy.stats import norm
import torch
from . import register_algorithm
from utils.seq_utils import hamming_distance, random_mutation
from flexs.utils import sequence_utils as s_utils

@register_algorithm("batchbo")
class ProximalExploration: 
    """
        batchbo
    """
    
    def __init__(self, args, model, alphabet, starting_sequence):
        method = "UCB_risk"
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
        self.rng = np.random.default_rng(1)
        self.population_size = 20
        self.children_proportion = 0.2
        self.eps=0.9
        self.model_queries_per_batch=args.num_model_queries_per_round

        ## mutation config
        beta=0.01
        self.beta=beta
        parent_selection_proportion=0.3
        valid_parent_selection_strategies = ["top-proportion", "wright-fisher"]
        parent_selection_strategy='top-proportion'
        if parent_selection_strategy not in valid_parent_selection_strategies:
            raise ValueError(
                f"parent_selection_strategy must be one of "
                f"{valid_parent_selection_strategies}"
            )
        if (
            parent_selection_strategy == "top-proportion"
            and parent_selection_proportion is None
        ):
            raise ValueError(
                "if top-proportion, parent_selection_proportion cannot be None"
            )
        if parent_selection_strategy == "wright-fisher" and beta is None:
            raise ValueError("if wright-fisher, beta cannot be None")
        self.parent_selection_strategy = parent_selection_strategy
        self.parent_selection_proportion = parent_selection_proportion





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

    def _choose_parents(self, scores, num_parents):
        """Return parent indices according to `self.parent_selection_strategy`."""
        if self.parent_selection_strategy == "top-proportion":
            k = int(self.parent_selection_proportion * self.population_size)
            return self.rng.choice(np.argsort(scores)[-k:], num_parents)

        # Then self.parent_selection_strategy == "wright-fisher":
        fitnesses = np.exp(scores / self.beta)
        probs = torch.Tensor(fitnesses / np.sum(fitnesses))
        return torch.multinomial(probs, num_parents, replacement=True).numpy()

    def construct_candidate_pool(self,measured_sequences,mutation_times,len_pool):
        """Propose top `sequences_batch_size` sequences for evaluation."""
        # Set the torch seed by generating a random integer from the pre-seeded self.rng
        torch.manual_seed(self.rng.integers(-(2**31), 2**31))

        measured_sequence_set = set(measured_sequences["sequence"])

        # Create initial population by choosing parents from `measured_sequences`
        initial_pop_inds = self._choose_parents(
            measured_sequences["true_score"].to_numpy(),
            self.population_size,
        )
        pop = measured_sequences["sequence"].to_numpy()[initial_pop_inds]
        scores = measured_sequences["true_score"].to_numpy()[initial_pop_inds]

        sequences = {}
        count=0
        while (
            count + self.population_size
            < mutation_times
        ):
            # Create "children" by recombining parents selected from population
            # according to self.parent_selection_strategy and
            # self.recombination_strategy
            count+=1
            num_children = int(self.children_proportion * self.population_size)
            parents = pop[self._choose_parents(scores, num_children)]

            # Single-point mutation of children (for now)
            children = []
            for seq in parents:
                child = s_utils.generate_random_mutant(seq, 1 / len(seq), self.alphabet)

                if child not in measured_sequence_set and child not in sequences:
                    children.append(child)

            if len(children) == 0:
                continue

            children = np.array(children)
            child_scores = self.model.get_fitness(children)

            # Now kick out the worst samples and replace them with the new children
            argsorted_scores = np.argsort(scores)
            pop[argsorted_scores[: len(children)]] = children
            scores[argsorted_scores[: len(children)]] = child_scores

            sequences.update(zip(children, child_scores))

        # We propose the top `self.sequences_batch_size`
        # new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -len_pool : -1]

        return new_seqs[sorted_order]

    def _propose_sequences(self, measured_sequences,score_max):

        candidate_pool = self.construct_candidate_pool(measured_sequences,mutation_times=2000,len_pool=200)

    

        # use exploer to fine tune the candidate pool
        new_state_string, _ = self.pick_action(
            candidate_pool, score_max
        ) 

        # return np.array(candidate_pool)
        return new_state_string
