"""Define a baseline genetic algorithm implementation."""
from typing import Optional, Tuple
from . import register_algorithm
import numpy as np
import pandas as pd
import torch

import flexs
from flexs.utils import sequence_utils as s_utils


@register_algorithm("genetic")
class GeneticAlgorithm(): ## if inherit it, it can improve score
    """A genetic algorithm explorer with single point mutations and recombination.

    Based on the `parent_selection_strategy`, this class implements one of three
    genetic algorithms:

        1. If `parent_selection_strategy == 'top-k'`, we have a traditional
           genetic algorithm where the top-k scoring sequences in the
           population become parents.

        2. If `parent_selection_strategy == 'wright-fisher'`, we have a
           genetic algorithm based off of the Wright-Fisher model of evolution,
           where members of the population become parents with a probability
           exponential to their fitness (softmax the scores then sample).
    """

    def __init__(self, args, model, alphabet, starting_sequence):

        """Create genetic algorithm."""
        parent_selection_strategy='top-proportion'
        population_size=20
        
        name = (
            f"GeneticAlgorithm_pop_size={population_size}_"
            f"parents={parent_selection_strategy}"
        )
        beta=0.01

        self.model_queries_per_batch=args.num_model_queries_per_round
        self.sequences_batch_size=args.num_queries_per_round
        self.alphabet = alphabet
        self.population_size = 20
        self.starting_sequence=starting_sequence
        self.rounds=args.num_rounds
        self.model=model
        parent_selection_proportion=0.3
        # Validate parent_selection_strategy
        valid_parent_selection_strategies = ["top-proportion", "wright-fisher"]
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
        self.beta = beta
        self.children_proportion = 0.2
        self.parent_selection_proportion = parent_selection_proportion

        self.rng = np.random.default_rng(1)

    def _choose_parents(self, scores, num_parents):
        """Return parent indices according to `self.parent_selection_strategy`."""
        if self.parent_selection_strategy == "top-proportion":
            k = int(self.parent_selection_proportion * self.population_size)
            return self.rng.choice(np.argsort(scores)[-k:], num_parents)

        # Then self.parent_selection_strategy == "wright-fisher":
        fitnesses = np.exp(scores / self.beta)
        probs = torch.Tensor(fitnesses / np.sum(fitnesses))
        return torch.multinomial(probs, num_parents, replacement=True).numpy()

    def propose_sequences(self, measured_sequences, score_max):
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
            < self.model_queries_per_batch
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
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]