"""Defines the Random explorer class."""
from typing import Optional, Tuple
import random
import numpy as np
import pandas as pd
from . import register_algorithm
import flexs
from flexs.utils import sequence_utils as s_utils

@register_algorithm("random")
class Random(flexs.Explorer):
    """A simple random explorer.

    Chooses a random previously measured sequence and mutates it.

    A good baseline to compare other search strategies against.

    Since random search is not data-driven, the model is only used to score
    sequences, but not to guide the search strategy.
    """

    def __init__(
        self,
        args,
        model: flexs.Model,
        # model,
        alphabet: str,
        starting_sequence: str,
 
    ):
        """
        Create a random search explorer.

        Args:
            mu: Average number of residue mutations from parent for generated sequences.
            elitist: If true, will propose the top `sequences_batch_size` sequences
                generated according to `model`. If false, randomly proposes
                `sequences_batch_size` sequences without taking model score into
                account (true random search).
            seed: Integer seed for random number generator.

        """
        mu=float(1)
        name = f"Random_mu={mu}"

        # super().__init__(
        #     model,
        #     name,
        #     # rounds,
        #     # sequences_batch_size,
        #     # model_queries_per_batch,
        #     starting_sequence,
        #     # log_file,
        # )
        self.seed=random.randint(0, 10),
        self.model=model
        self.mu=mu
        self.rng = np.random.default_rng(self.seed)
        self.alphabet = alphabet
        self.elitist = False
        self.model_queries_per_batch=args.num_model_queries_per_round
        self.sequences_batch_size=args.batch_size
        self.rounds=args.num_queries_per_round
        self.data_range=args.datasetrange
    
    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        old_sequences = measured_sequences["sequence"]
        old_sequence_set = set(old_sequences)
        new_seqs = set()

        while len(new_seqs) <= self.model_queries_per_batch:
            seq = self.rng.choice(old_sequences)
            new_seq = s_utils.generate_random_mutant(
                seq, self.mu / len(seq), alphabet=self.alphabet
            )

            if new_seq not in old_sequence_set and int(new_seq,2)<self.data_range:

                new_seqs.add(new_seq)

        new_seqs = np.array(list(new_seqs))
        preds = self.model.get_fitness(new_seqs)

        if self.elitist:
            idxs = np.argsort(preds)[: -self.sequences_batch_size : -1]
        else:
            idxs = self.rng.integers(0, len(new_seqs), size=self.sequences_batch_size)
        # import random
        # idxs= random.sample(idxs,self.rounds)
        return new_seqs[idxs[0:self.rounds]], preds[idxs]