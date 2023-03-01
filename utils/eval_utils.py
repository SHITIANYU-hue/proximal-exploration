import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List
import torch

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from utils.seq_utils import levenshtein_distance


class Runner:
    """
        The interface of landscape/model/explorer is compatible with FLEXS benchmark.
        - Fitness Landscape EXploration Sandbox (FLEXS)
          https://github.com/samsinai/FLEXS
    """
    
    def __init__(self, args):
        self.num_rounds = args.num_rounds
        self.num_queries_per_round = args.num_queries_per_round

    def run(self, landscape, starting_sequence, model, explorer, name, runs, out_dir):
        self.results = pd.DataFrame()
        starting_fitness = landscape.get_fitness([starting_sequence])[0]
        output_dir = Path(out_dir).expanduser().resolve()
        output_dir.mkdir(exist_ok=True, parents=True)

        seed_everything(runs)
        self.results = pd.DataFrame()
        starting_fitness = landscape.get_fitness([starting_sequence])[0]
        _, _, _, _, _ ,_= self.update_results(0, [starting_sequence], [starting_fitness], 0)
        rounds_ = []
        score_maxs = []
        mutation: list[int] = []
        mutation_counts = []
        rts = []
        searched_seq_ = []
        score_max_this_round_=[]
        loss_ = [None]
        round_min_seq = starting_sequence
        score_max = starting_fitness
        
        for round in range(1, self.num_rounds+1):
            round_start_time = time.time()
            if len(self.sequence_buffer) > 1:
                loss=model.train(self.sequence_buffer, self.fitness_buffer)
                loss_.append(loss)
            sequences, model_scores = explorer.propose_sequences(self.results,score_max)
            assert len(sequences) <= self.num_queries_per_round
            true_scores = landscape.get_fitness(sequences)
            # if model_scores[0] and true_scores[0]:
            #     rmse = np.sqrt(np.mean(np.square(np.array(model_scores) - np.array(true_scores))))
                # wandb.log({"RMSE": rmse, "round": round})

            round_mutation = []
            for seq in sequences:
                round_mutation.append(levenshtein_distance(s1=starting_sequence, s2=seq))
            mutation += round_mutation

            round_running_time = time.time() - round_start_time
            round_min_seq = sequences[np.argmin(round_mutation)]
            roundss, score_max, score_max_this_round, rt, mutcounts, searched_seq = self.update_results(
                round, sequences, true_scores, np.average(mutation), round_running_time
            )
            # wandb.log({"mutations": mutcounts, "best_score": score_max, "score_max_this_round":score_max_this_round, "round": round})
            mutation_counts.append(mutcounts)
            rounds_.append(roundss)
            score_maxs.append(score_max)
            rts.append(rt)
            searched_seq_.append(searched_seq)
            score_max_this_round_.append(score_max_this_round)
            result = pd.DataFrame(
                {
                    "round": rounds_,
                    "scoremax": score_maxs,
                    "scoremaxthisround":score_max_this_round_,
                    "run_time": rts,
                    "mutcounts": mutation_counts,
                    "loss": loss_,
                    "searched_seq": searched_seq_,
                }
            )
            result.to_csv(output_dir / f"{name}_seed{runs}.csv", index=False)
    
    def update_results(
        self,
        round: int,
        sequences: List[str],
        true_scores: List[float],
        mutcounts: float,
        running_time: float = 0.0,
    ):
        self.results = pd.concat(
            (
                self.results,
                pd.DataFrame(
                    {
                        "round": round,
                        "sequence": sequences,
                        "true_score": true_scores,
                        "score_max_this_round":max(true_scores),
                        "mutcounts": mutcounts,
                    }
                ),
            ),
            axis=0,
            ignore_index=True,
        )
        print(
            "round: {}  max fitness score: {:.3f} score max this round: {:.3f}  running time: {:.2f} (sec) mutation counts:{:.3f} searched sequence number {}".format(
                round, self.results["true_score"].max(), max(true_scores), running_time, mutcounts, len(self.results)
            )
        )
        return round, self.results["true_score"].max(), max(true_scores), running_time, mutcounts, len(self.results)
    
    @property
    def sequence_buffer(self):
        return self.results['sequence'].to_numpy()

    @property
    def fitness_buffer(self):
        return self.results['true_score'].to_numpy()
