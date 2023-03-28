"""
generate fake samples by double mutation on CDR-H2/H3,CDR-L3

step 1: numbering heavy chain via imgt or kabat
step 2: make double mutation on CDR-H2/H3,CDR-L3
step3: place the generated CDR-H2/H3,CDR-L3 back into the original regions

Reference 1: amino acid property and volume, pls refers to
    http://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/IMGTclasses.html

Reference 2:
    kd = 1e-9
    RT = 0.592126
    deltag_10_nM = RT * math.log(10*kd)
    deltag_1_nM = RT * math.log(1*kd)
    deltag_01_nM = RT * math.log(0.1*kd)

    deltag_10_nM = -10.90736400619354
    deltag_1_nM = -12.27078450696773
    deltag_01_nM = -13.634205007741924

"""
import random
import argparse
import sys
import os
import pandas as pd
import hashlib

from typing import Dict
# from muation_map import generate_mutation_mapping, NEGATIVE_MUTATIONS, CANDIDATE_MUTATIONS


def single_seq_mutation(cdr_seq: str, mutation_map: Dict, dropout_p:float , mutate_num = 2):
    """
        :param seq: cdr sequence of vh or vl chain
        :param mutation_map: possible mutants(value, list of aa) of a given aa(key, char)
        :param dropout_p: mutation probablility
        :param mutate_num: number of mutation sites in given cdr_seq
    """
    if random.random() < dropout_p:
        origin_seq = cdr_seq
        seq = list(cdr_seq)
        index = list(range(len(seq)))
        random.shuffle(index)
        n_mutate = 0
        for i in index:
            if seq[i] in mutation_map:
                possible_aas = mutation_map[seq[i]]
                possible_aa_num = len(possible_aas)
                if possible_aa_num == 0:
                    continue
                else:
                    mut_idx = random.sample(range(possible_aa_num),1)[0]
                    seq[i] = possible_aas[mut_idx]  # mutate
                    n_mutate += 1
            if n_mutate == mutate_num:
                break
        if n_mutate == mutate_num:
            return "".join(seq)
        else:
            return origin_seq
    else:   # not mutate
        return cdr_seq

# if __name__ == "__main__":
#     baseline = "AREAYWYDKYFDY"
#     map_config = "size_based"
#     if map_config == "candidate":
#         # the mutants may not affect binding affinity
#         mutation_map = CANDIDATE_MUTATIONS
#     elif map_config == "negative":
#         # the mutants may bring negative effect to binding affinity
#         mutation_map = NEGATIVE_MUTATIONS
#     elif map_config == "size_based":
#         # the mutant is similar in size to the original amino acid
#         mutation_map = generate_mutation_mapping()
#     else:
#         raise NotImplementedError()

#     for _ in range(10):
#         print(single_seq_mutation(
#             cdr_seq=baseline,
#             mutation_map=mutation_map,
#             dropout_p=0.5,
#             mutate_num=2
#         ))

