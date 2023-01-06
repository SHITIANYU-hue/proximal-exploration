import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import random
from copy import deepcopy
from itertools import groupby
import re

def hamming_distance(seq_1, seq_2, config=None): #config: list, e.g. [2, 3, 4, 5] -- denotes there are 4 categorical variables, with numbers of categories
                                                 # being 2, 3, 4, and 5 respectively.
    return sum([x!=y for x, y in zip(seq_1, seq_2)])

def convert_str(data, name):
    id=int(data,2)
    if id>=len(name):
        id=np.random.randint(len(name))
    return name[id]
    # if len(data)==20:
    #     return name[int(data,2)]
    # else:
    #     seq=[]
    #     for i in range(len(data)):
    #         seq.append(name[int(data[i],2)])
    #     return seq

def levenshteinDistance(s1_, s2_,name):
    id1=int(s1_,2)
    id2=int(s2_,2)
    if id1>=len(name) or id2>=len(name):
        return 5
    else:
        s1=name[id1]
        s2 = name[id2]

        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

def levenshteinDistance_(s1_, seq_batch, s2_,name):
    id1=int(s1_,2)
    id2=int(s2_,2)
    # print('seq batch',seq_batch)
    if id1>=len(name) or id2>=len(name):
        return 5
    else:
        s1=name[id1]
        s2 = name[id2]

        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]


def dec2bin(num,length=16):
    l = []
    while length>=0:
        num, remainder = divmod(num, 2)
        l.append(str(remainder))
        length=length-1
    
    return ''.join(l[::-1])

def random_mutation(sequence, alphabet, num_mutations,range):

    idx=np.random.randint(range)
    return dec2bin(idx)

def random_mutation_(sequence, alphabet, num_mutations): ##origional mutation function
    wt_seq = list(sequence)
    for _ in range(num_mutations):
        idx = np.random.randint(len(sequence))
        wt_seq[idx] = alphabet[np.random.randint(len(alphabet))]
    new_seq = ''.join(wt_seq)
    return new_seq

def sequence_to_one_hot(sequence, alphabet):
    # Input:  - sequence: [sequence_length]
    #         - alphabet: [alphabet_size]
    # Output: - one_hot:  [sequence_length, alphabet_size]
    alphabet_dict = {x: idx for idx, x in enumerate(alphabet)}
    one_hot = F.one_hot(torch.tensor([alphabet_dict[x] for x in sequence]).long(), num_classes=len(alphabet))
    return one_hot

def sequences_to_tensor(sequences, alphabet):
    # Input:  - sequences: [batch_size, sequence_length]
    #         - alphabet:  [alphabet_size]
    # Output: - one_hots:  [batch_size, alphabet_size, sequence_length]
    
    one_hots = torch.stack([sequence_to_one_hot(seq, alphabet) for seq in sequences], dim=0)
    one_hots = torch.permute(one_hots, [0, 2, 1]).float()
    return one_hots

def sequences_to_mutation_sets(sequences, alphabet, wt_sequence, context_radius):
    # Input:  - sequences:          [batch_size, sequence_length]
    #         - alphabet:           [alphabet_size]
    #         - wt_sequence:        [sequence_length]
    #         - context_radius:     integer
    # Output: - mutation_sets:      [batch_size, max_mutation_num, alphabet_size, 2*context_radius+1]
    #         - mutation_sets_mask: [batch_size, max_mutation_num]

    context_width = 2 * context_radius + 1
    max_mutation_num = max(1, np.max([hamming_distance(seq, wt_sequence) for seq in sequences]))
    
    mutation_set_List, mutation_set_mask_List = [], []
    for seq in sequences:
        one_hot = sequence_to_one_hot(seq, alphabet).numpy()
        one_hot_padded = np.pad(one_hot, ((context_radius, context_radius), (0, 0)), mode='constant', constant_values=0.0)
        
        mutation_set = [one_hot_padded[i:i+context_width] for i in range(len(seq)) if seq[i]!=wt_sequence[i]]
        padding_len = max_mutation_num - len(mutation_set)
        mutation_set_mask = [1.0] * len(mutation_set) + [0.0] * padding_len
        mutation_set += [np.zeros(shape=(context_width, len(alphabet)))] * padding_len
            
        mutation_set_List.append(mutation_set)
        mutation_set_mask_List.append(mutation_set_mask)
    
    mutation_sets = torch.tensor(np.array(mutation_set_List)).permute([0, 1, 3, 2]).float()
    mutation_sets_mask = torch.tensor(np.array(mutation_set_mask_List)).float()
    return mutation_sets, mutation_sets_mask


def random_sample_within_discrete_tr_ordinal(x_center, max_hamming_dist, n_categories):
    """Same as above, but here we assume a ordinal representation of the categorical variables."""
    # random.seed(random.randint(0, 1e6))
    if max_hamming_dist < 1:
        bit_change = int(max(max_hamming_dist * len(n_categories), 1))
    else:
        bit_change = int(min(max_hamming_dist, len(n_categories)))
    x_pert = deepcopy(x_center)
    modified_bits = random.sample(range(len(n_categories)), bit_change)
    for bit in modified_bits:
        options = np.arange(n_categories[bit])
        x_pert[bit] = int(random.choice(options))
    return x_pert

def check_cdr_constraints_all(x, x_center_local=None, hamming=None, config=None):
    COUNT_AA = 5
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AA_to_idx = {aa: i for i, aa in enumerate(AA)}
    idx_to_AA = {value: key for key, value in AA_to_idx.items()}
    N_glycosylation_pattern = 'N[^P][ST][^P]'
    # Constraints on CDR3 sequence
    x_to_seq = ''.join(idx_to_AA[int(aa)] for aa in x)
    
    # prot = ProteinAnalysis(x_to_seq)
    # charge = prot.charge_at_pH(7.4)
    # Counting number of consecutive keys
    count = max([sum(1 for _ in group) for _, group in groupby(x_to_seq)])
    if count > 5:
        c1 = False
    else:
        c1 = True
    charge = 0
    for char in x_to_seq:
        charge += int(char == 'R' or char == 'K') + 0.1 * int(char == 'H') - int(char == 'D' or char == 'E')
    if (charge > 2.0 or charge < -2.0):
        c2 = False
    else:
        c2 = True
    if re.search(N_glycosylation_pattern, x_to_seq):
        c3 = False
    else:
        c3 = True

    if x_center_local is not None:
        # 1 if met (True)
        c4 = hamming_distance(x_center_local, x, config) <= hamming
        # Return 0 if True
        return int(not (c1)), int(not (c2)), int(not (c3)), int(not (c4))

    return int(not (c1)), int(not (c2)), int(not (c3))

