FUNCTIONAL_EQUIVALENCE_MUTATIONS = {
    'A': ('V', 'L', 'I', 'G'),
    'R': ('K', 'Q', 'N'),
    'N': ('Q', 'H', 'D', 'K', 'R'),
    'D': ('E', 'N'),
    'C': ('S', 'A'),
    'Q': ('N', 'D', 'E', 'K', 'R', 'H'),
    'E': ('D', 'Q'),
    'G': ('A',),
    'H': ('R', 'N', 'Q', 'K'),
    'I': ('L', 'V', 'M', 'A', 'F'),
    'L': ('I', 'V', 'M', 'A', 'F'),
    'K': ('R', 'N', 'Q'),
    'M': ('L', 'F', 'I', 'V'),
    'F': ('Y', 'L', 'V', 'I', 'A', 'W'),
    'P': ('A',),
    'S': ('T',),
    'T': ('S',),
    'W': ('Y', 'F'),
    'Y': ('F', 'W', 'T', 'S'),
    'V': ('L', 'I', 'M', 'F', 'A'),
}
# CANDIDATE_MUTATIONS contains all FUNCTIONAL_EQUIVALENCE_MUTATIONS, and add other frequently appeared mutations that are not contained in T113 patents
CANDIDATE_MUTATIONS = {
    'A': ('V', 'L', 'I', 'G', 'S'),
    'I': ('L', 'V', 'M', 'A', 'F'),
    'L': ('I', 'V', 'M', 'A', 'F'),
    'V': ('L', 'I', 'M', 'F', 'A'),
    'M': ('L', 'F', 'I', 'V', 'T'),
    'F': ('Y', 'L', 'V', 'I', 'A', 'W'),
    'W': ('Y', 'F', 'A', 'E', 'H'),
    'Y': ('F', 'W', 'T', 'S', 'H'),
    'G': ('A', 'S', 'D', 'N', 'K'),
    'P': ('A', 'G', 'S'),
    'C': ('S', 'A'),
    'S': ('T', 'G', 'A', 'Y', 'D'),
    'T': ('S', 'I', 'A', 'L', 'Y'),
    'N': ('Q', 'H', 'D', 'K', 'R'),
    'Q': ('N', 'D', 'E', 'K', 'R', 'H'),
    'D': ('E', 'N', 'S', 'G', 'V'),
    'E': ('D', 'Q', 'S', 'N', 'H'),
    'H': ('R', 'N', 'Q', 'K', 'L'),
    'K': ('R', 'N', 'Q', 'G', 'S'),
    'R': ('K', 'Q', 'N', 'A', 'S'),
}
# Frequently appeared mutations that do not belong to FUNCTIONAL_EQUIVALENCE_MUTATIONS and not contained in T113 patents
NEGATIVE_MUTATIONS = {
    'A': ('S', 'Y', 'R', 'T', 'W'),
    'I': ('T', 'S', 'N', 'G', 'Y'),
    'L': ('H', 'N', 'Y', 'S', 'T'),
    'V': ('G', 'D', 'T', 'S', 'Y'),
    'M': ('T', 'S', 'G', 'A', 'Y'),
    'F': ('N', 'S', 'E', 'P', 'D'),
    'W': ('A', 'E', 'H', 'N', 'G'),
    'Y': ('H', 'A', 'N', 'L', 'D'),
    'G': ('S', 'D', 'N', 'K', 'P'),
    'P': ('G', 'S', 'V', 'N', 'F'),
    'C': (),
    'S': ('G', 'A', 'Y', 'D', 'E'),
    'T': ('I', 'A', 'L', 'Y', 'N'),
    'N': ('L', 'Y', 'E', 'G', 'S'),
    'Q': ('G', 'L', 'Y', 'S'),
    'D': ('S', 'G', 'V', 'Y', 'K'),
    'E': ('S', 'N', 'H', 'Y', 'K'),
    'H': ('L', 'Y', 'E', 'G', 'W'),
    'K': ('G', 'S', 'E', 'D', 'T'),
    'R': ('A', 'S', 'Y', 'T', 'G'),
}

SMALL_AA = ['A', 'G', 'S', 'N', 'D', 'P', 'T']
MEDIUM_AA = ['Q', 'E', 'H', 'R', 'K']
LARGE_AA = ['V', 'I', 'L', 'M']
VERY_LARGE_AA = ['L', 'F', 'W', 'Y']

def generate_mutation_mapping():
    """
    mutation mapping dictionary
    """
    from collections import defaultdict
    mutation_map = defaultdict(list)
    aa_sets = [SMALL_AA, MEDIUM_AA, LARGE_AA, VERY_LARGE_AA]
    for aa_s in aa_sets:
        line = "".join(aa_s)
        for aa in aa_s:
            mutation_map[aa].extend(list(line.replace(aa, "")))
    return mutation_map
