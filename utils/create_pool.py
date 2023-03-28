from antiberty import AntiBERTyRunner
import torch.nn as nn# define the adaptive average pooling layer
import torch
import pandas as pd
import numpy as np


def AntiBertypool(names,thre=0.8):
    antiberty = AntiBERTyRunner()
    scores=[]
    for i in range(len(names)):
        sequence=names[i]
        embeddings, attentions = antiberty.embed(sequence, return_attention=True)
        # print('scores:',eval(analyze_data)['pseudo_ppl'])
        scores.append(torch.mean(torch.cat(embeddings).float()).cpu().numpy())

    sorted_names = [x[0] for x in sorted(zip(names, scores), key=lambda x: x[1])]

    top_n_percent=int(len(names)*thre)

    return sorted_names[0:top_n_percent]

