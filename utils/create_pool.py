from antiberty import AntiBERTyRunner
import torch.nn as nn# define the adaptive average pooling layer
import torch
import pandas as pd
import numpy as np
import json
import time

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


def get_ppl_service(ab_seq) -> json:
    import urllib3

    http = urllib3.PoolManager(cert_reqs='CERT_NONE')
    http_request = http.request(
        method="POST",
        url="http://58.18.174.244:37024/231773288/ppl-score?raw_data=" + ab_seq,
        headers={
            'Content-Type': 'application/json;charset=UTF-8'
        },
    )
    if http_request.status == 200:
        return str(http_request.data, "GBK")
    time.sleep(5)

def APIpool(names,thre=0.8):
    scores=[]
    for i in range(len(names)):
        sequence= f"H,[human],{names[i]}"
        analyze_data = get_ppl_service(sequence)
        # print('scores:',eval(analyze_data)['pseudo_ppl'])
        scores.append(eval(analyze_data)['pseudo_ppl'])

    sorted_names = [x[0] for x in sorted(zip(names, scores), key=lambda x: x[1])]

    top_n_percent=int(len(names)*thre)

    return sorted_names[0:top_n_percent]