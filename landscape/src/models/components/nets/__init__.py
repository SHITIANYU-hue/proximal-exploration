from src.models.components.nets.convolution import (
    Seq32x1_16,
    Seq32x2_16,
    Seq64x1_16,
    Seq_emb_32x1_16,
    Seq32x1_16_filt3,
)
from src.models.components.nets.dense import Seq_32_32

net_collection = {
    "Seq_32_32": Seq_32_32,
    "Seq32x1_16": Seq32x1_16,
    "Seq32x2_16": Seq32x2_16,
    "Seq64x1_16": Seq64x1_16,
    "Seq32x1_16_filt3": Seq32x1_16_filt3,
    "Seq_emb_32x1_16": Seq_emb_32x1_16,
}


def get_net(name):
    return net_collection[name]