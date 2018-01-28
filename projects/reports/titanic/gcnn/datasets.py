import pandas as pd
import numpy as np
import os
from torchvision import datasets

current_dir = os.path.dirname(__file__)


def load_icebergs(dataset: str):
    uri = os.path.join(current_dir, '../data/{}.json'.format(dataset))
    icebergs = pd.read_json(uri).set_index('id')
    icebergs = icebergs.assign(
        inc_angle=pd.to_numeric(icebergs.inc_angle, 'coerce'),
        band_1=icebergs.band_1.apply(np.array),
        band_2=icebergs.band_2.apply(np.array),
    )
    return icebergs


def load_mnist(train: bool):
    uri = os.path.join(current_dir, '../data')
    x, y = zip(*datasets.MNIST(uri, train=train, download=True))
    return pd.DataFrame(dict(x=x, y=y))
