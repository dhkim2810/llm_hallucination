from .modules import *


def load_dataset(name, **kwargs):
    dsets = {
        "sciq": SciQDataset,
        "scienceqa": ScienceQADataset,
    }
    return dsets[name](**kwargs)
