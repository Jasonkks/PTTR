from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)


# from .pointnet_cosine import PointnetCosine
from .pointnet_transformer import PointnetTransformerSiamese

def get_model(name, *args, **kwargs):
    if name in ['C', 'cosine', 'c']:
        return PointnetCosine(*args, **kwargs)
    elif name in ['T', 'transformer', 't']:
        return PointnetTransformerSiamese(*args, **kwargs)
    else:
        raise ValueError()
