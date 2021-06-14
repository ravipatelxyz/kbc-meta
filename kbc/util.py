# -*- coding: utf-8 -*-

import numpy as np
import random
import torch

from typing import List, Tuple

try:
    from apex import amp  # noqa: F401

    _has_apex = True
except ImportError:
    _has_apex = False

try:
    import torch_xla.core.xla_model as xm  # noqa: F401

    _torch_tpu_available = True  # pylint: disable=
except ImportError:
    _torch_tpu_available = False


def is_apex_available():
    return _has_apex


def is_torch_tpu_available():
    return _torch_tpu_available


def make_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res


def set_seed(seed: int, is_deterministic=True):
    # set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return
