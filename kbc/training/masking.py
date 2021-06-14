# -*- coding: utf-8 -*-

import numpy as np

from typing import List, Tuple, Dict


def compute_masks(triples: List[Tuple[str, str, str]],
                  all_triples: List[Tuple[str, str, str]],
                  entity_to_idx: Dict[str, int],
                  predicate_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nb_examples = len(triples)
    nb_entities = len(entity_to_idx)
    nb_predicates = len(predicate_to_idx)

    sp_to_o_lst: Dict[Tuple[int, int], List[int]] = dict()
    po_to_s_lst: Dict[Tuple[int, int], List[int]] = dict()
    so_to_p_lst: Dict[Tuple[int, int], List[int]] = dict()

    for s, p, o in all_triples:
        s_idx, p_idx, o_idx = entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o]
        key_sp = (s_idx, p_idx)
        key_po = (p_idx, o_idx)
        key_so = (s_idx, o_idx)

        if key_sp not in sp_to_o_lst:
            sp_to_o_lst[key_sp] = []
        sp_to_o_lst[key_sp] += [o_idx]

        if key_po not in po_to_s_lst:
            po_to_s_lst[key_po] = []
        po_to_s_lst[key_po] += [s_idx]

        if key_so not in so_to_p_lst:
            so_to_p_lst[key_so] = []
        so_to_p_lst[key_so] += [p_idx]
    
    mask_sp = np.zeros(shape=(nb_examples, nb_entities), dtype=np.float)
    mask_po = np.zeros(shape=(nb_examples, nb_entities), dtype=np.float)
    mask_so = np.zeros(shape=(nb_examples, nb_predicates), dtype=np.float)

    for i, (s, p, o) in enumerate(triples):
        s_idx, p_idx, o_idx = entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o]

        key_sp = (s_idx, p_idx)
        key_po = (p_idx, o_idx)
        key_so = (s_idx, o_idx)

        if key_sp in sp_to_o_lst:
            for _o_idx in sp_to_o_lst[key_sp]:
                if _o_idx != o_idx:
                    mask_sp[i, _o_idx] = - np.inf

        if key_po in po_to_s_lst:
            for _s_idx in po_to_s_lst[key_po]:
                if _s_idx != s_idx:
                    mask_po[i, _s_idx] = - np.inf

        if key_so in so_to_p_lst:
            for _p_idx in so_to_p_lst[key_so]:
                if _p_idx != p_idx:
                    mask_so[i, _p_idx] = - np.inf

    return mask_sp, mask_po, mask_so
