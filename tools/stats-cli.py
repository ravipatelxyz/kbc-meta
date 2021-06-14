#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import seaborn as sns

path = sys.argv[1]

with open(path, 'r') as f:
    lines = f.readlines()

triples = [triple.strip().split('\t') for triple in lines]

word_to_count = {}

for t in triples:
    for s in t:
        if s not in word_to_count:
            word_to_count[s] = 0
        word_to_count[s] += 1

_counts = [c for k, c in word_to_count.items()]
counts = [c for c in _counts if c < 100]

ax = sns.countplot(counts)
fig = ax.get_figure()

fig.savefig("output.png")

print(len([c for c in _counts if c < 10]), len(_counts))
