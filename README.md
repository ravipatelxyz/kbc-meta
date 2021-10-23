# Thesis title: Gradient-based Regularisation Hyperparameter Optimisation for Neural Link Predictors

**Candidate number**: NXLX9

**Supervisors:** Yihong Chen, Pasquale Minervini, Pontus Stenetorp

This repository contains all code corresponding to the stated thesis, submitted as part of **UCL Machine Learning MSc**.


# Abstract

*See thesis PDF for Abstract citations*

Innovations in neural link prediction models have been plentiful of late, with representation learning approaches continuing to achieve state of the art performance on benchmark datasets. Model performance has been shown to be highly dependent on hyperparameter choice, including the method and strength of regularisation. Sparsity in knowledge graph datasets, with long-tailed entity/relation frequency distributions due to infrequent entities and relations, presents a special challenge to mitigate over overfitting through regularisation. Work to date has focused on $L2$-norm or tensor $N3$-norm penalty regularisation with regularisation strength controlled by a single scalar hyperparameter. This leaves significant scope for innovation in the regulariser, including consideration of heavily hyperparameterised regularisers, to improve model generalisation performance.

Traditional approaches to hyperparameter optimisation have limited applicability beyond settings with small or medium hyperparameter dimensionality. Therefore, we explored gradient-based hyperparameter optimisation as an alternative more suited to scaling to high dimensions, and novel in the setting of neural link prediction. We empirically evaluated backward mode iterative differentiation for hyper-gradient calculation, first in a toy bilevel optimisation setting, then in a single regularisation hyperparameter neural link prediction setting, and finally in a high-dimensional per-embedding hyperparameter setting. Multiple method failure modes were characterised and resolved, through techniques including enforcement of positive-only regularisation hyperparameter values, gradient clipping, gradient-induced re-initialisation, and gradient accumulation and averaging. Optimisation of per-embedding regularisation hyperparameters, with full-batch training, was associated with improved validation loss and validation mean reciprocal rank compared to a grid search baseline. This represents new evidence that the method may improve neural link prediction performance in combination with fine-grained regularisation hyperparameters. Preliminary results suggest evolutionary strategies provide a promising, more memory-efficient alternative to backpropagation for hyper-gradient estimation.
