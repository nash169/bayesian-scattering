#!/usr/bin/env python

from bayesian_scattering.utils.dataset import sample_loader, sample_index, sample_duplicates
from bayesian_scattering.utils.math import average_distance
from bayesian_scattering.utils.train import train_exact_gp, train_approx_gp
from bayesian_scattering.utils.test import test_regression, test_classification
from bayesian_scattering.utils.helpers import get_dataset, get_feature, get_model, get_kernel, train_model, get_results

__all__ = [
]
