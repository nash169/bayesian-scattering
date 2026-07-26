from bayesian_scattering.features.wavelet_scattering import WaveletScattering
from bayesian_scattering.features.timm import TorchImageModel
from bayesian_scattering.features.identity import Identity
from bayesian_scattering.features.pca import PCA
from bayesian_scattering.features.random_conv import RandomConv

__all__ = ["WaveletScattering", "TorchImageModel", "Unimol", "Identity", "PCA", "RandomConv"]


def __getattr__(name):
    if name == "Unimol":
        from bayesian_scattering.features.unimol import Unimol

        return Unimol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
