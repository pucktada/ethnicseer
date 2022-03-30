from .ethnic_classifier import EthnicClassifier
from .experiment import load_from_directory, train_model, train_and_evaluate

__all__ = (
    '__version__',
    'my_package_name'
)
import pkg_resources  # part of setuptools
__version__ = pkg_resources.get_distribution("ethnicseer").version