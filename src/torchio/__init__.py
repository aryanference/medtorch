"""Top-level package for MedTorch."""

__author__ = """MedTorch Team"""
__email__ = 'hello@medtorch.dev'
__version__ = '1.2.0'


from . import datasets
from . import metrics
from . import reference
from . import utils
from .constants import *  # noqa: F401, F403
from .data import GridAggregator
from .data import GridSampler
from .data import Image
from .data import LabelMap
from .data import LabelSampler
from .data import Queue
from .data import ScalarImage
from .data import Subject
from .data import SubjectsDataset
from .data import SubjectsLoader
from .data import UniformSampler
from .data import WeightedSampler
from .data import inference
from .data import io
from .data import sampler
from .data.multi_subject_transform import MultiSubjectTransform
from .transforms import *  # noqa: F401, F403

__all__ = [
    'utils',
    'io',
    'sampler',
    'inference',
    'metrics',
    'SubjectsDataset',
    'SubjectsLoader',
    'Image',
    'ScalarImage',
    'LabelMap',
    'Queue',
    'Subject',
    'datasets',
    'reference',
    'WeightedSampler',
    'UniformSampler',
    'LabelSampler',
    'GridSampler',
    'GridAggregator',
    'MultiSubjectTransform',
]
