"""
Author: @mesparza
Script for general Path management etc
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PATHS:
    data_path: Path = './data'
    preprocessed_data_path: Path = 'preprocessed_data_81Y'
    output_path: Path = Path('./output')
