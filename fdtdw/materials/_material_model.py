from abc import ABC, abstractmethod
import numpy as np
from typing import List


class MaterialModel(ABC):
    def __init__(self, dx: float, S: float):
        self.dx = float(dx)
        self.S = float(S)

    @property
    @abstractmethod
    def coeff_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_vacuum_params(self, shape: tuple) -> dict:
        pass

    @abstractmethod
    def get_coeffs(self, params: dict) -> dict:
        pass

    @abstractmethod
    def get_params(self, coeffs: dict) -> dict:
        pass
