import numpy as np
from abc import ABC, abstractmethod


class CameraSource(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray: ...  # H×W×3, RGB, uint8

    @abstractmethod
    def stop(self) -> None: ...
