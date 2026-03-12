from abc import ABC, abstractmethod

import numpy as np

from .types import Pose


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray) -> Pose | None:
        """Returns None if no person detected."""
        ...
