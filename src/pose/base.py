from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .types import Pose


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray) -> Optional[Pose]:
        """Returns None if no person detected."""
        ...
