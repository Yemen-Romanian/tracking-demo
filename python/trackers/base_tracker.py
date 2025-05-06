from abc import abstractmethod
from typing import Tuple, List

class BaseTracker:
    @abstractmethod
    def init(self, frame, bbox):
        raise NotImplementedError("Init is not implemented")
    
    @abstractmethod
    def update(self, frame) -> Tuple[float, List]:
        raise NotImplementedError("Update is not implemented")