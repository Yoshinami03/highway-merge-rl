from abc import ABC, abstractmethod
from typing import Dict, Any
from .types import VehicleState, ControlInput

class PhysicsModel(ABC):
    @abstractmethod
    def step(self, state: VehicleState, u: ControlInput, dt: float) -> VehicleState: ...

class ControlAdapter(ABC):
    @abstractmethod
    def act(self, obs): ...

class EnvBridge(ABC):
    @abstractmethod
    def notify_punish(self, kind: str, magnitude: float): ...
    @abstractmethod
    def notify_reward(self, kind: str, magnitude: float): ...
    @abstractmethod
    def read_state(self) -> Dict[str, Any]: ...
