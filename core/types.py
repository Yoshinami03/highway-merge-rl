from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class VehicleState:
    x: float
    lane: int
    v: float
    yaw: float

@dataclass
class ControlInput:
    throttle: float
    lane_change: int

@dataclass
class StepResult:
    obs: Tuple
    reward: float
    done: bool
    info: Dict
