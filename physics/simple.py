import numpy as np
from ..core.types import VehicleState, ControlInput
from ..core.interfaces import PhysicsModel

class Simple1DPhysics(PhysicsModel):
    def __init__(self, a_max=3.0, a_min=-5.0, v_min=0.0, v_max=33.0):
        self.a_max = a_max
        self.a_min = a_min
        self.v_min = v_min
        self.v_max = v_max
    def step(self, state: VehicleState, u: ControlInput, dt: float) -> VehicleState:
        a = np.clip(u.throttle, -1.0, 1.0)
        a = self.a_min if a < 0 else self.a_max * a
        v = np.clip(state.v + a * dt, self.v_min, self.v_max)
        x = state.x + v * dt
        lane = state.lane
        yaw = 0.0
        return VehicleState(x=x, lane=lane, v=v, yaw=yaw)
