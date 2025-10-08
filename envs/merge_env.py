import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Tuple, Dict
from ..core.types import VehicleState, ControlInput
from ..physics.simple import Simple1DPhysics

@dataclass
class RoadSpec:
    length: float
    merge_start: float
    merge_end: float
    main_lane_id: int
    merge_lane_id: int

class MergeEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, seed: int = 0, road: RoadSpec = RoadSpec(1000.0, 300.0, 600.0, 1, 0), dt: float = 0.1, spawn_rate: float = 0.05):
        self.np_random = np.random.default_rng(seed)
        self.dt = dt
        self.road = road
        self.physics = Simple1DPhysics()
        self.agent = VehicleState(x=0.0, lane=road.merge_lane_id, v=0.0, yaw=0.0)
        self.traffic: List[VehicleState] = []
        self.spawn_rate = spawn_rate
        self.min_gap = 8.0
        self.honk_gap = 12.0
        self.brake_penalty = 0.01
        self.collision_penalty = 5.0
        self.honk_penalty = 0.05
        self.merge_bonus = 0.5
        self.progress_scale = 0.01
        self.action_space = spaces.MultiDiscrete([3, 2])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self._done = False
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.agent = VehicleState(x=0.0, lane=self.road.merge_lane_id, v=0.0, yaw=0.0)
        self.traffic = []
        self._done = False
        return self._obs(), {}
    def step(self, action: np.ndarray):
        if self._done:
            return self._obs(), 0.0, True, False, {}
        throttle = {-1: -1.0, 0: 0.0, 1: 1.0}[int(action[0]) - 1]
        lane_change = int(action[1])
        u = ControlInput(throttle=throttle, lane_change=lane_change)
        prev_x = self.agent.x
        prev_v = self.agent.v
        self._spawn_traffic()
        if self._can_merge() and u.lane_change == 1 and self.agent.lane == self.road.merge_lane_id:
            if self._safe_gap():
                self.agent = VehicleState(x=self.agent.x, lane=self.road.main_lane_id, v=self.agent.v, yaw=0.0)
                merged = True
            else:
                self._done = True
                r = -self.collision_penalty
                return self._obs(), r, True, False, {"event": "unsafe_merge"}
        else:
            merged = False
        self.agent = self.physics.step(self.agent, u, self.dt)
        self._move_traffic()
        r = 0.0
        r += (self.agent.x - prev_x) * self.progress_scale
        if merged:
            r += self.merge_bonus
        if u.throttle < 0:
            r -= self.brake_penalty
        if self._collision():
            self._done = True
            r -= self.collision_penalty
            return self._obs(), r, True, False, {"event": "collision"}
        if self._honk():
            r -= self.honk_penalty
        if self.agent.x >= self.road.length:
            self._done = True
            return self._obs(), r, True, False, {"event": "finish"}
        return self._obs(), r, False, False, {}
    def _obs(self):
        a = self.agent
        f = self._front_car(a.x)
        b = self._back_car(a.x)
        d_f = (f.x - a.x) if f is not None else 1e6
        v_f = f.v if f is not None else 0.0
        d_b = (a.x - b.x) if b is not None else 1e6
        v_b = b.v if b is not None else 0.0
        in_merge_zone = 1.0 if self._can_merge() and a.lane == self.road.merge_lane_id else 0.0
        at_merge_lane = 1.0 if a.lane == self.road.merge_lane_id else 0.0
        return np.array([a.x, a.v, d_f, v_f, d_b, v_b, in_merge_zone, at_merge_lane], dtype=np.float32)
    def _spawn_traffic(self):
        if self.np_random.random() < self.spawn_rate:
            x0 = self.agent.x + 80.0 + float(self.np_random.uniform(0, 60))
            v0 = float(self.np_random.uniform(12.0, 25.0))
            self.traffic.append(VehicleState(x=x0, lane=self.road.main_lane_id, v=v0, yaw=0.0))
        if len(self.traffic) == 0:
            x0 = 120.0
            v0 = 18.0
            self.traffic.append(VehicleState(x=x0, lane=self.road.main_lane_id, v=v0, yaw=0.0))
    def _move_traffic(self):
        for i in range(len(self.traffic)):
            s = self.traffic[i]
            self.traffic[i] = VehicleState(x=s.x + s.v * self.dt, lane=s.lane, v=s.v, yaw=0.0)
        self.traffic = [c for c in self.traffic if c.x <= self.road.length + 50.0]
        self.traffic.sort(key=lambda z: z.x)
    def _front_car(self, x):
        ahead = [c for c in self.traffic if c.lane == self.road.main_lane_id and c.x >= x]
        return ahead[0] if len(ahead) > 0 else None
    def _back_car(self, x):
        behind = [c for c in self.traffic if c.lane == self.road.main_lane_id and c.x < x]
        return behind[-1] if len(behind) > 0 else None
    def _safe_gap(self):
        f = self._front_car(self.agent.x)
        b = self._back_car(self.agent.x)
        d_f = (f.x - self.agent.x) if f is not None else 1e6
        d_b = (self.agent.x - b.x) if b is not None else 1e6
        return d_f > self.min_gap and d_b > self.min_gap
    def _can_merge(self):
        return self.road.merge_start <= self.agent.x <= self.road.merge_end
    def _collision(self):
        if self.agent.lane != self.road.main_lane_id:
            return False
        for c in self.traffic:
            if c.lane == self.road.main_lane_id and abs(c.x - self.agent.x) < self.min_gap:
                return True
        return False
    def _honk(self):
        b = self._back_car(self.agent.x)
        if b is None:
            return False
        return (self.agent.x - b.x) < self.honk_gap and self.agent.v < 1.0
