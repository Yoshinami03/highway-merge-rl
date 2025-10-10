import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from core.types import VehicleState, ControlInput
from physics.simple import Simple1DPhysics

@dataclass
class RoadSpec:
    length: float
    merge_start: float
    merge_end: float
    main_lane_id: int
    merge_lane_id: int

class MergeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, seed: int = 0, road: RoadSpec = RoadSpec(1000.0, 300.0, 600.0, 1, 0), dt: float = 0.1, spawn_rate: float = 0.05, render_mode: str = None):
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
        
        # Rendering setup
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.agent_patch = None
        self.traffic_patches = []
        self.road_patches = []
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
    
    def render(self):
        if self.render_mode is None:
            return
        
        if self.fig is None:
            self._setup_render()
        
        self._update_render()
        
        if self.render_mode == "human":
            plt.pause(0.01)
        elif self.render_mode == "rgb_array":
            # Convert figure to RGB array
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return buf
    
    def _setup_render(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.ax.set_xlim(0, self.road.length)
        self.ax.set_ylim(-1, 3)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('Position (m)', fontsize=12)
        self.ax.set_ylabel('Lane', fontsize=12)
        self.ax.set_title('Highway Merge Environment - HighwayEnv Style', fontsize=14, fontweight='bold')
        
        # HighwayEnv style: remove axis ticks and labels for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Draw road
        self._draw_road()
        
        # Initialize vehicle patches (HighwayEnv style)
        self.agent_patch = patches.Rectangle((0, 0), 4, 0.8, 
                                           facecolor='#FF6B6B', edgecolor='#2C3E50', linewidth=2)
        self.ax.add_patch(self.agent_patch)
        
        # Add lane change indicator
        self.lane_change_indicator = None
        
        plt.ion()
        plt.show()
    
    def _draw_road(self):
        # HighwayEnv style: Two lanes side by side (horizontal layout)
        # Main lane (right side)
        main_lane_y = 1.5
        main_lane_width = 0.8
        main_lane = patches.Rectangle((0, main_lane_y), self.road.length, main_lane_width,
                                    facecolor='#34495E', edgecolor='#2C3E50', alpha=0.8, linewidth=2)
        self.ax.add_patch(main_lane)
        
        # Merge lane (left side)
        merge_lane_y = 0.5
        merge_lane_width = 0.8
        merge_lane = patches.Rectangle((0, merge_lane_y), self.road.length, merge_lane_width,
                                     facecolor='#34495E', edgecolor='#2C3E50', alpha=0.8, linewidth=2)
        self.ax.add_patch(merge_lane)
        
        # Lane boundaries (HighwayEnv style - white dashed lines)
        self.ax.axhline(y=main_lane_y, color='white', linewidth=2, alpha=0.8, linestyle='--')
        self.ax.axhline(y=merge_lane_y, color='white', linewidth=2, alpha=0.8, linestyle='--')
        self.ax.axhline(y=main_lane_y + main_lane_width, color='white', linewidth=2, alpha=0.8, linestyle='--')
        self.ax.axhline(y=merge_lane_y + merge_lane_width, color='white', linewidth=2, alpha=0.8, linestyle='--')
        
        # Merge zone (HighwayEnv style - subtle highlight)
        merge_zone_main = patches.Rectangle((self.road.merge_start, main_lane_y), 
                                          self.road.merge_end - self.road.merge_start, main_lane_width,
                                          facecolor='#F39C12', alpha=0.3, edgecolor='#E67E22', linewidth=2)
        self.ax.add_patch(merge_zone_main)
        
        merge_zone_merge = patches.Rectangle((self.road.merge_start, merge_lane_y), 
                                           self.road.merge_end - self.road.merge_start, merge_lane_width,
                                           facecolor='#F39C12', alpha=0.3, edgecolor='#E67E22', linewidth=2)
        self.ax.add_patch(merge_zone_merge)
        
        # Merge arrow (HighwayEnv style)
        merge_arrow_x = (self.road.merge_start + self.road.merge_end) / 2
        self.ax.annotate('', xy=(merge_arrow_x, main_lane_y), xytext=(merge_arrow_x, merge_lane_y + merge_lane_width),
                        arrowprops=dict(arrowstyle='->', color='#E67E22', lw=3))
        
        # Lane labels (HighwayEnv style - minimal)
        self.ax.text(50, main_lane_y + main_lane_width/2, 'Main Lane', 
                    ha='left', va='center', fontsize=10, fontweight='bold', color='white')
        self.ax.text(50, merge_lane_y + merge_lane_width/2, 'Merge Lane', 
                    ha='left', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Distance markers (HighwayEnv style - subtle)
        for x in range(0, int(self.road.length) + 1, 200):
            self.ax.axvline(x, color='white', linestyle=':', alpha=0.4, linewidth=1)
            self.ax.text(x, -0.2, f'{x}m', ha='center', va='top', fontsize=8, color='white', alpha=0.7)
        
        # Merge zone indicators
        self._draw_lane_change_indicators()
    
    def _draw_lane_change_indicators(self):
        """車線変更可能エリアの視覚的インジケーターを描画 (HighwayEnv style)"""
        # Merge zone boundaries (HighwayEnv style - subtle)
        self.ax.axvline(self.road.merge_start, color='#E67E22', linestyle='-', linewidth=2, alpha=0.6)
        self.ax.axvline(self.road.merge_end, color='#E67E22', linestyle='-', linewidth=2, alpha=0.6)
        
        # Merge zone labels (HighwayEnv style - minimal)
        self.ax.text(self.road.merge_start, 2.5, 'Merge Start', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='#E67E22', alpha=0.8)
        self.ax.text(self.road.merge_end, 2.5, 'Merge End', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='#E67E22', alpha=0.8)
    
    def _update_render(self):
        # Update agent position (HighwayEnv style)
        if self.agent_patch is not None:
            self.agent_patch.set_x(self.agent.x)
            if self.agent.lane == self.road.merge_lane_id:
                # 合流車線にいる場合（下側）
                self.agent_patch.set_y(0.5)
                self.agent_patch.set_facecolor('#FF6B6B')  # HighwayEnv red
                self.agent_patch.set_edgecolor('#2C3E50')
                self.agent_patch.set_linewidth(2)
            else:
                # 本線にいる場合（上側）
                self.agent_patch.set_y(1.5)
                self.agent_patch.set_facecolor('#E74C3C')  # Darker red for main lane
                self.agent_patch.set_edgecolor('#2C3E50')
                self.agent_patch.set_linewidth(2)
        
        # Clear and redraw traffic (HighwayEnv style)
        for patch in self.traffic_patches:
            patch.remove()
        self.traffic_patches.clear()
        
        # Draw traffic vehicles (HighwayEnv style - blue vehicles)
        for vehicle in self.traffic:
            if vehicle.lane == self.road.main_lane_id:
                traffic_patch = patches.Rectangle((vehicle.x, 1.5), 4, 0.8,
                                                facecolor='#3498DB', edgecolor='#2C3E50', alpha=0.9, linewidth=2)
                self.ax.add_patch(traffic_patch)
                self.traffic_patches.append(traffic_patch)
        
        # Status display (HighwayEnv style - minimal)
        status_text = f"Agent: {self.agent.x:.0f}m, {self.agent.v:.1f}m/s"
        if self.agent.lane == self.road.merge_lane_id:
            status_text += " (Merge)"
            lane_color = '#FF6B6B'
        else:
            status_text += " (Main)"
            lane_color = '#E74C3C'
        
        # Update status text (HighwayEnv style - top right corner)
        if hasattr(self, 'status_text'):
            self.status_text.remove()
        self.status_text = self.ax.text(self.road.length - 50, 2.5, status_text, fontsize=10, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=lane_color, linewidth=1))
        
        # Lane change indicator update
        self._update_lane_change_indicator()
        
        # Update display
        self.fig.canvas.draw()
    
    def _update_lane_change_indicator(self):
        """車線変更可能状態のインジケーターを更新 (HighwayEnv style)"""
        # 既存のインジケーターを削除
        if self.lane_change_indicator is not None:
            self.lane_change_indicator.remove()
            self.lane_change_indicator = None
        
        # 合流ゾーン内で合流車線にいる場合のみ表示
        if (self._can_merge() and 
            self.agent.lane == self.road.merge_lane_id and 
            self.agent.x >= self.road.merge_start and 
            self.agent.x <= self.road.merge_end):
            
            # HighwayEnv style indicator (minimal)
            indicator_text = "MERGE"
            self.lane_change_indicator = self.ax.text(
                self.agent.x + 15, 0.1, indicator_text, 
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='#E67E22', alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='#E67E22', linewidth=1)
            )
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
