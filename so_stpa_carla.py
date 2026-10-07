from __future__ import annotations
import argparse
import glob
import math
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np


def append_carla_path() -> None:
    directories = [
        os.environ.get('CARLA_PYTHONAPI', ''),
        '/home/carlauser/CarlaUE4/PythonAPI/carla',
        '/home/carlauser/CarlaUE4/PythonAPI',
        '/root/CarlaUE4/PythonAPI/carla',
        '/root/CarlaUE4/PythonAPI',
        './CARLA_0.9.14/PythonAPI/carla',
        './CARLA_0.9.14/PythonAPI',
    ]
    for directory in directories:
        if directory and os.path.exists(directory) and directory not in sys.path:
            sys.path.append(directory)
    patterns = [
        os.environ.get('CARLA_EGG', ''),
        '/home/carlauser/CarlaUE4/PythonAPI/carla/dist/carla-*linux-x86_64.egg',
        '/root/CarlaUE4/PythonAPI/carla/dist/carla-*linux-x86_64.egg',
        './CARLA_0.9.14/PythonAPI/carla/dist/carla-*linux-x86_64.egg',
        '/home/carlauser/CarlaUE4/PythonAPI/carla/dist/carla-*.whl',
        '/root/CarlaUE4/PythonAPI/carla/dist/carla-*.whl',
        './CARLA_0.9.14/PythonAPI/carla/dist/carla-*.whl',
    ]
    for pattern in patterns:
        if not pattern:
            continue
        for match in glob.glob(pattern):
            if match not in sys.path:
                sys.path.append(match)


append_carla_path()

try:
    import carla
except Exception as exc:
    raise RuntimeError('Cannot import the CARLA Python API. Configure CARLA_PYTHONAPI, CARLA_EGG, or PYTHONPATH for CARLA 0.9.14.') from exc

try:
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
except Exception as exc:
    raise RuntimeError('stable-baselines3 is required.') from exc

try:
    SB3_MAJOR = int(stable_baselines3.__version__.split('.')[0])
except Exception:
    SB3_MAJOR = 2

if SB3_MAJOR >= 2:
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception as exc:
        raise RuntimeError('SB3 2.x requires gymnasium.') from exc
    USE_GYMNASIUM_API = True
else:
    try:
        import gym
        from gym import spaces
    except Exception as exc:
        raise RuntimeError('SB3 1.x requires gym.') from exc
    USE_GYMNASIUM_API = False

try:
    import torch
except Exception as exc:
    raise RuntimeError('PyTorch is required by Stable-Baselines3.') from exc


@dataclass
class CarlaConfig:
    host: str = '127.0.0.1'
    port: int = 2000
    timeout_seconds: float = 60.0
    fixed_delta_seconds: float = 0.05
    no_rendering_mode: bool = True
    route_traffic_light_id: Optional[int] = None


@dataclass
class ScenarioConfig:
    weather_set: List[str] = field(default_factory=lambda: ['clear', 'cloudy', 'foggy', 'rainy'])
    speed_limit_fallback_kmh: float = 60.0
    speed_limit_override_kmh: Optional[float] = 60.0
    vehicle_filter: str = 'vehicle.tesla.model3'
    max_episode_seconds: float = 30.0
    route_step_m: float = 1.0
    initial_speed_mps: float = 12.0
    max_target_speed_ratio: float = 1.1
    collision_terminates: bool = True
    perception_degradation_enabled: bool = False
    ego_initial_distance_to_stop_m: float = 72.5
    lead_initial_distance_to_stop_m: float = 18.0
    lead_braking_duration_s: float = 3.0
    red_duration_min_s: float = 6.0
    red_duration_max_s: float = 10.0
    goal_distance_after_stop_m: float = 25.0
    lateral_initial_offset_m: float = 0.3
    initial_yaw_error_deg: float = 2.0
    steering_residual_limit: float = 0.16
    severe_offroute_threshold_m: float = 4.0
    lane_change_corridor_m: float = 420.0
    lane_change_obstacle_spacing_m: float = 30.0
    lane_change_safe_distance_m: float = 10.0
    lane_change_fraction_rate_per_s: float = 0.8
    lane_change_signal_threshold: float = 0.0
    lane_change_goal_margin_m: float = 2.0
    lane_change_offcorridor_threshold_m: float = 2.5


@dataclass
class RewardConfig:
    omega_time: float = 0.1
    omega_speed: float = 0.05
    collision_penalty: float = -10.0
    comfort_good: float = 1.0
    comfort_bad: float = -1.0
    comfort_jerk_threshold_mps3: float = 0.6
    red_light_penalty: float = -5.0
    dangerous_following_penalty: float = -2.0
    lane_keeping_reward: float = 1.0
    adverse_weather_speed_penalty: float = -1.5
    perception_failure_penalty: float = -1.0
    obstacle_proximity_penalty: float = -2.0
    unsafe_target_gap_penalty: float = -2.0
    missing_turn_signal_penalty: float = -1.0


@dataclass
class PPOConfig:
    total_timesteps: int = 100000
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ent_coef: float = 0.0
    policy_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])


@dataclass
class ExperimentConfig:
    training_seeds: List[int] = field(default_factory=lambda: [0])
    output_dir: str = 'models'


@dataclass
class Config:
    carla: CarlaConfig = field(default_factory=CarlaConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def speed_mps(actor: Any) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)


def yaw_to_unit(yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(yaw_deg)
    return math.cos(yaw), math.sin(yaw)


def right_unit(yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(yaw_deg)
    return -math.sin(yaw), math.cos(yaw)


def parse_int_spec(text: str) -> List[int]:
    values: List[int] = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk[1:]:
            left, right = chunk.split('-', 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(chunk))
    if not values:
        raise ValueError('No integer values were parsed')
    return values


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_weather_speed_kmh(speed_limit_kmh: float) -> float:
    if speed_limit_kmh <= 50.0:
        return speed_limit_kmh
    if speed_limit_kmh <= 80.0:
        return 0.8 * speed_limit_kmh
    return 65.0


def weather_parameters(name: str) -> Any:
    if name == 'clear':
        return carla.WeatherParameters.ClearNoon
    if name == 'cloudy':
        return carla.WeatherParameters.CloudyNoon
    if name == 'rainy':
        return carla.WeatherParameters.HardRainNoon
    if name == 'foggy':
        weather = carla.WeatherParameters.CloudyNoon
        weather.fog_density = 75.0
        weather.fog_distance = 20.0
        weather.fog_falloff = 1.0
        return weather
    raise ValueError('Unsupported weather profile: %s' % name)


@dataclass(frozen=True)
class RouteProjection:
    progress_m: float
    lateral_offset_m: float
    route_yaw_rad: float
    distance_to_route_m: float


class RoutePolyline:
    def __init__(self, xyz: np.ndarray):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2:
            raise ValueError('A route requires at least two 3-D points')
        self.xyz = xyz.astype(np.float64)
        delta = self.xyz[1:, :2] - self.xyz[:-1, :2]
        segment_length = np.linalg.norm(delta, axis=1)
        keep = segment_length > 1e-5
        if not np.all(keep):
            filtered = [self.xyz[0]]
            for i in range(1, len(self.xyz)):
                if np.linalg.norm(self.xyz[i, :2] - filtered[-1][:2]) > 0.05:
                    filtered.append(self.xyz[i])
            self.xyz = np.asarray(filtered, dtype=np.float64)
            delta = self.xyz[1:, :2] - self.xyz[:-1, :2]
            segment_length = np.linalg.norm(delta, axis=1)
        if len(segment_length) == 0 or np.any(segment_length < 1e-5):
            raise ValueError('Invalid route geometry')
        self.segment_vector = delta
        self.segment_length = segment_length
        self.cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
        self.length_m = float(self.cumulative[-1])

    @classmethod
    def from_waypoints(cls, waypoints: Sequence[Any]) -> 'RoutePolyline':
        xyz = np.array([[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp in waypoints], dtype=np.float64)
        return cls(xyz)

    def project(self, location: Any) -> RouteProjection:
        point = np.array([location.x, location.y], dtype=np.float64)
        start = self.xyz[:-1, :2]
        segment = self.segment_vector
        relative = point - start
        t = np.sum(relative * segment, axis=1) / np.maximum(self.segment_length ** 2, 1e-9)
        t = np.clip(t, 0.0, 1.0)
        closest = start + t[:, None] * segment
        difference = point - closest
        squared_distance = np.sum(difference ** 2, axis=1)
        index = int(np.argmin(squared_distance))
        tangent = segment[index] / self.segment_length[index]
        cross_z = tangent[0] * difference[index, 1] - tangent[1] * difference[index, 0]
        magnitude = math.sqrt(float(squared_distance[index]))
        signed_offset = math.copysign(magnitude, cross_z if abs(cross_z) > 1e-12 else 1.0)
        progress = float(self.cumulative[index] + t[index] * self.segment_length[index])
        yaw = math.atan2(tangent[1], tangent[0])
        return RouteProjection(progress, signed_offset, yaw, magnitude)

    def sample(self, progress_m: float) -> Tuple[np.ndarray, float]:
        distance = float(np.clip(progress_m, 0.0, self.length_m))
        index = int(np.searchsorted(self.cumulative, distance, side='right') - 1)
        index = min(max(index, 0), len(self.segment_length) - 1)
        local = (distance - self.cumulative[index]) / self.segment_length[index]
        xyz = self.xyz[index] + local * (self.xyz[index + 1] - self.xyz[index])
        yaw = math.atan2(self.segment_vector[index, 1], self.segment_vector[index, 0])
        return xyz, yaw


def select_straight(current: Any, candidates: Iterable[Any]) -> Optional[Any]:
    items = [candidate for candidate in candidates if candidate is not None and candidate.lane_type == carla.LaneType.Driving]
    if not items:
        return None
    current_yaw = current.transform.rotation.yaw
    items.sort(key=lambda wp: (angle_difference_deg(wp.transform.rotation.yaw, current_yaw), abs(wp.lane_id - current.lane_id)))
    return items[0]


def walk_waypoint(start_waypoint: Any, distance_m: float, forward: bool, step_m: float) -> Optional[Any]:
    current = start_waypoint
    travelled = 0.0
    while travelled + 1e-6 < distance_m:
        step = min(step_m, distance_m - travelled)
        candidates = current.next(step) if forward else current.previous(step)
        next_waypoint = select_straight(current, candidates)
        if next_waypoint is None:
            return None
        current = next_waypoint
        travelled += step
    return current


def build_forward_waypoints(start_waypoint: Any, length_m: float, step_m: float) -> List[Any]:
    points = [start_waypoint]
    current = start_waypoint
    travelled = 0.0
    while travelled + 1e-6 < length_m:
        step = min(step_m, length_m - travelled)
        next_waypoint = select_straight(current, current.next(step))
        if next_waypoint is None:
            break
        if angle_difference_deg(next_waypoint.transform.rotation.yaw, start_waypoint.transform.rotation.yaw) > 20.0:
            break
        points.append(next_waypoint)
        current = next_waypoint
        travelled += step
    return points


@dataclass
class SignalizedRoute:
    route: RoutePolyline
    traffic_light: Any
    stop_progress_m: float
    ego_start_progress_m: float
    lead_start_progress_m: float
    goal_progress_m: float


def discover_signalized_routes(world: Any, scenario: ScenarioConfig, requested_light_id: Optional[int]) -> List[SignalizedRoute]:
    lights = list(world.get_actors().filter('traffic.traffic_light*'))
    lights.sort(key=lambda actor: actor.id)
    if requested_light_id is not None:
        lights = [light for light in lights if int(light.id) == int(requested_light_id)]
    required_back = scenario.ego_initial_distance_to_stop_m
    required_forward = scenario.goal_distance_after_stop_m + 10.0
    candidates: List[Tuple[Any, ...]] = []
    for light in lights:
        try:
            stop_waypoints = list(light.get_stop_waypoints())
        except Exception:
            continue
        stop_waypoints.sort(key=lambda wp: (wp.road_id, wp.section_id, wp.lane_id, wp.s))
        for stop_waypoint in stop_waypoints:
            start_waypoint = walk_waypoint(stop_waypoint, required_back, False, scenario.route_step_m)
            if start_waypoint is None:
                continue
            waypoints = build_forward_waypoints(start_waypoint, required_back + required_forward, scenario.route_step_m)
            if len(waypoints) < int(0.75 * (required_back + required_forward) / scenario.route_step_m):
                continue
            try:
                route = RoutePolyline.from_waypoints(waypoints)
            except ValueError:
                continue
            stop_projection = route.project(stop_waypoint.transform.location)
            if stop_projection.progress_m < required_back * 0.8:
                continue
            candidates.append(((light.id, stop_waypoint.road_id, stop_waypoint.section_id, stop_waypoint.lane_id, stop_waypoint.s), route, light, stop_projection.progress_m))
    if not candidates:
        raise RuntimeError('No suitable signalized route was found in Town03')
    candidates.sort(key=lambda item: item[0])
    routes: List[SignalizedRoute] = []
    for _, route, light, stop_progress in candidates:
        ego_progress = max(0.0, stop_progress - scenario.ego_initial_distance_to_stop_m)
        lead_progress = max(ego_progress + 10.0, stop_progress - scenario.lead_initial_distance_to_stop_m)
        goal_progress = min(route.length_m - 1.0, stop_progress + scenario.goal_distance_after_stop_m)
        routes.append(SignalizedRoute(route, light, float(stop_progress), float(ego_progress), float(lead_progress), float(goal_progress)))
    return routes


@dataclass
class LaneCorridor:
    lane0: RoutePolyline
    lane1: RoutePolyline
    lane1_is_left: bool
    length_m: float


def adjacent_driving_lane(waypoint: Any) -> Optional[Tuple[Any, bool]]:
    left = waypoint.get_left_lane()
    right = waypoint.get_right_lane()
    candidates: List[Tuple[Any, bool]] = []
    if left is not None and left.lane_type == carla.LaneType.Driving and angle_difference_deg(left.transform.rotation.yaw, waypoint.transform.rotation.yaw) < 30.0:
        candidates.append((left, True))
    if right is not None and right.lane_type == carla.LaneType.Driving and angle_difference_deg(right.transform.rotation.yaw, waypoint.transform.rotation.yaw) < 30.0:
        candidates.append((right, False))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (0 if item[1] else 1, abs(item[0].lane_id - waypoint.lane_id)))
    return candidates[0]


def discover_lane_corridor(world: Any, scenario: ScenarioConfig) -> LaneCorridor:
    road_map = world.get_map()
    spawn_points = list(road_map.get_spawn_points())
    spawn_points.sort(key=lambda transform: (round(transform.location.x, 2), round(transform.location.y, 2), round(transform.rotation.yaw, 2)))
    required = scenario.lane_change_corridor_m + 10.0
    for transform in spawn_points:
        wp0 = road_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp0 is None:
            continue
        adjacent = adjacent_driving_lane(wp0)
        if adjacent is None:
            continue
        wp1, lane1_is_left = adjacent
        route0_wps = build_forward_waypoints(wp0, required, scenario.route_step_m)
        route1_wps = build_forward_waypoints(wp1, required, scenario.route_step_m)
        if len(route0_wps) < int(0.95 * required / scenario.route_step_m) or len(route1_wps) < int(0.95 * required / scenario.route_step_m):
            continue
        try:
            route0 = RoutePolyline.from_waypoints(route0_wps)
            route1 = RoutePolyline.from_waypoints(route1_wps)
        except ValueError:
            continue
        length_m = min(route0.length_m, route1.length_m, scenario.lane_change_corridor_m)
        if length_m < scenario.lane_change_corridor_m - 5.0:
            continue
        p0_start, _ = route0.sample(0.0)
        p1_start, _ = route1.sample(0.0)
        p0_end, _ = route0.sample(length_m)
        p1_end, _ = route1.sample(length_m)
        start_gap = float(np.linalg.norm(p0_start[:2] - p1_start[:2]))
        end_gap = float(np.linalg.norm(p0_end[:2] - p1_end[:2]))
        if not (2.5 <= start_gap <= 5.5 and 2.5 <= end_gap <= 5.5):
            continue
        return LaneCorridor(route0, route1, lane1_is_left, float(length_m))
    raise RuntimeError('No suitable two-lane 420 m corridor was found in Town04')


@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    integral_limit: float
    previous_error: float = 0.0
    integral: float = 0.0

    def reset(self) -> None:
        self.previous_error = 0.0
        self.integral = 0.0

    def step(self, error: float, dt: float) -> float:
        self.integral = clamp(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        derivative = (error - self.previous_error) / max(dt, 1e-6)
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class VehicleController:
    def __init__(self, dt: float):
        self.dt = dt
        self.longitudinal = PIDController(0.35, 0.04, 0.02, 10.0)

    def reset(self) -> None:
        self.longitudinal.reset()

    def control_to_point(self, vehicle: Any, current_speed_mps: float, target_speed_mps: float, target_xyz: np.ndarray) -> Any:
        speed_error = target_speed_mps - current_speed_mps
        longitudinal_command = self.longitudinal.step(speed_error, self.dt)
        if longitudinal_command >= 0.0:
            throttle = clamp(longitudinal_command, 0.0, 0.85)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(-longitudinal_command, 0.0, 1.0)
        transform = vehicle.get_transform()
        dx = float(target_xyz[0] - transform.location.x)
        dy = float(target_xyz[1] - transform.location.y)
        target_heading = math.atan2(dy, dx)
        current_heading = math.radians(transform.rotation.yaw)
        alpha = wrap_angle_rad(target_heading - current_heading)
        distance = max(3.0, math.sqrt(dx * dx + dy * dy))
        steering_angle = math.atan2(2.0 * 2.8 * math.sin(alpha), distance)
        steer = clamp(steering_angle / math.radians(35.0), -1.0, 1.0)
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)

    def control_on_route(self, vehicle: Any, route: RoutePolyline, projection: RouteProjection, current_speed_mps: float, target_speed_mps: float, steering_residual: float = 0.0) -> Any:
        lookahead = clamp(5.0 + 0.55 * current_speed_mps, 5.0, 14.0)
        target_xyz, _ = route.sample(projection.progress_m + lookahead)
        control = self.control_to_point(vehicle, current_speed_mps, target_speed_mps, target_xyz)
        control.steer = clamp(float(control.steer) + steering_residual, -1.0, 1.0)
        return control


LIGHT_RED = 0
LIGHT_YELLOW = 1
LIGHT_GREEN = 2
LIGHT_UNKNOWN = 3


@dataclass(frozen=True)
class PerceptionOutput:
    lead_distance_m: float
    relative_speed_mps: float
    light_state: int
    confidence: float
    failure: bool


WEATHER_DEGRADATION: Dict[str, Dict[str, float]] = {
    'clear': {'dropout': 0.002, 'wrong': 0.002, 'distance_sigma': 0.15, 'fault_scale': 0.25},
    'cloudy': {'dropout': 0.008, 'wrong': 0.006, 'distance_sigma': 0.35, 'fault_scale': 0.55},
    'foggy': {'dropout': 0.05, 'wrong': 0.025, 'distance_sigma': 1.5, 'fault_scale': 1.0},
    'rainy': {'dropout': 0.035, 'wrong': 0.02, 'distance_sigma': 1.0, 'fault_scale': 0.9},
}


class PerceptionDegrader:
    def __init__(self, rng: np.random.Generator, weather: str, red_duration_s: float):
        self.rng = rng
        self.weather = weather
        self.red_duration_s = red_duration_s
        self.parameters = WEATHER_DEGRADATION[weather]
        self.last_distance = 80.0
        self.last_relative_speed = 0.0
        self.signal_fault = self._sample_signal_fault()
        scale = self.parameters['fault_scale']
        self.red_delay_s = float(self.rng.uniform(0.35, 1.2) * scale)
        self.early_green_s = float(self.rng.uniform(0.35, 1.2) * scale)

    def _sample_signal_fault(self) -> str:
        scale = self.parameters['fault_scale']
        p_delayed = 0.1 * scale
        p_missing = 0.08 * scale
        p_early = 0.1 * scale
        draw = float(self.rng.random())
        if draw < p_delayed:
            return 'delayed_red'
        if draw < p_delayed + p_missing:
            return 'missing_red'
        if draw < p_delayed + p_missing + p_early:
            return 'early_green'
        return 'normal'

    def observe(self, simulation_time_s: float, actual_light_state: int, true_lead_distance_m: float, true_relative_speed_mps: float) -> PerceptionOutput:
        failure = False
        confidence = 1.0
        dropout = bool(self.rng.random() < self.parameters['dropout'])
        wrong = bool(self.rng.random() < self.parameters['wrong'])
        if dropout:
            lead_distance = self.last_distance
            relative_speed = self.last_relative_speed
            failure = True
            confidence = 0.0
        else:
            sigma = self.parameters['distance_sigma'] * (3.0 if wrong else 1.0)
            lead_distance = max(0.0, true_lead_distance_m + float(self.rng.normal(0.0, sigma)))
            relative_speed = true_relative_speed_mps + float(self.rng.normal(0.0, 0.25 * sigma))
            self.last_distance = lead_distance
            self.last_relative_speed = relative_speed
            if wrong:
                failure = True
                confidence = 0.25
            else:
                confidence = max(0.45, 1.0 - sigma / 4.0)
        perceived_light = actual_light_state
        if self.signal_fault == 'delayed_red' and actual_light_state == LIGHT_RED and simulation_time_s < self.red_delay_s:
            perceived_light = LIGHT_GREEN
            failure = True
            confidence = min(confidence, 0.25)
        elif self.signal_fault == 'missing_red' and actual_light_state == LIGHT_RED:
            perceived_light = LIGHT_UNKNOWN
            failure = True
            confidence = min(confidence, 0.1)
        elif self.signal_fault == 'early_green' and actual_light_state == LIGHT_RED and 0.0 < self.red_duration_s - simulation_time_s <= self.early_green_s:
            perceived_light = LIGHT_GREEN
            failure = True
            confidence = min(confidence, 0.25)
        return PerceptionOutput(float(lead_distance), float(relative_speed), int(perceived_light), float(confidence), bool(failure))


class CollisionCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0

    def callback(self, _event: Any) -> None:
        with self.lock:
            self.count += 1

    def consume(self) -> int:
        with self.lock:
            value = self.count
            self.count = 0
        return value


class CarlaEnvBase(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config: Config, reward_strategy: str, town: str, seed: Optional[int]):
        super().__init__()
        self.config = config
        self.reward_strategy = reward_strategy
        self.town = town
        self.client = carla.Client(config.carla.host, config.carla.port)
        self.client.set_timeout(config.carla.timeout_seconds)
        self.world = self._load_world()
        self.original_settings = self.world.get_settings()
        self._configure_world()
        self.actor_list: List[Any] = []
        self.ego = None
        self.collision_sensor = None
        self.collision_counter = CollisionCounter()
        self.weather_name = 'clear'
        self.adverse_weather = False
        self.speed_limit_kmh = config.scenario.speed_limit_fallback_kmh
        self.speed_limit_mps = self.speed_limit_kmh / 3.6
        self.safe_weather_speed_mps = safe_weather_speed_kmh(self.speed_limit_kmh) / 3.6
        self.simulation_time_s = 0.0
        self.episode_steps = 0
        self.max_episode_steps = int(round(config.scenario.max_episode_seconds / config.carla.fixed_delta_seconds))
        self._legacy_seed = seed
        if seed is not None and not USE_GYMNASIUM_API:
            self.seed(seed)

    def _load_world(self) -> Any:
        world = self.client.get_world()
        current_name = world.get_map().name.split('/')[-1]
        if current_name != self.town:
            world = self.client.load_world(self.town)
            time.sleep(2.0)
        return world

    def _configure_world(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.carla.fixed_delta_seconds
        settings.no_rendering_mode = self.config.carla.no_rendering_mode
        try:
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
        except Exception:
            pass
        self.world.apply_settings(settings)
        for _ in range(5):
            self.world.tick()

    def seed(self, seed: Optional[int] = None):
        self._legacy_seed = seed
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def _prepare_reset(self, seed: Optional[int]) -> None:
        if USE_GYMNASIUM_API:
            super().reset(seed=seed)
        elif seed is not None:
            self.seed(seed)
        elif not hasattr(self, 'np_random'):
            self.seed(self._legacy_seed)
        self._destroy_actors()
        self.weather_name = str(self.np_random.choice(self.config.scenario.weather_set))
        self.adverse_weather = self.weather_name in {'foggy', 'rainy'}
        self.world.set_weather(weather_parameters(self.weather_name))
        self.simulation_time_s = 0.0
        self.episode_steps = 0
        self.collision_counter.consume()

    def _destroy_actors(self) -> None:
        for actor in reversed(self.actor_list):
            try:
                if actor is not None and actor.is_alive:
                    if hasattr(actor, 'stop'):
                        actor.stop()
                    actor.destroy()
            except Exception:
                pass
        self.actor_list.clear()
        self.ego = None
        self.collision_sensor = None
        for _ in range(2):
            try:
                self.world.tick()
            except Exception:
                break

    def _spawn_vehicle(self, transform: Any, role_name: str, color: str) -> Any:
        blueprints = list(self.world.get_blueprint_library().filter(self.config.scenario.vehicle_filter))
        if not blueprints:
            blueprints = list(self.world.get_blueprint_library().filter('vehicle.*'))
        if not blueprints:
            raise RuntimeError('No vehicle blueprint is available')
        blueprint = blueprints[0]
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', role_name)
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', color)
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            for dz in (0.8, 1.2, 1.8):
                shifted = carla.Transform(carla.Location(transform.location.x, transform.location.y, transform.location.z + dz), transform.rotation)
                actor = self.world.try_spawn_actor(blueprint, shifted)
                if actor is not None:
                    break
        if actor is None:
            raise RuntimeError('Failed to spawn vehicle')
        actor.set_autopilot(False)
        self.actor_list.append(actor)
        return actor

    def _spawn_collision_sensor(self) -> None:
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self.ego)
        sensor.listen(self.collision_counter.callback)
        self.collision_sensor = sensor
        self.actor_list.append(sensor)

    def _sample_speed_limit(self) -> None:
        override = self.config.scenario.speed_limit_override_kmh
        if override is not None:
            self.speed_limit_kmh = float(override)
        else:
            reported = float(self.ego.get_speed_limit()) if self.ego is not None else 0.0
            self.speed_limit_kmh = reported if reported > 1.0 else self.config.scenario.speed_limit_fallback_kmh
        self.speed_limit_mps = self.speed_limit_kmh / 3.6
        self.safe_weather_speed_mps = safe_weather_speed_kmh(self.speed_limit_kmh) / 3.6

    def _common_reward(self, collision_event: bool, elapsed_time_s: float, speed_value_mps: float, jerk_mps3: float) -> float:
        cfg = self.config.reward
        collision_reward = cfg.collision_penalty if collision_event else 0.0
        time_reward = -elapsed_time_s
        speed_reward = -abs(speed_value_mps - self.speed_limit_mps)
        efficiency_reward = cfg.omega_time * time_reward + cfg.omega_speed * speed_reward
        comfort_reward = cfg.comfort_good if abs(jerk_mps3) < cfg.comfort_jerk_threshold_mps3 else cfg.comfort_bad
        return float(collision_reward + efficiency_reward + comfort_reward)

    def _so_strategy(self) -> bool:
        return self.reward_strategy.upper().replace('_', '-') == 'SO-STPA'

    def close(self) -> None:
        self._destroy_actors()
        try:
            self.world.apply_settings(self.original_settings)
        except Exception:
            pass


class CarFollowingEnv(CarlaEnvBase):
    def __init__(self, config: Config, reward_strategy: str, seed: Optional[int] = None):
        super().__init__(config, reward_strategy, 'Town03', seed)
        self.route_infos = discover_signalized_routes(self.world, config.scenario, config.carla.route_traffic_light_id)
        self.route_info = self.route_infos[0]
        self.route_rng = np.random.default_rng(seed)
        try:
            self.world.freeze_all_traffic_lights(True)
        except Exception:
            pass
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float32)
        self.ego_controller = VehicleController(config.carla.fixed_delta_seconds)
        self.lead_controller = VehicleController(config.carla.fixed_delta_seconds)
        self.lead = None
        self.perception: Optional[PerceptionDegrader] = None
        self.red_duration_s = 8.0
        self.current_actual_light_state = LIGHT_RED
        self.previous_speed_mps = config.scenario.initial_speed_mps
        self.filtered_acceleration_mps2 = 0.0
        self.filtered_jerk_mps3 = 0.0
        self.previous_progress_m = self.route_info.ego_start_progress_m
        self.red_violation_recorded = False
        self.previous_target_fraction = 0.5

    def _transform_at(self, progress_m: float, lateral_m: float = 0.0, yaw_error_deg: float = 0.0) -> Any:
        xyz, yaw = self.route_info.route.sample(progress_m)
        right_x, right_y = right_unit(math.degrees(yaw))
        location = carla.Location(x=float(xyz[0] + lateral_m * right_x), y=float(xyz[1] + lateral_m * right_y), z=float(xyz[2] + 0.45))
        rotation = carla.Rotation(yaw=math.degrees(yaw) + yaw_error_deg)
        return carla.Transform(location, rotation)

    def _set_initial_velocity(self, vehicle: Any, progress_m: float, speed: float) -> None:
        _, yaw = self.route_info.route.sample(progress_m)
        forward_x, forward_y = yaw_to_unit(math.degrees(yaw))
        vehicle.set_target_velocity(carla.Vector3D(x=forward_x * speed, y=forward_y * speed, z=0.0))

    def _set_traffic_light_state(self, state: int) -> None:
        light = self.route_info.traffic_light
        desired = carla.TrafficLightState.Red if state == LIGHT_RED else carla.TrafficLightState.Green
        try:
            for grouped_light in light.get_group_traffic_lights():
                grouped_light.set_state(carla.TrafficLightState.Red)
            light.set_state(desired)
        except Exception:
            light.set_state(desired)
        self.current_actual_light_state = state

    def _lead_target_speed(self) -> float:
        elapsed = self.simulation_time_s
        duration = self.config.scenario.lead_braking_duration_s
        initial = self.config.scenario.initial_speed_mps
        if elapsed <= duration:
            return initial - 0.5 * initial * (1.0 - math.cos(math.pi * elapsed / duration))
        if elapsed < self.red_duration_s:
            return 0.0
        return self.speed_limit_mps

    def _reset_internal(self, seed: Optional[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._prepare_reset(seed)
        if seed is not None:
            self.route_rng = np.random.default_rng(seed)
        self.route_info = self.route_infos[int(self.route_rng.integers(0, len(self.route_infos)))]
        self.ego_controller.reset()
        self.lead_controller.reset()
        self.red_duration_s = float(self.np_random.uniform(self.config.scenario.red_duration_min_s, self.config.scenario.red_duration_max_s))
        self.current_actual_light_state = LIGHT_GREEN
        self._set_traffic_light_state(LIGHT_RED)
        lateral_offset = float(self.np_random.uniform(-self.config.scenario.lateral_initial_offset_m, self.config.scenario.lateral_initial_offset_m))
        yaw_error = float(self.np_random.uniform(-self.config.scenario.initial_yaw_error_deg, self.config.scenario.initial_yaw_error_deg))
        ego_progress = self.route_info.ego_start_progress_m
        lead_progress = self.route_info.lead_start_progress_m
        self.ego = self._spawn_vehicle(self._transform_at(ego_progress, lateral_offset, yaw_error), 'hero', '0,0,255')
        self.lead = self._spawn_vehicle(self._transform_at(lead_progress), 'lead', '255,128,0')
        self._spawn_collision_sensor()
        self._set_initial_velocity(self.ego, ego_progress, self.config.scenario.initial_speed_mps)
        self._set_initial_velocity(self.lead, lead_progress, self.config.scenario.initial_speed_mps)
        self.world.tick()
        self._sample_speed_limit()
        perception_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self.perception = PerceptionDegrader(np.random.default_rng(perception_seed), self.weather_name, self.red_duration_s)
        self.previous_speed_mps = speed_mps(self.ego)
        self.filtered_acceleration_mps2 = 0.0
        self.filtered_jerk_mps3 = 0.0
        self.previous_progress_m = self.route_info.route.project(self.ego.get_location()).progress_m
        self.red_violation_recorded = False
        self.previous_target_fraction = 0.5
        observation, state = self._observation_and_state()
        return observation, {'state': state}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        del options
        observation, info = self._reset_internal(seed)
        if USE_GYMNASIUM_API:
            return observation, info
        return observation

    def _perceived_state(self, front_distance: float, relative_speed: float) -> PerceptionOutput:
        if not self.config.scenario.perception_degradation_enabled:
            return PerceptionOutput(front_distance, relative_speed, self.current_actual_light_state, 1.0, False)
        if self.perception is None:
            raise RuntimeError('PerceptionDegrader is not initialized')
        return self.perception.observe(self.simulation_time_s, self.current_actual_light_state, front_distance, relative_speed)

    def _observation_and_state(self) -> Tuple[np.ndarray, Dict[str, float]]:
        ego_transform = self.ego.get_transform()
        lead_transform = self.lead.get_transform()
        ego_projection = self.route_info.route.project(ego_transform.location)
        lead_projection = self.route_info.route.project(lead_transform.location)
        ego_speed = speed_mps(self.ego)
        lead_speed = speed_mps(self.lead)
        dt = self.config.carla.fixed_delta_seconds
        raw_acceleration = (ego_speed - self.previous_speed_mps) / dt
        acceleration = 0.12 * raw_acceleration + 0.88 * self.filtered_acceleration_mps2
        raw_jerk = (acceleration - self.filtered_acceleration_mps2) / dt
        jerk = 0.08 * raw_jerk + 0.92 * self.filtered_jerk_mps3
        self.filtered_acceleration_mps2 = float(acceleration)
        self.filtered_jerk_mps3 = float(jerk)
        front_distance = max(0.0, lead_projection.progress_m - ego_projection.progress_m - 4.5)
        relative_speed = ego_speed - lead_speed
        perceived = self._perceived_state(front_distance, relative_speed)
        yaw_error = wrap_angle_rad(math.radians(ego_transform.rotation.yaw) - ego_projection.route_yaw_rad)
        distance_to_stop = self.route_info.stop_progress_m - ego_projection.progress_m
        light_one_hot = np.zeros(4, dtype=np.float32)
        light_one_hot[int(np.clip(perceived.light_state, 0, 3))] = 1.0
        weather_one_hot = np.zeros(4, dtype=np.float32)
        weather_one_hot[self.config.scenario.weather_set.index(self.weather_name)] = 1.0
        observation = np.array([
            clamp(ego_speed / max(self.speed_limit_mps, 1e-3), 0.0, 2.0) - 1.0,
            clamp(acceleration / 6.0, -1.0, 1.0),
            clamp(jerk / 12.0, -1.0, 1.0),
            clamp(ego_projection.lateral_offset_m / 3.0, -1.0, 1.0),
            clamp(yaw_error / math.pi, -1.0, 1.0),
            clamp(perceived.lead_distance_m / 80.0, 0.0, 1.0),
            clamp(perceived.relative_speed_mps / 20.0, -1.0, 1.0),
            *light_one_hot.tolist(),
            clamp(distance_to_stop / 100.0, -1.0, 1.0),
            clamp(ego_projection.progress_m / max(self.route_info.goal_progress_m, 1.0), 0.0, 1.0),
            clamp(self.speed_limit_mps / 30.0, 0.0, 1.0),
            clamp(self.safe_weather_speed_mps / 30.0, 0.0, 1.0),
            *weather_one_hot.tolist(),
            1.0 if self.adverse_weather else 0.0,
            clamp(perceived.confidence, 0.0, 1.0),
            clamp(self.previous_target_fraction, 0.0, 1.0),
            clamp(self.simulation_time_s / max(self.config.scenario.max_episode_seconds, 1e-6), 0.0, 1.0),
        ], dtype=np.float32)
        state = {
            'ego_speed_mps': float(ego_speed),
            'lead_speed_mps': float(lead_speed),
            'accel_mps2': float(acceleration),
            'jerk_mps3': float(jerk),
            'ego_progress_m': float(ego_projection.progress_m),
            'lateral_offset_m': float(ego_projection.lateral_offset_m),
            'front_distance_m': float(front_distance),
            'perception_failure': float(perceived.failure),
        }
        return np.clip(observation, -1.0, 1.0).astype(np.float32), state

    def _step_internal(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        dt = self.config.carla.fixed_delta_seconds
        self._set_traffic_light_state(LIGHT_RED if self.simulation_time_s < self.red_duration_s else LIGHT_GREEN)
        ego_projection = self.route_info.route.project(self.ego.get_location())
        lead_projection = self.route_info.route.project(self.lead.get_location())
        ego_speed = speed_mps(self.ego)
        lead_speed = speed_mps(self.lead)
        lead_control = self.lead_controller.control_on_route(self.lead, self.route_info.route, lead_projection, lead_speed, self._lead_target_speed())
        self.lead.apply_control(lead_control)
        target_fraction = 0.5 * (float(action[0]) + 1.0) * self.config.scenario.max_target_speed_ratio
        target_speed = target_fraction * self.speed_limit_mps
        steering_residual = float(action[1]) * self.config.scenario.steering_residual_limit
        ego_control = self.ego_controller.control_on_route(self.ego, self.route_info.route, ego_projection, ego_speed, target_speed, steering_residual)
        self.ego.apply_control(ego_control)
        self.world.tick()
        self.simulation_time_s += dt
        self.episode_steps += 1
        self._set_traffic_light_state(LIGHT_RED if self.simulation_time_s < self.red_duration_s else LIGHT_GREEN)
        observation, state = self._observation_and_state()
        collision_event = self.collision_counter.consume() > 0
        crossed_stop_line = self.previous_progress_m < self.route_info.stop_progress_m <= state['ego_progress_m']
        red_violation_event = bool(crossed_stop_line and self.current_actual_light_state == LIGHT_RED and not self.red_violation_recorded)
        if red_violation_event:
            self.red_violation_recorded = True
        safe_following_distance_m = self.speed_limit_kmh
        dangerous_following = 0.0 < state['front_distance_m'] < safe_following_distance_m
        reward = self._common_reward(collision_event, self.simulation_time_s, state['ego_speed_mps'], state['jerk_mps3'])
        if red_violation_event:
            reward += self.config.reward.red_light_penalty
        if dangerous_following:
            reward += self.config.reward.dangerous_following_penalty
        if self._so_strategy():
            if abs(state['lateral_offset_m']) < 1.5:
                reward += self.config.reward.lane_keeping_reward
            if self.adverse_weather and state['ego_speed_mps'] > self.safe_weather_speed_mps:
                reward += self.config.reward.adverse_weather_speed_penalty
            if bool(state['perception_failure']):
                reward += self.config.reward.perception_failure_penalty
        reached_goal = state['ego_progress_m'] >= self.route_info.goal_progress_m
        severe_offroute = abs(state['lateral_offset_m']) >= self.config.scenario.severe_offroute_threshold_m
        terminated = bool(reached_goal or (collision_event and self.config.scenario.collision_terminates))
        truncated = bool(self.episode_steps >= self.max_episode_steps or severe_offroute)
        self.previous_speed_mps = state['ego_speed_mps']
        self.previous_progress_m = state['ego_progress_m']
        self.previous_target_fraction = clamp(target_fraction / max(self.config.scenario.max_target_speed_ratio, 1e-6), 0.0, 1.0)
        info = {'state': state}
        return observation, float(reward), terminated, truncated, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._step_internal(action)
        if USE_GYMNASIUM_API:
            return observation, reward, terminated, truncated, info
        done = bool(terminated or truncated)
        if truncated and not terminated:
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def close(self) -> None:
        try:
            self.world.freeze_all_traffic_lights(False)
        except Exception:
            pass
        super().close()


class LaneChangingEnv(CarlaEnvBase):
    def __init__(self, config: Config, reward_strategy: str, seed: Optional[int] = None):
        super().__init__(config, reward_strategy, 'Town04', seed)
        self.corridor = discover_lane_corridor(self.world, config.scenario)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(25,), dtype=np.float32)
        self.controller = VehicleController(config.carla.fixed_delta_seconds)
        self.obstacles: List[Any] = []
        self.obstacle_progress: List[float] = []
        self.obstacle_lanes: List[int] = []
        self.previous_speed_mps = config.scenario.initial_speed_mps
        self.filtered_acceleration_mps2 = 0.0
        self.filtered_jerk_mps3 = 0.0
        self.command_lane_fraction = 0.0
        self.previous_signal_active = False

    def _route(self, lane_index: int) -> RoutePolyline:
        return self.corridor.lane0 if lane_index == 0 else self.corridor.lane1

    def _transform_at(self, lane_index: int, progress_m: float, z_offset: float = 0.45) -> Any:
        xyz, yaw = self._route(lane_index).sample(progress_m)
        return carla.Transform(carla.Location(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2] + z_offset)), carla.Rotation(yaw=math.degrees(yaw)))

    def _set_initial_velocity(self, vehicle: Any, lane_index: int, progress_m: float, speed: float) -> None:
        _, yaw = self._route(lane_index).sample(progress_m)
        fx, fy = yaw_to_unit(math.degrees(yaw))
        vehicle.set_target_velocity(carla.Vector3D(x=fx * speed, y=fy * speed, z=0.0))

    def _spawn_obstacles(self) -> None:
        self.obstacles = []
        self.obstacle_progress = []
        self.obstacle_lanes = []
        spacing = self.config.scenario.lane_change_obstacle_spacing_m
        positions = np.arange(0.5 * spacing, self.corridor.length_m - 0.25 * spacing, spacing)
        for index, progress in enumerate(positions):
            lane_index = 1 if index % 2 == 0 else 0
            actor = self._spawn_vehicle(self._transform_at(lane_index, float(progress)), 'obstacle_%d' % index, '255,128,0')
            try:
                actor.set_simulate_physics(False)
            except Exception:
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            self.obstacles.append(actor)
            self.obstacle_progress.append(float(progress))
            self.obstacle_lanes.append(lane_index)

    def _lane_fraction_and_progress(self) -> Tuple[float, float, float, RouteProjection, RouteProjection]:
        location = self.ego.get_location()
        p0 = self.corridor.lane0.project(location)
        p1 = self.corridor.lane1.project(location)
        denom = p0.distance_to_route_m + p1.distance_to_route_m
        if denom < 1e-6:
            fraction = self.command_lane_fraction
        else:
            fraction = clamp(p0.distance_to_route_m / denom, 0.0, 1.0)
        progress = p0.progress_m if p0.distance_to_route_m <= p1.distance_to_route_m else p1.progress_m
        offcorridor = min(p0.distance_to_route_m, p1.distance_to_route_m)
        return float(fraction), float(progress), float(offcorridor), p0, p1

    def _target_xyz(self, progress_m: float, fraction: float, speed_value: float) -> np.ndarray:
        lookahead = clamp(7.0 + 0.55 * speed_value, 7.0, 16.0)
        xyz0, _ = self.corridor.lane0.sample(progress_m + lookahead)
        xyz1, _ = self.corridor.lane1.sample(progress_m + lookahead)
        return (1.0 - fraction) * xyz0 + fraction * xyz1

    def _obstacle_gaps(self, ego_progress: float, target_lane: int) -> Tuple[float, float, float, float, int]:
        front_target = float('inf')
        rear_target = float('inf')
        nearest_distance = float('inf')
        nearest_longitudinal = 0.0
        nearest_lane = -1
        ego_location = self.ego.get_location()
        for actor, progress, lane_index in zip(self.obstacles, self.obstacle_progress, self.obstacle_lanes):
            if actor is None or not actor.is_alive:
                continue
            distance = ego_location.distance(actor.get_location())
            if distance < nearest_distance:
                nearest_distance = float(distance)
                nearest_longitudinal = float(progress - ego_progress)
                nearest_lane = int(lane_index)
            if lane_index == target_lane:
                delta = progress - ego_progress
                if delta >= 0.0:
                    front_target = min(front_target, delta)
                else:
                    rear_target = min(rear_target, -delta)
        return float(front_target), float(rear_target), float(nearest_distance), float(nearest_longitudinal), int(nearest_lane)

    def _set_turn_signal(self, active: bool, target_lane: int, current_lane: int) -> None:
        if not active or target_lane == current_lane:
            state = carla.VehicleLightState.NONE
        else:
            moving_to_lane1 = target_lane == 1
            left = self.corridor.lane1_is_left if moving_to_lane1 else not self.corridor.lane1_is_left
            state = carla.VehicleLightState.LeftBlinker if left else carla.VehicleLightState.RightBlinker
        try:
            self.ego.set_light_state(state)
        except Exception:
            pass

    def _reset_internal(self, seed: Optional[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._prepare_reset(seed)
        self.controller.reset()
        self.command_lane_fraction = 0.0
        self.previous_signal_active = False
        self.ego = self._spawn_vehicle(self._transform_at(0, 0.0), 'hero', '0,0,255')
        self._spawn_obstacles()
        self._spawn_collision_sensor()
        self._set_initial_velocity(self.ego, 0, 0.0, self.config.scenario.initial_speed_mps)
        self.world.tick()
        self._sample_speed_limit()
        self.previous_speed_mps = speed_mps(self.ego)
        self.filtered_acceleration_mps2 = 0.0
        self.filtered_jerk_mps3 = 0.0
        observation, state = self._observation_and_state()
        return observation, {'state': state}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        del options
        observation, info = self._reset_internal(seed)
        if USE_GYMNASIUM_API:
            return observation, info
        return observation

    def _observation_and_state(self) -> Tuple[np.ndarray, Dict[str, float]]:
        ego_speed = speed_mps(self.ego)
        dt = self.config.carla.fixed_delta_seconds
        raw_acceleration = (ego_speed - self.previous_speed_mps) / dt
        acceleration = 0.12 * raw_acceleration + 0.88 * self.filtered_acceleration_mps2
        raw_jerk = (acceleration - self.filtered_acceleration_mps2) / dt
        jerk = 0.08 * raw_jerk + 0.92 * self.filtered_jerk_mps3
        self.filtered_acceleration_mps2 = float(acceleration)
        self.filtered_jerk_mps3 = float(jerk)
        lane_fraction, progress, offcorridor, p0, p1 = self._lane_fraction_and_progress()
        current_lane = 0 if p0.distance_to_route_m <= p1.distance_to_route_m else 1
        target_lane = 1 if self.command_lane_fraction >= 0.5 else 0
        front_target, rear_target, nearest_distance, nearest_longitudinal, nearest_lane = self._obstacle_gaps(progress, target_lane)
        lane0_front, lane0_rear, _, _, _ = self._obstacle_gaps(progress, 0)
        lane1_front, lane1_rear, _, _, _ = self._obstacle_gaps(progress, 1)
        weather_one_hot = np.zeros(4, dtype=np.float32)
        weather_one_hot[self.config.scenario.weather_set.index(self.weather_name)] = 1.0
        finite_cap = 120.0
        observation = np.array([
            clamp(ego_speed / max(self.speed_limit_mps, 1e-3), 0.0, 2.0) - 1.0,
            clamp(acceleration / 6.0, -1.0, 1.0),
            clamp(jerk / 12.0, -1.0, 1.0),
            clamp(progress / max(self.corridor.length_m, 1.0), 0.0, 1.0),
            2.0 * lane_fraction - 1.0,
            2.0 * self.command_lane_fraction - 1.0,
            float(current_lane * 2 - 1),
            float(target_lane * 2 - 1),
            clamp(min(front_target, finite_cap) / finite_cap, 0.0, 1.0),
            clamp(min(rear_target, finite_cap) / finite_cap, 0.0, 1.0),
            clamp(min(nearest_distance, finite_cap) / finite_cap, 0.0, 1.0),
            clamp(nearest_longitudinal / finite_cap, -1.0, 1.0),
            0.0 if nearest_lane < 0 else float(nearest_lane * 2 - 1),
            clamp(min(lane0_front, finite_cap) / finite_cap, 0.0, 1.0),
            clamp(min(lane0_rear, finite_cap) / finite_cap, 0.0, 1.0),
            clamp(min(lane1_front, finite_cap) / finite_cap, 0.0, 1.0),
            clamp(min(lane1_rear, finite_cap) / finite_cap, 0.0, 1.0),
            clamp(self.speed_limit_mps / 30.0, 0.0, 1.0),
            clamp(self.safe_weather_speed_mps / 30.0, 0.0, 1.0),
            *weather_one_hot.tolist(),
            1.0 if self.adverse_weather else 0.0,
            clamp(self.simulation_time_s / max(self.config.scenario.max_episode_seconds, 1e-6), 0.0, 1.0),
        ], dtype=np.float32)
        state = {
            'ego_speed_mps': float(ego_speed),
            'accel_mps2': float(acceleration),
            'jerk_mps3': float(jerk),
            'progress_m': float(progress),
            'lane_fraction': float(lane_fraction),
            'command_lane_fraction': float(self.command_lane_fraction),
            'current_lane': float(current_lane),
            'target_lane': float(target_lane),
            'front_target_gap_m': float(front_target),
            'rear_target_gap_m': float(rear_target),
            'nearest_obstacle_distance_m': float(nearest_distance),
            'offcorridor_m': float(offcorridor),
        }
        return np.clip(observation, -1.0, 1.0).astype(np.float32), state

    def _step_internal(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)
        dt = self.config.carla.fixed_delta_seconds
        lane_fraction, progress, _, p0, p1 = self._lane_fraction_and_progress()
        current_lane = 0 if p0.distance_to_route_m <= p1.distance_to_route_m else 1
        requested_fraction = 0.5 * (float(action[1]) + 1.0)
        max_delta = self.config.scenario.lane_change_fraction_rate_per_s * dt
        self.command_lane_fraction += clamp(requested_fraction - self.command_lane_fraction, -max_delta, max_delta)
        self.command_lane_fraction = clamp(self.command_lane_fraction, 0.0, 1.0)
        target_lane = 1 if self.command_lane_fraction >= 0.5 else 0
        lane_change_execution = bool(abs(self.command_lane_fraction - lane_fraction) > 0.12 or 0.12 < lane_fraction < 0.88)
        signal_active = bool(float(action[2]) > self.config.scenario.lane_change_signal_threshold and lane_change_execution)
        self._set_turn_signal(signal_active, target_lane, current_lane)
        target_speed_fraction = 0.5 * (float(action[0]) + 1.0) * self.config.scenario.max_target_speed_ratio
        target_speed = target_speed_fraction * self.speed_limit_mps
        target_xyz = self._target_xyz(progress, self.command_lane_fraction, speed_mps(self.ego))
        control = self.controller.control_to_point(self.ego, speed_mps(self.ego), target_speed, target_xyz)
        self.ego.apply_control(control)
        self.world.tick()
        self.simulation_time_s += dt
        self.episode_steps += 1
        observation, state = self._observation_and_state()
        collision_event = self.collision_counter.consume() > 0
        target_gap = min(state['front_target_gap_m'], state['rear_target_gap_m'])
        obstacle_proximity = state['nearest_obstacle_distance_m'] < self.config.scenario.lane_change_safe_distance_m
        unsafe_target_gap = lane_change_execution and target_gap < self.config.scenario.lane_change_safe_distance_m
        reward = self._common_reward(collision_event, self.simulation_time_s, state['ego_speed_mps'], state['jerk_mps3'])
        if obstacle_proximity:
            reward += self.config.reward.obstacle_proximity_penalty
        if self._so_strategy():
            if self.adverse_weather and state['ego_speed_mps'] > self.safe_weather_speed_mps:
                reward += self.config.reward.adverse_weather_speed_penalty
            if unsafe_target_gap:
                reward += self.config.reward.unsafe_target_gap_penalty
            if lane_change_execution and not signal_active:
                reward += self.config.reward.missing_turn_signal_penalty
        reached_goal = state['progress_m'] >= self.corridor.length_m - self.config.scenario.lane_change_goal_margin_m
        offcorridor = state['offcorridor_m'] > self.config.scenario.lane_change_offcorridor_threshold_m
        terminated = bool(reached_goal or (collision_event and self.config.scenario.collision_terminates))
        truncated = bool(self.episode_steps >= self.max_episode_steps or offcorridor)
        self.previous_speed_mps = state['ego_speed_mps']
        self.previous_signal_active = signal_active
        info = {'state': state}
        return observation, float(reward), terminated, truncated, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._step_internal(action)
        if USE_GYMNASIUM_API:
            return observation, reward, terminated, truncated, info
        done = bool(terminated or truncated)
        if truncated and not terminated:
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info


def strategy_folder(strategy: str) -> str:
    return strategy.replace('-', '_')


def scenario_folder(scenario_name: str) -> str:
    return scenario_name.replace('-', '_')


def make_environment(config: Config, scenario_name: str, strategy: str, seed: int) -> gym.Env:
    if scenario_name == 'car-following':
        return CarFollowingEnv(config, strategy, seed)
    if scenario_name == 'lane-changing':
        return LaneChangingEnv(config, strategy, seed)
    raise ValueError('Unknown scenario: %s' % scenario_name)


def train_one(config: Config, scenario_name: str, strategy: str, seed: int, timesteps: int, device: str) -> Path:
    set_all_seeds(seed)
    output_root = Path(config.experiment.output_dir)
    model_dir = output_root / scenario_folder(scenario_name) / strategy_folder(strategy) / ('seed_%d' % seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    environment = make_environment(config, scenario_name, strategy, seed)
    if SB3_MAJOR >= 2:
        network_architecture: Any = dict(pi=list(config.ppo.policy_hidden_sizes), vf=list(config.ppo.policy_hidden_sizes))
    else:
        network_architecture = [dict(pi=list(config.ppo.policy_hidden_sizes), vf=list(config.ppo.policy_hidden_sizes))]
    policy_kwargs = {'activation_fn': torch.nn.Tanh, 'net_arch': network_architecture}
    model = PPO(
        'MlpPolicy',
        environment,
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        ent_coef=config.ppo.ent_coef,
        seed=seed,
        verbose=0,
        policy_kwargs=policy_kwargs,
        device=device,
    )
    try:
        model.learn(total_timesteps=timesteps)
        save_path = model_dir / 'final_model'
        model.save(str(save_path))
        return save_path.with_suffix('.zip')
    finally:
        environment.close()


def run_training(config: Config, scenarios: Sequence[str], strategies: Sequence[str], seeds: Sequence[int], timesteps: int, device: str) -> None:
    for scenario_name in scenarios:
        for strategy in strategies:
            for seed in seeds:
                train_one(config, scenario_name, strategy, int(seed), timesteps, device)


def build_config_from_args(args: argparse.Namespace) -> Config:
    config = Config()
    config.carla.host = args.host
    config.carla.port = args.port
    config.carla.timeout_seconds = args.timeout
    config.carla.no_rendering_mode = not args.render
    config.carla.route_traffic_light_id = args.route_light_id
    config.scenario.speed_limit_override_kmh = args.speed_limit_kmh
    config.scenario.perception_degradation_enabled = args.perception_degradation
    config.reward.omega_time = args.omega_time
    config.reward.omega_speed = args.omega_speed
    config.experiment.output_dir = args.output_dir
    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['car-following', 'lane-changing', 'all'], default='all')
    parser.add_argument('--strategy', choices=['STPA', 'SO-STPA', 'all'], default='all')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--route-light-id', type=int, default=None)
    parser.add_argument('--speed-limit-kmh', type=float, default=60.0)
    parser.add_argument('--perception-degradation', action='store_true')
    parser.add_argument('--omega-time', type=float, default=0.1)
    parser.add_argument('--omega-speed', type=float, default=0.05)
    parser.add_argument('--train-seeds', default='0')
    parser.add_argument('--train-steps', type=int, default=100000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', default='models')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = build_config_from_args(args)
    training_seeds = parse_int_spec(args.train_seeds)
    scenarios = ['car-following', 'lane-changing'] if args.scenario == 'all' else [args.scenario]
    strategies = ['STPA', 'SO-STPA'] if args.strategy == 'all' else [args.strategy]
    run_training(config, scenarios, strategies, training_seeds, args.train_steps, args.device)


if __name__ == '__main__':
    main()
