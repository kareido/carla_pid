import sys
from collections import deque
import numpy as np
import math

sys.path.append('carla910/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg')
import carla

class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=30)

    def pid_control(self, target_speed, current_speed, enable_brake=False):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        if enable_brake:
            throttle_min_clip = -1.0
        else:
            throttle_min_clip = 0.0

        return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt),
            throttle_min_clip, 1.0)

class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """
    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def pid_control(self, target_transform, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param target_transform: target waypoint's transform (waypoint.transform)
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([target_transform.location.x -
                          v_begin.x, target_transform.location.y -
                          v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
            (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
            self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)


class PIDController:
    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        self.longitudinal = PIDLongitudinalController(K_P, K_D, K_I, dt)
        self.lateral = PIDLateralController(K_P, K_D, K_I, dt)

    def pid_control(self, target_transform, vehicle_transform, target_speed, current_speed, enable_brake=False):
        """ Retrun [steer, throttle] pair
        """
        steer = self.lateral.pid_control(target_transform, vehicle_transform)
        throttle = self.longitudinal.pid_control(target_speed, current_speed, enable_brake)
        return [steer, throttle]


if __name__ == '__main__':
    kargs_dict = {
        'K_P': 0.1,
        'K_D': 0.0005,
        'K_I': 0.4,
        'dt': 1 / 10,
    }
    pid_controller = PIDController(**kargs_dict)
    curr_loc = carla.Location(0, 0, 0)
    curr_rot = carla.Rotation(0, 0, 0)
    curr_speed = 15.
    curr_tsfm = carla.Transform(curr_loc, curr_rot)

    target_loc = carla.Location(2, 5, 0)
    target_rot = carla.Rotation(0, 0, 0)
    target_speed = 20.
    target_tsfm = carla.Transform(target_loc, target_rot)

    steer, throttle = pid_controller.pid_control(target_tsfm, curr_tsfm, target_speed, curr_speed, enable_brake=True)

    print(steer, throttle)
