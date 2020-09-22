from __future__ import annotations

import time
import numpy as np
from typing import Optional, Union, Tuple, Generator, List
import math
from tkinter import *

MAX_ERROR = 1e-3
SCALE = 10
WIDTH = 900
HEIGHT = 600
MAX_STEPS = 100
STEP_TIME = 0.5


# Because I could not find the constraints of this specific robot arm:
# https://www.trossenrobotics.com/p/phantomx-ax-12-reactor-robot-arm.aspx
# I made a program where you can input your own robot arm.

# First the program will ask some questions in the console,
# after which a window will be opened to show how the arms find the goal.
# You can set the amount of arm segments, the length, angle and angle_limit (symmetric from 0)
# of each of the arm segments. You can also choose the position of the goal yourself.
# If the goal is to far, that is, the distance from 0 is greater than the total arm length, then you'll be notified,
# but the arm will try it's hardest.

# Koen van Wel
# Student no: 500757443


# -------- IK --------
class ArmSegment:
    def __init__(self, length: float, angle: float, parent: Optional[ArmSegment] = None, limit: float = 180):
        self.length = length
        self.angle = angle
        self.parent = parent
        self.limit = limit

    def calc_endpoint(self) -> Tuple[float, float]:
        actual_angle = self.calc_actual_angle()
        dx = self.length * math.sin(math.radians(actual_angle))
        dy = self.length * math.cos(math.radians(actual_angle))
        if self.parent is None:
            return dx, dy
        else:
            (px, py) = self.parent.calc_endpoint()
            return px + dx, py + dy

    def calc_actual_angle(self) -> float:
        if self.parent is None:
            return self.angle % 360
        else:
            return (self.angle + self.parent.calc_actual_angle()) % 360

    def set_angle(self, new_angle: float):
        new_angle = new_angle % 360
        if new_angle < 180:
            self.angle = min(new_angle, self.limit)
        else:
            self.angle = max(new_angle, 360 - self.limit)

    def get_arm(self) -> List[ArmSegment]:
        if self.parent is None:
            return [self]
        else:
            return [self] + self.parent.get_arm()


def damping(angle: float) -> float:
    abs_angle = abs(angle)
    if abs_angle > 90:
        return angle / damping_scale
    elif abs_angle > 45:
        return angle / (damping_scale / 2)
    elif abs_angle > 22.5:
        return angle / (damping_scale / 4)
    elif abs_angle > 12.5:
        return angle / (damping_scale / 8)
    else:
        return angle


def inverse_kine(goal: Tuple[float, float], last_arm: ArmSegment, current_arm: Optional[ArmSegment] = None,
                 step_count: int = MAX_STEPS, use_damping: bool = True) -> Generator[List[ArmSegment]]:
    if current_arm is None:
        current_arm = last_arm
    if step_count > 0 and not all(i < MAX_ERROR for i in (abs(x - y) for x, y in zip(last_arm.calc_endpoint(), goal))):
        parent_base = (0, 0) if current_arm.parent is None else current_arm.parent.calc_endpoint()

        # endpoints relative to parent base
        endpoint_a = np.subtract(last_arm.calc_endpoint(), parent_base)
        endpoint_g = np.subtract(goal, parent_base)

        # normalized vectors
        vec_a = endpoint_a / np.linalg.norm(endpoint_a)
        vec_g = endpoint_g / np.linalg.norm(endpoint_g)
        # NB: when a vector is given to np.linalg.norm (and no other arguments are passed) it will return the L2 norm,
        # or Euclidean length. Dividing the fields in the vector by this value will give you the normalized vector
        # It's generally equivalent to the pythagorean formula sqrt(x**2 + y**2):
        assert np.linalg.norm(endpoint_a) == math.sqrt(endpoint_a[0] ** 2 + endpoint_a[1] ** 2)

        # angles
        angle_diff = math.degrees(math.acos(np.dot(vec_a, vec_g)))
        print("Step:", MAX_STEPS - step_count, "Current angle:", current_arm.angle,
              "Angle difference:", angle_diff, "(Damped:", damping(angle_diff), ")")

        if use_damping:
            angle_diff = damping(angle_diff)
        sign = 1 if np.cross(np.append(vec_a, 0.0), np.append(vec_g, 0.0))[2] > 0 else -1

        # rotate
        current_arm.set_angle((current_arm.angle - sign * damping(angle_diff)) % 360)
        yield last_arm.get_arm()
        yield from inverse_kine(goal, last_arm, current_arm.parent, step_count - 1, use_damping)


# -------- GUI --------
def to_canvas_pos(pos: Union[Tuple[float, float], float], ypos: Optional[float] = None) -> Tuple[float, float]:
    if type(ypos) == float and type(pos) == float:
        pos = (pos, ypos)
    return (WIDTH / 2) + pos[0] * SCALE, (HEIGHT / 2) - pos[1] * SCALE


def get_line_pos(segment: ArmSegment):
    parent_endpoint = segment.parent.calc_endpoint() if segment.parent is not None else (0, 0)
    parent_pos = to_canvas_pos(parent_endpoint)
    segment_pos = to_canvas_pos(segment.calc_endpoint())
    return parent_pos, segment_pos


class ArmCanvas(Canvas):

    def __init__(self, master):
        super().__init__(master, width=WIDTH, height=HEIGHT)
        self.master.title("Inverse Kinematica")
        self.line_lib = {}

    def run(self, last_segment: ArmSegment, goal: Tuple[float, float], step_time: float):
        self.add_arm(last_segment.get_arm())
        self.draw_goal(goal)
        super().update()
        time.sleep(step_time)
        for arm in inverse_kine(goal, last_segment):
            self.update_arm(arm)
            print(arm[0].calc_endpoint())
            time.sleep(step_time)
            super().update()

    def add_arm(self, arm: List[ArmSegment]):
        for segment in arm:
            parent_pos, segment_pos = get_line_pos(segment)
            self.line_lib[segment] = super().create_line(
                parent_pos[0], parent_pos[1],
                segment_pos[0], segment_pos[1],
                fill="black"
            )

    def draw_goal(self, goal: Tuple[float, float], size: float = 0.5):
        goal_tl = to_canvas_pos(goal[0] - size, goal[1] - size)
        goal_br = to_canvas_pos(goal[0] + size, goal[1] + size)
        super().create_oval(goal_tl[0], goal_tl[1], goal_br[0], goal_br[1], fill="red")

    def update_arm(self, arm: List[ArmSegment]):
        for segment in arm:
            parent_pos, segment_pos = get_line_pos(segment)
            super().coords(self.line_lib[segment], parent_pos[0], parent_pos[1], segment_pos[0], segment_pos[1])
            self.line_lib[segment] = super().create_line(
                parent_pos[0], parent_pos[1],
                segment_pos[0], segment_pos[1],
                fill="black"
            )


# -------- MAIN --------
if __name__ == '__main__':
    segment_count = int(input("How many segments? "))
    total_length = 0
    current_segment = None
    for i in range(segment_count):
        print("Segment {}:".format(i))
        length = float(input("Segment length? "))
        total_length += length
        angle = float(input("Segment angle? "))
        limit = float(input("Limit in angle from 0 degrees? (Max 180) "))
        assert limit <= 180
        current_segment = ArmSegment(length, angle % 360, current_segment, limit)

    x = float(input("Goal x? "))
    y = float(input("Goal y? "))

    damping_scale = float(input("How much damping? (1 is off, less than 1 is not advised) "))

    if np.linalg.norm([x, y]) > total_length:
        print("Goal unreachable (but we'll try)")

    root = Tk()
    armcanvas = ArmCanvas(root)
    armcanvas.pack()
    armcanvas.run(current_segment, (x, y), STEP_TIME)

    root.mainloop()
