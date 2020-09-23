from __future__ import annotations

from tkinter import Tk, Canvas
from typing import Optional, Union, Tuple, Generator, List
from time import sleep
from math import sin, cos, acos, radians, degrees, sqrt  # sqrt is only used in assertion proving linalg.norm is useful
from numpy import subtract, linalg, dot, cross, append


SCALE = 10
WIDTH = 900
HEIGHT = 600
MAX_STEPS = 100
STEP_TIME = 0.5
MAX_ERROR = 1e-3


# Because I could not find the constraints of this specific robot arm:
# https://www.trossenrobotics.com/p/phantomx-ax-12-reactor-robot-arm.aspx
# I made a program where you can input your own robot arm.

# First the program will ask some questions in the console,
# after which a window will be opened to show how the arms find the goal.
# You can set the amount of arm segments and, starting at the base, the length, angle and angle_limit (symmetric from 0)
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
        dx = self.length * sin(radians(actual_angle))
        dy = self.length * cos(radians(actual_angle))
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


def damping(angle: float, damping_scale: float) -> float:
    if damping_scale < 1:
        return angle
    abs_angle = abs(angle)
    if abs_angle > 90:
        return angle / damping_scale
    elif abs_angle > 45 and damping_scale > 2:
        return angle / (damping_scale / 2)
    elif abs_angle > 22.5 and damping_scale > 4:
        return angle / (damping_scale / 4)
    elif abs_angle > 12.5 and damping_scale > 8:
        return angle / (damping_scale / 8)
    else:
        return angle


def inverse_kine(
        goal: Tuple[float, float],
        last_arm: ArmSegment,
        damping_scale: float,
        max_steps: int,
        max_error: float,
        _current_arm: Optional[ArmSegment] = None,
        _step_count: int = 0
) -> Generator[List[ArmSegment]]:
    if _current_arm is None:
        _current_arm = last_arm
    if _step_count < max_steps and not all(
            i < max_error for i in (abs(x - y) for x, y in zip(last_arm.calc_endpoint(), goal))):
        parent_base = (0, 0) if _current_arm.parent is None else _current_arm.parent.calc_endpoint()

        # endpoints relative to parent base
        endpoint_a = subtract(last_arm.calc_endpoint(), parent_base)
        endpoint_g = subtract(goal, parent_base)

        # normalized vectors
        vec_a = endpoint_a / linalg.norm(endpoint_a)
        vec_g = endpoint_g / linalg.norm(endpoint_g)
        # NB: when a vector is given to np.linalg.norm (and no other arguments are passed) it will return the L2 norm,
        # or Euclidean length. Dividing the fields in the vector by this value will give you the normalized vector
        # It's generally equivalent to the pythagorean formula sqrt(x**2 + y**2):
        assert linalg.norm(endpoint_a) == sqrt(endpoint_a[0] ** 2 + endpoint_a[1] ** 2)

        # angles
        angle_diff = degrees(acos(dot(vec_a, vec_g)))
        print("Step:", MAX_STEPS - _step_count, "Current angle:", _current_arm.angle,
              "Angle difference:", angle_diff, "(Damped:", damping(angle_diff, damping_scale), ")")

        sign = 1 if cross(append(vec_a, 0.0), append(vec_g, 0.0))[2] > 0 else -1

        # rotate
        _current_arm.set_angle((_current_arm.angle - sign * damping(angle_diff, damping_scale)) % 360)
        yield last_arm.get_arm()
        yield from inverse_kine(goal, last_arm, damping_scale, max_steps, max_error, _current_arm.parent,
                                _step_count + 1)


# -------- GUI --------
class ArmCanvas(Canvas):

    def __init__(self, master, width: int, height: int, scale: float):
        super().__init__(master, width=width, height=height)
        self.draw_scale = scale
        self.master.title("Inverse Kinematica")
        self.line_lib = {}

    def run(
            self,
            last_segment: ArmSegment,
            goal: Tuple[float, float],
            damping_scale: float,
            step_time: float,
            max_error: float,
            max_steps: int
    ):
        self.__add_arm(last_segment.get_arm())
        self.__draw_goal(goal)
        super().update()
        sleep(step_time)
        for arm in inverse_kine(goal, last_segment, damping_scale, max_steps, max_error):
            self.__update_arm(arm)
            print(arm[0].calc_endpoint())
            sleep(step_time)
            super().update()

    def __add_arm(self, arm: List[ArmSegment]):
        for segment in arm:
            parent_pos, segment_pos = self.__get_line_pos(segment)
            self.line_lib[segment] = super().create_line(
                parent_pos[0], parent_pos[1],
                segment_pos[0], segment_pos[1],
                fill="black"
            )

    def __draw_goal(self, goal: Tuple[float, float], size: float = 0.5):
        goal_tl = self.__to_canvas_pos(goal[0] - size, goal[1] - size)
        goal_br = self.__to_canvas_pos(goal[0] + size, goal[1] + size)
        super().create_oval(goal_tl[0], goal_tl[1], goal_br[0], goal_br[1], fill="red")

    def __update_arm(self, arm: List[ArmSegment]):
        for segment in arm:
            parent_pos, segment_pos = self.__get_line_pos(segment)
            super().coords(self.line_lib[segment], parent_pos[0], parent_pos[1], segment_pos[0], segment_pos[1])

    def __to_canvas_pos(self, pos: Union[Tuple[float, float], float], ypos: Optional[float] = None
                        ) -> Tuple[float, float]:

        if type(ypos) == float and type(pos) == float:
            pos = (pos, ypos)
        return (self.winfo_width() / 2) + pos[0] * self.draw_scale, (self.winfo_height() / 2) - pos[1] * self.draw_scale

    def __get_line_pos(self, segment: ArmSegment):
        parent_endpoint = segment.parent.calc_endpoint() if segment.parent is not None else (0, 0)
        parent_pos = self.__to_canvas_pos(parent_endpoint)
        segment_pos = self.__to_canvas_pos(segment.calc_endpoint())
        return parent_pos, segment_pos


# -------- MAIN --------
def main():
    segment_count = int(input("How many segments? "))
    total_length = 0
    current_segment = None
    for i in range(segment_count):
        print("Segment {}:".format(i))
        length = float(input("Segment length? "))
        total_length += length
        angle = float(input("Segment angle? "))
        limit = min(float(input("Limit in angle from 0 degrees? (Max 180) ")), 180)

        current_segment = ArmSegment(length, angle % 360, current_segment, limit)

    x = float(input("Goal x? "))
    y = float(input("Goal y? "))
    if linalg.norm([x, y]) > total_length:
        print("Goal unreachable (but we'll try)")

    damping_scale = float(input("How much damping? (1 or less will means that damping won't be used) "))

    # Can prompt for other variables here (but opted to save them as constants)

    root = Tk()
    arm_canvas = ArmCanvas(root, WIDTH, HEIGHT, SCALE)
    arm_canvas.pack()
    arm_canvas.run(current_segment, (x, y), damping_scale, STEP_TIME, MAX_ERROR, MAX_STEPS)

    root.mainloop()


if __name__ == '__main__':
    main()
