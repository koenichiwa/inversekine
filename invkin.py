from __future__ import annotations

from tkinter import Tk, Canvas
from typing import Optional, Union, Tuple, Generator, Iterable, Sequence
from time import sleep
from math import sin, cos, acos, radians, degrees
from numpy import subtract, linalg, dot, cross, append, add
from functools import reduce
from itertools import islice, cycle, count, chain

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
        self.parent = parent
        self.limit = min(limit, 180)
        self.__angle = 0
        self.set_angle(angle)

    def __str__(self):
        string = ""
        for segment, depth in zip(chain(self.get_arm(), [None]), count(1)):
            if segment is not None:
                string += "(Angle: {}, Angle limit: {}, Length {}: Position: {}, Parent: \n"\
                    .format(self.__angle, self.limit, self.length, self.calc_endpoint())
                string += "\t" * depth
            else:
                string += "None" + (")" * (depth-1))
        return string

    def calc_endpoint(self) -> Tuple[float, float]:
        def endpoint_to_parent(segment: ArmSegment):
            actual_angle = segment.calc_actual_angle()
            dx = segment.length * sin(radians(actual_angle))
            dy = segment.length * cos(radians(actual_angle))
            return dx, dy
        return reduce(add, map(endpoint_to_parent, self.get_arm()))

    def calc_actual_angle(self) -> float:
        return sum(segment.get_angle() for segment in self.get_arm()) % 360

    def set_angle(self, new_angle: float):
        new_angle = new_angle % 360
        if new_angle < 180:
            self.__angle = min(new_angle, self.limit)
        else:
            self.__angle = max(new_angle, 360 - self.limit)

    def get_angle(self) -> float:
        return self.__angle

    def get_arm(self) -> Generator[ArmSegment]:
        yield self
        if self.parent is not None:
            yield from self.parent.get_arm()


def damping(angle: float, damping_scale: float) -> float:
    abs_angle = abs(angle)
    if abs_angle > 90 and damping_scale > 1:
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
        arm: Sequence[ArmSegment],
        damping_scale: float,
        max_steps: int,
        max_error: float
) -> Generator[None]:
    for segment in islice(cycle(arm), max_steps):
        if linalg.norm(subtract(arm[0].calc_endpoint(), goal)) < max_error:
            break

        parent_base = (0, 0) if segment.parent is None else segment.parent.calc_endpoint()
        endpoint_a = subtract(arm[0].calc_endpoint(), parent_base)
        endpoint_g = subtract(goal, parent_base)

        vec_a = endpoint_a / linalg.norm(endpoint_a)
        vec_g = endpoint_g / linalg.norm(endpoint_g)

        angle_diff = degrees(acos(dot(vec_a, vec_g)))

        sign = -1 if cross(append(vec_a, 0.0), append(vec_g, 0.0))[2] > 0 else 1

        segment.set_angle((segment.get_angle() + sign * damping(angle_diff, damping_scale)))
        print("Current segment:\n", segment,
              "\nAngle difference:", angle_diff, "(Damped:", damping(angle_diff, damping_scale), ")"
              "\n\"Hand\" position", arm[0].calc_endpoint(), "Goal position", goal)
        yield


# -------- GUI --------
class ArmCanvas(Canvas):

    def __init__(self, master, width: int, height: int, scale: float):
        super().__init__(master, width=width, height=height)
        self.width = width
        self.height = height
        self.draw_scale = scale
        self.line_lib = {}

    def run(
            self,
            arm: Iterable[ArmSegment],
            goal: Tuple[float, float],
            damping_scale: float,
            step_time: float,
            max_error: float,
            max_steps: int
    ):
        arm = list(arm)
        self.__add_arm(arm)
        self.__draw_goal(goal)
        super().update()
        sleep(step_time)
        for _ in inverse_kine(goal, arm, damping_scale, max_steps, max_error):
            self.__update_arm(arm)
            super().update()
            sleep(step_time)

    def __add_arm(self, arm: Iterable[ArmSegment]):
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

    def __update_arm(self, arm: Iterable[ArmSegment]):
        for segment in arm:
            parent_pos, segment_pos = self.__get_line_pos(segment)
            super().coords(self.line_lib[segment], parent_pos[0], parent_pos[1], segment_pos[0], segment_pos[1])

    def __to_canvas_pos(
            self,
            pos: Union[Tuple[float, float], float],
            ypos: Optional[float] = None
    ) -> Tuple[float, float]:
        if type(ypos) == float and type(pos) == float:
            pos = (pos, ypos)
        return (self.width / 2) + pos[0] * self.draw_scale, (self.height / 2) - pos[1] * self.draw_scale

    def __get_line_pos(self, segment: ArmSegment):
        parent_endpoint = segment.parent.calc_endpoint() if segment.parent is not None else (0, 0)
        parent_pos = self.__to_canvas_pos(parent_endpoint)
        segment_pos = self.__to_canvas_pos(segment.calc_endpoint())
        return parent_pos, segment_pos


# -------- MAIN --------
def main():
    arm = []
    segment_count = int(input("How many segments? "))
    total_length = 0
    current_segment = None
    for i in range(segment_count):
        print("Segment {}:".format(i))
        length = float(input("Segment length? "))
        total_length += length
        angle = float(input("Segment angle? "))
        limit = float(input("Limit in angle from 0 degrees? (Max 180) "))

        current_segment = ArmSegment(length, angle, current_segment, limit)
        arm = [current_segment] + arm

    x = float(input("Goal x? "))
    y = float(input("Goal y? "))
    if linalg.norm([x, y]) > total_length:
        print("Goal unreachable (but we'll try)")

    damping_scale = float(input("How much damping? (1 or less will means that damping won't be used) "))

    # Can prompt for other variables here (but opted to save them as constants)

    tk = Tk()
    tk.title("Inverse Kinematica")

    arm_canvas = ArmCanvas(tk, WIDTH, HEIGHT, SCALE)
    arm_canvas.pack()
    arm_canvas.run(arm, (x, y), damping_scale, STEP_TIME, MAX_ERROR, MAX_STEPS)

    tk.mainloop()


if __name__ == '__main__':
    main()
