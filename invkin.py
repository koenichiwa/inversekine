import math
import numpy as np
from typing import Optional


MAX_ERROR = 1e-3


class ArmSegment(object):
    pass


class ArmSegment:
    def __init__(self, length: float, angle: float, parent: ArmSegment):
        self.length = length
        self.angle = angle
        self.parent = parent

    def calc_endpoint(self) -> (float, float):
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
            return self.angle
        else:
            return self.angle + self.parent.calc_actual_angle()


def inverse_kine(goal: (float, float), last_arm: ArmSegment, current_arm: Optional[ArmSegment] = None):
    if current_arm is None:
        current_arm = last_arm
    if not all(i < MAX_ERROR for i in (abs(x - y) for x, y in zip(last_arm.calc_endpoint(), goal))):
        parent_base = (0, 0) if current_arm.parent is None else current_arm.parent.calc_endpoint()

        # endpoints relative to parent base
        endpoint_a = np.subtract(current_arm.calc_endpoint(), parent_base)
        endpoint_g = np.subtract(goal, parent_base)

        # normalized vectors
        vec_a = endpoint_a / np.linalg.norm(endpoint_a)
        vec_g = endpoint_g / np.linalg.norm(endpoint_g)

        rd = np.dot(vec_a, vec_g)
        dr = np.cross(vec_a, vec_g)

        # angles

        angle_a = math.degrees(math.atan2(vec_a[1], vec_a[0])) # ik weet  dat atan2 niet hoort, maar wat dan wel?
        angle_g = math.degrees(math.atan2(vec_g[1], vec_g[0]))
        print(current_arm.angle, angle_g, last_arm.calc_endpoint(), goal)
        current_arm.angle = -angle_g
        current_arm.angle = current_arm.angle % 360
        inverse_kine(goal, last_arm, current_arm.parent)


if __name__ == '__main__':
    a = ArmSegment(10, 45, None)
    b = ArmSegment(10, 45, a)
    c = ArmSegment(10, 45, b)

    inverse_kine((10.071067811865476, 7.071067811865476), b)

    print(b.calc_endpoint())
