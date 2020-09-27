from typing import Any
from unittest import TestCase
from invkin import ArmSegment


class TestArmSegment(TestCase):
    # create assertAlmostEqual method that accept tuples
    def assertAlmostEqualTuple(self, first_tuple: tuple, second_tuple: tuple, places: int = 7,
                               msg: Any = None, delta: float = None) -> None:
        for first, second in zip(first_tuple, second_tuple):
            self.assertAlmostEqual(first, second, places, msg, delta)

    def assertEqualTuple(self, first_tuple: tuple, second_tuple: tuple, msg: Any = None) -> None:
        for first, second in zip(first_tuple, second_tuple):
            self.assertEqual(first, second, msg)

    def test_calc_endpoint(self):
        # First arm
        a = ArmSegment(1, 0)
        self.assertEqualTuple((0, 1), a.calc_endpoint())

        # Child arm of same length
        b = ArmSegment(1, 0, a)
        self.assertEqualTuple((0, 2), b.calc_endpoint())
        # Child of child of same length
        c = ArmSegment(1, 0, b)
        self.assertEqualTuple((0, 3), c.calc_endpoint())

        del b, c

        # Child arm of base double lenght
        d = ArmSegment(2, 0, a)
        self.assertEqualTuple((0, 3), d.calc_endpoint())

        del d

        # Child arm with 90 degrees
        e = ArmSegment(1, 90, a)
        self.assertAlmostEqualTuple((1, 1), e.calc_endpoint())

        f = ArmSegment(1, 90, e)
        self.assertAlmostEqualTuple((1, 0), f.calc_endpoint())

        del a, e, f

        g = ArmSegment(0, 0)

        self.assertEqual((0, 0), g.calc_endpoint())

        h = ArmSegment(0, 1)
        self.assertEqual((0, 0), h.calc_endpoint())

    def test_calc_actual_angle(self):
        a = ArmSegment(1, 0)
        self.assertEqual(a.calc_actual_angle(), 0)
        b = ArmSegment(1, 90, a)
        self.assertEqual(b.calc_actual_angle(), 90)
        c = ArmSegment(1, 90, b)
        self.assertEqual(c.calc_actual_angle(), 180)
        d = ArmSegment(1, 45, b)
        self.assertEqual(d.calc_actual_angle(), 135)
        e = ArmSegment(1, 181, c)
        self.assertEqual(e.calc_actual_angle(), 1)
        f = ArmSegment(1, 180.1, c)
        self.assertAlmostEqual(f.calc_actual_angle(), 0.1)

    def test_set_angle(self):
        a = ArmSegment(1, 0)
        self.assertEqual(a.get_angle(), 0)
        a.set_angle(10)
        self.assertEqual(a.get_angle(), 10)
        a.set_angle(370)
        self.assertEqual(a.get_angle(), 10)
        a.set_angle(-350)
        self.assertEqual(a.get_angle(), 10)

    def test_get_arm(self):
        a = ArmSegment(1, 0)
        b = ArmSegment(1, 0, a)
        c = ArmSegment(1, 0, b)
        self.assertListEqual(list(a.get_arm()), [a])
        self.assertListEqual(list(b.get_arm()), [b, a])
        self.assertListEqual(list(c.get_arm()), [c, b, a])


if __name__ == '__main__':
    TestArmSegment().run()
