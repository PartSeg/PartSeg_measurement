# pylint: disable=no-self-use
import inspect

from sympy import symbols

from PartSeg_measurement._measurement_wrap import (
    MeasurementCache,
    MeasurementFunctionWrap,
)


class TestMeasurementFunctionWrap:
    def test_wraps(self):
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = MeasurementFunctionWrap(
            measurement_func=func, name=func, units="m"
        )
        assert func.__doc__ == wrap.__doc__
        assert inspect.signature(func) == inspect.signature(wrap)

    def test_filtering_parameters(self):
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = MeasurementFunctionWrap(
            measurement_func=func, name=func, units="m"
        )
        assert wrap(a=1, b=2, c=3) == 3

    def test_filtering_parameters_with_kwargs(self):
        c_in = False

        def func(a: int, b: float, **kwargs) -> float:
            """Sample docstring"""
            nonlocal c_in
            c_in = "c" in kwargs
            return a + b

        wrap = MeasurementFunctionWrap(
            measurement_func=func, name=func, units="m"
        )
        assert wrap(a=1, b=2, c=3) == 3
        assert c_in
        assert wrap(a=1, b=2, d=3) == 3
        assert not c_in

    def test_rename_kwargs(self):
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = MeasurementFunctionWrap(
            measurement_func=func,
            name=func,
            units="m",
            rename_kwargs={"a": "x"},
        ).rename_parameter("b", "y")
        assert wrap(x=1, y=2) == 3
        sig = inspect.signature(wrap)
        assert sig.parameters["x"].name == "x"
        assert sig.parameters["x"].annotation == int
        assert sig.parameters["y"].name == "y"
        assert sig.parameters["y"].annotation == float


class TestMeasurementCombinationWrap:
    def test_operations_on_wraps_div(self):
        def func1(a: int, b: float) -> float:
            return a + b

        def func2(a: int, b: float) -> float:
            return a - b

        wrap1 = MeasurementFunctionWrap(
            measurement_func=func1, name="func1", units="m"
        )
        wrap2 = MeasurementFunctionWrap(
            measurement_func=func2, name="func2", units="m"
        )
        divided = wrap1 / wrap2
        assert str(divided) == "func1 / func2"
        assert divided._units == 1

    def test_operations_on_wraps_mul(self):
        def func1(a: int, b: float) -> float:
            return a + b

        def func2(a: int, b: float) -> float:
            return a - b

        wrap1 = MeasurementFunctionWrap(
            measurement_func=func1, name="func1", units="m"
        )
        wrap2 = MeasurementFunctionWrap(
            measurement_func=func2, name="func2", units="m"
        )
        mul = wrap1 * wrap2
        assert str(mul) == "func1 * func2"
        assert mul._units == symbols("m") ** 2


class TestMeasurementCache:
    def test_cache_is_empty(self):
        called = 0

        def _called(a: int, b: float) -> float:
            nonlocal called
            called += 1
            return a + b

        cache = MeasurementCache()
        assert cache.calculate(_called, a=1, b=2) == 3
        assert called == 1
        assert cache.calculate(_called, a=1, b=2) == 3
        assert called == 1
        assert cache.calculate(_called, a=1, b=3) == 4
        assert called == 2
        assert cache.calculate(_called, a=1, b=2) == 3
        assert called == 2
