# pylint: disable=no-self-use
import inspect
import json

import nme
from sympy import Rational, symbols

from PartSeg_measurement.measurement_wrap import (
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
            name="func",
            units="m",
            rename_kwargs={"x": "a"},
        ).rename_parameter("b", "y")
        assert wrap(x=1, y=2) == 3
        sig = inspect.signature(wrap)
        assert sig.parameters["x"].name == "x"
        assert sig.parameters["x"].annotation == int
        assert sig.parameters["y"].name == "y"
        assert sig.parameters["y"].annotation == float

    def test_rename_kwargs2(self):
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = MeasurementFunctionWrap(
            measurement_func=func,
            name="func",
            units="m",
            rename_kwargs={"x": "a"},
        ).rename_parameter("x", "y")
        assert wrap(y=1, b=2) == 3
        sig = inspect.signature(wrap)
        assert sig.parameters["y"].name == "y"
        assert sig.parameters["y"].annotation == int
        assert sig.parameters["b"].name == "b"
        assert sig.parameters["b"].annotation == float

    def test_bind(self):
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = MeasurementFunctionWrap(
            measurement_func=func,
            units="m",
            name="func",
        ).bind(a=1)
        assert wrap(b=2) == 3
        sig = inspect.signature(wrap)
        assert "a" not in sig.parameters
        assert "b" in sig.parameters

    def test_serialize(self, tmp_path, clean_register):
        @nme.register_class
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = (
            MeasurementFunctionWrap(
                measurement_func=func,
                units="m",
                name="func",
            )
            .bind(a=1)
            .rename_parameter("b", "y")
        )

        with open(tmp_path / "test.json", "w") as f:
            json.dump(wrap, f, cls=nme.NMEEncoder)

        with open(tmp_path / "test.json") as f:
            wrap2 = json.load(f, object_hook=nme.nme_object_hook)

        assert wrap2.name == "func"
        assert wrap2.units == symbols("m")
        assert wrap2(y=2) == 3


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

    def test_power(self):
        def func1(a: int, b: float) -> float:
            return a + b

        wrap = MeasurementFunctionWrap(
            measurement_func=func1, name="func1", units="m"
        )
        pow1 = wrap**2.0
        assert str(pow1) == "func1 ** 2.0"
        assert pow1.units == symbols("m") ** Rational(2.0)
        pow2 = pow1**3.0
        assert pow2.units == symbols("m") ** Rational(6.0)
        # FIXME assert str(pow2) == "func1 ** 6.0"


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
