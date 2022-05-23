import inspect

from sympy import symbols

from PartSeg_measurement._measurement_wrap import MeasurementFunctionWrap


def test_wraps():
    def func(a: int, b: float) -> float:
        """Sample docstring"""
        return a + b

    wrap = MeasurementFunctionWrap(measurement_func=func, name=func, units="m")
    assert func.__doc__ == wrap.__doc__
    assert inspect.signature(func) == inspect.signature(wrap)


def test_filtering_parameters():
    def func(a: int, b: float) -> float:
        """Sample docstring"""
        return a + b

    wrap = MeasurementFunctionWrap(measurement_func=func, name=func, units="m")
    assert wrap(a=1, b=2, c=3) == 3


def test_filtering_parameters_with_kwargs():
    c_in = False

    def func(a: int, b: float, **kwargs) -> float:
        """Sample docstring"""
        nonlocal c_in
        c_in = "c" in kwargs
        return a + b

    wrap = MeasurementFunctionWrap(measurement_func=func, name=func, units="m")
    assert wrap(a=1, b=2, c=3) == 3
    assert c_in
    assert wrap(a=1, b=2, d=3) == 3
    assert not c_in


def test_operations_on_wraps_div():
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


def test_operations_on_wraps_mul():
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
    divided = wrap1 * wrap2
    assert str(divided) == "func1 * func2"
    assert divided._units == symbols("m") ** 2
