# pylint: disable=no-self-use
import inspect
import json
import operator

import docstring_parser
import nme
import pytest

from PartSeg_measurement.measurement_wrap import (
    MeasurementCache,
    MeasurementCalculation,
    MeasurementCombinationWrap,
    MeasurementFunctionWrap,
    measurement,
)


class TestMeasurementFunctionWrap:
    def test_wraps(self):
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = MeasurementFunctionWrap(measurement_func=func)
        assert func.__doc__ == wrap.__doc__
        sig = inspect.signature(wrap)
        assert sig.parameters["a"].annotation is int
        assert sig.parameters["a"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["b"].annotation is float
        assert sig.parameters["b"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_filtering_parameters(self):
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = MeasurementFunctionWrap(measurement_func=func)
        assert wrap(a=1, b=2, c=3) == 3

    def test_filtering_parameters_with_kwargs(self):
        c_in = False

        def func(a: int, b: float, **kwargs) -> float:
            """Sample docstring"""
            nonlocal c_in
            c_in = "c" in kwargs
            return a + b

        wrap = MeasurementFunctionWrap(measurement_func=func)
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
        ).bind(a=1)
        assert wrap(b=2) == 3
        sig = inspect.signature(wrap)
        assert "a" not in sig.parameters
        assert "b" in sig.parameters

    def test_rename_bind(self):
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = (
            MeasurementFunctionWrap(
                measurement_func=func,
            )
            .rename_parameter("a", "x")
            .bind(x=1)
        )
        assert wrap(b=2) == 3

    def test_serialize(self, tmp_path, clean_register):
        @nme.register_class
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = (
            MeasurementFunctionWrap(
                measurement_func=func,
            )
            .bind(a=1)
            .rename_parameter("b", "y")
        )

        with open(tmp_path / "test.json", "w") as f:
            json.dump(wrap, f, cls=nme.NMEEncoder)

        with open(tmp_path / "test.json") as f:
            wrap2 = json.load(f, object_hook=nme.nme_object_hook)

        assert wrap2.name == "Func"
        assert wrap2(y=2) == 3
        assert wrap2._measurement_func is func

    def test_lack_of_kwarg(self):
        def func(*, a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = MeasurementFunctionWrap(
            measurement_func=func,
        )
        with pytest.raises(TypeError):
            wrap(a=1)

        with pytest.raises(TypeError):
            wrap.rename_parameter("b", "y")(a=1)

    def test_positional_only_error(self):
        def func(a: int, /, b: float) -> float:
            """Sample docstring"""
            return a + b

        with pytest.raises(TypeError):
            MeasurementFunctionWrap(
                measurement_func=func,
                units="m",
                name="func",
            )

    def test_prepare_doc_rename(self):
        def func(a: int, b: float) -> float:
            """
            Sample docstring

            Parameters
            ----------
            a : int
                a
            b : float
                b
            """
            return a + b

        wrap = MeasurementFunctionWrap(measurement_func=func, name="func")
        assert wrap.__doc__ == docstring_parser.compose(
            docstring_parser.parse(func.__doc__)
        )
        wrap2 = wrap.rename_parameter("b", "y")
        assert wrap2.__doc__ == docstring_parser.compose(
            docstring_parser.parse(
                func.__doc__.replace("b : float", "y : float")
            )
        )

    def test_prepare_doc_bind(self):
        def func(a: int, b: float) -> float:
            """
            Sample docstring

            Parameters
            ----------
            a : int
                a
            b : float
                b
            """
            return a + b

        wrap = MeasurementFunctionWrap(
            measurement_func=func, name="func"
        ).bind(a=1)
        assert len(docstring_parser.parse(wrap.__doc__).params) == 1
        assert docstring_parser.parse(wrap.__doc__).params[0].arg_name == "b"


class TestMeasurementCombinationWrap:
    def test_div(self):
        def func1(a: int, b: float) -> float:
            return a + b

        def func2(a: int, b: float) -> float:
            return a - b

        wrap1 = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        wrap2 = MeasurementFunctionWrap(measurement_func=func2, name="func2")
        divided = wrap1 / wrap2
        assert str(divided) == "func1 / func2"

    def test_div_2(self):
        def func1(a: int, b: float) -> float:
            return a + b

        wrap = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        wrap2 = wrap / 2
        assert wrap2(a=2, b=4) == 3
        assert str(wrap2) == "func1 / 2"

    def test_mul(self):
        def func1(a: int, b: float) -> float:
            return a + b

        def func2(a: int, b: float) -> float:
            return a - b

        wrap1 = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        wrap2 = MeasurementFunctionWrap(measurement_func=func2, name="func2")
        mul = wrap1 * wrap2
        assert str(mul) == "func1 * func2"

    def test_mul_2(self):
        def func1(a: int, b: float) -> float:
            return a + b

        wrap = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        wrap2 = wrap * 2
        assert wrap2(a=1, b=2) == 6
        assert str(wrap2) == "func1 * 2"

    def test_power(self):
        def func1(a: int, b: float) -> float:
            return a + b

        wrap = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        pow1 = wrap**2.0
        assert str(pow1) == "func1 ** 2.0"
        pow2 = pow1**3.0
        assert str(pow2) == "func1 ** 2.0 ** 3.0"
        # FIXME assert str(pow2) == "func1 ** 6.0"

    def test_proper_signature(self):
        def func1(*, a: int, b: float) -> float:
            return a + b

        def func2(*, a: int, c: float) -> float:
            return a + c

        wrap1 = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        wrap2 = MeasurementFunctionWrap(measurement_func=func2, name="func2")
        assert inspect.signature(wrap1) == inspect.signature(func1)
        comb = wrap1 * wrap2
        sig = inspect.signature(comb)
        assert len(sig.parameters) == 3
        assert sig.parameters["a"].annotation is int
        assert sig.parameters["a"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["b"].annotation is float
        assert sig.parameters["b"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["c"].annotation is float
        assert sig.parameters["c"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_to_low_num_sources(self):
        def func1(a: int, b: float) -> float:
            return a + b

        wrap = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        with pytest.raises(RuntimeError):
            MeasurementCombinationWrap(operator.mul, [wrap], name="test")

    def test_annotation_collision(self):
        def func1(a: int, b: float) -> float:
            return a + b

        def func2(a: int, b: int) -> float:
            return a + b

        wrap1 = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        wrap2 = MeasurementFunctionWrap(measurement_func=func2, name="func2")
        with pytest.raises(
            RuntimeError, match="Different annotations for parameter b"
        ):
            wrap1 * wrap2

    def test_create_docstring_base(self):
        def func1(a: int):
            """
            Func 1 docstring

            Parameters
            ----------
            a : int
                a parameter desc
            """
            return a

        def func2(a: int, b: float):
            """
            Func 2 docstring

            Parameters
            ----------
            a : int
                a parameter info
            b : float
                b parameter
            """
            return a + b

        wrap1 = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        wrap2 = MeasurementFunctionWrap(measurement_func=func2, name="func2")
        comb = wrap1 * wrap2
        assert "Func 1 docstring" in comb.__doc__
        assert "Func 2 docstring" in comb.__doc__
        assert "a : int" in comb.__doc__
        assert "a parameter desc" in comb.__doc__
        assert "a parameter info" not in comb.__doc__
        assert "b : float" in comb.__doc__
        assert "b parameter" in comb.__doc__
        comb2 = comb.bind(a=1)
        assert "Func 1 docstring" in comb2.__doc__
        assert "a : int" not in comb2.__doc__
        comb3 = comb.rename_parameter("a", "y")
        assert "y : int" in comb3.__doc__

    def test_serialize_bind_rename(self, tmp_path, clean_register):
        @nme.register_class
        def func1(a: int, b: float):
            return a + b

        @nme.register_class
        def func2(a: int, b: float):
            return a * b

        wrap1 = MeasurementFunctionWrap(measurement_func=func1, name="func")
        wrap2 = MeasurementFunctionWrap(measurement_func=func2, name="func4")
        combine1 = wrap1 * wrap2
        with open(tmp_path / "combine1.json", "w") as f_p:
            json.dump(combine1, f_p, cls=nme.NMEEncoder)

        with open(tmp_path / "combine1.json") as f_p:
            combine1_1 = json.load(f_p, object_hook=nme.nme_object_hook)

        assert isinstance(combine1_1, MeasurementCombinationWrap)
        assert combine1_1(a=2, b=4) == 48


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

    def test_cache_fail_hash(self):
        def local_max(a):
            return max(a)

        cache = MeasurementCache()
        with pytest.warns(UserWarning, match="Error then try to"):
            assert cache.calculate(local_max, a=[1, 2, 3]) == 3
        assert not cache._cache[local_max]
        assert cache.calculate(local_max, a=(1, 2, 3)) == 3
        assert cache._cache[local_max]


class TestMeasurementDecorator:
    def test_basic(self):
        @measurement
        def func(a: int, b: float) -> float:
            return a + b

        assert isinstance(func, MeasurementFunctionWrap)
        assert func(a=1, b=7) == 8

    def test_basic_serialization(self, clean_register, tmp_path):
        @measurement
        def func(a: int, b: float) -> float:
            return a + b

        with open(tmp_path / "func.json", "w") as f_p:
            json.dump(func, f_p, cls=nme.NMEEncoder)

        with open(tmp_path / "func.json") as f_p:
            func_1 = json.load(f_p, object_hook=nme.nme_object_hook)

        assert func_1(a=1, b=7) == 8

    def test_combination_serialization(self, clean_register, tmp_path):
        @measurement
        def func1(a: int, b: float) -> float:
            return a + b

        @measurement
        def func2(a: int, b: float) -> float:
            return a * b

        comb = func1 * func2
        with open(tmp_path / "comb.json", "w") as f_p:
            json.dump(comb, f_p, cls=nme.NMEEncoder)

        with open(tmp_path / "comb.json") as f_p:
            comb_1 = json.load(f_p, object_hook=nme.nme_object_hook)

        assert comb_1(a=1, b=7) == 56


class TestMeasurementCalculation:
    def test_calculate_no_args(self, clean_register):
        @measurement
        def func1(a: int, b: float):
            return a + b

        @measurement
        def func2(a: int, b: float):
            return a * b

        meas = MeasurementCalculation([func1, func2])
        assert meas(a=1, b=7) == [8, 7]

    def test_signature(self, clean_register):
        @measurement
        def func1(a: int, b: float):
            return a + b

        @measurement
        def func2(a: int, c: float):
            return a * c

        meas = MeasurementCalculation([func1, func2])
        assert meas(a=1, b=7, c=2) == [8, 2]
        signature = inspect.signature(meas)
        assert len(signature.parameters) == 3
        assert signature.parameters["a"].annotation is int
        assert signature.parameters["b"].annotation is float
        assert signature.parameters["c"].annotation is float

    def test_serialize(self, clean_register, tmp_path):
        @measurement
        def func1(a: int, b: float):
            return a + b

        @measurement
        def func2(a: int, c: float):
            return a * c

        meas = MeasurementCalculation([func1, func2, func1 * func2])
        with open(tmp_path / "meas.json", "w") as f_p:
            json.dump(meas, f_p, cls=nme.NMEEncoder)

        with open(tmp_path / "meas.json") as f_p:
            meas_1 = json.load(f_p, object_hook=nme.nme_object_hook)

        assert meas_1(a=1, b=7, c=2) == [8, 2, 16]
