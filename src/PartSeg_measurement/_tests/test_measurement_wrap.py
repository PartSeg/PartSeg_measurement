# pylint: disable=no-self-use
import inspect
import json
import operator

import docstring_parser
import nme
import numpy as np
import pytest

from PartSeg_measurement.measurement_wrap import (
    BoundInfo,
    MeasurementCache,
    MeasurementCalculation,
    MeasurementCombinationWrap,
    MeasurementFunctionWrap,
    measurement,
)
from PartSeg_measurement.types import Image, Labels


class TestBoundInfo:
    def test_base(self):
        bound_info = BoundInfo(np.array([1, 10, 10]), np.array([20, 20, 20]))
        assert np.all(bound_info.lower == np.array([1, 10, 10]))
        assert np.all(bound_info.upper == np.array([20, 20, 20]))
        assert np.all(bound_info.box_size() == np.array([20, 11, 11]))
        assert "[1, 10, 10]" in str(bound_info)

    def test_slice(self):
        bound_info = BoundInfo(np.array([1, 10, 10]), np.array([20, 20, 20]))
        assert bound_info.get_slices(0) == [
            slice(1, 21),
            slice(10, 21),
            slice(10, 21),
        ]
        assert bound_info.get_slices(1) == [
            slice(0, 22),
            slice(9, 22),
            slice(9, 22),
        ]
        assert bound_info.get_slices(2) == [
            slice(0, 23),
            slice(8, 23),
            slice(8, 23),
        ]

    def test_del_dim(self):
        bound_info = BoundInfo(np.array([1, 10, 10]), np.array([20, 20, 20]))
        bound_info2 = bound_info.del_dim(1)
        assert np.all(bound_info2.lower == np.array([1, 10]))
        assert np.all(bound_info2.upper == np.array([20, 20]))


class TestMeasurementFunctionWrap:
    def test_wraps(self):
        def func(a: int, b: float) -> float:
            """Sample docstring"""
            return a + b

        wrap = MeasurementFunctionWrap(measurement_func=func)
        assert wrap(a=2, b=5) == 7
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
            return a + b  # pragma: no cover

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
            return a + b  # pragma: no cover

        with pytest.raises(
            TypeError, match="Positional only parameters not supported"
        ):
            MeasurementFunctionWrap(
                measurement_func=func,
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
        assert wrap2(a=3, y=6) == 9

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
        assert wrap(b=2) == 3

    def test_eq(self):
        def func1(a: int, b: float) -> float:
            return a + b

        def func2(a: int, b: float) -> float:
            return a + b  # pragma: no cover

        wrap1 = MeasurementFunctionWrap(measurement_func=func1)
        wrap1_1 = MeasurementFunctionWrap(measurement_func=func1)
        assert wrap1 == wrap1_1
        wrap2 = MeasurementFunctionWrap(measurement_func=func2)
        assert wrap1 != wrap2
        wrap3 = wrap1.bind(a=1).rename_parameter("b", "y")
        assert wrap1 != wrap3
        wrap4 = wrap1.rename_parameter("b", "y").bind(a=1)
        assert wrap3 == wrap4
        wrap5 = (
            wrap1.rename_parameter("a", "c")
            .bind(c=1)
            .rename_parameter("b", "y")
        )
        assert wrap4 == wrap5
        assert wrap1(a=1, b=2) == 3
        assert wrap3(y=2) == 3

    def test_repr(self):
        def func(a: int, b: float) -> float:
            return a + b  # pragma: no cover

        wrap = MeasurementFunctionWrap(measurement_func=func)
        assert wrap.__repr__().startswith(
            "MeasurementFunctionWrap(<function TestMeasurementFunctionWrap."
            "test_repr.<locals>.func at "
        )
        assert "{'y': 'b'}" in repr(wrap.rename_parameter("b", "y"))
        assert "{'b': 1}" in repr(wrap.bind(b=1))

    def test_per_component_add(self):
        def func(a: Labels, b: Image) -> float:
            return b[a > 0].sum()

        wrap = MeasurementFunctionWrap(measurement_func=func)
        sig = inspect.signature(wrap)
        assert sig.parameters["a"].annotation == Labels
        assert sig.parameters["per_component"].annotation == bool


class TestMeasurementCombinationWrap:
    def test_div(self):
        def func1(a: int, b: float) -> float:
            return a + b

        def func2(a: int, b: float) -> float:
            return a - b

        wrap1 = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        wrap2 = MeasurementFunctionWrap(measurement_func=func2, name="func2")
        divided = wrap1 / wrap2
        assert divided(a=3, b=1) == 2
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

        wrap1 = MeasurementFunctionWrap(measurement_func=func1)
        wrap2 = MeasurementFunctionWrap(measurement_func=func2, name="func2")
        mul = wrap1 * wrap2
        assert mul(a=2, b=1) == 3
        assert str(mul) == "Func1 * func2"

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
        assert pow2(a=1, b=2) == 3**6
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
        assert comb(a=1, b=2, c=3) == 12
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
            return a + b  # pragma: no cover

        wrap = MeasurementFunctionWrap(measurement_func=func1, name="func1")
        with pytest.raises(RuntimeError):
            MeasurementCombinationWrap(operator.mul, [wrap], name="test")

    def test_annotation_collision(self):
        def func1(a: int, b: float) -> float:
            return a + b  # pragma: no cover

        def func2(a: int, b: int) -> float:
            return a + b  # pragma: no cover

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
        assert comb(a=1, b=2) == 3
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

    def test_eq(self):
        def func1(a: int, b: float):
            return a + b  # pragma: no cover

        def func2(a: int, b: float):
            return a * b  # pragma: no cover

        wrap1 = MeasurementFunctionWrap(measurement_func=func1)
        wrap2 = MeasurementFunctionWrap(measurement_func=func2)
        comb1 = wrap1 * wrap2
        comb2 = wrap1 * wrap2
        assert comb1 == comb2
        assert comb1 is not comb2


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

    def test_cache_measurement_wrap(self):
        called = 0

        def _called(a: int, b: float) -> float:
            nonlocal called
            called += 1
            return a + b

        wrap = MeasurementFunctionWrap(_called, name="func")
        cache = MeasurementCache()
        assert cache.calculate(wrap, a=1, b=2) == 3
        assert called == 1
        assert cache.calculate(wrap, a=1, b=2) == 3
        assert called == 1
        assert cache.calculate(wrap, a=1, b=3) == 4
        assert called == 2
        assert cache.calculate(wrap, a=1, b=2) == 3
        assert called == 2

    def test_cache_measurement_wrap_with_kwargs(self):
        called = 0

        def _called(a: int, b: float) -> float:
            nonlocal called
            called += 1
            return a + b

        wrap = MeasurementFunctionWrap(_called, name="func")
        wrap2 = wrap.bind(a=1)
        wrap3 = wrap2.rename_parameter("b", "c")
        wrap4 = wrap.rename_parameter("b", "c").bind(a=1)
        cache = MeasurementCache()
        assert cache.calculate(wrap, a=1, b=2) == 3
        assert called == 1
        assert cache.calculate(wrap2, b=2) == 3
        assert called == 2
        assert cache.calculate(wrap2, b=2) == 3
        assert called == 2
        assert cache.calculate(wrap3, c=2) == 3
        assert called == 3
        assert cache.calculate(wrap3, c=2) == 3
        assert called == 3
        assert cache.calculate(wrap4, c=2) == 3
        assert called == 3

    def test_cache_measurement_combine(self):
        called = 0

        def _called(a: int, b: float) -> float:
            nonlocal called
            called += 1
            return a + b

        wrap = MeasurementFunctionWrap(_called, name="func")
        wrap2 = wrap * 2

        cache = MeasurementCache()
        assert cache.calculate(wrap, a=1, b=2) == 3
        assert called == 1
        assert cache.calculate(wrap2, a=1, b=2) == 6
        assert called == 2
        assert cache.calculate(wrap2, a=1, b=2) == 6
        assert called == 2
        assert cache.calculate(wrap2, a=1, b=4) == 10
        assert called == 3
        assert cache.calculate(wrap2, a=1, b=4) == 10
        assert called == 3


class TestMeasurementDecorator:
    def test_basic(self, clean_register):
        @measurement
        def func(a: int, b: float) -> float:
            return a + b

        assert isinstance(func, MeasurementFunctionWrap)
        assert func(a=1, b=7) == 8

    def test_name(self, clean_register):
        @measurement(name="test")
        def func(a: int, b: float) -> float:
            return a + b

        assert func.name == "test"
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
        assert len(meas) == 2
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
        assert len(meas) == 3
        with open(tmp_path / "meas.json", "w") as f_p:
            json.dump(meas, f_p, cls=nme.NMEEncoder)

        with open(tmp_path / "meas.json") as f_p:
            meas_1 = json.load(f_p, object_hook=nme.nme_object_hook)

        assert meas_1(a=1, b=7, c=2) == [8, 2, 16]

    def test_modification(self, clean_register):
        @measurement
        def func1(a: int, b: float):
            return a + b

        @measurement
        def func2(a: int, c: float):
            return a * c

        meas = MeasurementCalculation([func1, func2])
        assert len(meas) == 2
        assert "b" in inspect.signature(meas).parameters

        del meas[0]
        assert len(meas) == 1
        assert meas(a=1, b=7, c=2) == [2]
        assert "b" not in inspect.signature(meas).parameters
        assert "c" in inspect.signature(meas).parameters

        meas[0] = func1
        assert len(meas) == 1
        assert meas(a=1, b=7, c=2) == [8]
        assert "b" in inspect.signature(meas).parameters
        assert "c" not in inspect.signature(meas).parameters

        meas.insert(0, func2)
        assert len(meas) == 2
        assert meas(a=1, b=7, c=2) == [2, 8]
        assert "b" in inspect.signature(meas).parameters
        assert "c" in inspect.signature(meas).parameters

        assert meas[0] is func2
        meas[:] = [func1, func2]
        assert meas[0] is func1
        assert meas(a=1, b=7, c=2) == [8, 2]

    def test_wrong_setitem(self, clean_register):
        @measurement
        def func1(a: int, b: float):
            return a + b

        meas = MeasurementCalculation([func1])
        with pytest.raises(TypeError):
            meas[0] = [func1, func1]

        meas.append(func1)
        with pytest.raises(TypeError):
            meas[1:2] = func1

        assert meas(a=1, b=7) == [8, 8]

    def test_non_callable(self):
        with pytest.raises(TypeError):
            MeasurementCalculation([1])

    def test_not_wrapped_function(self, clean_register):
        @measurement
        def func1(a: int, b: float):
            return a + b

        def func2(a: int, c: float):
            return a + c

        meas = MeasurementCalculation([func1, func2])
        assert len(meas) == 2
        assert meas(a=1, b=7, c=2) == [8, 3]

        assert "c" in inspect.signature(meas).parameters

    def test_append_extend(self, clean_register):
        @measurement
        def func1(a: int, b: float):
            return a + b

        def func2(a: int, c: float):
            return a + c

        meas = MeasurementCalculation([])
        meas.append(func1)
        assert len(meas) == 1
        assert "c" not in inspect.signature(meas).parameters
        meas.extend([func2])
        assert len(meas) == 2
        assert "c" in inspect.signature(meas).parameters
        assert meas(a=1, b=7, c=2) == [8, 3]
