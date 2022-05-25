import functools
import inspect
import operator
import typing
import warnings
from abc import ABC
from copy import copy

from sympy import Symbol, symbols


class UnitsException(Exception):
    pass


class MeasurementWrapBase(ABC):
    """
    Base class for measurement wrappers.

    This class is used to wrap a measurement function and provide a
    consistent interface to the measurement function.
    """

    def __init__(
        self,
        name: str,
        units: typing.Union[str, Symbol],
        long_description: str = "",
        power=1,
    ):
        if isinstance(units, str):
            units = symbols(units)
        self._name = name
        self._power = power
        self._long_description = long_description
        self._units = units

    def __call__(self, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return (
            self._name if self._power == 1 else f"{self._name}^{self._power}"
        )

    def all_additional_parameters_set(self):
        """
        If all additional parameters are set.

        Functions could have additional parameters, different from layers data

        """
        return False

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise RuntimeError("Modulo not supported")
        res = copy(self)
        res._power = power
        return res

    def __mul__(self, other):
        if isinstance(other, MeasurementWrapBase):
            units = self._units * other._units
        else:
            units = self._units
        return MeasurementCombinationWrap(
            operator=operator.mul,
            sources=(copy(self), copy(other)),
            name=f"{self} * {other}",
            units=units,
        )

    def __truediv__(self, other):
        if isinstance(other, MeasurementWrapBase):
            units = self._units / other._units
        else:
            units = self._units
        return MeasurementCombinationWrap(
            operator=operator.truediv,
            sources=(copy(self), copy(other)),
            name=f"{self} / {other}",
            units=units,
        )


class MeasurementCache:
    def __init__(self):
        self.cache = {}

    def calculate(self, func: MeasurementWrapBase, **kwargs):
        try:
            if func not in self.cache:
                self.cache[func] = {}
            key = tuple(kwargs.items())
            if key not in self.cache[func]:
                self.cache[func][key] = func(**kwargs)
            return self.cache[func][key]
        except Exception as e:
            warnings.warn(
                f"Error then try to cache in measurement {func}: {e}"
            )
            return func(**kwargs)


class MeasurementFunctionWrap(MeasurementWrapBase):
    def __init__(self, measurement_func, **kwargs):
        signature = inspect.signature(measurement_func)
        pass_args = self._check_signature(signature)
        super().__init__(**kwargs)
        self._measurement_func = measurement_func
        self._pass_args = pass_args
        self.rename_args = {}
        functools.wraps(measurement_func)(self)

    @staticmethod
    def _check_signature(signature: inspect.Signature):
        if any(
            x.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
            )
            for x in signature.parameters.values()
        ):
            raise RuntimeError("Positional only parameters not supported")
        if any(
            x.kind == inspect.Parameter.VAR_KEYWORD
            for x in signature.parameters.values()
        ):
            return tuple()
        return tuple(signature.parameters.keys())

    def __call__(self, **kwargs):
        try:
            for from_, to in self.rename_args.items():
                kwargs[to] = kwargs.pop(from_)
        except KeyError:
            raise RuntimeError(
                "Not all parameters are set for measurement function"
            )

        if self._pass_args:
            return (
                self._measurement_func(
                    **{name: kwargs[name] for name in self._pass_args}
                )
                ** self._power
            )
        return self._measurement_func(**kwargs) ** self._power


class MeasurementCombinationWrap(MeasurementWrapBase):
    def __init__(self, operator, sources, **kwargs):
        signature = inspect.signature(operator)
        if not (
            len(
                [
                    v
                    for v in signature.parameters.values()
                    if v.default == inspect.Parameter.empty
                ]
            )
            <= len(sources)
            <= len(signature.parameters)
        ):
            raise RuntimeError("operator could not handle all sources")
        super().__init__(**kwargs)
        self._operator = operator
        self._sources = tuple(sources)

    def __hash__(self):
        return hash((self._operator, self._sources))

    def __call__(self, **kwargs):
        return (
            self._operator(
                source(**kwargs)
                if isinstance(source, MeasurementWrapBase)
                else source
                for source in self._sources
            )
            ** self._power
        )


def measurement(units, name="", long_description="", power=1):
    def _func(func):
        nonlocal name
        if name == "":
            name = func.__name__.replace("_", " ").capitalize()
        return MeasurementFunctionWrap(
            func,
            name=name,
            units=units,
            long_description=long_description,
            power=power,
        )

    return _func
