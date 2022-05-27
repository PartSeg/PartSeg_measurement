import inspect
import operator
import typing
import warnings
from abc import ABC
from copy import copy

from sympy import Symbol, symbols


class UnitsException(Exception):
    """Raised where units do not match"""


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

    def as_dict(self):
        return {
            "name": self._name,
            "power": self._power,
            "long_description": self._long_description,
            "units": self._units,
        }

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
    """
    Cache for measurement functions.
    For speedup of repeated calls.
    """

    def __init__(self):
        self._cache = {}

    def calculate(self, func: MeasurementWrapBase, **kwargs):
        """
        Try to get result from cache. If not found, calculate and store result.

        Parameters
        ----------
        func: MeasurementWrapBase
            Measurement function to be called.
        kwargs: dict
            Additional parameters for the measurement function.

        Returns
        -------
        result: Any
            Result of the measurement function.
        """
        try:
            if func not in self._cache:
                self._cache[func] = {}
            key = tuple(kwargs.items())
            if key not in self._cache[func]:
                self._cache[func][key] = func(**kwargs)
            return self._cache[func][key]
        except Exception as e:
            warnings.warn(
                f"Error then try to cache in measurement {func}: {e}"
            )
            return func(**kwargs)


@typing.final
class MeasurementFunctionWrap(MeasurementWrapBase):
    """
    Wrapper for measurement functions.
    """

    def __init__(self, measurement_func, rename_kwargs=None, **kwargs):
        if isinstance(measurement_func, MeasurementFunctionWrap):
            measurement_func = measurement_func._measurement_func
        signature = inspect.signature(measurement_func)
        pass_args = self._check_signature(signature)
        super().__init__(**kwargs)
        self._measurement_func = measurement_func
        self._pass_args = pass_args
        self._rename_kwargs = {} if rename_kwargs is None else rename_kwargs
        # functools.wraps(measurement_func)(self)

        annotations = copy(measurement_func.__annotations__)
        parameters = dict(**signature.parameters)
        for new_name, original_name in self._rename_kwargs.items():
            annotations[new_name] = annotations.pop(original_name)
            parameters[new_name] = parameters.pop(original_name).replace(
                name=new_name
            )
        self.__annotations__ = annotations

        self.__signature__ = inspect.Signature(
            parameters=list(parameters.values()),
            return_annotation=signature.return_annotation,
        )
        self.__doc__ = measurement_func.__doc__
        self.__name__ = measurement_func.__name__

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

    def __copy__(self):
        return MeasurementFunctionWrap(**self.as_dict())

    def as_dict(self):
        res = super().as_dict()
        res["measurement_func"] = self._measurement_func
        res["rename_kwargs"] = copy(self._rename_kwargs)
        return res

    def rename_parameter(
        self, current_name, new_name
    ) -> "MeasurementFunctionWrap":
        """
        Return a copy of the measurement function with renamed parameter.

        Parameters
        ----------
        current_name : str
            Current parameter name.
        new_name
            New parameter name.

        Returns
        -------
        func: MeasurementFunctionWrap
            Copy of the measurement function with renamed parameter.
        """
        dkt = self.as_dict()
        if current_name in dkt["rename_kwargs"]:
            dkt["rename_kwargs"][new_name] = dkt["rename_kwargs"].pop(
                current_name
            )
        else:
            dkt["rename_kwargs"][new_name] = current_name
        return self.__class__(**dkt)

    def __call__(self, **kwargs):
        try:
            for current_name, original_name in self._rename_kwargs.items():
                kwargs[original_name] = kwargs.pop(current_name)
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
    """
    Represents combination of measurement functions.
    """

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


def measurement(
    units: typing.Union[str, Symbol],
    name: str = "",
    long_description: str = "",
):
    """
    Decorator for measurement functions.

    Parameters
    ----------
    units: str or sympy.Symbol
    name: str, optional
        Name for measurement function. If not calculated from
        ``function.__name__`` by replace ``_`` with spaces and
        capitalize firs letter.
    long_description: str, optional
        Long description for measurement function. Could be used for render in
        user interface.

    Returns
    -------
    func: typing.Callable[[typing.Callable], MeasurementWrapBase]
        decorated function

    """

    def _func(func):
        nonlocal name
        if name == "":
            name = func.__name__.replace("_", " ").capitalize()
        return MeasurementFunctionWrap(
            func,
            name=name,
            units=units,
            long_description=long_description,
            power=1,
        )

    return _func
