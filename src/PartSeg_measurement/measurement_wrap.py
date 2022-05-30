import inspect
import operator
import typing
import warnings
from abc import ABC
from copy import copy

import docstring_parser
import nme
from sympy import Rational, Symbol, parse_expr

MeasurementWrapType = typing.TypeVar(
    "MeasurementWrapType", bound="MeasurementWrapBase"
)


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
        rename_kwargs: typing.Optional[typing.Dict[str, str]] = None,
        bind_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ):
        if isinstance(units, str):
            units = parse_expr(units)
        self._name = name
        self._long_description = long_description
        self._units = units
        self._rename_kwargs = {} if rename_kwargs is None else rename_kwargs
        self._bind_args = {} if bind_args is None else bind_args

    @property
    def name(self):
        return self._name

    @property
    def units(self):
        return self._units

    def prepare_kwargs(self, **kwargs) -> typing.Dict[str, typing.Any]:
        """
        Prepare the kwargs for the measurement function.

        Parameters
        ----------
        kwargs: dict
            The kwargs to prepare.

        Returns
        -------
        dict
            The prepared kwargs.

        """
        missed_kwargs = []
        problematic_kwargs = {"kwargs"}

        for current_name, original_name in self._rename_kwargs.items():
            try:
                kwargs[original_name] = kwargs.pop(current_name)
            except KeyError:
                missed_kwargs.append(current_name)
                problematic_kwargs.add(original_name)

        for name, value in self._bind_args.items():
            kwargs[name] = value

        missed_kwargs.extend(
            name
            for name, value in inspect.signature(self).parameters.items()
            if self._rename_kwargs.get(name, name) not in kwargs
            and name not in problematic_kwargs
            and value.kind != inspect.Parameter.VAR_KEYWORD
        )

        if len(missed_kwargs) == 1:
            raise TypeError(
                f"{self.name}() missing 1 required keyword-only argument:"
                f" '{missed_kwargs[0]}'"
            )
        if missed_kwargs:
            raise TypeError(
                f"{self.name}() missing {len(missed_kwargs)} required "
                f"keyword-only arguments: '{', '.join(missed_kwargs[:-1])}' "
                f"and '{missed_kwargs[-1]}'"
            )
        return kwargs

    def __call__(self, **kwargs):
        raise NotImplementedError

    def __copy__(self):
        return self.__class__(**self.as_dict(serialize=False))

    def __str__(self):
        # FIXME add signature
        return self.name

    def as_dict(self, serialize=True) -> typing.Dict[str, typing.Any]:
        """
        Return a dictionary representation of the measurement.
        Parameters
        ----------
        serialize: bool
            If True, the units will be serialized to a string.

        Returns
        -------

        """
        return {
            "name": self._name,
            "long_description": self._long_description,
            "units": str(self._units) if serialize else self._units,
            "rename_kwargs": copy(self._rename_kwargs),
            "bind_args": copy(self._bind_args),
        }

    def all_additional_parameters_set(self):
        """
        If all additional parameters are set.

        Functions could have additional parameters, different from layers data

        """
        return False

    def bind(self: MeasurementWrapType, **kwargs) -> MeasurementWrapType:
        dkt = self.as_dict(serialize=False)
        for name, value in kwargs.items():
            if name in dkt["rename_kwargs"]:
                name = dkt["rename_kwargs"].pop(name)
            dkt["bind_args"][name] = value
        return self.__class__(**dkt)

    def rename_parameter(
        self: MeasurementWrapType, current_name, new_name
    ) -> MeasurementWrapType:
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
        func:
            measurement combination with power set.
        """
        dkt = self.as_dict(serialize=False)
        if current_name in dkt["rename_kwargs"]:
            dkt["rename_kwargs"][new_name] = dkt["rename_kwargs"].pop(
                current_name
            )
        else:
            dkt["rename_kwargs"][new_name] = current_name
        return self.__class__(**dkt)

    def __pow__(self, power, modulo=None):
        res = copy(self)
        res._power = power
        return MeasurementCombinationWrap(
            operator=pow,
            sources=(copy(self), power, modulo),
            name=f"{self.name} ** {power}",
            units=self._units ** Rational(power),
        )

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

    def calculate(self, func: typing.Callable, **kwargs):
        """
        Try to get result from cache. If not found, calculate and store result.

        Parameters
        ----------e
        func: typing.Callable
            Measurement function to be called. Need to be hashable
        kwargs
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

    def __init__(
        self,
        measurement_func: typing.Callable,
        **kwargs,
    ):
        if isinstance(measurement_func, str):
            measurement_func = nme.REGISTER.get_class(measurement_func)
        if isinstance(measurement_func, MeasurementFunctionWrap):
            measurement_func = measurement_func._measurement_func
        signature = inspect.signature(measurement_func)
        pass_args = self._check_signature(signature)
        super().__init__(**kwargs)
        self._measurement_func: typing.Callable = measurement_func
        self._pass_args = pass_args

        # functools.wraps(measurement_func)(self)

        annotations = copy(measurement_func.__annotations__)
        parameters = dict(**signature.parameters)
        for name in self._bind_args:
            del annotations[name]
            del parameters[name]

        for new_name, original_name in self._rename_kwargs.items():
            annotations[new_name] = annotations.pop(original_name)
            parameters[new_name] = parameters.pop(original_name).replace(
                name=new_name
            )
        for name in list(parameters):
            if (
                parameters[name].kind
                == inspect.Parameter.POSITIONAL_OR_KEYWORD
            ):
                parameters[name] = parameters[name].replace(
                    kind=inspect.Parameter.KEYWORD_ONLY
                )

        self.__annotations__ = annotations

        self.__signature__ = inspect.Signature(
            parameters=list(parameters.values()),
            return_annotation=signature.return_annotation,
        )
        self.__doc__ = self._prepare_docs(measurement_func.__doc__)
        self.__name__ = measurement_func.__name__

    def _prepare_docs(self, doc: typing.Optional[str]) -> str:
        reverse_rename_kwargs = {y: x for x, y in self._rename_kwargs.items()}
        parsed = docstring_parser.parse(doc)
        for param in parsed.params:
            if param.arg_name in reverse_rename_kwargs:
                param.arg_name = reverse_rename_kwargs[param.arg_name]
        for param in list(parsed.params):
            if param.arg_name in self._bind_args:
                parsed.meta.remove(param)
        return docstring_parser.compose(parsed)

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
            raise TypeError("Positional only parameters not supported")
        if any(
            x.kind == inspect.Parameter.VAR_KEYWORD
            for x in signature.parameters.values()
        ):
            return tuple()
        return tuple(signature.parameters.keys())

    def as_dict(self, serialize: bool = True) -> typing.Dict[str, typing.Any]:
        res = super().as_dict(serialize=serialize)
        res["measurement_func"] = (
            nme.class_to_str(self._measurement_func)
            if serialize
            else self._measurement_func
        )
        return res

    def __call__(self, **kwargs):
        kwargs = self.prepare_kwargs(**kwargs)

        if self._pass_args:
            return self._measurement_func(
                **{name: kwargs[name] for name in self._pass_args}
            )
        return self._measurement_func(**kwargs)


class MeasurementCombinationWrap(MeasurementWrapBase):
    """
    Represents combination of measurement functions.
    """

    def __init__(self, operator, sources, **kwargs):
        if isinstance(operator, str):
            operator = nme.REGISTER.get_class(operator)
        if not self._check_operator(operator, sources):
            raise RuntimeError("operator could not handle all sources")
        super().__init__(**kwargs)
        self._operator = operator
        self._sources = tuple(sources)

        self.__signature__ = self._calculate_signature(
            self._sources, self._operator
        )
        self.__doc__ = self._prepare_doc(self._sources)

    def _prepare_doc(self, sources: typing.Sequence) -> str:
        """
        Prepare docstring base on docstring of sources.

        Parameters
        ----------
        sources: typing.Sequence
            Sequence of sources to prepare docstring for.

        Returns
        -------
        str
            Prepared docstring.
        """
        reverse_rename_kwargs = {y: x for x, y in self._rename_kwargs.items()}

        args = {}
        style = None
        raises = []
        descriptions = [self.name]
        for source in sources:
            if not (
                isinstance(source, MeasurementWrapBase)
                and hasattr(source, "__doc__")
            ):
                continue
            parsed = docstring_parser.parse(source.__doc__)
            raises.extend(parsed.raises)
            descriptions.append(f"{source.name}: {parsed.short_description}")
            if style is None:
                style = parsed.style
            for param in parsed.params:
                if param.arg_name in self._bind_args:
                    continue
                if param.arg_name in reverse_rename_kwargs:
                    param.arg_name = reverse_rename_kwargs[param.arg_name]
                if param.arg_name not in args:
                    args[param.arg_name] = param

        target_doc = docstring_parser.Docstring(style=style)
        target_doc.meta.extend(args.values())
        target_doc.meta.extend(raises)
        target_doc.short_description = "\n\n".join(descriptions)

        return docstring_parser.compose(target_doc)

    @staticmethod
    def _calculate_signature(sources, operator=None):
        sig_parameters = {}
        for source in sources:
            if not isinstance(source, MeasurementWrapBase):
                continue
            sub_signature = inspect.signature(source)
            for param in sub_signature.parameters.values():
                if param.name in sig_parameters:
                    if (
                        param.annotation
                        != sig_parameters[param.name].annotation
                    ):
                        raise RuntimeError(
                            f"Different annotations for parameter {param.name}"
                        )
                    continue
                sig_parameters[param.name] = param
        return inspect.Signature(parameters=list(sig_parameters.values()))

    @staticmethod
    def _check_operator(
        operator: typing.Callable, sources: typing.Sequence
    ) -> bool:
        signature = inspect.signature(operator)
        return (
            len(
                [
                    v
                    for v in signature.parameters.values()
                    if v.default == inspect.Parameter.empty
                ]
            )
            <= len(sources)
            <= len(signature.parameters)
        )

    def as_dict(self, serialize=True) -> typing.Dict[str, typing.Any]:
        res = super().as_dict(serialize=serialize)
        res["operator"] = (
            nme.class_to_str(self._operator) if serialize else self._operator
        )
        res["sources"] = self._sources
        return res

    def __hash__(self):
        return hash((self._operator, self._sources))

    def __call__(self, **kwargs):
        kwargs = self.prepare_kwargs(**kwargs)
        return self._operator(
            *[
                source(**kwargs)
                if isinstance(source, MeasurementWrapBase)
                else source
                for source in self._sources
            ]
        )


class MeasurementCalculation(typing.MutableSequence[MeasurementWrapBase]):
    """
    A class that represents a calculation of multiple measurements.
    """

    def __init__(self, initial_measurements: typing.Sequence[typing.Callable]):
        self._list: typing.List[MeasurementWrapBase] = [
            self._verify_measurement(m) for m in initial_measurements
        ]
        self.__signature__ = None
        self._update_signature()

    def _verify_measurement(self, measurement: typing.Callable):
        if isinstance(measurement, MeasurementWrapBase):
            return measurement
        if not callable(measurement):
            raise TypeError(f"{measurement} is not a callable")
        return MeasurementFunctionWrap(measurement)

    def _update_signature(self):
        self.__signature__ = MeasurementCombinationWrap._calculate_signature(
            self._list
        )

    def __call__(self, **kwargs):
        return [source(**kwargs) for source in self]

    def insert(self, index: int, value: typing.Callable) -> None:
        self._list.insert(index, self._verify_measurement(value))

    @typing.overload
    def __getitem__(self, i: int) -> MeasurementWrapBase:
        ...

    @typing.overload
    def __getitem__(self, s: slice) -> "MeasurementCalculation":
        ...

    def __getitem__(
        self, i: typing.Union[int, slice]
    ) -> typing.Union[MeasurementWrapBase, "MeasurementCalculation"]:
        return (
            self.__class__(self._list[i])
            if isinstance(i, slice)
            else self._list[i]
        )

    @typing.overload
    def __setitem__(self, i: int, o: typing.Callable) -> None:
        ...

    @typing.overload
    def __setitem__(
        self, s: slice, o: typing.Iterable[typing.Callable]
    ) -> None:
        ...

    def __setitem__(
        self,
        i: typing.Union[int, slice],
        o: typing.Union[typing.Callable, typing.Iterable[typing.Callable]],
    ) -> None:
        if isinstance(i, slice):
            if not isinstance(o, typing.Iterable):
                raise TypeError(
                    f"{o} is not iterable, but a slice was requested"
                )
            self._list[i] = [self._verify_measurement(m) for m in o]
        elif isinstance(o, typing.Iterable):
            raise TypeError(
                f"{o} is iterable, but a single index was requested"
            )
        else:
            self._list[i] = self._verify_measurement(o)
        self._update_signature()

    @typing.overload
    def __delitem__(self, i: int) -> None:
        ...

    @typing.overload
    def __delitem__(self, i: slice) -> None:
        ...

    def __delitem__(self, i: typing.Union[int, slice]) -> None:
        self._list.__delitem__(i)
        self._update_signature()

    def __len__(self) -> int:
        return len(self._list)


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
        nme.register_class(func)
        return MeasurementFunctionWrap(
            func,
            name=name,
            units=units,
            long_description=long_description,
        )

    return _func
