# Copyright 2023 sepes authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constructor code generation from type annotations."""

# this modules contains lots of functionality similar to `dataclasses` and attrs.
# however, notable differences are:
# - allow marking fields as positional only, keyword only, variable positional,...
# - allow applying functions on the field values during initialization using descriptors.
# - does not allow mutable defaults.
# - allow registering additional types to be excluded from `autoinit`. e.g. raise an error.
# - only code generation is supported is done. other functionality like `__repr__`,
#   `__eq__`, `__hash__`, esp. are not done here.

# one design choice is that `autoinit` and `Field` are not tightly coupled.
# Field` can be used without `autoinit` as a descriptor to apply functions on
# the field values during initialization. Moreover, `TreeClass` is not coupled with
# `autoinit` or `Field` and can be used without them. this simplifies the code
# by separating the functionality.

from __future__ import annotations

import functools as ft
import sys
from collections import defaultdict
from collections.abc import Callable, MutableMapping, MutableSequence, MutableSet
from typing import Any, Literal, Sequence, TypeVar, get_args

from typing_extensions import dataclass_transform

T = TypeVar("T")
PyTree = Any
EllipsisType = type(Ellipsis)
ArgKindType = Literal["POS_ONLY", "POS_OR_KW", "VAR_POS", "KW_ONLY", "VAR_KW"]
ArgKind = get_args(ArgKindType)
EXCLUDED_FIELD_NAMES: set[str] = {"self", "__post_init__", "__annotations__"}


@ft.singledispatch
def check_excluded_type(value: T) -> None:
    ...


@check_excluded_type.register(MutableSequence)
@check_excluded_type.register(MutableMapping)
@check_excluded_type.register(MutableSet)
def _(value) -> None:
    raise TypeError(f"Mutable {value=} is not allowed.")


class Null:
    __slots__ = []
    __repr__ = lambda _: "NULL"
    __bool__ = lambda _: False


NULL = Null()


def generate_field_doc(field: Field) -> str:
    out: list[str] = ["Field Information:"]
    out += [f"\tName:\t\t``{field.name}``"]
    out += [f"\tDefault:\t``{field.default}``"] if field.default is not NULL else []
    out += [f"Description:\n\t{field.doc}"] if field.doc else []

    if field.on_setattr or field.on_getattr:
        out += ["Callbacks:"]

    if field.on_setattr:
        out += ["\t- On setting attribute:\n"]
        out += [f"\t\t- ``{func}``" for func in field.on_setattr]

    if field.on_getattr:
        out += ["\t- On getting attribute:\n"]
        out += [f"\t\t- ``{func}``" for func in field.on_getattr]

    return "\n".join(out)


def slots(klass) -> tuple[str, ...]:
    return getattr(klass, "__slots__", ())


class Field:
    """Field descriptor placeholder

    Args:
        name: The field name.
        type: The field type.
        default: The default value of the field.
        init: Whether the field is included in the object's ``__init__`` function.
        repr: Whether the field is included in the object's ``__repr__`` function.
        kind: Argument kind, one of:

            - ``POS_ONLY``: positional only argument (e.g. ``x`` in ``def f(x, /):``)
            - ``VAR_POS``: variable positional argument (e.g. ``*x`` in ``def f(*x):``)
            - ``POS_OR_KW``: positional or keyword argument (e.g. ``x`` in ``def f(x):``)
            - ``KW_ONLY``: keyword only argument (e.g. ``x`` in ``def f(*, x):``)
            - ``VAR_KW``: variable keyword argument (e.g. ``**x`` in ``def f(**x):``)

        metadata: A mapping of user-defined data for the field.
        on_setattr: A sequence of functions called on ``__setattr__``.
        on_getattr: A sequence of functions called on ``__getattr__``.
        alias: An a alias for the field name in the constructor. e.g ``name=x``,
            ``alias=y`` will allow ``obj = Class(y=1)`` to be equivalent to
            ``obj = Class(x=1)``.
        doc: The field documentation.
    """

    __slots__ = [
        "name",
        "type",
        "default",
        "init",
        "repr",
        "kind",
        "metadata",
        "on_setattr",
        "on_getattr",
        "alias",
        "doc",
    ]

    def __init__(
        self,
        *,
        name: str | Null = NULL,
        type: type | Null = NULL,
        default: Any = NULL,
        init: bool = True,
        repr: bool = True,
        kind: ArgKind = "POS_OR_KW",
        metadata: dict[str, Any] | None = None,
        on_setattr: Sequence[Callable[[Any], Any]] = (),
        on_getattr: Sequence[Callable[[Any], Any]] = (),
        alias: str | None = None,
        doc: str = "",
    ):
        self.name = name
        self.type = type
        self.default = default
        self.init = init
        self.repr = repr
        self.kind = kind
        self.metadata = metadata
        self.on_setattr = on_setattr
        self.on_getattr = on_getattr
        self.alias = alias
        self.doc = doc

    def replace(self, **kwargs) -> Field:
        """Replace the field attributes."""
        # define a `replace` method similar to `dataclasses.replace` or namedtuple
        # to allow the user to replace the field attributes.
        return type(self)(**{k: kwargs.get(k, getattr(self, k)) for k in slots(Field)})

    def pipe(self, funcs: Sequence[Callable[[Any], Any]], value: Any):
        """Apply a sequence of functions on the field value."""
        for func in funcs:
            # for a given sequence of unary functions, apply them on the field value
            # and return the result. if an error is raised, emit a descriptive error
            try:
                value = func(value)
            except Exception as e:
                # emit a *descriptive* error message with the name of the attribute
                # associated with the field and the name of the function that raised
                # the error.
                cname = getattr(func, "__name__", func)
                raise type(e)(f"On applying {cname} for field=`{self.name}`:\n{e}")
        return value

    def __set_name__(self, owner, name: str) -> None:
        """Set the field name."""
        # set the name of the field to the attribute name in the class
        # and the type to the type hint of the attribute if it exists
        self.name = name
        # in case the user uses `field` as a descriptor without annotating the class
        if "__annotations__" in (variables := vars(owner)):
            # set the type to the type hint of the attribute if it exists
            self.type = variables.get(name, NULL)

    @property
    def __doc__(self) -> str:
        """Return the field documentation."""
        return generate_field_doc(field=self)

    def __get__(self: T, instance, _) -> T | Any:
        """Return the field value."""
        if instance is None:
            return self
        return self.pipe(self.on_getattr, vars(instance)[self.name])

    def __set__(self: T, instance, value) -> None:
        """Set the field value."""
        vars(instance)[self.name] = self.pipe(self.on_setattr, value)

    def __delete__(self: T, instance) -> None:
        """Delete the field value."""
        del vars(instance)[self.name]


def field(
    *,
    default: Any = NULL,
    init: bool = True,
    repr: bool = True,
    kind: ArgKindType = "POS_OR_KW",
    metadata: dict[str, Any] | None = None,  # type: ignore
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
    doc: str = "",
) -> Field:
    """Field placeholder for type hinted attributes.

    Args:
        default: The default value of the field.
        init: Whether the field is included in the object's ``__init__`` function.
        repr: Whether the field is included in the object's ``__repr__`` function.
        kind: Argument kind, one of:

            - ``POS_ONLY``: positional only argument (e.g. ``x`` in ``def f(x, /):``)
            - ``VAR_POS``: variable positional argument (e.g. ``*x`` in ``def f(*x):``)
            - ``POS_OR_KW``: positional or keyword argument (e.g. ``x`` in ``def f(x):``)
            - ``KW_ONLY``: keyword only argument (e.g. ``x`` in ``def f(*, x):``)
            - ``VAR_KW``: variable keyword argument (e.g. ``**x`` in ``def f(**x):``)

        metadata: A mapping of user-defined data for the field.
        on_setattr: A sequence of functions to called on ``__setattr__``.
        on_getattr: A sequence of functions to called on ``__getattr__``.
        alias: An a alias for the field name in the constructor. e.g ``name=x``,
            ``alias=y`` will allow ``obj = Class(y=1)`` to be equivalent to
            ``obj = Class(x=1)``.
        doc: extra documentation for the :func:.`field` .the complete documentation
            of the field includes the field name, the field doc, and the
            default value, and function callbacks applied on the field value.
            Mainly used for documenting the field callbacks.

            .. code-block:: python

                >>> import sepes as sp
                >>> @sp.autoinit
                ... class Tree:
                ...    leaf: int = sp.field(default=1, doc="Leaf node of the tree.", on_setattr=[lambda x:x])

                >>> print(Tree.leaf.__doc__)  # doctest: +SKIP
                Field Information:
                        Name:           ``leaf``
                        Default:        ``1``
                Description:
                        Leaf node of the tree.
                Callbacks:
                        - On setting attribute:

                                - ``<function Tree.<lambda> at 0x11c53dc60>``

    Example:
        Type and range validation using :attr:`on_setattr`:

        >>> import sepes as sp
        >>> @sp.autoinit
        ... class IsInstance(sp.TreeClass):
        ...    klass: type
        ...    def __call__(self, x):
        ...        assert isinstance(x, self.klass)
        ...        return x
        <BLANKLINE>
        >>> @sp.autoinit
        ... class Range(sp.TreeClass):
        ...    start: int|float = float("-inf")
        ...    stop: int|float = float("inf")
        ...    def __call__(self, x):
        ...        assert self.start <= x <= self.stop
        ...        return x
        <BLANKLINE>
        >>> @sp.autoinit
        ... class Employee(sp.TreeClass):
        ...    # assert employee ``name`` is str
        ...    name: str = sp.field(on_setattr=[IsInstance(str)])
        ...    # use callback compostion to assert employee ``age`` is int and positive
        ...    age: int = sp.field(on_setattr=[IsInstance(int), Range(1)])
        >>> employee = Employee(name="Asem", age=10)
        >>> print(employee)
        Employee(name=Asem, age=10)

    Example:
        Private attribute using :attr:`alias`:

        >>> import sepes as sp
        >>> @sp.autoinit
        ... class Employee(sp.TreeClass):
        ...     # `alias` is the name used in the constructor
        ...    _name: str = sp.field(alias="name")
        >>> employee = Employee(name="Asem")  # use `name` in the constructor
        >>> print(employee)  # `_name` is the private attribute name
        Employee(_name=Asem)

    Example:
        Buffer creation using :attr:`on_getattr`:

        >>> import sepes as sp
        >>> import jax.numpy as jnp
        >>> @sp.autoinit
        ... class Tree(sp.TreeClass):
        ...     buffer: jax.Array = sp.field(on_getattr=[jax.lax.stop_gradient])
        >>> tree = Tree(buffer=jnp.array((1.0, 2.0)))
        >>> def sum_buffer(tree):
        ...     return tree.buffer.sum()
        >>> print(jax.grad(sum_buffer)(tree))  # no gradient on `buffer`
        Tree(buffer=[0. 0.])

    Example:
        Parameterization using :attr:`on_getattr`:

        >>> import sepes as sp
        >>> import jax.numpy as jnp
        >>> def symmetric(array: jax.Array) -> jax.Array:
        ...    triangle = jnp.triu(array)  # upper triangle
        ...    return triangle + triangle.transpose(-1, -2)
        >>> @sp.autoinit
        ... class Tree(sp.TreeClass):
        ...    symmetric_matrix: jax.Array = sp.field(on_getattr=[symmetric])
        >>> tree = Tree(symmetric_matrix=jnp.arange(9).reshape(3, 3))
        >>> print(tree.symmetric_matrix)
        [[ 0  1  2]
         [ 1  8  5]
         [ 2  5 16]]

    Note:
        - :func:`field` is commonly used to annotate the class attributes to be
          used by the :func:`autoinit` decorator to generate the ``__init__``
          method similar to ``dataclasses.dataclass``.

        - :func:`field` can be used without the :func:`autoinit` as a descriptor
          to apply functions on the field values during initialization using
          the ``on_setattr`` / ``on_getattr`` argument.

            >>> import sepes as sp
            >>> def print_and_return(x):
            ...    print(f"Setting {x}")
            ...    return x
            >>> class Tree:
            ...    # `a` must be defined as a class attribute for the descriptor to work
            ...    a: int = sp.field(on_setattr=[print_and_return])
            ...    def __init__(self, a):
            ...        self.a = a
            >>> tree = Tree(1)
            Setting 1
    """
    if not isinstance(alias, (str, type(None))):
        raise TypeError(f"Non-string {alias=} argument provided to `field`")

    if not isinstance(metadata, (dict, type(None))):
        raise TypeError(f"Non-dict {metadata=} argument provided to `field`")

    if kind not in ArgKind:
        raise ValueError(f"{kind=} not in {ArgKind}")

    if not isinstance(on_setattr, Sequence):
        raise TypeError(f"Non-sequence {on_setattr=} argument provided to `field`")

    if not isinstance(on_getattr, Sequence):
        raise TypeError(f"Non-sequence {on_getattr=} argument provided to `field`")

    if not isinstance(init, bool):
        raise TypeError(f"Non-bool {init=} argument provided to `field`")

    for func in on_setattr:
        if not isinstance(func, Callable):  # type: ignore
            raise TypeError(f"Non-callable {func=} provided to `field` on_setattr")

    for func in on_getattr:
        if not isinstance(func, Callable):
            raise TypeError(f"Non-callable {func=} provided to `field` on_getattr")

    return Field(
        default=default,
        init=init,
        repr=repr,
        kind=kind,
        metadata=metadata,  # type: ignore
        on_setattr=on_setattr,
        on_getattr=on_getattr,
        alias=alias,
        doc=doc,
    )


def build_field_map(klass: type) -> dict[str, Field]:
    field_map: dict[str, Field] = dict()

    if klass is object:
        return dict(field_map)

    for base in reversed(klass.__mro__[1:]):
        field_map.update(build_field_map(base))

    if (hint_map := vars(klass).get("__annotations__", NULL)) is NULL:
        return dict(field_map)

    if EXCLUDED_FIELD_NAMES.intersection(hint_map):
        raise ValueError(f"`Field` name cannot be in {EXCLUDED_FIELD_NAMES}")

    for key, hint in hint_map.items():
        # get the current base key
        value = vars(klass).get(key, NULL)

        if not isinstance(value, Field):
            # non-`Field` annotation is ignored
            # non-autoinit base class type hints are ignored
            continue

        # in case the user uses mutable defaults or any other user-defined
        # excluded types, raise an error
        check_excluded_type(value.default)

        # case: `x: Any = field(default=1)`
        field_map[key] = value.replace(name=key, type=hint)

    return field_map


def fields(x: Any) -> tuple[Field, ...]:
    """Returns a tuple of ``Field`` objects for the given instance or class.

    ``Field`` objects are generated from the class type hints and contains
    the information about the field information.if the user uses
    the ``sepes.field`` to annotate.

    Note:
        - If the class is not annotated, an empty tuple is returned.
        - The ``Field`` generation is cached for class and its bases.
    """
    return tuple(build_field_map(x if isinstance(x, type) else type(x)).values())


def convert_hints_to_fields(klass: type[T]) -> type[T]:
    # convert klass hints to `Field` objects for the current decorated class
    if (hint_map := vars(klass).get("__annotations__", NULL)) is NULL:
        return klass

    for key, hint in hint_map.items():
        if not isinstance(value := vars(klass).get(key, NULL), Field):
            setattr(klass, key, Field(default=value, type=hint, name=key))
    return klass


def check_duplicate_var_kind(field_map: dict[str, Field]) -> None:
    # check for duplicate `VAR_POS` and `VAR_KW` arguments
    seen: set[Literal["VAR_POS", "VAR_KW"]] = set()
    for field in field_map.values():
        if field.kind in ("VAR_POS", "VAR_KW"):
            # disallow multiple `VAR_POS` and `VAR_KW` arguments
            # for example more than one field(kind="VAR_POS") is not allowed
            if field.kind in seen:
                raise TypeError(f"Duplicate {field.kind=} for {field.name=}")
            seen.add(field.kind)


def build_init_method(klass: type[T]) -> type[T]:
    field_map: dict[str, Field] = build_field_map(klass)
    check_duplicate_var_kind(field_map)
    hints = {"return": None}  # annotations

    body: list[str] = []
    head: list[str] = ["self"]
    heads: dict[str, list[str]] = defaultdict(list)

    for field in field_map.values():
        if field.init:
            # add to field to head and body
            hints[field.name] = field.type
            # how to name the field in the constructor
            alias = field.alias or field.name
            body += [f"self.{field.name}={alias}"]

            if field.default is NULL:
                # e.g. def __init__(.., x)
                heads[field.kind] += [alias]
            else:
                # e.g def __init__(.., x=value) but
                # pass reference to the default value
                heads[field.kind] += [f"{alias}=refmap['{field.name}'].default"]
        else:
            if field.default is not NULL:
                # case for fields with `init=False` and no default value
                # usaully declared in __post_init__
                body += [f"self.{field.name}=refmap['{field.name}'].default"]

    has_post = (key := "__post_init__") in vars(klass)
    body += [f"self.{key}()"] if has_post else ["pass"]

    # organize the arguments order:
    # (POS_ONLY, POS_OR_KW, VAR_POS, KW_ONLY, VAR_KW)
    head += (heads["POS_ONLY"] + ["/"]) if heads["POS_ONLY"] else []
    head += heads["POS_OR_KW"]
    head += ["*" + "".join(heads["VAR_POS"])] if heads["VAR_POS"] else []
    # case for ...(*a, b) and ...(a, *, b)
    head += ["*"] if (heads["KW_ONLY"] and not heads["VAR_POS"]) else []
    head += heads["KW_ONLY"]
    head += ["**" + "".join(heads["VAR_KW"])] if heads["VAR_KW"] else []

    # generate the code for the method
    code = "def closure(refmap):\n"
    code += f"\tdef __init__({','.join(head)}):"
    field_map["__annotations__"] = hints

    code += f"\n\t\t{';'.join(body)}"
    code += f"\n\t__init__.__qualname__ = '{klass.__qualname__}.__init__'"
    code += f"\n\t__init__.__annotations__ = refmap['__annotations__']"
    code += "\n\treturn __init__"

    # execute the code in the class namespace to generate the method
    exec(code, vars(sys.modules[klass.__module__]), namespace := dict())
    method = namespace["closure"](field_map)
    # add the method to the class
    setattr(klass, "__init__", method)
    return klass


@dataclass_transform(field_specifiers=(Field, field))
def autoinit(klass: type[T]) -> type[T]:
    """A class decorator that generates the ``__init__`` method from type hints.

    Using the ``autoinit`` decorator, the user can define the class attributes
    using type hints and the ``__init__`` method will be generated automatically

    >>> import sepes as sp
    >>> @sp.autoinit
    ... class Tree:
    ...     x: int
    ...     y: int

    Is equivalent to:

    >>> class Tree:
    ...     def __init__(self, x: int, y: int):
    ...         self.x = x
    ...         self.y = y

    Example:
        >>> import sepes as sp
        >>> @sp.autoinit
        ... class Tree:
        ...     x: int
        ...     y: int
        >>> tree = Tree(1, 2)
        >>> tree.x, tree.y
        (1, 2)

    Example:
        >>> # define fields with different argument kinds
        >>> import sepes as sp
        >>> @sp.autoinit
        ... class Tree:
        ...     kw_only_field: int = sp.field(default=1, kind="KW_ONLY")
        ...     pos_only_field: int = sp.field(default=2, kind="POS_ONLY")

    Example:
        >>> # define a converter to apply ``abs`` on the field value
        >>> @sp.autoinit
        ... class Tree:
        ...     a:int = sp.field(on_setattr=[abs])
        >>> Tree(a=-1).a
        1

    .. warning::
        - The ``autoinit`` decorator will is no-op if the class already has a
          user-defined ``__init__`` method.

    Note:
        - In case of inheritance, the ``__init__`` method is generated from the
          the type hints of the current class and any base classes that
          are decorated with ``autoinit``.

        >>> import sepes as sp
        >>> import inspect
        >>> @sp.autoinit
        ... class Base:
        ...     x: int
        >>> @sp.autoinit
        ... class Derived(Base):
        ...     y: int
        >>> obj = Derived(x=1, y=2)
        >>> inspect.signature(obj.__init__)
        <Signature (x: int, y: int) -> None>

        - Base classes that are not decorated with ``autoinit`` are ignored during
          synthesis of the ``__init__`` method.

        >>> import sepes as sp
        >>> import inspect
        >>> class Base:
        ...     x: int
        >>> @sp.autoinit
        ... class Derived(Base):
        ...     y: int
        >>> obj = Derived(y=2)
        >>> inspect.signature(obj.__init__)
        <Signature (y: int) -> None>

    Note:
        Use ``autoinit`` instead of ``dataclasses.dataclass`` if you want to
        use ``jax.Array`` as a field default value. As ``dataclasses.dataclass``
        will incorrectly raise an error starting from python 3.11 complaining
        that ``jax.Array`` is not immutable.

    Note:
        By default ``autoinit`` will raise an error if the user uses mutable defaults.
        To register an additional type to be excluded from ``autoinit``, use
        :func:`autoinit.register_excluded_type`, with an optional ``reason``
        for excluding the type.

        >>> import sepes as sp
        >>> class T:
        ...     pass
        >>> sp.autoinit.register_excluded_type(T, reason="not allowed")
        >>> @sp.autoinit
        ... class Tree:
        ...     x: T = sp.field(default=T())  # doctest: +SKIP
        Traceback (most recent call last):
            ...
    """
    return (
        klass
        # if the class already has a user-defined __init__ method
        # then return the class as is without any modification
        if "__init__" in vars(klass)
        # first convert the current class hints to fields
        # then build the __init__ method from the fields of the current class
        # and any base classes that are decorated with `autoinit`
        else build_init_method(convert_hints_to_fields(klass))
    )


def register_excluded_type(klass: type, reason: str | None = None) -> None:
    """Exclude a type from being used in the ``autoinit`` decorator.

    Args:
        klass: The type to be excluded.
        reason: The reason for excluding the type.
    """
    reason = f" {reason=}" if reason is not None else ""

    @check_excluded_type.register(klass)
    def _(value) -> None:
        raise TypeError(f"{value=} is excluded from `autoinit`.{reason}")


autoinit.register_excluded_type = register_excluded_type
