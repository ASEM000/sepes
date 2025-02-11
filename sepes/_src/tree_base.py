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

"""Define a class that convert a class to a compatible tree structure."""

from __future__ import annotations

import abc
from typing import Any, Hashable, TypeVar

from typing_extensions import Self, Unpack

import sepes
from sepes._src.code_build import fields
from sepes._src.tree_index import at
from sepes._src.tree_pprint import PPSpec, tree_repr, tree_str
from sepes._src.tree_util import is_tree_equal, tree_copy, tree_hash, value_and_tree

T = TypeVar("T", bound=Hashable)
S = TypeVar("S")
PyTree = Any
EllipsisType = type(Ellipsis)  # TODO: use typing.EllipsisType when available
_mutable_instance_registry: set[int] = set()


def add_mutable_entry(node) -> None:
    _mutable_instance_registry.add(id(node))


def discard_mutable_entry(node) -> None:
    # use discard instead of remove to avoid raising KeyError
    # if the node has been removed in a parent node.
    _mutable_instance_registry.discard(id(node))


class TreeClassMeta(abc.ABCMeta):
    def __call__(klass: type[T], *a, **k) -> T:
        tree = getattr(klass, "__new__")(klass, *a, **k)
        # allow the setattr/delattr to set/delete attributes in the initialization
        # phase by flagging the instance as mutable.
        add_mutable_entry(tree)
        # initialize the instance with the instance marked as mutable.
        getattr(klass, "__init__")(tree, *a, **k)
        # remove the mutable flag after the initialization. to disallow
        # setattr/delattr to set/delete attributes after the initialization.
        discard_mutable_entry(tree)
        return tree


class TreeClass(metaclass=TreeClassMeta):
    """Convert a class to a pytree by inheriting from :class:`.TreeClass`.

    A pytree is any nested structure of containers and leaves. A container is
    a pytree can be a container or a leaf. Container examples are: a ``tuple``,
    ``list``, or ``dict``. A leaf is a non-container data structure like an
    ``int``, ``float``, ``string``, or ``Array``. :class:`.TreeClass` is a
    container pytree that holds other pytrees in its attributes.

    Note:
        :class:`.TreeClass` is immutable by default. This means that setting or
        deleting attributes after initialization is not allowed. This behavior
        is intended to prevent accidental mutation of the tree. All tree modifications
        on `TreeClass` are out-of-place. This means that all tree modifications
        return a new instance of the tree with the modified values.

        There are two ways to set or delete attributes after initialization:

        1. Using :attr:`.at` property to modify an *existing* leaf of the tree.

           >>> import sepes as sp
           >>> class Tree(sp.TreeClass):
           ...     def __init__(self, leaf: int):
           ...         self.leaf = leaf
           >>> tree = Tree(leaf=1)
           >>> new_tree = tree.at["leaf"].set(100)
           >>> tree is new_tree  # new instance is created
           False

        2. Using :func:`.value_and_tree` to call a method that mutates the tree.
           and apply the mutation on a *copy* of the tree. This option allows
           writing methods that mutate the tree instance but with these updates
           applied on a copy of the tree.

           >>> import sepes as sp
           >>> class Tree(sp.TreeClass):
           ...     def __init__(self, foo: int):
           ...         self.foo = foo
           ...     def add_bar(self, value:int) -> None:
           ...         # this method mutates the tree instance
           ...         # and will raise an `AttributeError` if called directly.
           ...         setattr(self, "bar", value)
           >>> tree = Tree(foo=1)
           >>> # now lets try to call `add_bar` directly
           >>> tree.add_bar(value=100)  # doctest: +SKIP
           Cannot set attribute value=100 to `key='bar'`  on an immutable instance of `Tree`.
           >>> output, tree_ = sp.value_and_tree(lambda T: T.add_bar(100))(tree)
           >>> tree, tree_
           (Tree(foo=1), Tree(foo=1, bar=100))

           This pattern is useful to write freely mutating methods, but with
           The expense of having to call through `at["method_name"]` instead of
           calling the method directly.

    Note:
        ``sepes`` offers two methods to construct the ``__init__`` method:

        1. Manual ``__init__`` method

           >>> import sepes as sp
           >>> class Tree(sp.TreeClass):
           ...     def __init__(self, a:int, b:float):
           ...         self.a = a
           ...         self.b = b
           >>> tree = Tree(a=1, b=2.0)

        2. Auto generated ``__init__`` method from type annotations.

           Using :func:`.autoinit` decorator where the type annotations are used to
           generate the ``__init__`` method. :func:`.autoinit`` with :func:`field`
           objects can be used to apply functions on the field values during
           initialization, support multiple argument kinds, and can apply functions
           on field values on getting the value. For more details see :func:`.autoinit`
           and :func:`.field`.

           >>> import sepes as sp
           >>> @sp.autoinit
           ... class Tree(sp.TreeClass):
           ...     a:int
           ...     b:float
           >>> tree = Tree(a=1, b=2.0)

    Note:
        Leaf-wise math operations are supported  using ``leafwise`` decorator.
        ``leafwise`` decorator adds ``__add__``, ``__sub__``, ``__mul__``, ... etc
        to registered pytrees. These methods apply math operations to each leaf of
        the tree. for example:

        >>> @sp.leafwise
        ... class Tree(sp.TreeClass):
        ...     def __init__(self, a:int, b:float):
        ...         self.a = a
        ...         self.b = b
        >>> tree = Tree(a=1, b=2.0)
        >>> tree + 1  # will add 1 to each leaf
        Tree(a=2, b=3.0)

    Note:
        Advanced indexing is supported using ``at`` property. Indexing can be
        used to ``get``, ``set``, or ``apply`` a function to a leaf or a group of
        leaves using ``leaf`` name, index or by a boolean mask.

        >>> class Tree(sp.TreeClass):
        ...     def __init__(self, a:int, b:float):
        ...         self.a = a
        ...         self.b = b
        >>> tree = Tree(a=1, b=2.0)
        >>> tree.at["a"].get()
        Tree(a=1, b=None)
        >>> tree.at["a"].get()
        Tree(a=1, b=None)

    Note:
        ``AttributeError`` is raised, If a method that mutates the instance
        is called directly. Instead use :func:`.value_and_tree` to call
        the method on a copy of the tree. :func:`.value_and_tree` calls the function
        on copied input arguments to ensure non-mutating behavior.

        >>> import sepes as sp
        >>> class Counter(sp.TreeClass):
        ...     def __init__(self, count: int):
        ...         self.count = count
        ...     def increment(self, value):
        ...         self.count += value
        ...         return self.count
        >>> counter = Counter(0)
        >>> sp.value_and_tree(lambda C: C.increment(1))(counter)
        (1, Counter(count=1))

    Note:
        :class:`.TreeClass` inherits from ``abc.ABC`` meaning that it can `abc`
        features like ``@abc.abstractmethod`` can be used to define abstract
        behavior that can be implemented by subclasses.

    Warning:
        The structure should be organized as a tree. In essence, *cyclic references*
        are not allowed. The leaves of the tree are the values of the tree and
        the branches are the containers that hold the leaves.
    """

    def __init_subclass__(klass: type[T], **k):
        # disallow setattr/delattr to be overridden as they are used
        # to implement the immutable/controlled mutability behavior.
        if "__setattr__" in vars(klass):
            raise TypeError(f"Reserved method `__setattr__` defined in `{klass}`.")
        if "__delattr__" in vars(klass):
            raise TypeError(f"Reserved method `__delattr__` defined in `{klass}`.")
        super().__init_subclass__(**k)
        # - register the class with the proper tree backend.
        # - the registration envolves defining two rules: how to flatten the nested
        #   structure of the class and how to unflatten the flattened structure.
        #   The flatten rule for `TreeClass` is equivalent to vars(self). and the
        #   unflatten rule is equivalent to `klass(**flat_tree)`. The flatten/unflatten
        #   rule is exactly same as the flatten rule for normal dictionaries.
        treelib = sepes._src.backend.treelib
        treelib.register_treeclass(klass)

    def __setattr__(self, key: str, value: Any) -> None:
        # - implements the controlled mutability behavior.
        # - In essence, setattr is allowed to set attributes during initialization
        #   and during functional call using `value_and_tree(method)(*, **)` by marking the
        #   instnace as mutable. Otherwise, setattr is disallowed.
        # - recall that during the functional call using `value_and_tree(method)(*, **)`
        #   the tree is always copied and the copy is marked as mutable, thus
        #   setattr is allowed to set attributes on the copy not the original.
        if id(self) not in _mutable_instance_registry:
            raise AttributeError(
                f"Cannot set attribute {value=} to `{key=}`  "
                f"on an immutable instance of `{type(self).__name__}`."
            )

        getattr(object, "__setattr__")(self, key, value)

    def __delattr__(self, key: str) -> None:
        # - same as __setattr__ but for delattr.
        # - both __setattr__ and __delattr__ are used to implement the
        # - controlled mutability behavior during initialization and
        #   during functional call using `value_and_tree(method)(*, **)`.
        # - recall that during the functional call using `value_and_tree(method)(*, **)`
        #   the tree is always copied and the copy is marked as mutable, thus
        #   setattr is allowed to set attributes on the copy not the original.
        if id(self) not in _mutable_instance_registry:
            raise AttributeError(
                f"Cannot delete attribute `{key}` "
                f"on immutable instance of `{type(self).__name__}`."
            )
        getattr(object, "__delattr__")(self, key)

    @property
    def at(self) -> at[Self]:
        """Immutable out-of-place indexing.

        - ``.at[***].get()``:
            Return a new instance with the value at the index otherwise None.
        - ``.at[***].set(value)``:
            Set the `value` and return a new instance with the updated value.
        - ``.at[***].apply(func)``:
            Apply a ``func`` and return a new instance with the updated value.

        *Acceptable indexing types are:*
            - ``str`` for mapping keys or class attributes.
            - ``int`` for positional indexing for sequences.
            - ``...`` to select all leaves.
            - a boolean mask of the same structure as the tree
            - a tuple of the above types to index multiple keys at same level.

        Example:
            >>> import sepes as sp
            >>> class Tree(sp.TreeClass):
            ...    def __init__(self, a:int, b:float):
            ...        self.a = a
            ...        self.b = b
            ...    def add(self, x: int) -> int:
            ...        self.a += x
            ...        return self.a
            >>> tree = Tree(a=1, b=2.0)
            >>> tree.at["a"].get()
            Tree(a=1, b=None)
            >>> tree.at["a"].set(100)
            Tree(a=100, b=2.0)
            >>> tree.at["a"].apply(lambda x: 100)
            Tree(a=100, b=2.0)

        Note:
            - ``pytree.at[*][**]`` is equivalent to selecting pytree.*.** .
            - ``pytree.at[*, **]`` is equivalent selecting pytree.* and pytree.**
        """
        # NOTE: use `at` as a property to enable chaining syntax.
        # instead of at(at(tree)[...].apply(...))[...].set(...)
        # chaining syntax is tree.at[...].apply(...).at[...].set(...)
        return at(self)

    def __repr__(self) -> str:
        return tree_repr(self)

    def __str__(self) -> str:
        return tree_str(self)

    def __copy__(self):
        return tree_copy(self)

    def __hash__(self) -> int:
        return tree_hash(self)

    def __eq__(self, other: Any) -> bool:
        return is_tree_equal(self, other)


@tree_repr.def_type(TreeClass)
def _(node: TreeClass, **spec: Unpack[PPSpec]) -> str:
    name = type(node).__name__
    skip = [f.name for f in fields(node) if not f.repr]
    kvs = tuple((k, v) for k, v in vars(node).items() if k not in skip)
    return name + "(" + tree_repr.pps(tree_repr.av_pp, kvs, **spec) + ")"


@tree_str.def_type(TreeClass)
def _(node: TreeClass, **spec: Unpack[PPSpec]) -> str:
    name = type(node).__name__
    skip = [f.name for f in fields(node) if not f.repr]
    kvs = tuple((k, v) for k, v in vars(node).items() if k not in skip)
    return name + "(" + tree_str.pps(tree_str.av_pp, kvs, **spec) + ")"


@value_and_tree.def_mutator(TreeClass)
def _(node: TreeClass) -> None:
    add_mutable_entry(node)


@value_and_tree.def_immutator(TreeClass)
def _(node: TreeClass) -> None:
    discard_mutable_entry(node)
