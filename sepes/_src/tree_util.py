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

"""Utility functions for pytrees."""

from __future__ import annotations

import copy
import functools as ft
import operator as op
from math import ceil, floor, trunc
from typing import Any, Callable, Generic, Hashable, Iterator, Sequence, Tuple, TypeVar

from typing_extensions import ParamSpec

import sepes
import sepes._src.backend.arraylib as arraylib
from sepes._src.backend import is_package_avaiable

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
P = ParamSpec("P")
PyTree = Any
EllipsisType = TypeVar("EllipsisType")
KeyEntry = TypeVar("KeyEntry", bound=Hashable)
TypeEntry = TypeVar("TypeEntry", bound=type)
TraceEntry = Tuple[KeyEntry, TypeEntry]
KeyPath = Tuple[KeyEntry, ...]
TypePath = Tuple[TypeEntry, ...]
KeyTypePath = Tuple[KeyPath, TypePath]


def tree_hash(*trees: PyTree) -> int:
    treelib = sepes._src.backend.treelib
    leaves, treedef = treelib.flatten(trees)
    return hash((*leaves, treedef))


def tree_copy(tree: T) -> T:
    """Return a copy of the tree."""
    # the dispatcher calls copy on the leaves of the tree
    # by default as an extra measure - beside flatten/unflatten-
    # to ensure that the tree is copied completely
    treelib = sepes._src.backend.treelib
    types = tuple(set(tree_copy.copy_dispatcher.registry) - {object})

    def is_leaf(node) -> bool:
        return isinstance(node, types)

    return treelib.map(tree_copy.copy_dispatcher, tree, is_leaf=is_leaf)


# default behavior is to copy the tree elements except for registered types
# like jax arrays which are immutable by default and should not be copied
tree_copy.copy_dispatcher = ft.singledispatch(copy.copy)
tree_copy.def_type = tree_copy.copy_dispatcher.register


@tree_copy.def_type(int)
@tree_copy.def_type(float)
@tree_copy.def_type(complex)
@tree_copy.def_type(str)
@tree_copy.def_type(bytes)
def _(x: T) -> T:
    # skip applying `copy.copy` on immutable atom types
    return x


def is_array_like(node) -> bool:
    return hasattr(node, "shape") and hasattr(node, "dtype")


def _is_leaf_rhs_equal(leaf, rhs):
    if is_array_like(leaf):
        if is_array_like(rhs):
            if leaf.shape != rhs.shape:
                return False
            if leaf.dtype != rhs.dtype:
                return False
            try:
                verdict = arraylib.all(leaf == rhs)
            except NotImplementedError:
                verdict = leaf == rhs
            try:
                return bool(verdict)
            except Exception:
                return verdict  # fail under `jit`
        return False
    return leaf == rhs


def is_tree_equal(*trees: Any) -> bool:
    """Return ``True`` if all pytrees are equal.

    Note:
        trees are compared using their leaves and treedefs.
    """
    treelib = sepes._src.backend.treelib
    tree0, *rest = trees
    leaves0, treedef0 = treelib.flatten(tree0)
    verdict = True

    for tree in rest:
        leaves, treedef = treelib.flatten(tree)
        if (treedef != treedef0) or verdict is False:
            return False
        verdict = ft.reduce(op.and_, map(_is_leaf_rhs_equal, leaves0, leaves), verdict)
    return verdict


class Static(Generic[T]):
    def __init_subclass__(klass, **k) -> None:
        # register subclasses as an empty pytree node
        # written like this to enforce selection of the proper backend
        # every time a subclass is created
        super().__init_subclass__(**k)
        # register with the proper backend
        treelib = sepes._src.backend.treelib
        treelib.register_static(klass)


class partial(ft.partial):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)  # type: ignore
        return self.func(*args, *iargs, **{**self.keywords, **kwargs})


def bcmap(
    func: Callable[P, T],
    broadcast_to: int | str | None = None,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Callable[P, T]:
    """Map a function over pytree leaves with automatic broadcasting for scalar arguments.

    Args:
        func: the function to be mapped over the pytree.
        broadcast_to: Accepts integer for broadcasting to a specific argument
            or string for broadcasting to a specific keyword argument.
            If ``None``, then the function is broadcasted to the first argument
            or the first keyword argument if no positional arguments are provided.
            Defaults to ``None``.
        is_leaf: a predicate function that returns True if the node is a leaf.

    Example:
        Transform `numpy` functions to work with pytrees:

        >>> import sepes as sp
        >>> import jax.numpy as jnp
        >>> tree_of_arrays = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}
        >>> tree_add = sp.bcmap(jnp.add)
        >>> # both lhs and rhs are pytrees
        >>> print(sp.tree_str(tree_add(tree_of_arrays, tree_of_arrays)))
        dict(a=[2 4 6], b=[ 8 10 12])
        >>> # rhs is a scalar
        >>> print(sp.tree_str(tree_add(tree_of_arrays, 1)))
        dict(a=[2 3 4], b=[5 6 7])
    """
    treelib = sepes._src.backend.treelib

    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        cargs = []
        ckwargs = {}
        leaves = []
        kwargs_keys: list[str] = []

        bdcst_to = (
            (0 if len(args) else next(iter(kwargs)))
            if broadcast_to is None
            else broadcast_to
        )

        treedef0 = (
            # reference treedef is the first positional argument
            treelib.flatten(args[bdcst_to], is_leaf=is_leaf)[1]
            if len(args)
            # reference treedef is the first keyword argument
            else treelib.flatten(kwargs[bdcst_to], is_leaf=is_leaf)[1]
        )

        for arg in args:
            if treedef0 == treelib.flatten(arg, is_leaf=is_leaf)[1]:
                cargs += [...]
                leaves += [treedef0.flatten_up_to(arg)]
            else:
                cargs += [arg]

        for key in kwargs:
            if treedef0 == treelib.flatten(kwargs[key], is_leaf=is_leaf)[1]:
                ckwargs[key] = ...
                leaves += [treedef0.flatten_up_to(kwargs[key])]
                kwargs_keys += [key]
            else:
                ckwargs[key] = kwargs[key]

        split_index = len(leaves) - len(kwargs_keys)
        all_leaves = []
        bfunc = partial(func, *cargs, **ckwargs)
        for args_kwargs_values in zip(*leaves):
            args = args_kwargs_values[:split_index]
            kwargs = dict(zip(kwargs_keys, args_kwargs_values[split_index:]))
            all_leaves += [bfunc(*args, **kwargs)]
        return treelib.unflatten(treedef0, all_leaves)

    return wrapper


def swop(func):
    # swaping the arguments of a two-arg function
    return ft.wraps(func)(lambda leaf, rhs: func(rhs, leaf))


def leafwise(klass: type[T]) -> type[T]:
    """A class decorator that adds leafwise operators to a class.

    Leafwise operators are operators that are applied to the leaves of a pytree.
    For example leafwise ``__add__`` is equivalent to:

    - ``tree_map(lambda x: x + rhs, tree)`` if ``rhs`` is a scalar.
    - ``tree_map(lambda x, y: x + y, tree, rhs)`` if ``rhs`` is a pytree
      with the same structure as ``tree``.

    Args:
        klass: The class to be decorated.

    Returns:
        The decorated class.

    Example:
        Use ``numpy`` functions on :class:`TreeClass`` classes decorated with :func:`leafwise`

        >>> import sepes as sp
        >>> import jax.numpy as jnp
        >>> @sp.leafwise
        ... @sp.autoinit
        ... class Point(sp.TreeClass):
        ...    x: float = 0.5
        ...    y: float = 1.0
        ...    description: str = "point coordinates"
        >>> # use :func:`tree_mask` to mask the non-inexact part of the tree
        >>> # i.e. mask the string leaf ``description`` to ``Point`` work
        >>> # with ``jax.numpy`` functions
        >>> co = sp.tree_mask(Point())
        >>> print(sp.bcmap(jnp.where)(co > 0.5, co, 1000))
        Point(x=1000.0, y=1.0, description=#point coordinates)

    Note:
        If a mathematically equivalent operator is already defined on the class,
        then it is not overridden.

    ==================      ============
    Method                  Operator
    ==================      ============
    ``__add__``              ``+``
    ``__and__``              ``&``
    ``__ceil__``             ``math.ceil``
    ``__divmod__``           ``divmod``
    ``__eq__``               ``==``
    ``__floor__``            ``math.floor``
    ``__floordiv__``         ``//``
    ``__ge__``               ``>=``
    ``__gt__``               ``>``
    ``__invert__``           ``~``
    ``__le__``               ``<=``
    ``__lshift__``           ``<<``
    ``__lt__``               ``<``
    ``__matmul__``           ``@``
    ``__mod__``              ``%``
    ``__mul__``              ``*``
    ``__ne__``               ``!=``
    ``__neg__``              ``-``
    ``__or__``               ``|``
    ``__pos__``              ``+``
    ``__pow__``              ``**``
    ``__round__``            ``round``
    ``__sub__``              ``-``
    ``__truediv__``          ``/``
    ``__trunc__``            ``math.trunc``
    ``__xor__``              ``^``
    ==================      ============
    """
    treelib = sepes._src.backend.treelib

    def uop(func):
        def wrapper(self):
            return treelib.map(func, self)

        return ft.wraps(func)(wrapper)

    def bop(func):
        def wrapper(leaf, rhs=None):
            if isinstance(rhs, type(leaf)):
                return treelib.map(func, leaf, rhs)
            return treelib.map(lambda x: func(x, rhs), leaf)

        return ft.wraps(func)(wrapper)

    for key, method in (
        ("__abs__", uop(abs)),
        ("__add__", bop(op.add)),
        ("__and__", bop(op.and_)),
        ("__ceil__", uop(ceil)),
        ("__divmod__", bop(divmod)),
        ("__eq__", bop(op.eq)),
        ("__floor__", uop(floor)),
        ("__floordiv__", bop(op.floordiv)),
        ("__ge__", bop(op.ge)),
        ("__gt__", bop(op.gt)),
        ("__invert__", uop(op.invert)),
        ("__le__", bop(op.le)),
        ("__lshift__", bop(op.lshift)),
        ("__lt__", bop(op.lt)),
        ("__matmul__", bop(op.matmul)),
        ("__mod__", bop(op.mod)),
        ("__mul__", bop(op.mul)),
        ("__ne__", bop(op.ne)),
        ("__neg__", uop(op.neg)),
        ("__or__", bop(op.or_)),
        ("__pos__", uop(op.pos)),
        ("__pow__", bop(op.pow)),
        ("__radd__", bop(swop(op.add))),
        ("__rand__", bop(swop(op.and_))),
        ("__rdivmod__", bop(swop(divmod))),
        ("__rfloordiv__", bop(swop(op.floordiv))),
        ("__rlshift__", bop(swop(op.lshift))),
        ("__rmatmul__", bop(swop(op.matmul))),
        ("__rmod__", bop(swop(op.mod))),
        ("__rmul__", bop(swop(op.mul))),
        ("__ror__", bop(swop(op.or_))),
        ("__round__", bop(round)),
        ("__rpow__", bop(swop(op.pow))),
        ("__rrshift__", bop(swop(op.rshift))),
        ("__rshift__", bop(op.rshift)),
        ("__rsub__", bop(swop(op.sub))),
        ("__rtruediv__", bop(swop(op.truediv))),
        ("__rxor__", bop(swop(op.xor))),
        ("__sub__", bop(op.sub)),
        ("__truediv__", bop(op.truediv)),
        ("__trunc__", uop(trunc)),
        ("__xor__", bop(op.xor)),
    ):
        if key not in vars(klass):
            # do not override any user defined methods
            # this behavior similar is to `dataclasses.dataclass`
            setattr(klass, key, method)
    return klass


def tree_type_path_leaves(
    tree: PyTree,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
    is_path_leaf: Callable[[KeyTypePath], bool] | None = None,
) -> Sequence[tuple[KeyTypePath, Any]]:
    treelib = sepes._src.backend.treelib
    _, atomicdef = treelib.flatten(1)

    # mainly used for visualization
    def flatten_one_level(type_path: KeyTypePath, tree: PyTree):
        # predicate and type path
        if (is_leaf and is_leaf(tree)) or (is_path_leaf and is_path_leaf(type_path)):
            yield type_path, tree
            return

        def one_level_is_leaf(node) -> bool:
            if is_leaf and is_leaf(node):
                return True
            if id(node) == id(tree):
                return False
            return True

        path_leaf, treedef = treelib.path_flatten(tree, is_leaf=one_level_is_leaf)

        if treedef == atomicdef:
            yield type_path, tree
            return

        for key, value in path_leaf:
            keys, types = type_path
            path = ((*keys, *key), (*types, type(value)))
            yield from flatten_one_level(path, value)

    return list(flatten_one_level(((), ()), tree))


class Node:
    # mainly used for visualization
    __slots__ = ["data", "parent", "children", "__weakref__"]

    def __init__(
        self,
        data: tuple[TraceEntry, Any],
        parent: Node | None = None,
    ):
        self.data = data
        self.parent = parent
        self.children: dict[TraceEntry, Node] = {}

    def add_child(self, child: Node) -> None:
        # add child node to this node and set
        # this node as the parent of the child
        if not isinstance(child, Node):
            raise TypeError(f"`child` must be a `Node`, got {type(child)}")
        ti, _ = child.data
        if ti not in self.children:
            # establish parent-child relationship
            child.parent = self
            self.children[ti] = child

    def __iter__(self) -> Iterator[Node]:
        # iterate over children nodes
        return iter(self.children.values())

    def __repr__(self) -> str:
        return f"Node(data={self.data})"

    def __contains__(self, key: TraceEntry) -> bool:
        return key in self.children


def is_path_leaf_depth_factory(depth: int | float):
    # generate `is_path_leaf` function to stop tracing at a certain `depth`
    # in essence, depth is the length of the trace entry
    def is_path_leaf(trace) -> bool:
        keys, _ = trace
        # stop tracing if depth is reached
        return False if depth is None else (depth <= len(keys))

    return is_path_leaf


def construct_tree(
    tree: PyTree,
    is_leaf: Callable[[Any], bool] | None = None,
    is_path_leaf: Callable[[KeyTypePath], bool] | None = None,
) -> Node:
    # construct a tree with `Node` objects using `tree_type_path_leaves`
    # to establish parent-child relationship between nodes

    traces_leaves = tree_type_path_leaves(
        tree,
        is_leaf=is_leaf,
        is_path_leaf=is_path_leaf,
    )

    ti = (None, type(tree))
    vi = tree
    root = Node(data=(ti, vi))

    for trace, leaf in traces_leaves:
        keys, types = trace
        cur = root
        for i, ti in enumerate(zip(keys, types)):
            if ti in cur:
                # common parent node
                cur = cur.children[ti]
            else:
                # new path
                vi = leaf if i == len(keys) - 1 else None
                child = Node(data=(ti, vi))
                cur.add_child(child)
                cur = child
    return root


def value_and_tree(func: Callable[..., T], argnums: int | Sequence[int] = 0):
    """Call a function on copied input argument and return the value and the tree.

    Input arguments are copied before calling the function, and the argument
    specified by ``argnums`` are returned as a tree.

    Args:
        func: A function.
        argnums: The argument number of the tree that will be returned. If multiple
            arguments are specified, the tree will be returned as a tuple.

    Returns:
        A function that returns the value and the tree.

    Example:
        Usage with mutable types:

        >>> import sepes as sp
        >>> mutable_tree = [1, 2, 3]
        >>> def mutating_func(tree):
        ...     tree[0] += 100
        ...     return tree
        >>> new_tree = mutating_func(mutable_tree)
        >>> assert new_tree is mutable_tree
        >>> # now with `value_and_tree` the function does not mutate the tree
        >>> new_tree, _ = sp.value_and_tree(mutating_func)(mutable_tree)
        >>> assert new_tree is not mutable_tree

    Example:
        Usage with immutable types (:class:`.TreeClass`) with support for in-place
        mutation via custom behavior registration using :func:`.value_and_tree.def_mutator`
        and :func:`.value_and_tree.def_immutator`:

        >>> import sepes as sp
        >>> class Counter(sp.TreeClass):
        ...     def __init__(self, count: int):
        ...         self.count = count
        ...     def increment(self, value):
        ...         self.count += value
        ...         return self.count
        >>> counter = Counter(0)
        >>> counter.increment(1)  # doctest: +SKIP
        AttributeError: Cannot set attribute value=1 to `key='count'`  on an immutable instance of `Counter`.
        >>> sp.value_and_tree(lambda counter: counter.increment(1))(counter)
        (1, Counter(count=1))

    Note:
        Use this function on function that:

        - Mutates the input arguments of mutable types (e.g. lists, dicts, etc.).
        - Mutates the input arguments of immutable types that do not support in-place
          mutation and needs special handling that can be registered (e.g. :class:`.TreeClass`)
          using :func:`.value_and_tree.def_mutator` and :func:`.value_and_tree.def_immutator`.

    Note:
        The default behavior of :func:`value_and_tree` is to copy the input
        arguments and then call the function on the copy. However if the function
        mutates some of the input arguments that does not support in-place mutation,
        then the function will fail. In this case, :func:`value_and_tree` enables
        registering custom behavior that modifies the copied input argument to
        allow in-place mutation. and custom function that restores the copied
        argument to its original state after the method call. The following example
        shows how to register custom functions for a simple class that allows
        in-place mutation if ``immutable`` Flag is set to ``False``.

        >>> import jax
        >>> from jax.util import unzip2
        >>> import sepes as sp
        >>> @jax.tree_util.register_pytree_node_class
        ... class MyNode:
        ...     def __init__(self):
        ...         self.counter = 0
        ...         self.immutable = True
        ...     def tree_flatten(self):
        ...         keys, values = unzip2(vars(self).items())
        ...         return tuple(values), tuple(keys)
        ...     @classmethod
        ...     def tree_unflatten(cls, keys, values):
        ...         self = object.__new__(cls)
        ...         vars(self).update(dict(zip(keys, values)))
        ...         return self
        ...     def __setattr__(self, name, value):
        ...         if getattr(self, "immutable", False) is True:
        ...             raise AttributeError("MyNode is immutable")
        ...         object.__setattr__(self, name, value)
        ...     def __repr__(self):
        ...         params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        ...         return f"MyNode({params})"
        ...     def increment(self) -> None:
        ...         self.counter += 1
        >>> @sp.value_and_tree.def_mutator(MyNode)
        ... def mutable(node) -> None:
        ...     vars(node)["immutable"] = False
        >>> @sp.value_and_tree.def_immutator(MyNode)
        ... def immutable(node) -> None:
        ...     vars(node)["immutable"] = True
        >>> node = MyNode()
        >>> sp.value_and_tree(lambda node: node.increment())(node)
        (None, MyNode(counter=1, immutable=True))
    """
    treelib = sepes._src.backend.treelib
    is_int_argnum = isinstance(argnums, int)
    argnums = [argnums] if is_int_argnum else argnums

    def mutate_is_leaf(node):
        value_and_tree.mutator_dispatcher(node)
        return False

    def immutate_is_leaf(node):
        value_and_tree.immutator_dispatcher(node)
        return False

    @ft.wraps(func)
    def stateless_func(*args, **kwargs) -> tuple[T, PyTree | tuple[PyTree, ...]]:
        # copy the incoming inputs
        (args, kwargs) = tree_copy((args, kwargs))
        # and edit the node/record to make it mutable (if there is a rule for it)
        treelib.map(lambda _: _, (args, kwargs), is_leaf=mutate_is_leaf)
        output = func(*args, **kwargs)
        # traverse each node in the tree depth-first manner
        # to undo the mutation (if there is a rule for it)
        treelib.map(lambda _: _, (args, kwargs), is_leaf=immutate_is_leaf)
        out_args = tuple(a for i, a in enumerate(args) if i in argnums)
        out_args = out_args[0] if is_int_argnum else out_args
        return output, out_args

    return stateless_func


value_and_tree.mutator_dispatcher = ft.singledispatch(lambda node: node)
value_and_tree.immutator_dispatcher = ft.singledispatch(lambda node: node)
value_and_tree.def_mutator = value_and_tree.mutator_dispatcher.register
value_and_tree.def_immutator = value_and_tree.immutator_dispatcher.register


if is_package_avaiable("jax"):
    import jax

    # basically avoid calling copy on jax arrays because they
    # are immutable by default
    @tree_copy.def_type(jax.Array)
    def _(node: jax.Array) -> jax.Array:
        return node

    # avoid calling __copy__ on jitted functions becasue they loses their
    # wrapped function attributes (maybe a bug in jax)
    @tree_copy.def_type(type(jax.jit(lambda x: x)))
    def _(node: T1) -> T1:
        return node
