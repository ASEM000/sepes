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

"""Define lens-like indexing/masking for pytrees."""

# enable get/set/apply/scan/reduce operations on selected parts of a nested
# structure -pytree- in out-of-place manner. this process invovles defining two
# parts: 1) *where* to select the parts of the pytree and 2) *what* to do with
# the selected parts. the *where* part is defined either by a path or a boolean
# mask. the *what* part is defined by a set value, or a function to apply to
# the selected parts. once we have a *final* boolean mask that encompasses all
# path and the boolean mask, we can use `tree_map` to apply the *what* part to
# the *where* part. for example, for a tree = [[1, 2], 3, 4] and boolean mask
# [[True, False], False, True] and path mask [0][1], then we select only leaf
# 1 that is at the intersection of the boolean mask and the path mask. then we
# apply the *what* part to the *where* part.

from __future__ import annotations

import abc
import functools as ft
import re
from typing import Any, Callable, Hashable, NamedTuple, Tuple, TypeVar

from sepes._src.backend import arraylib, treelib
from sepes._src.backend.treelib.base import ParallelConfig

T = TypeVar("T")
S = TypeVar("S")
PyTree = Any
EllipsisType = TypeVar("EllipsisType")
KeyEntry = TypeVar("KeyEntry", bound=Hashable)
TypeEntry = TypeVar("TypeEntry", bound=type)
TraceEntry = Tuple[KeyEntry, TypeEntry]
KeyPath = Tuple[KeyEntry, ...]
TypePath = Tuple[TypeEntry, ...]
TraceType = Tuple[KeyPath, TypePath]
_no_initializer = object()

SequenceKeyType = type(treelib.sequence_key(0))
DictKeyType = type(treelib.dict_key("key"))
GetAttrKeyType = type(treelib.attribute_key("name"))


class BaseKey(abc.ABC):
    """Parent class for all match classes.

    - Subclass this class to create custom match keys by implementing
      the `__eq__` method. The ``__eq__`` method should return True if the
      key matches the given path entry and False otherwise. The path entry
      refers to the entry defined in the ``tree_flatten_with_keys`` method of
      the pytree class.

    - Typical path entries in ``jax`` are:

        - ``jax.tree_util.GetAttrKey`` for attributes
        - ``jax.tree_util.DictKey`` for mapping keys
        - ``jax.tree_util.SequenceKey`` for sequence indices

    - When implementing the ``__eq__`` method you can use the ``singledispatchmethod``
      to unpack the path entry for example:

        - ``jax.tree_util.GetAttrKey`` -> `key.name`
        - ``jax.tree_util.DictKey`` -> `key.key`
        - ``jax.tree_util.SequenceKey`` -> `key.index`


        See Examples for more details.

    Example:
        >>> # define an match strategy to match a leaf with a given name and type
        >>> import sepes as sp
        >>> from typing import NamedTuple
        >>> import jax
        >>> class NameTypeContainer(NamedTuple):
        ...     name: str
        ...     type: type
        >>> @jax.tree_util.register_pytree_with_keys_class
        ... class Tree:
        ...    def __init__(self, a, b) -> None:
        ...        self.a = a
        ...        self.b = b
        ...    def tree_flatten_with_keys(self):
        ...        ak = (NameTypeContainer("a", type(self.a)), self.a)
        ...        bk = (NameTypeContainer("b", type(self.b)), self.b)
        ...        return (ak, bk), None
        ...    @classmethod
        ...    def tree_unflatten(cls, aux_data, children):
        ...        return cls(*children)
        ...    @property
        ...    def at(self):
        ...        return sp.AtIndexer(self)
        >>> tree = Tree(1, 2)
        >>> class MatchNameType(sp.BaseKey):
        ...    def __init__(self, name, type):
        ...        self.name = name
        ...        self.type = type
        ...    def __eq__(self, other):
        ...        if isinstance(other, NameTypeContainer):
        ...            return other == (self.name, self.type)
        ...        return False
        >>> tree = tree.at[MatchNameType("a", int)].get()
        >>> assert jax.tree_util.tree_leaves(tree) == [1]

    Note:
        - use ``BaseKey.def_alias(type, func)`` to define an index type alias
          for `BaseKey` subclasses. This is useful for convience when
          creating new match strategies.

            >>> import sepes as sp
            >>> import functools as ft
            >>> from types import FunctionType
            >>> import jax.tree_util as jtu
            >>> # lets define a new match strategy called `FuncKey` that applies
            >>> # a function to the path entry and returns True if the function
            >>> # returns True and False otherwise.
            >>> # for example `FuncKey(lambda x: x.startswith("a"))` will match
            >>> # all leaves that start with "a".
            >>> class FuncKey(sp.BaseKey):
            ...    def __init__(self, func):
            ...        self.func = func
            ...    @ft.singledispatchmethod
            ...    def __eq__(self, key):
            ...        return self.func(key)
            ...    @__eq__.register(jtu.GetAttrKey)
            ...    def _(self, key: jtu.GetAttrKey):
            ...        # unpack the GetAttrKey
            ...        return self.func(key.name)
            ...    @__eq__.register(jtu.DictKey)
            ...    def _(self, key: jtu.DictKey):
            ...        # unpack the DictKey
            ...        return self.func(key.key)
            ...    @__eq__.register(jtu.SequenceKey)
            ...    def _(self, key: jtu.SequenceKey):
            ...        return self.func(key.index)
            >>> # instead of using ``FuncKey(function)`` we can define an alias
            >>> # for `FuncKey`, for this example we will define any FunctionType
            >>> # as a `FuncKey` by default.
            >>> @sp.BaseKey.def_alias(FunctionType)
            ... def _(func):
            ...    return FuncKey(func)
            >>> # create a simple pytree
            >>> @sp.autoinit
            ... class Tree(sp.TreeClass):
            ...    a: int
            ...    b: str
            >>> tree = Tree(1, "string")
            >>> # now we can use the `FuncKey` alias to match all leaves that
            >>> # are strings and start with "a"
            >>> tree.at[lambda x: isinstance(x, str) and x.startswith("a")].get()
            Tree(a=1, b=None)
    """

    @abc.abstractmethod
    def __eq__(self, entry: KeyEntry) -> bool:
        pass


class IntKey(BaseKey):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    @ft.singledispatchmethod
    def __eq__(self, _: KeyEntry) -> bool:
        return False

    @__eq__.register(int)
    def _(self, other: int) -> bool:
        return self.idx == other

    @__eq__.register(SequenceKeyType)
    def _(self, other: SequenceKeyType) -> bool:
        return self.idx == other.idx


class NameKey(BaseKey):
    def __init__(self, name: str) -> None:
        self.name = name

    @ft.singledispatchmethod
    def __eq__(self, _: KeyEntry) -> bool:
        return False

    @__eq__.register(str)
    def _(self, other: str) -> bool:
        return self.name == other

    @__eq__.register(GetAttrKeyType)
    def _(self, other: GetAttrKeyType) -> bool:
        return self.name == other.name

    @__eq__.register(DictKeyType)
    def _(self, other: DictKeyType) -> bool:
        return self.name == other.key


class EllipsisKey(BaseKey):
    """Match all leaves."""

    def __init__(self, _):
        del _

    def __eq__(self, _: KeyEntry) -> bool:
        return True


class MultiKey(BaseKey):
    """Match a leaf with multiple keys at the same level."""

    def __init__(self, *keys: tuple[BaseKey, ...]):
        self.keys = tuple(keys)

    def __eq__(self, entry) -> bool:
        return any(entry == key for key in self.keys)


class RegexKey(BaseKey):
    """Match a leaf with a regex pattern inside 'at' property.

    Args:
        pattern: regex pattern to match.

    Example:
        >>> import sepes as sp
        >>> import re
        >>> @sp.autoinit
        ... class Tree(sp.TreeClass):
        ...     weight_1: float = 1.0
        ...     weight_2: float = 2.0
        ...     weight_3: float = 3.0
        ...     bias: float = 0.0
        >>> tree = Tree()
        >>> tree.at[re.compile(r"weight_.*")].set(100.0)  # set all weights to 100.0
        Tree(weight_1=100.0, weight_2=100.0, weight_3=100.0, bias=0.0)
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    @ft.singledispatchmethod
    def __eq__(self, _: KeyEntry) -> bool:
        return False

    @__eq__.register(str)
    def _(self, other: str) -> bool:
        return re.fullmatch(self.pattern, other) is not None

    @__eq__.register(GetAttrKeyType)
    def _(self, other) -> bool:
        return re.fullmatch(self.pattern, other.name) is not None

    @__eq__.register(DictKeyType)
    def _(self, other) -> bool:
        return re.fullmatch(self.pattern, other.key) is not None


# dispatch on type of indexer to convert input item to at indexer
# `__getitem__` to the appropriate key
# avoid using container pytree types to avoid conflict between
# matching as a mask or as an instance of `BaseKey`
indexer_dispatcher = ft.singledispatch(lambda x: x)
indexer_dispatcher.register(type(...), EllipsisKey)
indexer_dispatcher.register(int, IntKey)
indexer_dispatcher.register(str, NameKey)
indexer_dispatcher.register(re.Pattern, RegexKey)

BaseKey.def_alias = indexer_dispatcher.register


_NOT_IMPLEMENTED_INDEXING = """Indexing with {} is not implemented, supported indexing types are:
- `str` for mapping keys or class attributes.
- `int` for positional indexing for sequences.
- `...` to select all leaves.
- Boolean mask of the same structure as the tree
- `re.Pattern` to index all keys matching a regex pattern.
- Instance of `BaseKey` with custom logic to index a pytree.
- `tuple` of the above types to match multiple leaves at the same level.
"""


def _generate_path_mask(
    tree: PyTree,
    where: tuple[BaseKey, ...],
    is_leaf: Callable[[Any], None] | None = None,
) -> PyTree:
    # generate a boolean mask for `where` path in `tree`
    # where path is a tuple of indices or keys, for example
    # where=("a",) wil set all leaves of `tree` with key "a" to True and
    # all other leaves to False
    match = False

    def map_func(path, _: Any):
        if len(where) > len(path):
            # path is shorter than `where` path. for example
            # where=("a", "b") and the current path is ("a",) then
            # the current path is not a match
            return False
        for wi, ki in zip(where, path):
            if not (wi == ki):
                return False

        nonlocal match
        match = True
        return match

    mask = treelib.tree_path_map(map_func, tree, is_leaf=is_leaf)

    if not match:
        raise LookupError(f"No leaf match is found for {where=}.")

    return mask


def _combine_bool_leaves(*leaves):
    verdict = True
    for leaf in leaves:
        verdict &= leaf
    return verdict


def _is_bool_leaf(leaf: Any) -> bool:
    if isinstance(leaf, arraylib.ndarray):
        return arraylib.is_bool(leaf)
    return isinstance(leaf, bool)


def _resolve_where(
    tree: T,
    where: tuple[Any, ...],  # type: ignore
    is_leaf: Callable[[Any], None] | None = None,
) -> T | None:
    # given a pytree `tree` and a `where` path, that is composed of keys or
    # boolean masks, generate a boolean mask that will be eventually used to
    # with `tree_map` to select the leaves at the specified location.
    mask = None
    bool_masks: list[T] = []
    path_masks: list[BaseKey] = []
    _, treedef0 = treelib.tree_flatten(tree, is_leaf=is_leaf)
    seen_tuple = False  # handle multiple keys at the same level
    level_paths = []

    def verify_and_aggregate_is_leaf(x) -> bool:
        # use is_leaf with non-local to traverse the tree depth-first manner
        # required for verifying if a pytree is a valid indexing pytree
        nonlocal seen_tuple, level_paths, bool_masks
        # used to check if a pytree is a valid indexing pytree
        # used with `is_leaf` argument of any `tree_*` function
        leaves, treedef = treelib.tree_flatten(x)

        if treedef == treedef0 and all(map(_is_bool_leaf, leaves)):
            # boolean pytrees of same structure as `tree` is a valid indexing pytree
            bool_masks += [x]
            return True

        if isinstance(resolved_key := indexer_dispatcher(x), BaseKey):
            # valid resolution of `BaseKey` is a valid indexing leaf
            # makes it possible to dispatch on multi-leaf pytree
            level_paths += [resolved_key]
            return False

        if type(x) is tuple and seen_tuple is False:
            # e.g. `at[1,2,3]` but not `at[1,(2,3)]``
            seen_tuple = True
            return False

        # not a container of other keys or a pytree of same structure
        raise NotImplementedError(_NOT_IMPLEMENTED_INDEXING.format(x))

    for level_keys in where:
        # each for loop iteration is a level in the where path
        # this means that if where = ("a", "b", "c") then this means
        # we are travering the tree at level "a" then level "b" then level "c"
        treelib.tree_flatten(level_keys, is_leaf=verify_and_aggregate_is_leaf)
        # if len(level_paths) > 1 then this means that we have multiple keys
        # at the same level, for example where = ("a", ("b", "c")) then this
        # means that for a parent "a", select "b" and "c".
        path_masks += [MultiKey(*level_paths)] if len(level_paths) > 1 else level_paths
        level_paths = []
        seen_tuple = False

    if path_masks:
        mask = _generate_path_mask(tree, path_masks, is_leaf=is_leaf)

    if bool_masks:
        all_masks = [mask, *bool_masks] if mask else bool_masks
        mask = treelib.tree_map(_combine_bool_leaves, *all_masks)

    return mask


class AtIndexer(NamedTuple):
    """Index a pytree at a given path using a path or mask.

    Args:
        tree: pytree to index
        where: one of the following:

            - ``str`` for mapping keys or class attributes.
            - ``int`` for positional indexing for sequences.
            - ``...`` to select all leaves.
            - a boolean mask of the same structure as the tree
            - ``re.Pattern`` to index all keys matching a regex pattern.
            - an instance of ``BaseKey`` with custom logic to index a pytree.
            - a tuple of the above to match multiple keys at the same level.

    Example:
        >>> # use `AtIndexer` on a pytree (e.g. dict,list,tuple,esp.)
        >>> import jax
        >>> import sepes as sp
        >>> tree = {"level1_0": {"level2_0": 100, "level2_1": 200}, "level1_1": 300}
        >>> indexer = sp.AtIndexer(tree)
        >>> indexer["level1_0"]["level2_0"].get()
        {'level1_0': {'level2_0': 100, 'level2_1': None}, 'level1_1': None}
        >>> # get multiple keys at once at the same level
        >>> indexer["level1_0"]["level2_0", "level2_1"].get()
        {'level1_0': {'level2_0': 100, 'level2_1': 200}, 'level1_1': None}
        >>> # get with a mask
        >>> mask = {"level1_0": {"level2_0": True, "level2_1": False}, "level1_1": True}
        >>> indexer[mask].get()
        {'level1_0': {'level2_0': 100, 'level2_1': None}, 'level1_1': 300}

    Example:
        >>> # use ``AtIndexer`` in a class
        >>> import jax.tree_util as jtu
        >>> import sepes as sp
        >>> @jax.tree_util.register_pytree_with_keys_class
        ... class Tree:
        ...    def __init__(self, a, b):
        ...        self.a = a
        ...        self.b = b
        ...    def tree_flatten_with_keys(self):
        ...        kva = (jtu.GetAttrKey("a"), self.a)
        ...        kvb = (jtu.GetAttrKey("b"), self.b)
        ...        return (kva, kvb), None
        ...    @classmethod
        ...    def tree_unflatten(cls, aux_data, children):
        ...        return cls(*children)
        ...    @property
        ...    def at(self):
        ...        return sp.AtIndexer(self)
        ...    def __repr__(self) -> str:
        ...        return f"{self.__class__.__name__}(a={self.a}, b={self.b})"
        >>> Tree(1, 2).at["a"].get()
        Tree(a=1, b=None)
    """

    tree: PyTree
    where: tuple[BaseKey | PyTree] | tuple[()] = ()

    def __getitem__(self, where: Any) -> AtIndexer:
        # AtIndexer[where] will extend the current path with `where`
        # for example AtIndexer[where1][where2] will extend the current path
        # with `where1` and `where2` to indicate the path to the leaves to
        # select.
        return type(self)(self.tree, (*self.where, where))

    def get(
        self,
        *,
        is_leaf: Callable[[Any], None] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> PyTree:
        """Get the leaf values at the specified location.

        Args:
            is_leaf: a predicate function to determine if a value is a leaf.
            is_parallel: accepts the following:

                - ``bool``: apply ``func`` in parallel if ``True`` otherwise in serial.
                - ``dict``: a dict of of:
                    - ``max_workers``: maximum number of workers to use.
                    - ``kind``: kind of pool to use, either ``thread`` or ``process``.

        Returns:
            A _new_ pytree of leaf values at the specified location, with the
            non-selected leaf values set to None if the leaf is not an array.

        Example:
            >>> import sepes as sp
            >>> tree = {"a": 1, "b": [1, 2, 3]}
            >>> indexer = sp.AtIndexer(tree)  # construct an indexer
            >>> indexer["b"][0].get()  # get the first element of "b"
            {'a': None, 'b': [1, None, None]}

        Example:
            >>> import sepes as sp
            >>> @sp.autoinit
            ... class Tree(sp.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> # get ``a`` and return a new instance
            >>> # with ``None`` for all other leaves
            >>> tree.at['a'].get()
            Tree(a=1, b=None)
        """
        where = _resolve_where(self.tree, self.where, is_leaf)
        config = dict(is_leaf=is_leaf, is_parallel=is_parallel)

        def leaf_get(leaf: Any, where: Any):
            # support both array and non-array leaves
            # for array boolean mask we select **parts** of the array that
            # matches the mask, for example if the mask is Array([True, False, False])
            # and the leaf is Array([1, 2, 3]) then the result is Array([1])
            if isinstance(where, arraylib.ndarray) and arraylib.ndim(where) != 0:
                return leaf[where]
            # non-array boolean mask we select the leaf if the mask is True
            # and `None` otherwise
            return leaf if where else None

        return treelib.tree_map(leaf_get, self.tree, where, **config)

    def set(
        self,
        set_value: Any,
        *,
        is_leaf: Callable[[Any], None] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> PyTree:
        """Set the leaf values at the specified location.

        Args:
            set_value: the value to set at the specified location.
            is_leaf: a predicate function to determine if a value is a leaf.
            is_parallel: accepts the following:

                - ``bool``: apply ``func`` in parallel if ``True`` otherwise in serial.
                - ``dict``: a dict of of:
                    - ``max_workers``: maximum number of workers to use.
                    - ``kind``: kind of pool to use, either ``thread`` or ``process``.

        Returns:
            A pytree with the leaf values at the specified location
            set to ``set_value``.

        Example:
            >>> import sepes as sp
            >>> tree = {"a": 1, "b": [1, 2, 3]}
            >>> indexer = sp.AtIndexer(tree)
            >>> indexer["b"][0].set(100)  # set the first element of "b" to 100
            {'a': 1, 'b': [100, 2, 3]}

        Example:
            >>> import sepes as sp
            >>> @sp.autoinit
            ... class Tree(sp.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> # set ``a`` and return a new instance
            >>> # with all other leaves unchanged
            >>> tree.at['a'].set(100)
            Tree(a=100, b=2)
        """
        where = _resolve_where(self.tree, self.where, is_leaf)
        config = dict(is_leaf=is_leaf, is_parallel=is_parallel)

        def leaf_set(leaf: Any, where: Any, set_value: Any):
            # support both array and non-array leaves
            # for array boolean mask we select **parts** of the array that
            # matches the mask, for example if the mask is Array([True, False, False])
            # and the leaf is Array([1, 2, 3]) then the result is Array([1, 100, 100])
            # with set_value = 100
            if isinstance(where, arraylib.ndarray):
                return arraylib.where(where, set_value, leaf)
            return set_value if where else leaf

        _, lhsdef = treelib.tree_flatten(self.tree, is_leaf=is_leaf)
        _, rhsdef = treelib.tree_flatten(set_value, is_leaf=is_leaf)

        if lhsdef == rhsdef:
            # do not broadcast set_value if it is a pytree of same structure
            # for example tree.at[where].set(tree2) will set all tree leaves
            # to tree2 leaves if tree2 is a pytree of same structure as tree
            # instead of making each leaf of tree a copy of tree2
            # is design is similar to ``numpy`` design `np.at[...].set(Array)`
            return treelib.tree_map(leaf_set, self.tree, where, set_value, **config)

        # set_value is broadcasted to tree leaves
        # for example tree.at[where].set(1) will set all tree leaves to 1
        leaf_set_ = lambda leaf, where: leaf_set(leaf, where, set_value)
        return treelib.tree_map(leaf_set_, self.tree, where, **config)

    def apply(
        self,
        func: Callable[[Any], Any],
        *,
        is_leaf: Callable[[Any], None] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> PyTree:
        """Apply a function to the leaf values at the specified location.

        Args:
            func: the function to apply to the leaf values.
            is_leaf: a predicate function to determine if a value is a leaf.
            is_parallel: accepts the following:

                - ``bool``: apply ``func`` in parallel if ``True`` otherwise in serial.
                - ``dict``: a dict of of:
                    - ``max_workers``: maximum number of workers to use.
                    - ``kind``: kind of pool to use, either ``thread`` or ``process``.

        Returns:
            A pytree with the leaf values at the specified location set to
            the result of applying ``func`` to the leaf values.

        Example:
            >>> import sepes as sp
            >>> tree = {"a": 1, "b": [1, 2, 3]}
            >>> indexer = sp.AtIndexer(tree)
            >>> indexer["b"][0].apply(lambda x: x + 100)  # add 100 to the first element of "b"
            {'a': 1, 'b': [101, 2, 3]}

        Example:
            >>> import sepes as sp
            >>> @sp.autoinit
            ... class Tree(sp.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> # apply to ``a`` and return a new instance
            >>> # with all other leaves unchanged
            >>> tree.at['a'].apply(lambda _: 100)
            Tree(a=100, b=2)

        Example:
            >>> # read images in parallel
            >>> import sepes as sp
            >>> from matplotlib.pyplot import imread
            >>> indexer = sp.AtIndexer({"lenna": "lenna.png", "baboon": "baboon.png"})
            >>> images = indexer[...].apply(imread, parallel=dict(max_workers=2))  # doctest: +SKIP
        """
        where = _resolve_where(self.tree, self.where, is_leaf)
        config = dict(is_leaf=is_leaf, is_parallel=is_parallel)

        def leaf_apply(leaf: Any, where: bool):
            # same as `leaf_set` but with `func` applied to the leaf
            # one thing to note is that, the where mask select an array
            # then the function needs work properly when applied to the selected
            # array elements
            if isinstance(where, arraylib.ndarray):
                return arraylib.where(where, func(leaf), leaf)
            return func(leaf) if where else leaf

        return treelib.tree_map(leaf_apply, self.tree, where, **config)

    def scan(
        self,
        func: Callable[[Any, S], tuple[Any, S]],
        state: S,
        *,
        is_leaf: Callable[[Any], None] | None = None,
    ) -> tuple[PyTree, S]:
        """Apply a function while carrying a state.

        Args:
            func: the function to apply to the leaf values. the function accepts
                a running state and leaf value and returns a tuple of the new
                leaf value and the new state.
            state: the initial state to carry.
            is_leaf: a predicate function to determine if a value is a leaf. for
                example, ``lambda x: isinstance(x, list)`` will treat all lists
                as leaves and will not recurse into list items.

        Returns:
            A tuple of the final state and pytree with the leaf values at the
            specified location set to the result of applying ``func`` to the leaf
            values.

        Example:
            >>> import sepes as sp
            >>> tree = {"level1_0": {"level2_0": 100, "level2_1": 200}, "level1_1": 300}
            >>> def scan_func(leaf, state):
            ...     return 'SET', state + 1
            >>> init_state = 0
            >>> indexer = sp.AtIndexer(tree)
            >>> indexer["level1_0"]["level2_0"].scan(scan_func, state=init_state)
            ({'level1_0': {'level2_0': 'SET', 'level2_1': 200}, 'level1_1': 300}, 1)

        Example:
            >>> import sepes as sp
            >>> from typing import NamedTuple
            >>> class State(NamedTuple):
            ...     func_evals: int = 0
            >>> @sp.autoinit
            ... class Tree(sp.TreeClass):
            ...     a: int
            ...     b: int
            ...     c: int
            >>> tree = Tree(a=1, b=2, c=3)
            >>> def scan_func(leaf, state: State):
            ...     state = State(state.func_evals + 1)
            ...     return leaf + 1, state
            >>> # apply to ``a`` and ``b`` and return a new instance with all other
            >>> # leaves unchanged and the new state that counts the number of
            >>> # function evaluations
            >>> tree.at['a','b'].scan(scan_func, state=State())
            (Tree(a=2, b=3, c=3), State(func_evals=2))

        Note:
            ``scan`` applies a binary ``func`` to the leaf values while carrying
            a state and returning a tree leaves with the the ``func`` applied to
            them with final state. While ``reduce`` applies a binary ``func`` to the
            leaf values while carrying a state and returning a single value.
        """
        where = _resolve_where(self.tree, self.where, is_leaf)

        running_state = state

        def stateless_func(leaf):
            nonlocal running_state
            leaf, running_state = func(leaf, running_state)
            return leaf

        def leaf_apply(leaf: Any, where: bool):
            if isinstance(where, arraylib.ndarray):
                return arraylib.where(where, stateless_func(leaf), leaf)
            return stateless_func(leaf) if where else leaf

        out = treelib.tree_map(leaf_apply, self.tree, where, is_leaf=is_leaf)
        return out, running_state

    def reduce(
        self,
        func: Callable[[Any, Any], Any],
        *,
        initializer: Any = _no_initializer,
        is_leaf: Callable[[Any], None] | None = None,
    ) -> Any:
        """Reduce the leaf values at the specified location.

        Args:
            func: the function to reduce the leaf values.
            initializer: the initializer value for the reduction.
            is_leaf: a predicate function to determine if a value is a leaf.

        Returns:
            The result of reducing the leaf values at the specified location.

        Note:
            - If ``initializer`` is not specified, the first leaf value is used as
              the initializer.
            - ``reduce`` applies a binary ``func`` to each leaf values while accumulating
              a state a returns the final result. while ``scan`` applies ``func`` to each
              leaf value while carrying a state and returns the final state and
              the leaves of the tree with the result of applying ``func`` to each leaf.

        Example:
            >>> import sepes as sp
            >>> @sp.autoinit
            ... class Tree(sp.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> tree.at[...].reduce(lambda a, b: a + b, initializer=0)
            3
        """
        where = _resolve_where(self.tree, self.where, is_leaf)
        tree = self[where].get(is_leaf=is_leaf)  # type: ignore
        leaves, _ = treelib.tree_flatten(tree, is_leaf=is_leaf)
        if initializer is _no_initializer:
            return ft.reduce(func, leaves)
        return ft.reduce(func, leaves, initializer)
