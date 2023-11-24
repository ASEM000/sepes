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
from typing import Any, Callable, Hashable, Tuple, TypeVar, Generic

from typing_extensions import Self

import sepes
import sepes._src.backend.arraylib as arraylib
from sepes._src.backend.treelib import ParallelConfig
from types import SimpleNamespace
from sepes._src.tree_util import tree_copy

T = TypeVar("T")
S = TypeVar("S")
PyTree = Any
EllipsisType = TypeVar("EllipsisType")
KeyEntry = TypeVar("KeyEntry", bound=Hashable)
KeyPath = Tuple[KeyEntry, ...]
_no_initializer = object()


def recursive_getattr(tree: Any, where: tuple[str, ...]):
    # used to fetch methods from a class defined by path mask
    if not isinstance(where[0], str):
        raise TypeError(f"Expected string, got {type(where[0])!r}.")
    if len(where) == 1:
        return getattr(tree, where[0])
    return recursive_getattr(getattr(tree, where[0]), where[1:])


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

    broadcastable: bool = False


class IndexKey(BaseKey):
    """Match a leaf with a given index."""

    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __eq__(self, key: KeyEntry) -> bool:
        if isinstance(key, int):
            return self.idx == key
        treelib = sepes._src.backend.treelib
        if isinstance(key, type(treelib.sequence_key(0))):
            return self.idx == key.idx
        return False

    def __repr__(self) -> str:
        return f"{self.idx}"


class NameKey(BaseKey):
    """Match a leaf with a given key."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, key: KeyEntry) -> bool:
        if isinstance(key, str):
            return self.name == key
        treelib = sepes._src.backend.treelib
        if isinstance(key, type(treelib.attribute_key(""))):
            return self.name == key.name
        if isinstance(key, type(treelib.dict_key(""))):
            return self.name == key.key
        return False

    def __repr__(self) -> str:
        return f"{self.name}"


class EllipsisKey(BaseKey):
    """Match all leaves."""

    broadcastable = True

    def __init__(self, _):
        del _

    def __eq__(self, _: KeyEntry) -> bool:
        return True

    def __repr__(self) -> str:
        return "..."


class MultiKey(BaseKey):
    """Match a leaf with multiple keys at the same level."""

    def __init__(self, *keys: tuple[BaseKey, ...]):
        self.keys = tuple(keys)

    def __eq__(self, entry) -> bool:
        return any(entry == key for key in self.keys)

    def __repr__(self) -> str:
        return f"({', '.join(map(repr, self.keys))})"


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

    def __eq__(self, key: KeyEntry) -> bool:
        if isinstance(key, str):
            return re.fullmatch(self.pattern, key) is not None
        treelib = sepes._src.backend.treelib
        if isinstance(key, type(treelib.attribute_key(""))):
            return re.fullmatch(self.pattern, key.name) is not None
        if isinstance(key, type(treelib.dict_key(""))):
            return re.fullmatch(self.pattern, key.key) is not None
        return False

    def __repr__(self) -> str:
        return f"{self.pattern}"


# dispatch on type of indexer to convert input item to at indexer
# `__getitem__` to the appropriate key
# avoid using container pytree types to avoid conflict between
# matching as a mask or as an instance of `BaseKey`
indexer_dispatcher = ft.singledispatch(lambda x: x)
indexer_dispatcher.register(type(...), EllipsisKey)
indexer_dispatcher.register(int, IndexKey)
indexer_dispatcher.register(str, NameKey)
indexer_dispatcher.register(re.Pattern, RegexKey)

BaseKey.def_alias = indexer_dispatcher.register


_INVALID_INDEXER = """\
Indexing with {indexer} is not implemented, supported indexing types are:
  - `str` for mapping keys or class attributes.
  - `int` for positional indexing for sequences.
  - `...` to select all leaves.
  - Boolean mask of a compatible structure as the pytree.
  - `re.Pattern` to index all keys matching a regex pattern.
  - Instance of `BaseKey` with custom logic to index a pytree.
  - `tuple` of the above types to match multiple leaves at the same level.
"""

_NO_LEAF_MATCH = """\
No leaf match is found for where={where}. Available keys are {names}.
Check the following: 
  - If where is `str` then check if the key exists as a key or attribute.
  - If where is `int` then check if the index is in range.
  - If where is `re.Pattern` then check if the pattern matches any key.
  - If where is a `tuple` of the above types then check if any of the tuple elements match.
"""


def generate_path_mask(tree, where: tuple[BaseKey, ...], *, is_leaf=None):
    # given a pytree `tree` and a `where` path, that is composed of keys
    # generate a boolean mask that will be eventually used to with `tree_map`
    # to mark the leaves at the specified location.
    # for example for a tree = [[1, 2], 3, 4] and where = [0][1] then
    # generate [[False, True], False, False] mask
    match: bool = False
    treelib = sepes._src.backend.treelib

    def one_level_tree_path_map(func, tree):
        # apply func to the immediate children of tree
        def is_leaf_func(node) -> bool:
            # enable immediate children only
            if is_leaf and is_leaf(node) is True:
                return True
            if id(node) == id(tree):
                return False
            return True

        return treelib.tree_path_map(func, tree, is_leaf=is_leaf_func)

    if any(mask.broadcastable for mask in where):
        # should the selected subtree be broadcasted to the full tree
        # e.g. tree = [[1, 2], 3, 4] and where = [0], then
        # broadcast with True will be [[True, True], False, False]
        # and without broadcast will be [True, False, False]
        # the difference is that with broadcast the user defined value will
        # be broadcasted to the full subtree, for example if the user defined
        # value is 100 then the result will be [[100, 100], 3, 4]
        # and without broadcast the result will be [100, 3, 4]

        def bool_tree(value: bool, tree: Any):
            leaves, treedef = treelib.tree_flatten(tree, is_leaf=is_leaf)
            return treelib.tree_unflatten(treedef, [value] * len(leaves))

        true_tree = ft.partial(bool_tree, True)
        false_tree = ft.partial(bool_tree, False)

    else:
        # no broadcast, the user defined value will be applied to the selected
        # subtree only, for example if the user defined value is 100 then the
        true_tree = lambda _: True
        false_tree = lambda _: False

    def path_map_func(path, leaf):
        nonlocal match, where

        # ensure that the path is not empty
        if len(path) == len(where):
            for pi, ki in zip(path, where):
                if pi != ki:
                    return false_tree(leaf)
            match = True
            return true_tree(leaf)

        if len(path) and len(path) < len(where):
            # before traversing deeper into the tree, check if the current
            # path entry matches the current where entry, if not then return
            # a false tree to stop traversing deeper into the tree.
            (cur_where, *rest_where), (cur_path, *_) = where, path
            if cur_where == cur_path:
                # where is nonlocal to the function
                # so reduce the where path by one level and traverse deeper
                # then restore the where path to the original value before
                # returning the result
                where = rest_where
                # traverse deeper into the tree
                out_tree = one_level_tree_path_map(path_map_func, leaf)
                # return from the traversal
                where = (cur_where, *rest_where)
                return out_tree
            return false_tree(leaf)

        return false_tree(leaf)

    mask = one_level_tree_path_map(path_map_func, tree)

    if not match:
        path_leaf, _ = treelib.tree_path_flatten(tree, is_leaf=is_leaf)
        names = "".join("\n  - " + treelib.keystr(path) for path, _ in path_leaf)
        raise LookupError(_NO_LEAF_MATCH.format(where=where, names=names))

    return mask


def resolve_where(
    where: tuple[Any, ...],  # type: ignore
    tree: T,
    is_leaf: Callable[[Any], None] | None = None,
):
    treelib = sepes._src.backend.treelib

    def combine_bool_leaves(*leaves):
        # given a list of boolean leaves, combine them using `and`
        # this is used to combine multiple boolean masks resulting from
        # either path mask or boolean mask
        verdict = True
        for leaf in leaves:
            verdict &= leaf
        return verdict

    def is_bool_leaf(leaf: Any) -> bool:
        if isinstance(leaf, arraylib.ndarrays):
            return arraylib.is_bool(leaf)
        return isinstance(leaf, bool)

    # given a pytree `tree` and a `where` path, that is composed of keys or
    # boolean masks, generate a boolean mask that will be eventually used to
    # with `tree_map` to select the leaves at the specified location.
    mask = None
    bool_masks: list[T] = []
    path_masks: list[BaseKey] = []
    seen_tuple = False  # handle multiple keys at the same level
    level_paths = []

    def verify_and_aggregate_is_leaf(node: Any) -> bool:
        # use is_leaf with non-local to traverse the tree depth-first manner
        # required for verifying if a pytree is a valid indexing pytree
        nonlocal seen_tuple, level_paths, bool_masks
        # used to check if a pytree is a valid indexing pytree
        # used with `is_leaf` argument of any `tree_*` function
        leaves, _ = treelib.tree_flatten(node)

        if all(map(is_bool_leaf, leaves)):
            # if all leaves are boolean then this is maybe a boolean mask.
            # Maybe because the boolean mask can be a valid pytree of same structure
            # as the pytree to be indexed or _compatible_ structure.
            # that can be flattend up to inside tree_map.
            # the following is an example showcase this:
            # >>> tree = [1, 2, [3, 4]]
            # >>> mask = [True, True, False]
            # >>> AtIndexer(tree)[mask].get()
            # in essence the user can mark full subtrees by `False` without
            # needing to populate the subtree with `False` values. if treedef
            # check is mandated then the user will need to populate the subtree
            # with `False` values. i.e. mask = [True, True, [False, False]]
            # Finally, invalid boolean mask will be caught by `jax.tree_util`
            bool_masks += [node]
            return True

        if isinstance(resolved_key := indexer_dispatcher(node), BaseKey):
            # valid resolution of `BaseKey` is a valid indexing leaf
            # makes it possible to dispatch on multi-leaf pytree
            level_paths += [resolved_key]
            return False

        if type(node) is tuple and seen_tuple is False:
            # e.g. `at[1,2,3]` but not `at[1,(2,3)]``
            # i.e. inside `__getitem__` mutliple entries are transformed to a tuple
            seen_tuple = True
            return False

        # not a container of other keys or a pytree of same structure
        # emit a descriptive error message to the user by pointing to the
        # the available keys in the pytree.
        raise NotImplementedError(_INVALID_INDEXER.format(indexer=node))

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
        mask = generate_path_mask(tree, path_masks, is_leaf=is_leaf)

    if bool_masks:
        all_masks = [mask, *bool_masks] if mask else bool_masks
        mask = treelib.tree_map(combine_bool_leaves, *all_masks)

    return mask


class AtIndexer(Generic[T]):
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
        >>> import jax
        >>> import sepes as sp
        >>> tree = {"level1_0": {"level2_0": 100, "level2_1": 200}, "level1_1": 300}
        >>> indexer = sp.AtIndexer(tree)
        <BLANKLINE>
        >>> indexer["level1_0"]["level2_0"].get()
        {'level1_0': {'level2_0': 100, 'level2_1': None}, 'level1_1': None}
        <BLANKLINE>
        >>> # get multiple keys at once at the same level
        >>> indexer["level1_0"]["level2_0", "level2_1"].get()
        {'level1_0': {'level2_0': 100, 'level2_1': 200}, 'level1_1': None}
        <BLANKLINE>
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
        ...        return f"{type(self).__name__}(a={self.a}, b={self.b})"
        >>> Tree(1, 2).at["a"].get()
        Tree(a=1, b=None)
    """

    def __init__(self, tree: T, where: tuple[BaseKey | Any] | tuple[()] = ()):
        self.tree = tree
        self.where = where

    def __getitem__(self, where: Any) -> Self:
        # syntax sugar for extending the current path with `where`
        # AtIndexer(tree)[`where1`][`where2`] -> AtIndexer(tree, (`where1`, `where2`))
        # no distinction between class attribute and mapping key is made
        # for example AtIndexer(tree)["a"]["b"] will match both
        # tree.a.b and tree["a"]["b"] if tree is a dict or a class instance
        return type(self)(self.tree, (*self.where, where))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(tree={self.tree!r}, where={self.where})"

    def get(
        self,
        *,
        is_leaf: Callable[[Any], None] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ):
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
        treelib = sepes._src.backend.treelib

        def leaf_get(where: Any, leaf: Any):
            # support both array and non-array leaves
            # for array boolean mask we select **parts** of the array that
            # matches the mask, for example if the mask is Array([True, False, False])
            # and the leaf is Array([1, 2, 3]) then the result is Array([1])
            if isinstance(where, arraylib.ndarrays) and len(arraylib.shape(where)):
                return leaf[where]
            # non-array boolean mask we select the leaf if the mask is True
            # and `None` otherwise
            return leaf if where else None

        return treelib.tree_map(
            leaf_get,
            resolve_where(self.where, self.tree, is_leaf),
            self.tree,
            is_leaf=is_leaf,
            is_parallel=is_parallel,
        )

    def set(
        self,
        set_value: Any,
        *,
        is_leaf: Callable[[Any], None] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ):
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
        treelib = sepes._src.backend.treelib

        def leaf_set(where: Any, leaf: Any, set_value: Any):
            # support both array and non-array leaves
            # for array boolean mask we select **parts** of the array that
            # matches the mask, for example if the mask is Array([True, False, False])
            # and the leaf is Array([1, 2, 3]) then the result is Array([1, 100, 100])
            # with set_value = 100
            if isinstance(where, arraylib.ndarrays):
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
            return treelib.tree_map(
                leaf_set,
                resolve_where(self.where, self.tree, is_leaf),
                self.tree,
                set_value,
                is_leaf=is_leaf,
                is_parallel=is_parallel,
            )

        return treelib.tree_map(
            ft.partial(leaf_set, set_value=set_value),
            resolve_where(self.where, self.tree, is_leaf),
            self.tree,
            is_leaf=is_leaf,
            is_parallel=is_parallel,
        )

    def apply(
        self,
        func: Callable[[Any], Any],
        *,
        is_leaf: Callable[[Any], None] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ):
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

        treelib = sepes._src.backend.treelib

        def leaf_apply(where: Any, leaf: Any):
            # same as `leaf_set` but with `func` applied to the leaf
            # one thing to note is that, the where mask select an array
            # then the function needs work properly when applied to the selected
            # array elements
            if isinstance(where, arraylib.ndarrays):
                return arraylib.where(where, func(leaf), leaf)
            return func(leaf) if where else leaf

        return treelib.tree_map(
            leaf_apply,
            resolve_where(self.where, self.tree, is_leaf),
            self.tree,
            is_leaf=is_leaf,
            is_parallel=is_parallel,
        )

    def scan(
        self,
        func: Callable[[Any, S], tuple[Any, S]],
        state: S,
        *,
        is_leaf: Callable[[Any], None] | None = None,
    ) -> tuple[Any, S]:
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
        treelib = sepes._src.backend.treelib
        running_state = state

        def stateless_func(leaf):
            nonlocal running_state
            leaf, running_state = func(leaf, running_state)
            return leaf

        def leaf_apply(where: Any, leaf: Any):
            if isinstance(where, arraylib.ndarrays):
                return arraylib.where(where, stateless_func(leaf), leaf)
            return stateless_func(leaf) if where else leaf

        out_tree = treelib.tree_map(
            leaf_apply,
            resolve_where(self.where, self.tree, is_leaf),
            self.tree,
            is_leaf=is_leaf,
        )
        return out_tree, running_state

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
        treelib = sepes._src.backend.treelib
        tree = self.get(is_leaf=is_leaf)  # type: ignore
        leaves, _ = treelib.tree_flatten(tree, is_leaf=is_leaf)
        if initializer is _no_initializer:
            return ft.reduce(func, leaves)
        return ft.reduce(func, leaves, initializer)

    def pluck(
        self,
        count: int | None = None,
        *,
        is_leaf: Callable[[Any], None] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> list[Any]:
        """Extract subtrees at the specified location.

        Note:
            ``pluck`` first applies ``get`` to the specified location and then
            extracts the immediate subtrees of the selected leaves. ``is_leaf``
            and ``is_parallel`` are passed to ``get``.

        Args:
            count: number of subtrees to extract, Default to ``None`` to
                extract all subtrees.
            is_leaf: a predicate function to determine if a value is a leaf.
            is_parallel: accepts the following:

                - ``bool``: apply ``func`` in parallel if ``True`` otherwise in serial.
                - ``dict``: a dict of of:

                    - ``max_workers``: maximum number of workers to use.
                    - ``kind``: kind of pool to use, either ``thread`` or ``process``.

        Returns:
            A list of subtrees at the specified location.

        Note:
            Compared to ``get``, ``pluck`` extracts subtrees at the specified
            location and returns a list of subtrees. While ``get`` returns a
            pytree with the leaf values at the specified location and set the
            non-selected leaf values to ``None``.

        Example:
            >>> import sepes as sp
            >>> tree = {"a": 1, "b": [1, 2, 3]}
            >>> indexer = sp.AtIndexer(tree)  # construct an indexer
            <BLANKLINE>
            >>> # `pluck` returns a list of selected subtrees
            >>> indexer["b"].pluck()
            [[1, 2, 3]]
            <BLANKLINE>
            >>> # `get` returns same pytree
            >>> indexer["b"].get()
            {'a': None, 'b': [1, 2, 3]}

        Example:
            ``pluck`` with mask

            >>> import sepes as sp
            >>> tree = {"a": 1, "b": [2, 3, 4]}
            >>> mask = {"a": True, "b": [False, True, False]}
            >>> indexer = sp.AtIndexer(tree)
            >>> indexer[mask].pluck()
            [1, 3]

            This is equivalent to the following:

            >>> [tree["a"], tree["b"][1]]  # doctest: +SKIP
        """
        treelib = sepes._src.backend.treelib
        tree = self.get(is_leaf=is_leaf, is_parallel=is_parallel)
        subtrees: list[Any] = []
        count = float("inf") if count is None else count

        def aggregate_subtrees(node: Any) -> bool:
            nonlocal subtrees, count
            if count < 1:
                # stop traversing the tree
                # if total number of subtrees is reached
                return True
            if id(node) == id(tree):
                # skip the root node
                # for example if tree = dict(a=1) and mask is dict(a=True)
                # then returns [1] and not [dict(a=1)]
                return False
            leaves, _ = treelib.tree_flatten(node, is_leaf=lambda x: x is None)
            # in essence if the subtree does not contain any None leaves
            # then it is a valid subtree to be plucked
            # this because `get` sets the non-selected leaves to None
            if any(leaf is None for leaf in leaves):
                return False
            subtrees += [node]
            count -= 1
            return True

        treelib.tree_flatten(tree, is_leaf=aggregate_subtrees)
        return subtrees

    def __call__(self, *args, **kwargs) -> tuple[Any, PyTree]:
        """Call and return a tuple of the result and copy of the tree.

        Executes the method defined by the ``where`` path on the tree on
        a copy of the tree and returns a tuple of the result and the copy.
        To avoid mutating in place, use this method instead of calling the
        method directly on the tree.

        Example:
            >>> import sepes as sp
            >>> import jax
            >>> @jax.tree_util.register_pytree_with_keys_class
            ... class Counter:
            ...    def __init__(self, count: int):
            ...        self.count = count
            ...    def tree_flatten_with_keys(self):
            ...        return (["count", self.count],), None
            ...    @classmethod
            ...    def tree_unflatten(cls, aux_data, children):
            ...        del aux_data
            ...        return cls(*children)
            ...    def increment_count(self) -> int:
            ...        # mutates the tree
            ...        self.count += 1
            ...        return self.count
            ...    def __repr__(self) -> str:
            ...        return f"Tree(count={self.count})"
            >>> counter = Counter(0)
            >>> indexer = sp.AtIndexer(counter)
            >>> cur_count, new_counter = indexer["increment_count"]()
            >>> counter, new_counter
            (Tree(count=0), Tree(count=1))

        Note:
            The default behavior of :class:`.AtIndexer` ``__call__`` is to copy
            the instance and then call the method on the copy. However certain
            classes (e.g. :class:`.TreeClass` or ``dataclasses.dataclass(frozen=True)``)
            do not support in-place mutation. In this case, :class:`.AtIndexer`
            enables registering custom function that modifies the instance
            to allow in-place mutation. and custom function that restores the
            instance to its original state after the method call.

            The following example shows how to register custom functions for
            a simple class that allows in-place mutation if ``immutable`` Flag
            is set to ``False``.

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
            >>> @sp.AtIndexer.custom_call.def_mutator(MyNode)
            ... def mutable(node) -> None:
            ...     vars(node)["immutable"] = False
            >>> @sp.AtIndexer.custom_call.def_immutator(MyNode)
            ... def immutable(node) -> None:
            ...     vars(node)["immutable"] = True
            >>> node = MyNode()
            >>> sp.AtIndexer(node)["increment"]()
            (None, MyNode(counter=1, immutable=True))
        """
        # copy the current tree
        tree = tree_copy(self.tree)
        # and edit the node/record to make it mutable (if there is a rule for it)
        tree_mutate(tree)
        # use the copied mutable version of the tree to call the method
        method = recursive_getattr(tree, self.where)
        output = method(*args, **kwargs)
        # traverse each node in the tree depth-first manner
        # to undo the mutation (if there is a rule for it)
        tree_immutate(tree)
        return output, tree


def tree_mutate(tree):
    treelib = sepes._src.backend.treelib

    def is_leaf(node):
        AtIndexer.custom_call.mutator_dispatcher(node)
        return False

    return treelib.tree_map(lambda x: x, tree, is_leaf=is_leaf)


def tree_immutate(tree):
    treelib = sepes._src.backend.treelib

    def is_leaf(node):
        AtIndexer.custom_call.immutator_dispatcher(node)
        return False

    return treelib.tree_map(lambda x: x, tree, is_leaf=is_leaf)


# define rules for mutating and restoring the tree after calling a method
# useful in case the class does not support in-place mutation
# thus a rule to mutate the tree before calling the method and
# a rule to restore the tree after calling the method is needed.
custom_call = SimpleNamespace()
custom_call.mutator_dispatcher = ft.singledispatch(lambda node: node)
custom_call.immutator_dispatcher = ft.singledispatch(lambda node: node)
custom_call.def_mutator = custom_call.mutator_dispatcher.register
custom_call.def_immutator = custom_call.immutator_dispatcher.register

AtIndexer.custom_call = custom_call
