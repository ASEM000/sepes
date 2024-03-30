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

"""Define lens-like indexing for pytrees

This module provides a way to index and mask pytrees (e.g. TreeClass) in an 
out-of-place manner.Out-of-place means that the original pytree is not modified, 
instead a new pytree with the selected leaves are modified.

The indexing is done through two concepts:

1) Selection (Where): Determines parts of the pytree for manipulation via a path or a boolean mask.
2) Operation (What): Defines actions on selected parts, such as setting values or applying functions.

For example, the following code defines a dict pytree with where of same structure 
as the tree. The where (Selection) defines which parts of the tree to select and 
the set (Operation) operation sets the selected parts to 100.

>>> import sepes as sp
>>> tree = {"a": 1, "b": [1, 2, 3]}
>>> where = {"a": True, "b": [False, True, False]}
>>> sp.at(tree)[where].set(100)
{'a': 100, 'b': [1, 100, 3]}
"""

from __future__ import annotations

import abc
import functools as ft
import re
from typing import Any, Callable, Generic, Hashable, TypeVar, Sequence

from typing_extensions import Self

import sepes
import sepes._src.backend.arraylib as arraylib
from sepes._src.backend import is_package_avaiable
from sepes._src.backend.treelib import ParallelConfig
from sepes._src.tree_pprint import tree_repr

T = TypeVar("T")
S = TypeVar("S")
PyTree = Any
EllipsisType = TypeVar("EllipsisType")
PathKeyEntry = TypeVar("PathKeyEntry", bound=Hashable)
_no_initializer = object()
_no_fill_value = object()


class BaseKey(abc.ABC):
    """Parent class for all match classes."""

    @abc.abstractmethod
    def __eq__(self, entry: PathKeyEntry) -> bool:
        pass

    @property
    @abc.abstractmethod
    def broadcast(self): ...


_INVALID_INDEXER = """\
Indexing with {indexer} is not implemented, supported indexing types are:
  - `str` for mapping keys or class attributes.
  - `int` for positional indexing for sequences.
  - `...` to select all leaves.
  - ``re.Pattern`` to match a leaf level path with a regex pattern.
  - Boolean mask of a compatible structure as the pytree.
  - `tuple` of the above types to match multiple leaves at the same level.
"""

_NO_LEAF_MATCH = """\
No leaf match is found for where={where}, Available keys are {names}
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

        return treelib.path_map(func, tree, is_leaf=is_leaf_func)

    if any(where_i.broadcast for where_i in where):
        # should the selected subtree be broadcasted to the full tree
        # e.g. tree = [[1, 2], 3, 4] and where = [0], then
        # broadcast with True will be [[True, True], False, False]
        # and without broadcast will be [True, False, False]
        # the difference is that with broadcast the user defined value will
        # be broadcasted to the full subtree, for example if the user defined
        # value is 100 then the result will be [[100, 100], 3, 4]
        # and without broadcast the result will be [100, 3, 4]

        def bool_tree(value: bool, tree: Any):
            leaves, treedef = treelib.flatten(tree, is_leaf=is_leaf)
            return treelib.unflatten(treedef, [value] * len(leaves))

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
        path_leaf, _ = treelib.path_flatten(tree, is_leaf=is_leaf)
        path = "/".join(str(where_i.input) for where_i in where)
        names = "".join("\n  - " + treelib.keystr(path) for path, _ in path_leaf)
        raise LookupError(_NO_LEAF_MATCH.format(where=path, names=names))

    return mask


def resolve_where(
    where: list[Any],
    tree: T,
    is_leaf: Callable[[Any], bool] | None = None,
):
    treelib = sepes._src.backend.treelib
    ndarrays = tuple(arraylib.ndarrays)

    def combine_bool_leaves(*leaves):
        # given a list of boolean leaves, combine them using `and`
        # this is used to combine multiple boolean masks resulting from
        # either path mask or boolean mask
        verdict = True
        for leaf in leaves:
            verdict &= leaf
        return verdict

    def is_bool_leaf(leaf: Any) -> bool:
        if isinstance(leaf, ndarrays):
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
        leaves, _ = treelib.flatten(node)

        if all(map(is_bool_leaf, leaves)):
            # if all leaves are boolean then this is maybe a boolean mask.
            # Maybe because the boolean mask can be a valid pytree of same structure
            # as the pytree to be indexed or _compatible_ structure.
            # that can be flattend up to inside tree_map.
            # the following is an example showcase this:
            # >>> tree = [1, 2, [3, 4]]
            # >>> mask = [True, True, False]
            # >>> at(tree)[mask].get()
            # in essence the user can mark full subtrees by `False` without
            # needing to populate the subtree with `False` values. if treedef
            # check is mandated then the user will need to populate the subtree
            # with `False` values. i.e. mask = [True, True, [False, False]]
            # Finally, invalid boolean mask will be caught by `jax.tree_util`
            bool_masks += [node]
            return True

        if isinstance(resolved_key := at.dispatcher(node), BaseKey):
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
        treelib.flatten(level_keys, is_leaf=verify_and_aggregate_is_leaf)
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
        mask = treelib.map(combine_bool_leaves, *all_masks)

    return mask


class at(Generic[T]):
    """Operate on a pytree at a given path using a path or mask in out-of-place manner.

    Args:
        tree: pytree to operate on.
        where: one of the following:

            - ``str`` for mapping keys or class attributes.
            - ``int`` for positional indexing for sequences.
            - ``...`` to select all leaves.
            - a boolean mask of the same structure as the tree
            - ``re.Pattern`` to match a leaf level path with a regex pattern.
            - a tuple of the above to match multiple keys at the same level.

    Example:
        >>> import jax
        >>> import sepes as sp
        >>> tree = {"a": 1, "b": [1, 2, 3]}
        >>> sp.at(tree)["a"].set(100)
        {'a': 100, 'b': [1, 2, 3]}
        >>> sp.at(tree)["b"][0].set(100)
        {'a': 1, 'b': [100, 2, 3]}
        >>> mask = jax.tree_map(lambda x: x > 1, tree)
        >>> sp.at(tree)[mask].set(100)
        {'a': 1, 'b': [1, 100, 100]}
    """
    def __init__(self, tree: T, where: list[Any] | None = None) -> None:
        self.tree = tree
        self.where = [] if where is None else where

    def __getitem__(self, where: Any) -> Self:
        """Index a pytree at a given path using a path or mask."""
        return type(self)(self.tree, [*self.where, where])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({tree_repr(self.tree)}, where={self.where})"

    def get(
        self,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
        is_parallel: bool | ParallelConfig = False,
        fill_value: Any = _no_fill_value,
    ):
        """Get the leaf values at the specified location.

        Args:
            is_leaf: a predicate function to determine if a value is a leaf.
            is_parallel: accepts the following:

                - ``bool``: apply ``func`` in parallel if ``True`` otherwise in serial.
                - ``dict``: a dict of of:

                    - ``max_workers``: maximum number of workers to use.
                    - ``kind``: kind of pool to use, either ``thread`` or ``process``.

            fill_value: the value to fill the non-selected leaves with.
                Useful to use with ``jax.jit`` to avoid variable size arrays
                leaves related errors.

        Returns:
            A _new_ pytree of leaf values at the specified location, with the
            non-selected leaf values set to None if the leaf is not an array.

        Example:
            >>> import sepes as sp
            >>> tree = {"a": 1, "b": [1, 2, 3]}
            >>> sp.at(tree)["b"][0].get()
            {'a': None, 'b': [1, None, None]}
        """
        treelib = sepes._src.backend.treelib
        ndarrays = tuple(arraylib.ndarrays)

        def leaf_get(where: Any, leaf: Any):
            # support both array and non-array leaves
            # for array boolean mask we select **parts** of the array that
            # matches the mask, for example if the mask is Array([True, False, False])
            # and the leaf is Array([1, 2, 3]) then the result is Array([1])
            # because of the variable resultant size of the output
            if isinstance(where, ndarrays) and len(arraylib.shape(where)):
                if fill_value is not _no_fill_value:
                    return arraylib.where(where, leaf, fill_value)
                return leaf[where]
            # non-array boolean mask we select the leaf if the mask is True
            # and `None` otherwise
            if fill_value is not _no_fill_value:
                return leaf if where else fill_value
            return leaf if where else None

        return treelib.map(
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
        is_leaf: Callable[[Any], bool] | None = None,
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
            >>> sp.at(tree)["b"][0].set(100)
            {'a': 1, 'b': [100, 2, 3]}
        """
        treelib = sepes._src.backend.treelib
        ndarrays = tuple(arraylib.ndarrays)

        def leaf_set(where: Any, leaf: Any, set_value: Any):
            # support both array and non-array leaves
            # for array boolean mask we select **parts** of the array that
            # matches the mask, for example if the mask is Array([True, False, False])
            # and the leaf is Array([1, 2, 3]) then the result is Array([1, 100, 100])
            # with set_value = 100
            if isinstance(where, ndarrays):
                return arraylib.where(where, set_value, leaf)
            return set_value if where else leaf

        _, lhsdef = treelib.flatten(self.tree, is_leaf=is_leaf)
        _, rhsdef = treelib.flatten(set_value, is_leaf=is_leaf)

        if lhsdef == rhsdef:
            # do not broadcast set_value if it is a pytree of same structure
            # for example tree.at[where].set(tree2) will set all tree leaves
            # to tree2 leaves if tree2 is a pytree of same structure as tree
            # instead of making each leaf of tree a copy of tree2
            # is design is similar to ``numpy`` design `np.at[...].set(Array)`
            return treelib.map(
                leaf_set,
                resolve_where(self.where, self.tree, is_leaf),
                self.tree,
                set_value,
                is_leaf=is_leaf,
                is_parallel=is_parallel,
            )

        return treelib.map(
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
        is_leaf: Callable[[Any], bool] | None = None,
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
            >>> sp.at(tree)["b"][0].apply(lambda x: x + 100)
            {'a': 1, 'b': [101, 2, 3]}

        Example:
            Read images in parallel

            >>> import sepes as sp
            >>> from matplotlib.pyplot import imread
            >>> path = {"img1": "path1.png", "img2": "path2.png"}
            >>> is_parallel = dict(max_workers=2)
            >>> images = sp.at(path)[...].apply(imread, is_parallel=is_parallel)  # doctest: +SKIP
        """
        treelib = sepes._src.backend.treelib
        ndarrays = tuple(arraylib.ndarrays)

        def leaf_apply(where: Any, leaf: Any):
            # same as `leaf_set` but with `func` applied to the leaf
            # one thing to note is that, the where mask select an array
            # then the function needs work properly when applied to the selected
            # array elements
            if isinstance(where, ndarrays):
                return arraylib.where(where, func(leaf), leaf)
            return func(leaf) if where else leaf

        return treelib.map(
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
        is_leaf: Callable[[Any], bool] | None = None,
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
            >>> tree = {"a": 1, "b": [1, 2, 3]}
            >>> def scan_func(leaf, running_max):
            ...     cur_max = max(leaf, running_max)
            ...     return leaf, cur_max
            >>> running_max = float("-inf")
            >>> _, running_max = sp.at(tree)["b"][0, 1].scan(scan_func, state=running_max)
            >>> running_max  # max of b[0] and b[1]
            2

        Note:
            ``scan`` applies a binary ``func`` to the leaf values while carrying
            a state and returning a tree leaves with the the ``func`` applied to
            them with final state. While ``reduce`` applies a binary ``func`` to the
            leaf values while carrying a state and returning a single value.
        """
        treelib = sepes._src.backend.treelib
        ndarrays = tuple(arraylib.ndarrays)
        running_state = state

        def stateless_func(leaf):
            nonlocal running_state
            leaf, running_state = func(leaf, running_state)
            return leaf

        def leaf_apply(where: Any, leaf: Any):
            if isinstance(where, ndarrays):
                return arraylib.where(where, stateless_func(leaf), leaf)
            return stateless_func(leaf) if where else leaf

        out_tree = treelib.map(
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
        is_leaf: Callable[[Any], bool] | None = None,
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
            >>> tree = {"a": 1, "b": [1, 2, 3]}
            >>> sp.at(tree)["b"].reduce(lambda x, y: x + y)
            6
        """
        treelib = sepes._src.backend.treelib
        tree = self.get(is_leaf=is_leaf)  # type: ignore
        leaves, _ = treelib.flatten(tree, is_leaf=is_leaf)
        if initializer is _no_initializer:
            return ft.reduce(func, leaves)
        return ft.reduce(func, leaves, initializer)

    def pluck(
        self,
        count: int | None = None,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
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
            <BLANKLINE>
            >>> # `pluck` returns a list of selected subtrees
            >>> sp.at(tree)["b"].pluck()
            [[1, 2, 3]]
            <BLANKLINE>
            >>> # `get` returns same pytree
            >>> sp.at(tree)["b"].get()
            {'a': None, 'b': [1, 2, 3]}

        Example:
            ``pluck`` with mask

            >>> import sepes as sp
            >>> tree = {"a": 1, "b": [2, 3, 4]}
            >>> mask = {"a": True, "b": [False, True, False]}
            >>> sp.at(tree)[mask].pluck()
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
            leaves, _ = treelib.flatten(node, is_leaf=lambda x: x is None)
            # in essence if the subtree does not contain any None leaves
            # then it is a valid subtree to be plucked
            # this because `get` sets the non-selected leaves to None
            if any(leaf is None for leaf in leaves):
                return False
            subtrees += [node]
            count -= 1
            return True

        treelib.flatten(tree, is_leaf=aggregate_subtrees)
        return subtrees


# pass through for boolean pytrees masks and tuple of keys
at.dispatcher = ft.singledispatch(lambda x: x)


def def_rule(
    user_type: type[T],
    path_compare_func: Callable[[T, PathKeyEntry], bool],
    *,
    broadcastable: bool = False,
) -> None:
    # remove the BaseKey abstraction from the user-facing function
    class UserKey(BaseKey):
        broadcast: bool = broadcastable

        def __init__(self, input: T):
            self.input = input

        def __eq__(self, key: PathKeyEntry) -> bool:
            return path_compare_func(self.input, key)

    at.dispatcher.register(user_type, UserKey)


at.def_rule = def_rule


# key rules to match user input to with the path entry


def str_compare(name: str, key: PathKeyEntry):
    """Match a leaf with a given name."""
    if isinstance(key, str):
        return name == key
    treelib = sepes._src.backend.treelib
    if isinstance(key, type(treelib.attribute_key(""))):
        return name == key.name
    if isinstance(key, type(treelib.dict_key(""))):
        return name == key.key
    return False


def int_compare(idx: int, key: PathKeyEntry) -> bool:
    """Match a leaf with a given index."""
    if isinstance(key, int):
        return idx == key
    treelib = sepes._src.backend.treelib
    if isinstance(key, type(treelib.sequence_key(0))):
        return idx == key.idx
    return False


def regex_compare(pattern: re.Pattern, key: PathKeyEntry) -> bool:
    """Match a path with a regex pattern inside 'at' property."""
    if isinstance(key, str):
        return re.fullmatch(pattern, key) is not None
    treelib = sepes._src.backend.treelib
    if isinstance(key, type(treelib.attribute_key(""))):
        return re.fullmatch(pattern, key.name) is not None
    if isinstance(key, type(treelib.dict_key(""))):
        return re.fullmatch(pattern, key.key) is not None
    return False


def ellipsis_compare(_, __):
    return True


at.def_rule(str, str_compare, broadcastable=False)
at.def_rule(int, int_compare, broadcastable=False)
at.def_rule(re.Pattern, regex_compare, broadcastable=False)
at.def_rule(type(...), ellipsis_compare, broadcastable=True)


class MultiKey(BaseKey):
    """Match a leaf with multiple keys at the same level."""

    def __init__(self, *keys):
        self.keys = tuple(keys)

    def __eq__(self, entry: PathKeyEntry) -> bool:
        return any(entry == key for key in self.keys)

    broadcast: bool = False


if is_package_avaiable("jax"):
    import jax.tree_util as jtu

    def jax_key_compare(input, key: PathKeyEntry) -> bool:
        """Enable indexing with jax keys directly in `at`."""
        return input == key

    at.def_rule(jtu.SequenceKey, jax_key_compare, broadcastable=False)
    at.def_rule(jtu.GetAttrKey, jax_key_compare, broadcastable=False)
    at.def_rule(jtu.DictKey, jax_key_compare, broadcastable=False)
