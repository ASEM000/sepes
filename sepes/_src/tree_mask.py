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

"""Utilities to work with non-inexact type tree leaves across function transformations."""

from __future__ import annotations

import functools as ft
import hashlib
from typing import Any, Callable, NamedTuple, TypeVar, Union

import sepes
import sepes._src.backend.arraylib as arraylib
from sepes._src.backend import is_package_avaiable
from sepes._src.tree_pprint import tree_repr, tree_str, tree_summary
from sepes._src.tree_util import Static, is_tree_equal, tree_copy, tree_hash

T = TypeVar("T")
MaskType = Union[T, Callable[[Any], bool]]


class _MaskedError(NamedTuple):
    opname: str

    def __call__(self, *a, **k):
        raise NotImplementedError(
            f"Cannot apply `{self.opname}` operation on a masked object "
            f"{', '.join(map(str, a))} "
            f"{', '.join(k + '=' + str(v) for k, v in k.items())}.\n"
            "Unmask the object first using `tree_unmask`"
        )


class _MaskBase(Static[T]):
    # the objective of this class is to wrap a pytree node with a custom wrapper
    # that yields no leaves when flattened. This is useful to avoid updating
    # the node by effectivly *hiding it* from function transformations that operates
    # on flattened pytrees.
    __slots__ = ["__wrapped__"]
    __wrapped__: T

    def __init__(self, node: T) -> None:
        object.__setattr__(self, "__wrapped__", node)

    def __setattr__(self, _, __) -> None:
        raise AttributeError("Cannot assign to frozen instance.")

    def __delattr__(self, _: str) -> None:
        raise AttributeError("Cannot delete from frozen instance.")

    def __repr__(self) -> str:
        return "#" + tree_repr(self.__wrapped__)

    def __str__(self) -> str:
        return "#" + tree_str(self.__wrapped__)

    def __copy__(self) -> _MaskBase[T]:
        return type(self)(tree_copy(self.__wrapped__))

    # raise helpful error message when trying to interact with frozen object
    __add__ = __radd__ = __iadd__ = _MaskedError("+")
    __sub__ = __rsub__ = __isub__ = _MaskedError("-")
    __mul__ = __rmul__ = __imul__ = _MaskedError("*")
    __matmul__ = __rmatmul__ = __imatmul__ = _MaskedError("@")
    __truediv__ = __rtruediv__ = __itruediv__ = _MaskedError("/")
    __floordiv__ = __rfloordiv__ = __ifloordiv__ = _MaskedError("//")
    __mod__ = __rmod__ = __imod__ = _MaskedError("%")
    __pow__ = __rpow__ = __ipow__ = _MaskedError("**")
    __lshift__ = __rlshift__ = __ilshift__ = _MaskedError("<<")
    __rshift__ = __rrshift__ = __irshift__ = _MaskedError(">>")
    __and__ = __rand__ = __iand__ = _MaskedError("and")
    __xor__ = __rxor__ = __ixor__ = _MaskedError("")
    __or__ = __ror__ = __ior__ = _MaskedError("or")
    __neg__ = __pos__ = __abs__ = __invert__ = _MaskedError("unary")
    __lt__ = __le__ = __gt__ = __ge__ = _MaskedError("comparison")
    __call__ = _MaskedError("__call__")


@tree_summary.def_type(_MaskBase)
def _(node) -> str:
    return f"#{tree_summary.type_dispatcher(node.__wrapped__)}"


class _MaskedHashable(_MaskBase):
    def __hash__(self) -> int:
        return tree_hash(self.__wrapped__)

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, _MaskedHashable):
            return False
        return is_tree_equal(self.__wrapped__, rhs.__wrapped__)


class _MaskedArray(_MaskBase):
    # wrap arrays with a custom wrapper that implements hash and equality
    # using the wrapped array's bytes representation and sha256 hash function
    # this is useful to select some array to hold without updating in the process
    # of training a model.
    def __hash__(self) -> int:
        bytes = arraylib.tobytes(self.__wrapped__)
        return int(hashlib.sha256(bytes).hexdigest(), 16)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _MaskedArray):
            return False
        lhs, rhs = self.__wrapped__, other.__wrapped__
        # fast path to avoid calling `all` on large arrays
        if arraylib.shape(lhs) != arraylib.shape(rhs):
            return False
        if arraylib.dtype(lhs) != arraylib.dtype(rhs):
            return False
        return arraylib.array_equal(lhs, rhs)


def mask(value: T) -> _MaskBase[T]:
    # dispatching is used to customize the type of the wrapper based on the type
    # of the value. For instance, hashable values dont need custom hash and
    # equality implementations, so they are wrapped with a simpler wrapper.
    # this approach avoids type logic in the wrapper equality and hash methods,
    # thus effectively improving performance of the wrapper.
    return mask.type_dispatcher(value)


mask.type_dispatcher = ft.singledispatch(_MaskedHashable)
mask.def_type = mask.type_dispatcher.register


for ndarray in arraylib.ndarrays:

    @mask.def_type(ndarray)
    def mask_array(value: T) -> _MaskedArray[T]:
        # wrap arrays with a custom wrapper that implements hash and equality
        # arrays can be hashed by converting them to bytes and hashing the bytes
        return _MaskedArray(value)


@mask.def_type(_MaskBase)
def _(value: _MaskBase[T]) -> _MaskBase[T]:
    # idempotent mask operation, meaning that mask(mask(x)) == mask(x)
    # this is useful to avoid recursive unwrapping of frozen values, plus its
    # meaningless to mask a frozen value.
    return value


def is_masked(value: Any) -> bool:
    """Returns True if the value is a frozen wrapper."""
    return isinstance(value, _MaskBase)


def unmask(value: T) -> T:
    return unmask.type_dispatcher(value)


unmask.type_dispatcher = ft.singledispatch(lambda x: x)
unmask.def_type = unmask.type_dispatcher.register


@unmask.def_type(_MaskBase)
def _(value: _MaskBase[T]) -> T:
    return getattr(value, "__wrapped__")


def is_nondiff(value: Any) -> bool:
    return is_nondiff.type_dispatcher(value)


is_nondiff.type_dispatcher = ft.singledispatch(lambda _: True)
is_nondiff.def_type = is_nondiff.type_dispatcher.register


for ndarray in arraylib.ndarrays:

    @is_nondiff.def_type(ndarray)
    def is_nondiff_array(value) -> bool:
        # return True if the node is non-inexact type, otherwise False
        if arraylib.is_inexact(value):
            return False
        return True


@is_nondiff.def_type(float)
@is_nondiff.def_type(complex)
def _(_: float | complex) -> bool:
    return False


def _tree_mask_map(
    tree: T,
    cond: Callable[[Any], bool],
    func: type | Callable[[Any], Any],
    *,
    is_leaf: Callable[[Any], None] | None = None,
):

    if not isinstance(cond, Callable):
        # a callable that accepts a leaf and returns a boolean
        # but *not* a tree with the same structure as tree with boolean values.
        raise TypeError(
            f"`cond` must be a callable that accepts a leaf and returns a boolean "
            f" Got {cond=} and {tree=}."
        )

    treelib = sepes._src.backend.treelib

    def map_func(x):
        return func(x) if cond(x) else x

    return treelib.map(map_func, tree, is_leaf=is_leaf)


def tree_mask(
    tree: T,
    cond: Callable[[Any], bool] = is_nondiff,
    *,
    is_leaf: Callable[[Any], None] | None = None,
):
    """Mask leaves of a pytree based on ``mask`` boolean pytree or callable.

    Masked leaves are wrapped with a wrapper that yields no leaves when
    ``tree_flatten`` is called on it.

    Args:
        tree: A pytree of values.
        cond: A callable that accepts a leaf and returns a boolean to mark the leaf
            for masking. Defaults to masking non-differentiable leaf nodes that
            are not instances of of python float, python complex, or inexact
            array types.
        is_leaf: A callable that accepts a leaf and returns a boolean. If
            provided, it is used to determine if a value is a leaf. for example,
            ``is_leaf=lambda x: isinstance(x, list)`` will treat lists as leaves
            and will not recurse into them.

    Example:
        >>> import sepes as sp
        >>> import jax
        >>> tree = [1, 2, {"a": 3, "b": 4.}]
        >>> # mask all non-differentiable nodes by default
        >>> masked_tree = sp.tree_mask(tree)
        >>> masked_tree
        [#1, #2, {'a': #3, 'b': 4.0}]
        >>> jax.tree_util.tree_leaves(masked_tree)
        [4.0]
        >>> sp.tree_unmask(masked_tree)
        [1, 2, {'a': 3, 'b': 4.0}]

    Example:
        Pass non-differentiable values to ``jax.grad``

        >>> import sepes as sp
        >>> import jax
        >>> @jax.grad
        ... def square(tree):
        ...     tree = sp.tree_unmask(tree)
        ...     return tree[0] ** 2
        >>> tree = (1., 2)  # contains a non-differentiable node
        >>> square(sp.tree_mask(tree))
        (Array(2., dtype=float32, weak_type=True), #2)
    """
    return _tree_mask_map(tree, cond=cond, func=mask, is_leaf=is_leaf)


def tree_unmask(tree: T, cond: Callable[[Any], bool] = lambda _: True):
    """Undo the masking of tree leaves according to ``cond``. defaults to unmasking all leaves.

    Args:
        tree: A pytree of values.
        cond: A callable that accepts a leaf and returns a boolean to mark the
            leaf to be unmasked. Defaults to always unmask.

    Example:
        >>> import sepes as sp
        >>> import jax
        >>> tree = [1, 2, {"a": 3, "b": 4.}]
        >>> # mask all non-differentiable nodes by default
        >>> masked_tree = sp.tree_mask(tree)
        >>> masked_tree
        [#1, #2, {'a': #3, 'b': 4.0}]
        >>> jax.tree_util.tree_leaves(masked_tree)
        [4.0]
        >>> sp.tree_unmask(masked_tree)
        [1, 2, {'a': 3, 'b': 4.0}]

    Example:
        Pass non-differentiable values to ``jax.grad``

        >>> import sepes as sp
        >>> import jax
        >>> @jax.grad
        ... def square(tree):
        ...     tree = sp.tree_unmask(tree)
        ...     return tree[0] ** 2
        >>> tree = (1., 2)  # contains a non-differentiable node
        >>> square(sp.tree_mask(tree))
        (Array(2., dtype=float32, weak_type=True), #2)
    """
    return _tree_mask_map(tree, cond=cond, func=unmask, is_leaf=is_masked)


if is_package_avaiable("jax"):
    import jax

    # do not touch jax.core.Tracer instances.
    # otherwise calling `freeze` inside a jax transformation on
    # a tracer will hide the tracer from jax and will cause leaked tracer
    # error.
    @mask.def_type(jax.core.Tracer)
    def _(value: jax.core.Tracer) -> jax.core.Tracer:
        return value
