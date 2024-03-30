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

"""Utilities for pretty printing pytrees."""

from __future__ import annotations

import functools as ft
import inspect
import math
from itertools import zip_longest
from math import prod
from types import FunctionType
from typing import Any, Callable, NamedTuple, Sequence

from typing_extensions import TypeAlias, TypedDict, Unpack

import sepes
import sepes._src.backend.arraylib as arraylib
from sepes._src.backend import is_package_avaiable
from sepes._src.tree_util import (
    Node,
    construct_tree,
    is_path_leaf_depth_factory,
    tree_type_path_leaves,
)


class PPSpec(TypedDict):
    indent: int
    width: int
    depth: int | float


PyTree = Any

PP = Callable[[Any, Unpack[PPSpec]], str]


class ShapeDTypePP(NamedTuple):
    shape: tuple[int, ...]
    dtype: Any


def pp(printer, node: Any, **spec: Unpack[PPSpec]) -> str:
    return (
        "..."
        if spec["depth"] < 0
        else format_width(printer.dispatch(node, **spec), width=spec["width"])
    )


def pps(pp: PP, xs: Sequence[Any], **spec: Unpack[PPSpec]) -> str:
    if spec["depth"] < 1:
        return "..."

    spec["indent"] += 1
    spec["depth"] -= 1

    text = (
        "\n"
        + "\t" * spec["indent"]
        + (", \n" + "\t" * spec["indent"]).join(pp(x, **spec) for x in xs)
        + "\n"
        + "\t" * (spec["indent"] - 1)
    )

    return format_width(text, width=spec["width"])


def kv_pp(pp: PP, x: tuple[str, Any], **spec: Unpack[PPSpec]) -> str:
    return f"{x[0]}:{pp(x[1], **spec)}"


def av_pp(pp: PP, x: tuple[str, Any], **spec: Unpack[PPSpec]) -> str:
    return f"{x[0]}={pp(x[1], **spec)}"


def tree_repr(
    tree: PyTree,
    *,
    width: int = 80,
    tabwidth: int = 2,
    depth: int | float = float("inf"),
):
    out = tree_repr.dispatch(tree, indent=0, width=width, depth=depth)
    return out.expandtabs(tabwidth)


def tree_str(
    tree: PyTree,
    *,
    width: int = 80,
    tabwidth: int = 2,
    depth: int | float = float("inf"),
):
    out = tree_str.dispatch(tree, indent=0, width=width, depth=depth)
    return out.expandtabs(tabwidth)


def repr_dispatch(node: Any, **spec: Unpack[PPSpec]) -> str:
    return (
        ("\n" + "\t" * (spec["indent"])).join(text.split("\n"))
        if "\n" in (text := repr(node))
        else text
    )


def str_dispatch(node: Any, **spec: Unpack[PPSpec]) -> str:
    return (
        ("\n" + "\t" * (spec["indent"])).join(text.split("\n"))
        if "\n" in (text := str(node))
        else text
    )


tree_repr.dispatch = ft.singledispatch(repr_dispatch)
tree_repr.def_type = tree_repr.dispatch.register
tree_repr.pps = pps
tree_repr.pp = ft.partial(pp, tree_repr)
tree_repr.kv_pp = ft.partial(kv_pp, tree_repr.pp)
tree_repr.av_pp = ft.partial(av_pp, tree_repr.pp)

tree_str.dispatch = ft.singledispatch(str_dispatch)
tree_str.def_type = tree_str.dispatch.register
tree_str.pps = pps
tree_str.pp = ft.partial(pp, tree_str)
tree_str.kv_pp = ft.partial(kv_pp, tree_str.pp)
tree_str.av_pp = ft.partial(av_pp, tree_str.pp)


@tree_repr.def_type(ShapeDTypePP)
@tree_str.def_type(ShapeDTypePP)
def _(node: Any, **spec: Unpack[PPSpec]) -> str:
    """Pretty print a node with dtype and shape."""
    shape = f"{node.shape}".replace(",", "")
    shape = shape.replace("(", "[")
    shape = shape.replace(")", "]")
    shape = shape.replace(" ", ",")
    dtype = f"{node.dtype}".replace("int", "i")
    dtype = dtype.replace("float", "f")
    dtype = dtype.replace("complex", "c")
    return dtype + shape


@tree_repr.def_type(FunctionType)
@tree_str.def_type(FunctionType)
def _(func: Callable, **spec: Unpack[PPSpec]) -> str:
    del spec
    fullargspec = inspect.getfullargspec(func)

    header: list[str] = []

    if len(fullargspec.args):
        header += fullargspec.args
    if fullargspec.varargs is not None:
        header += ["*" + fullargspec.varargs]
    if len(fullargspec.kwonlyargs):
        if fullargspec.varargs is None:
            header += ["*"]
        header += fullargspec.kwonlyargs
    if fullargspec.varkw is not None:
        header += ["**" + fullargspec.varkw]

    *_, name = getattr(func, "__qualname__", "").split(".")
    return f"{name}({', '.join(header)})"


@tree_str.def_type(ft.partial)
def _(node: ft.partial, **spec: Unpack[PPSpec]) -> str:
    func = tree_str.pp(node.func, **spec)
    args = tree_str.pps(tree_str.pp, node.args, **spec)
    keywords = tree_str.pps(tree_str.kv_pp, node.keywords, **spec)
    return "partial(" + ",".join([func, args, keywords]) + ")"


@tree_str.def_type(list)
def _(node: list, **spec: Unpack[PPSpec]) -> str:
    return "[" + tree_str.pps(tree_str.pp, node, **spec) + "]"


@tree_str.def_type(tuple)
def _(node: tuple, **spec: Unpack[PPSpec]) -> str:
    if not hasattr(node, "_fields"):
        return "(" + tree_str.pps(tree_str.pp, node, **spec) + ")"
    name = type(node).__name__
    kvs = node._asdict().items()
    return name + "(" + tree_str.pps(tree_str.av_pp, kvs, **spec) + ")"


@tree_str.def_type(set)
def _(node: set, **spec: Unpack[PPSpec]) -> str:
    return "{" + tree_str.pps(tree_str.pp, node, **spec) + "}"


@tree_str.def_type(dict)
def _(node: dict, **spec: Unpack[PPSpec]) -> str:
    name = type(node).__name__
    return name + "(" + tree_str.pps(tree_str.av_pp, node.items(), **spec) + ")"


@tree_repr.def_type(str)
@tree_str.def_type(str)
def _(node: str, **spec: Unpack[PPSpec]) -> str:
    return node


for ndarray in arraylib.ndarrays:

    @tree_repr.def_type(ndarray)
    def array_pp(node, **spec: Unpack[PPSpec]) -> str:
        shape_dtype = ShapeDTypePP(arraylib.shape(node), arraylib.dtype(node))
        base = tree_repr.pp(shape_dtype, **spec)

        if not (arraylib.is_floating(node) or arraylib.is_integer(node)):
            return base
        if prod(arraylib.shape(node)) == 0:
            return base

        # Extended repr for numpy array, with extended information
        # this part of the function is inspired by
        # lovely-jax https://github.com/xl0/lovely-jax
        L, H = arraylib.min(node), arraylib.max(node)
        interval = "(" if math.isinf(L) else "["
        interval += f"{L},{H}" if arraylib.is_integer(node) else f"{L:.2f},{H:.2f}"
        interval += ")" if math.isinf(H) else "]"
        interval = interval.replace("inf", "∞")
        mean, std = f"{arraylib.mean(node):.2f}", f"{arraylib.std(node):.2f}"
        return f"{base}(μ={mean}, σ={std}, ∈{interval})"


@tree_repr.def_type(ft.partial)
def _(node: ft.partial, **spec: Unpack[PPSpec]) -> str:
    func = tree_repr.pp(node.func, **spec)
    args = tree_repr.pps(tree_repr.pp, node.args, **spec)
    keywords = tree_repr.pps(tree_repr.kv_pp, node.keywords, **spec)
    return "Partial(" + ",".join([func, args, keywords]) + ")"


@tree_repr.def_type(list)
def _(node: list, **spec: Unpack[PPSpec]) -> str:
    return "[" + tree_repr.pps(tree_repr.pp, node, **spec) + "]"


@tree_repr.def_type(tuple)
def _(node: tuple, **spec: Unpack[PPSpec]) -> str:
    if not hasattr(node, "_fields"):
        return "(" + tree_repr.pps(tree_repr.pp, node, **spec) + ")"
    name = type(node).__name__
    kvs = node._asdict().items()
    return name + "(" + tree_repr.pps(tree_repr.av_pp, kvs, **spec) + ")"


@tree_repr.def_type(set)
def _(node: set, **spec: Unpack[PPSpec]) -> str:
    return "{" + tree_repr.pps(tree_repr.pp, node, **spec) + "}"


@tree_repr.def_type(dict)
def _(node: dict, **spec: Unpack[PPSpec]) -> str:
    name = type(node).__name__
    return name + "(" + tree_repr.pps(tree_repr.av_pp, node.items(), **spec) + ")"


def tree_diagram(
    tree: Any,
    *,
    depth: int | float = float("inf"),
    is_leaf: Callable[[Any], None] | None = None,
    tabwidth: int = 4,
):
    """Pretty print arbitrary pytrees tree with tree structure diagram.

    Args:
        tree: arbitrary pytree.
        depth: depth of the tree to print. default is max depth.
        is_leaf: function to determine if a node is a leaf. default is None.
        tabwidth: tab width of the repr string. default is 4.

    Example:
        >>> import sepes as sp
        >>> @sp.autoinit
        ... class A(sp.TreeClass):
        ...     x: int = 10
        ...     y: int = (20,30)
        ...     z: int = 40

        >>> @sp.autoinit
        ... class B(sp.TreeClass):
        ...     a: int = 10
        ...     b: tuple = (20,30, A())

        >>> print(sp.tree_diagram(B(), depth=0))
        B(...)

        >>> print(sp.tree_diagram(B(), depth=1))
        B
        ├── .a=10
        └── .b=(...)


        >>> print(sp.tree_diagram(B(), depth=2))
        B
        ├── .a=10
        └── .b:tuple
            ├── [0]=20
            ├── [1]=30
            └── [2]=A(...)
    """
    vmark = ("│\t")[:tabwidth]  # vertical mark
    lmark = ("└" + "─" * (tabwidth - 2) + (" \t"))[:tabwidth]  # last mark
    cmark = ("├" + "─" * (tabwidth - 2) + (" \t"))[:tabwidth]  # connector mark
    smark = (" \t")[:tabwidth]  # space mark

    def step(
        node: Node,
        depth: int = 0,
        is_last: bool = False,
        is_lasts: tuple[bool, ...] = (),
    ) -> str:
        indent = "".join(smark if is_last else vmark for is_last in is_lasts[:-1])
        branch = (lmark if is_last else cmark) if depth > 0 else ""

        if (child_count := len(node.children)) == 0:
            (key, _), value = node.data
            text = f"{indent}"
            text += f"{branch}{key}=" if key is not None else ""
            text += tree_repr(value, depth=0)
            return text + "\n"

        (key, type), _ = node.data

        text = f"{indent}{branch}"
        text += f"{key}:" if key is not None else ""
        text += f"{type.__name__}\n"

        for i, child in enumerate(node.children.values()):
            text += step(
                child,
                depth=depth + 1,
                is_last=(i == child_count - 1),
                is_lasts=is_lasts + (i == child_count - 1,),
            )
        return text

    is_path_leaf = is_path_leaf_depth_factory(depth)
    root = construct_tree(tree, is_leaf=is_leaf, is_path_leaf=is_path_leaf)
    text = step(root, is_last=len(root.children) == 1)
    return (text if tabwidth is None else text.expandtabs(tabwidth)).rstrip()


def format_width(string, width=60):
    """Strip newline/tab characters if less than max width."""
    children_length = len(string) - string.count("\n") - string.count("\t")
    if children_length > width:
        return string
    return string.replace("\n", "").replace("\t", "")


# table printing

Row: TypeAlias = Sequence[str]  # list of columns


def _table(rows: list[Row]) -> str:
    """Generate a table from a list of rows."""

    def line(text: Row, widths: list[int]) -> str:
        return "\n".join(
            "│"
            + "│".join(col.ljust(width) for col, width in zip(line_row, widths))
            + "│"
            for line_row in zip_longest(*[t.split("\n") for t in text], fillvalue="")
        )

    widths = [max(map(len, "\n".join(col).split("\n"))) for col in zip(*rows)]
    spaces: Row = ["─" * width for width in widths]

    return (
        ("┌" + "┬".join(spaces) + "┐")
        + "\n"
        + ("\n├" + "┼".join(spaces) + "┤\n").join(line(row, widths) for row in rows)
        + "\n"
        + ("└" + "┴".join(spaces) + "┘")
    )


def size_pp(size: int, **spec: Unpack[PPSpec]):
    del spec
    order_alpha = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    size_order = int(math.log(size, 1024)) if size else 0
    text = f"{(size)/(1024**size_order):.2f}{order_alpha[size_order]}"
    return text


def tree_summary(
    tree: PyTree,
    *,
    depth: int | float = float("inf"),
    is_leaf: Callable[[Any], None] | None = None,
) -> str:
    """Print a summary of an arbitrary pytree.

    Args:
        tree: A pytree.
        depth: max depth to display the tree. Defaults to maximum depth.
        is_leaf: function to determine if a node is a leaf. Defaults to ``None``

    Returns:
        String summary of the tree structure:
            - First column: path to the node.
            - Second column: type of the node. to control the displayed type use
              ``tree_summary.def_type(type, func)`` to define a custom type display function.
            - Third column: number of leaves in the node. for arrays the number of leaves
              is the number of elements in the array, otherwise its 1. to control the
              number of leaves of a node use ``tree_summary.def_count(type,func)``
            - Fourth column: size of the node in bytes. if the node is array the size
              is the size of the array in bytes, otherwise the size is not displayed.
              to control the size of a node use ``tree_summary.def_size(type,func)``
            - Last row: type of parent, number of leaves of the parent

    Example:
        >>> import sepes as sp
        >>> import jax.numpy as jnp
        >>> print(sp.tree_summary([1, [2, [3]], jnp.array([1, 2, 3])]))
        ┌─────────┬────────────────────────────────────┬─────┬──────┐
        │Name     │Type                                │Count│Size  │
        ├─────────┼────────────────────────────────────┼─────┼──────┤
        │[0]      │int                                 │1    │      │
        ├─────────┼────────────────────────────────────┼─────┼──────┤
        │[1][0]   │int                                 │1    │      │
        ├─────────┼────────────────────────────────────┼─────┼──────┤
        │[1][1][0]│int                                 │1    │      │
        ├─────────┼────────────────────────────────────┼─────┼──────┤
        │[2]      │i32[3]                              │3    │12.00B│
        ├─────────┼────────────────────────────────────┼─────┼──────┤
        │Σ        │list[int,list[int,list[int]],i32[3]]│6    │12.00B│
        └─────────┴────────────────────────────────────┴─────┴──────┘

    Example:
        Display flops of a function in tree summary

        >>> import jax
        >>> import functools as ft
        >>> import sepes as sp
        >>> def count_flops(func, *args, **kwargs) -> int:
        ...     cost_analysis = jax.jit(func).lower(*args, **kwargs).cost_analysis()
        ...     return cost_analysis["flops"] if "flops" in cost_analysis else 0
        >>> class Flops:
        ...     def __init__(self, func, *args, **kwargs):
        ...         self.func = ft.partial(func, *args, **kwargs)
        >>> @sp.tree_summary.def_count(Flops)
        ... def _(node: Flops) -> int:
        ...     return count_flops(node.func)
        >>> @sp.tree_summary.def_type(Flops)
        ... def _(node: Flops) -> str:
        ...     return f"Flops({sp.tree_repr(node.func.func)})"
        >>> tree = dict(a=1, b=Flops(jax.nn.relu, jax.numpy.ones((10, 1))))
        >>> print(sp.tree_summary(tree))
        ┌─────┬───────────────────┬─────┬────┐
        │Name │Type               │Count│Size│
        ├─────┼───────────────────┼─────┼────┤
        │['a']│int                │1    │    │
        ├─────┼───────────────────┼─────┼────┤
        │['b']│Flops(jit(relu(x)))│10.0 │    │
        ├─────┼───────────────────┼─────┼────┤
        │Σ    │dict               │11.0 │    │
        └─────┴───────────────────┴─────┴────┘

    Example:
        Register custom type size rule

        >>> import jax
        >>> import sepes as sp
        >>> def func(x):
        ...     print(sp.tree_summary(x))
        ...     return x
        >>> class AbstractZero: ...
        >>> @sp.tree_summary.def_size(AbstractZero)
        ... def _(node: AbstractZero) -> int:
        ...     return 0
        >>> print(sp.tree_summary(AbstractZero()))
        ┌────┬────────────┬─────┬────┐
        │Name│Type        │Count│Size│
        ├────┼────────────┼─────┼────┤
        │Σ   │AbstractZero│1    │    │
        └────┴────────────┴─────┴────┘
    """
    treelib = sepes._src.backend.treelib
    empty_trace = ((), ())
    total_rows: list[list[str]] = [["Name", "Type", "Count", "Size"]]
    total_count = total_size = 0

    def tree_size(tree: PyTree) -> int:
        def reduce_func(acc, node):
            return acc + tree_summary.size_dispatcher(node)

        leaves, _ = treelib.flatten(tree)
        return ft.reduce(reduce_func, leaves, 0)

    def tree_count(tree: PyTree) -> int:
        def reduce_func(acc, node):
            return acc + tree_summary.count_dispatcher(node)

        leaves, _ = treelib.flatten(tree)
        return ft.reduce(reduce_func, leaves, 0)

    traces_leaves = tree_type_path_leaves(
        tree=tree,
        is_leaf=is_leaf,
        is_path_leaf=is_path_leaf_depth_factory(depth),
    )

    for trace, leaf in traces_leaves:
        total_count += (count := tree_count(leaf))
        total_size += (size := tree_size(leaf))

        if trace == empty_trace:
            continue

        paths, _ = trace
        path_string = treelib.keystr(paths)
        type_string = tree_summary.type_dispatcher(leaf)
        count_string = f"{count:,}" if count else ""
        size_string = size_pp(size) if size else ""
        total_rows += [[path_string, type_string, count_string, size_string]]

    path_string = "Σ"
    type_string = tree_summary.type_dispatcher(tree)
    count_string = f"{total_count:,}" if total_count else ""
    size_string = size_pp(total_size) if total_size else ""
    total_rows += [[path_string, type_string, count_string, size_string]]
    return _table(total_rows)


tree_summary.count_dispatcher = ft.singledispatch(lambda x: 1)
tree_summary.def_count = tree_summary.count_dispatcher.register
tree_summary.size_dispatcher = ft.singledispatch(lambda x: 0)
tree_summary.def_size = tree_summary.size_dispatcher.register
tree_summary.type_dispatcher = ft.singledispatch(lambda x: type(x).__name__)
tree_summary.def_type = tree_summary.type_dispatcher.register


for ndarray in arraylib.ndarrays:

    @tree_summary.def_size(ndarray)
    def _(node) -> int:
        return arraylib.nbytes(node)

    @tree_summary.def_count(ndarray)
    def _(node) -> int:
        return prod(arraylib.shape(node))

    @tree_summary.def_type(ndarray)
    def _(node: Any) -> str:
        """Return the type repr of the node."""
        shape = arraylib.shape(node)
        dtype = arraylib.dtype(node)
        return tree_repr(ShapeDTypePP(shape, dtype))

@tree_summary.def_type(list)
@tree_summary.def_type(tuple)
def _(node: tuple) -> str:
    # - output Container[types,...] instead of just container type in the type col.
    # - usually this encounterd if the tree_summary depth is not inf
    #   so the tree leaves could contain non-atomic types.
    treelib = sepes._src.backend.treelib

    one_level_types = treelib.map(
        tree_summary.type_dispatcher,
        node,
        is_leaf=lambda x: False if id(x) == id(node) else True,
    )
    return f"{type(node).__name__}[{','.join(one_level_types)}]"


if is_package_avaiable("jax"):
    # jax pretty printing extra handlers
    import jax

    @tree_str.def_type(jax.ShapeDtypeStruct)
    @tree_repr.def_type(jax.ShapeDtypeStruct)
    def _(node: jax.ShapeDtypeStruct, **spec: Unpack[PPSpec]) -> str:
        shape = arraylib.shape(node)
        dtype = arraylib.dtype(node)
        return tree_repr.dispatch(ShapeDTypePP(shape, dtype), **spec)

    # more readable repr for jax custom_jvp functions
    # for instance: jax.nn.relu is displayed as relu(x)
    # instead of <jax._src.custom_derivatives.custom_jvp object at ...>
    @tree_str.def_type(jax.custom_jvp)
    @tree_repr.def_type(jax.custom_jvp)
    def _(node: jax.custom_jvp, **spec: Unpack[PPSpec]) -> str:
        node = getattr(node, "__wrapped__", "<unknown>")
        return tree_repr.dispatch(node, **spec)

    @tree_str.def_type(type(jax.jit(lambda x: x)))
    @tree_repr.def_type(type(jax.jit(lambda x: x)))
    def _(node, **spec: Unpack[PPSpec]) -> str:
        # on copy pjit loses the __wrapped__ attribute (maybe a bug in jax)
        # so we need to handle it here
        node = getattr(node, "__wrapped__", "<unknown>")
        return "jit(" + tree_repr.dispatch(node, **spec) + ")"

    # without this rule, Tracer will be handled by the array handler
    # that display min/max/mean/std of the array.However `Tracer` does not
    # have these attributes so this will cause an error upon calculation.
    @tree_summary.def_type(jax.core.Tracer)
    @tree_repr.def_type(jax.core.Tracer)
    @tree_str.def_type(jax.core.Tracer)
    def _(node, **spec: Unpack[PPSpec]) -> str:
        shape = node.aval.shape
        dtype = node.aval.dtype
        string = tree_repr.dispatch(ShapeDTypePP(shape, dtype), **spec)
        return f"{type(node).__name__}({string})"

    # handle the sharding info if it is sharded
    @tree_summary.def_type(jax.Array)
    def _(node: Any) -> str:
        """Return the type repr of the node."""
        # global shape
        global_shape = arraylib.shape(node)
        shard_shape = node.sharding.shard_shape(global_shape)
        dtype = arraylib.dtype(node)
        global_info = tree_repr(ShapeDTypePP(global_shape, dtype))

        if global_shape == shard_shape:
            return global_info
        shard_info = tree_repr(ShapeDTypePP(shard_shape, dtype))
        return f"G:{global_info}\nS:{shard_info}"
