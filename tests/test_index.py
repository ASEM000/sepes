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

from __future__ import annotations

import os
import re
from collections import namedtuple
from typing import NamedTuple

import pytest

from sepes._src.backend import arraylib, backend, treelib
from sepes._src.code_build import autoinit
from sepes._src.tree_base import TreeClass, _mutable_instance_registry
from sepes._src.tree_index import BaseKey, at
from sepes._src.tree_util import is_tree_equal, leafwise, value_and_tree

test_arraylib = os.environ.get("SEPES_TEST_ARRAYLIB", "numpy")

if test_arraylib == "jax":
    import jax.numpy as arraylib

    default_int = arraylib.int32
elif test_arraylib in ["numpy", "default"]:
    import numpy as arraylib

    default_int = arraylib.int64
elif test_arraylib == "torch":
    import torch as arraylib

    arraylib.array = arraylib.tensor
    default_int = arraylib.int64
else:
    raise ImportError("no backend installed")


@leafwise
class ClassTree(TreeClass):
    """Tree class for testing."""

    def __init__(self, a: int, b: dict, e: int):
        """Initialize."""
        self.a = a
        self.b = b
        self.e = e


@leafwise
class ClassSubTree(TreeClass):
    """Tree class for testing."""

    def __init__(self, c: int, d: int):
        """Initialize."""
        self.c = c
        self.d = d


# key
tree1 = dict(a=1, b=dict(c=2, d=3), e=4)
tree2 = ClassTree(1, dict(c=2, d=3), 4)
tree3 = ClassTree(1, ClassSubTree(2, 3), 4)

# index
tree4 = [1, [2, 3], 4]
tree5 = (1, (2, 3), 4)
tree6 = [1, ClassSubTree(2, 3), 4]

# mixed
tree7 = dict(a=1, b=[2, 3], c=4)
tree8 = dict(a=1, b=ClassSubTree(c=2, d=3), e=4)

# by mask
tree9 = ClassTree(1, dict(c=2, d=3), arraylib.array([4, 5, 6]))

_X = 1_000


@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, dict(a=None, b=dict(c=2, d=None), e=None), ("b", "c")],
        [tree2, ClassTree(None, dict(c=2, d=None), None), ("b", "c")],
        [tree3, ClassTree(None, ClassSubTree(2, None), None), ("b", "c")],
        # by index
        [tree4, [None, [2, None], None], (1, 0)],
        [tree5, (None, (2, None), None), (1, 0)],
        # mixed
        [tree7, dict(a=None, b=[2, None], c=None), ("b", 0)],
        # by regex
        [tree1, dict(a=None, b=dict(c=2, d=None), e=None), ("b", re.compile("c"))],
        [tree2, ClassTree(None, dict(c=2, d=None), None), ("b", re.compile("c"))],
        [tree3, ClassTree(None, ClassSubTree(2, None), None), ("b", re.compile("c"))],
        # by ellipsis
        [tree1, tree1, (...,)],
        [tree2, tree2, (...,)],
        [tree3, tree3, (...,)],
        [tree4, tree4, (...,)],
        [tree5, tree5, (...,)],
        [tree6, tree6, (...,)],
        [tree7, tree7, (...,)],
        [tree8, tree8, (...,)],
    ],
)
def test_indexer_get(tree, expected, where):
    indexer = at(tree, where=where)
    assert is_tree_equal(indexer.get(), expected)
    assert is_tree_equal(indexer.get(is_parallel=True), expected)


@pytest.mark.skipif(backend == "default", reason="no array backend installed")
@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by boolean mask
        [
            tree9,
            ClassTree(None, dict(c=None, d=3), arraylib.array([4, 5, 6])),
            (tree9 > 2,),
        ],
        [
            tree9,
            ClassTree(None, dict(c=None, d=None), arraylib.array([5, 6])),
            (tree9 > 4,),
        ],
        [tree9, tree9, (tree9 == tree9,)],
        [
            tree9,
            ClassTree(
                None, dict(c=None, d=None), arraylib.array([], dtype=default_int)
            ),
            (tree9 != tree9,),
        ],
        # by ellipsis
        [tree9, tree9, (...,)],
    ],
)
def test_array_indexer_get(tree, expected, where):
    indexer = at(tree, where=where)
    assert is_tree_equal(indexer.get(), expected)
    assert is_tree_equal(indexer.get(is_parallel=True), expected)


@pytest.mark.skipif(backend != "jax", reason="test jax jit with get")
def test_get_fill_value():
    import jax
    import jax.numpy as jnp

    tree = dict(a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6]))
    mask = dict(
        a=jnp.array([False, True, False]),
        b=jnp.array([False, True, False]),
    )

    @jax.jit
    def jit_func(tree):
        return at(tree)[mask].get(fill_value=0)

    out = jit_func(tree)
    a = out["a"]
    b = out["b"]
    assert jnp.all(a == jnp.array([0, 2, 0]))
    assert jnp.all(b == jnp.array([0, 5, 0]))


@pytest.mark.parametrize(
    ["tree", "expected", "where", "set_value"],
    [
        # by name
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", "c"), _X],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", "c"), _X],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", "c"), _X],
        # by index
        [tree4, [1, [_X, 3], 4], (1, 0), _X],
        [tree5, (1, (_X, 3), 4), (1, 0), _X],
        # mixed
        [tree7, dict(a=1, b=[2, _X], c=4), ("b", 1), _X],
        # by regex
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", re.compile("c")), _X],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", re.compile("c")), _X],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", re.compile("c")), _X],
        # by ellipsis
        [
            tree1,
            dict(a=_X, b=dict(c=_X, d=_X), e=_X),
            (...,),
            dict(a=_X, b=dict(c=_X, d=_X), e=_X),
        ],
        [tree2, ClassTree(_X, dict(c=_X, d=_X), _X), (...,), _X],
        [tree3, ClassTree(_X, ClassSubTree(_X, _X), _X), (...,), _X],
        [tree4, [_X, [_X, _X], _X], (...,), _X],
        [tree5, (_X, (_X, _X), _X), (...,), _X],
        [tree6, [_X, ClassSubTree(_X, _X), _X], (...,), _X],
        [tree7, dict(a=_X, b=[_X, _X], c=_X), (...,), _X],
        [tree8, dict(a=_X, b=ClassSubTree(c=_X, d=_X), e=_X), (...,), _X],
    ],
)
def test_indexer_set(tree, expected, where, set_value):
    indexer = at(tree, where=where)
    assert is_tree_equal(indexer.set(set_value), expected)
    assert is_tree_equal(indexer.set(set_value, is_parallel=True), expected)


@pytest.mark.skipif(backend == "default", reason="no array backend installed")
@pytest.mark.parametrize(
    ["tree", "expected", "where", "set_value"],
    [
        # by name
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", "c"), _X],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", "c"), _X],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", "c"), _X],
        # by index
        [tree4, [1, [_X, 3], 4], (1, 0), _X],
        [tree5, (1, (_X, 3), 4), (1, 0), _X],
        # mixed
        [tree7, dict(a=1, b=[2, _X], c=4), ("b", 1), _X],
        # by regex
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", re.compile("c")), _X],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", re.compile("c")), _X],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", re.compile("c")), _X],
        # by ellipsis
        [
            tree1,
            dict(a=_X, b=dict(c=_X, d=_X), e=_X),
            (...,),
            dict(a=_X, b=dict(c=_X, d=_X), e=_X),
        ],
        [tree2, ClassTree(_X, dict(c=_X, d=_X), _X), (...,), _X],
        [tree3, ClassTree(_X, ClassSubTree(_X, _X), _X), (...,), _X],
        [tree4, [_X, [_X, _X], _X], (...,), _X],
        [tree5, (_X, (_X, _X), _X), (...,), _X],
        [tree6, [_X, ClassSubTree(_X, _X), _X], (...,), _X],
        [tree7, dict(a=_X, b=[_X, _X], c=_X), (...,), _X],
        [tree8, dict(a=_X, b=ClassSubTree(c=_X, d=_X), e=_X), (...,), _X],
    ],
)
def test_array_indexer_set(tree, expected, where, set_value):
    indexer = at(tree, where=where)
    assert is_tree_equal(indexer.set(set_value), expected)
    assert is_tree_equal(indexer.set(set_value, is_parallel=True), expected)


@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", "c")],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", "c")],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", "c")],
        # by index
        [tree4, [1, [_X, 3], 4], (1, 0)],
        [tree5, (1, (_X, 3), 4), (1, 0)],
        # mixed
        [tree7, dict(a=1, b=[2, _X], c=4), ("b", 1)],
        # by regex
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", re.compile("c"))],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", re.compile("c"))],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", re.compile("c"))],
        # by ellipsis
        [tree1, dict(a=_X, b=dict(c=_X, d=_X), e=_X), (...,)],
        [tree2, ClassTree(_X, dict(c=_X, d=_X), _X), (...,)],
        [tree3, ClassTree(_X, ClassSubTree(_X, _X), _X), (...,)],
        [tree4, [_X, [_X, _X], _X], (...,)],
        [tree5, (_X, (_X, _X), _X), (...,)],
        [tree6, [_X, ClassSubTree(_X, _X), _X], (...,)],
        [tree7, dict(a=_X, b=[_X, _X], c=_X), (...,)],
        [tree8, dict(a=_X, b=ClassSubTree(c=_X, d=_X), e=_X), (...,)],
    ],
)
def test_indexer_apply(tree, expected, where):
    indexer = at(tree, where=where)
    assert is_tree_equal(indexer.apply(lambda _: _X), expected)
    assert is_tree_equal(
        indexer.apply(lambda _: _X, is_parallel=True),
        expected,
    )


@pytest.mark.skipif(backend == "default", reason="no array backend installed")
@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", "c")],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", "c")],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", "c")],
        # by index
        [tree4, [1, [_X, 3], 4], (1, 0)],
        [tree5, (1, (_X, 3), 4), (1, 0)],
        # mixed
        [tree7, dict(a=1, b=[2, _X], c=4), ("b", 1)],
        # by regex
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", re.compile("c"))],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", re.compile("c"))],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", re.compile("c"))],
        # by ellipsis
        [tree1, dict(a=_X, b=dict(c=_X, d=_X), e=_X), (...,)],
        [tree2, ClassTree(_X, dict(c=_X, d=_X), _X), (...,)],
        [tree3, ClassTree(_X, ClassSubTree(_X, _X), _X), (...,)],
        [tree4, [_X, [_X, _X], _X], (...,)],
        [tree5, (_X, (_X, _X), _X), (...,)],
        [tree6, [_X, ClassSubTree(_X, _X), _X], (...,)],
        [tree7, dict(a=_X, b=[_X, _X], c=_X), (...,)],
        [tree8, dict(a=_X, b=ClassSubTree(c=_X, d=_X), e=_X), (...,)],
    ],
)
def test_array_indexer_apply(tree, expected, where):
    indexer = at(tree, where=where)
    assert is_tree_equal(indexer.apply(lambda _: _X), expected)
    assert is_tree_equal(
        indexer.apply(lambda _: _X, is_parallel=True),
        expected,
    )


@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, 5, ("b", ("c", "d"))],
        [tree2, 5, ("b", ("c", "d"))],
        [tree3, 5, ("b", ("c", "d"))],
        # by index
        [tree4, 5, (1, (0, 1))],
        [tree5, 5, (1, (0, 1))],
        # mixed
        [tree7, 5, ("b", (0, 1))],
        # by regex
        [tree1, 5, ("b", re.compile("c|d"))],
        [tree2, 5, ("b", re.compile("c|d"))],
        [tree3, 5, ("b", re.compile("c|d"))],
        # by ellipsis
        [tree1, 1 + 2 + 3 + 4, (...,)],
        [tree2, 1 + 2 + 3 + 4, (...,)],
        [tree3, 1 + 2 + 3 + 4, (...,)],
        [tree4, 1 + 2 + 3 + 4, (...,)],
        [tree5, 1 + 2 + 3 + 4, (...,)],
        [tree6, 1 + 2 + 3 + 4, (...,)],
        [tree7, 1 + 2 + 3 + 4, (...,)],
        [tree8, 1 + 2 + 3 + 4, (...,)],
    ],
)
def test_indexer_reduce(tree, expected, where):
    indexer = at(tree, where=where)
    assert is_tree_equal(
        indexer.reduce(lambda x, y: x + y, initializer=0),
        expected,
    )


@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, 5, ("b", ("c", "d"))],
        [tree2, 5, ("b", ("c", "d"))],
        [tree3, 5, ("b", ("c", "d"))],
        # by index
        [tree4, 5, (1, (0, 1))],
        [tree5, 5, (1, (0, 1))],
        # mixed
        [tree7, 5, ("b", (0, 1))],
        # by regex
        [tree1, 5, ("b", re.compile("c|d"))],
        [tree2, 5, ("b", re.compile("c|d"))],
        [tree3, 5, ("b", re.compile("c|d"))],
        # by ellipsis
        [tree1, 1 + 2 + 3 + 4, (...,)],
        [tree2, 1 + 2 + 3 + 4, (...,)],
        [tree3, 1 + 2 + 3 + 4, (...,)],
        [tree4, 1 + 2 + 3 + 4, (...,)],
        [tree5, 1 + 2 + 3 + 4, (...,)],
        [tree6, 1 + 2 + 3 + 4, (...,)],
        [tree7, 1 + 2 + 3 + 4, (...,)],
        [tree8, 1 + 2 + 3 + 4, (...,)],
    ],
)
def test_array_indexer_reduce(tree, expected, where):
    indexer = at(tree, where=where)
    assert is_tree_equal(
        indexer.reduce(lambda x, y: x + y, initializer=0),
        expected,
    )


@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, (dict(a=1, b=dict(c=2, d=5), e=4), 3), ("b", ("c", "d"))],
        [tree2, (ClassTree(1, dict(c=2, d=5), 4), 3), ("b", ("c", "d"))],
        [tree3, (ClassTree(1, ClassSubTree(2, 5), 4), 3), ("b", ("c", "d"))],
        # by index
        [tree4, ([1, [2, 5], 4], 3), (1, (0, 1))],
        [tree5, ((1, (2, 5), 4), 3), (1, (0, 1))],
        # mixed
        [tree7, (dict(a=1, b=[2, 5], c=4), 3), ("b", (0, 1))],
        # [tree8, (dict(a=1, b=ClassSubTree(c=2, d=5), e=4), 3), ("b", (0, 1))],
        # by regex
        [tree1, (dict(a=1, b=dict(c=2, d=5), e=4), 3), ("b", re.compile("c|d"))],
        [tree2, (ClassTree(1, dict(c=2, d=5), 4), 3), ("b", re.compile("c|d"))],
        [tree3, (ClassTree(1, ClassSubTree(2, 5), 4), 3), ("b", re.compile("c|d"))],
    ],
)
def test_indexer_scan(tree, expected, where):
    indexer = at(tree, where=where)
    assert is_tree_equal(
        indexer.scan(lambda x, s: (x + s, x), state=0),
        expected,
    )


def test_method_call():
    @leafwise
    @autoinit
    class Tree(TreeClass):
        a: int = 1

        def increment(self):
            self.a += 1

        def show(self):
            return 1

    t = Tree()

    @autoinit
    class Tree2(TreeClass):
        b: Tree = Tree()

    assert is_tree_equal(value_and_tree(lambda T: T.increment())(t)[1], Tree(2))
    assert is_tree_equal(value_and_tree(lambda T: T.b.show())(Tree2())[0], 1)

    with pytest.raises(AttributeError):
        value_and_tree(t.bla)()

    with pytest.raises(TypeError):
        value_and_tree(t.a)()

    @leafwise
    @autoinit
    class A(TreeClass):
        a: int

        def __call__(self, x):
            self.a += x
            return x

    a = A(1)
    _, b = value_and_tree(lambda A: A(2))(a)

    assert treelib.flatten(a)[0] == [1]
    assert treelib.flatten(b)[0] == [3]

    with pytest.raises(TypeError):
        a.at[0](1)


def test_call_context():
    @autoinit
    class L2(TreeClass):
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    _mutable_instance_registry.add(id(t))
    t.delete("a")
    _mutable_instance_registry.discard(id(t))

    with pytest.raises(AttributeError):
        t.delete("a")


@pytest.mark.parametrize("where", [("a", [1]), (0, [1])])
def test_unsupported_where(where):
    t = namedtuple("a", ["x", "y"])(1, 2)
    with pytest.raises(NotImplementedError):
        at(t, where=where).get()


@pytest.mark.skipif(backend != "jax", reason="jax backend needed")
def test_custom_key_jax():
    class NameTypeContainer(NamedTuple):
        name: str
        type: type

    class Tree:
        def __init__(self, a, b) -> None:
            self.a = a
            self.b = b

        @property
        def at(self):
            return at(self)

    if backend == "jax":
        import jax.tree_util as jtu

        def tree_flatten(tree):
            return (tree.a, tree.b), None

        def tree_unflatten(aux_data, children):
            return Tree(*children)

        def tree_flatten_with_keys(tree):
            ak = (NameTypeContainer("a", type(tree.a)), tree.a)
            bk = (NameTypeContainer("b", type(tree.b)), tree.b)
            return (ak, bk), None

        jtu.register_pytree_with_keys(
            nodetype=Tree,
            flatten_func=tree_flatten,
            flatten_with_keys=tree_flatten_with_keys,
            unflatten_func=tree_unflatten,
        )


@pytest.mark.skipif(backend not in ["torch", "numpy"], reason="optree backend needed")
def test_custom_key_optreee():
    class NameTypeContainer(NamedTuple):
        name: str
        type: type

    class Tree:
        def __init__(self, a, b) -> None:
            self.a = a
            self.b = b

        @property
        def at(self):
            return at(self)

    import optree as ot

    def tree_flatten(tree):
        ka = NameTypeContainer("a", type(tree.a))
        kb = NameTypeContainer("b", type(tree.b))
        return (tree.a, tree.b), None, (ka, kb)

    def tree_unflatten(aux_data, children):
        return Tree(*children)

    ot.register_pytree_node(
        Tree,
        flatten_func=tree_flatten,
        unflatten_func=tree_unflatten,
        namespace="sepes",
    )

    tree = Tree(1, 2)

    class MatchNameType(BaseKey):
        def __init__(self, name, type):
            self.name = name
            self.type = type

        def __eq__(self, other):
            if isinstance(other, NameTypeContainer):
                return other == (self.name, self.type)
            return False

    assert treelib.tree_flatten(tree.at[MatchNameType("a", int)].get())[0] == [1]


def test_repr_str():
    @autoinit
    class Tree(TreeClass):
        a: int = 1
        b: int = 2

    t = Tree()

    assert repr(t.at["a"]) == "at(Tree(a=1, b=2), where=['a'])"
    assert str(t.at["a"]) == "at(Tree(a=1, b=2), where=['a'])"
    assert repr(t.at[...]) == "at(Tree(a=1, b=2), where=[Ellipsis])"


def test_compat_mask():
    tree = [1, 2, [3, 4]]
    tree_ = at(tree)[[False, False, True]].set(10)
    assert tree_ == [1, 2, 10]


def test_pluck():
    tree = [1, 2, [3, 4]]
    subtrees = at(tree)[2].pluck()
    assert subtrees[0] == [3, 4]
    assert at(tree)[0, 1].pluck(1) == [1]
    assert at(tree)[0, 1].pluck(2) == [1, 2]

    tree = dict(a=1, b=2)
    assert at(tree)[...].pluck() == [1, 2]


@pytest.mark.skipif(backend != "jax", reason="jax backend needed")
def test_call():
    import jax.tree_util as jtu

    @jtu.register_pytree_with_keys_class
    class Counter:
        def __init__(self, count: int):
            self.count = count

        def tree_flatten_with_keys(self):
            return (["count", self.count],), None

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            del aux_data
            return cls(*children)

        def increment_count(self) -> int:
            # mutates the tree
            self.count += 1
            return self.count

        def __repr__(self) -> str:
            return f"Tree(count={self.count})"

    counter = Counter(0)
    cur_count, new_counter = value_and_tree(lambda C: C.increment_count())(counter)
    assert counter.count == 0
    assert cur_count == 1
    assert new_counter.count == 1
    assert not (counter is new_counter)


@pytest.mark.skipif(backend != "jax", reason="jax backend needed")
def test_pytree_matcher():
    import jax
    import jax.numpy as jnp
    import jax.tree_util as jtu

    class NameDtypeShapeMatcher(NamedTuple):
        name: str
        dtype: str
        shape: tuple[int, ...]

    def compare(matcher: NameDtypeShapeMatcher, key, leaf) -> bool:
        if not isinstance(leaf, jax.Array):
            return False
        if isinstance(key, str):
            key = key
        elif isinstance(key, jtu.GetAttrKey):
            key = key.name
        elif isinstance(key, jtu.DictKey):
            key = key.key
        return (
            matcher.name == key
            and matcher.dtype == leaf.dtype
            and matcher.shape == leaf.shape
        )

    tree = dict(weight=jnp.arange(9).reshape(3, 3), bias=jnp.zeros(3))
    at.def_rule(NameDtypeShapeMatcher, compare)
    matcher = NameDtypeShapeMatcher("weight", jnp.int32, (3, 3))
    to_symmetric = lambda x: (x + x.T) / 2
    at(tree)[matcher].apply(to_symmetric)
