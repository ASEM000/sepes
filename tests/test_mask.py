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

import copy
from typing import Any

import pytest

from sepes._src.backend import backend, treelib
from sepes._src.code_build import autoinit
from sepes._src.tree_base import TreeClass
from sepes._src.tree_mask import (
    freeze,
    is_frozen,
    tree_mask,
    tree_unmask,
    unfreeze,
)
import os
from sepes._src.tree_util import is_tree_equal, leafwise, tree_hash

test_arraylib = os.environ.get("SEPES_TEST_ARRAYLIB", "numpy")

if test_arraylib == "jax":
    import jax.numpy as arraylib
elif test_arraylib in ["numpy", "default"]:
    import numpy as arraylib
elif test_arraylib == "torch":
    import torch as arraylib

    arraylib.array = arraylib.tensor
else:
    raise ImportError("no backend installed")


def test_freeze_unfreeze():
    @autoinit
    class A(TreeClass):
        a: int
        b: int

    a = A(1, 2)
    b = a.at[...].apply(freeze)
    c = (
        a.at["a"]
        .apply(unfreeze, is_leaf=is_frozen)
        .at["b"]
        .apply(unfreeze, is_leaf=is_frozen)
    )

    assert treelib.tree_flatten(a)[0] == [1, 2]
    assert treelib.tree_flatten(b)[0] == []
    assert treelib.tree_flatten(c)[0] == [1, 2]
    assert unfreeze(freeze(1.0)) == 1.0

    @autoinit
    class A(TreeClass):
        a: int
        b: int

    @autoinit
    class B(TreeClass):
        c: int = 3
        d: A = A(1, 2)

    @autoinit
    class A(TreeClass):
        a: int
        b: int

    a = A(1, 2)
    b = treelib.tree_map(freeze, a)
    c = (
        a.at["a"]
        .apply(unfreeze, is_leaf=is_frozen)
        .at["b"]
        .apply(unfreeze, is_leaf=is_frozen)
    )

    assert treelib.tree_flatten(a)[0] == [1, 2]
    assert treelib.tree_flatten(b)[0] == []
    assert treelib.tree_flatten(c)[0] == [1, 2]

    @autoinit
    class L0(TreeClass):
        a: int = 0

    @autoinit
    class L1(TreeClass):
        b: L0 = L0()

    @autoinit
    class L2(TreeClass):
        c: L1 = L1()

    t = treelib.tree_map(freeze, L2())

    assert treelib.tree_flatten(t)[0] == []
    assert treelib.tree_flatten(t.c)[0] == []
    assert treelib.tree_flatten(t.c.b)[0] == []

    class L1(TreeClass):
        def __init__(self):
            self.b = L0()

    class L2(TreeClass):
        def __init__(self):
            self.c = L1()

    t = treelib.tree_map(freeze, L2())
    assert treelib.tree_flatten(t.c)[0] == []
    assert treelib.tree_flatten(t.c.b)[0] == []


def test_freeze_errors():
    class T:
        pass

    @autoinit
    class Test(TreeClass):
        a: Any

    t = Test(T())

    # with pytest.raises(Exception):
    t.at[...].set(0)

    with pytest.raises(TypeError):
        t.at[...].apply(arraylib.sin)

    t.at[...].reduce(arraylib.sin)


def test_freeze_with_ops():
    @autoinit
    class A(TreeClass):
        a: int
        b: int

    @autoinit
    class B(TreeClass):
        c: int = 3
        d: A = A(1, 2)

    @autoinit
    class Test(TreeClass):
        a: int = 1
        b: float = freeze(1.0)
        c: str = freeze("test")

    t = Test()
    assert treelib.tree_flatten(t)[0] == [1]

    with pytest.raises(AttributeError):
        treelib.tree_map(freeze, t).a = 1

    with pytest.raises(AttributeError):
        treelib.tree_map(unfreeze, t).a = 1

    hash(t)

    t = Test()
    treelib.tree_map(unfreeze, t, is_leaf=is_frozen)
    treelib.tree_map(freeze, t)

    @autoinit
    class Test(TreeClass):
        a: int

    t = treelib.tree_map(freeze, (Test(100)))

    class Test(TreeClass):
        def __init__(self, x):
            self.x = x

    t = Test(arraylib.array([1, 2, 3]))
    assert is_tree_equal(t.at[...].set(None), Test(x=None))

    class T0:
        a: int = 1

    class T1:
        a: T0 = T0()

    t = T1()


def test_freeze_diagram():
    @autoinit
    class A(TreeClass):
        a: int
        b: int

    @autoinit
    class B(TreeClass):
        c: int = 3
        d: A = A(1, 2)

    a = B()
    a = a.at["d"].set(freeze(a.d))
    a = B()

    a = a.at["d"].set(freeze(a.d))  # = a.d.freeze()


def test_freeze_mask():
    @autoinit
    class Test(TreeClass):
        a: int = 1
        b: int = 2
        c: float = 3.0

    t = Test()

    assert treelib.tree_flatten(treelib.tree_map(freeze, t))[0] == []


def test_freeze_nondiff():
    @autoinit
    class Test(TreeClass):
        a: int = freeze(1)
        b: str = "a"

    t = Test()

    assert treelib.tree_flatten(t)[0] == ["a"]
    assert treelib.tree_flatten(treelib.tree_map(freeze, t))[0] == []
    assert treelib.tree_flatten(
        (treelib.tree_map(freeze, t)).at["b"].apply(unfreeze, is_leaf=is_frozen)
    )[0] == ["a"]

    @autoinit
    class T0(TreeClass):
        a: Test = Test()

    t = T0()

    assert treelib.tree_flatten(t)[0] == ["a"]
    assert treelib.tree_flatten(treelib.tree_map(freeze, t))[0] == []

    assert treelib.tree_flatten(t)[0] == ["a"]
    assert treelib.tree_flatten(treelib.tree_map(freeze, t))[0] == []


def test_freeze_nondiff_with_mask():
    @autoinit
    class L0(TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3

    @autoinit
    class L1(TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @autoinit
    class L2(TreeClass):
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    t = t.at["d"]["d"]["a"].apply(freeze)
    t = t.at["d"]["d"]["b"].apply(freeze)

    assert treelib.tree_flatten(t)[0] == [10, 20, 30, 1, 2, 3, 3]


def test_non_dataclass_input_to_freeze():
    assert treelib.tree_flatten(freeze(1))[0] == []


def test_tree_mask():
    @autoinit
    class L0(TreeClass):
        x: int = 2
        y: int = 3

    @leafwise
    @autoinit
    class L1(TreeClass):
        a: int = 1
        b: L0 = L0()

    tree = L1()

    assert treelib.tree_flatten(tree)[0] == [1, 2, 3]
    assert treelib.tree_flatten(treelib.tree_map(freeze, tree))[0] == []
    assert treelib.tree_flatten(treelib.tree_map(freeze, tree))[0] == []
    assert treelib.tree_flatten(tree.at[...].apply(freeze))[0] == []
    assert treelib.tree_flatten(tree.at[tree > 1].apply(freeze))[0] == [1]
    assert treelib.tree_flatten(tree.at[tree == 1].apply(freeze))[0] == [2, 3]
    assert treelib.tree_flatten(tree.at[tree < 1].apply(freeze))[0] == [1, 2, 3]

    assert treelib.tree_flatten(tree.at["a"].apply(freeze))[0] == [2, 3]
    assert treelib.tree_flatten(tree.at["b"].apply(freeze))[0] == [1]
    assert treelib.tree_flatten(tree.at["b"]["x"].apply(freeze))[0] == [1, 3]
    assert treelib.tree_flatten(tree.at["b"]["y"].apply(freeze))[0] == [1, 2]


def test_tree_unmask():
    @autoinit
    class L0(TreeClass):
        x: int = 2
        y: int = 3

    @leafwise
    @autoinit
    class L1(TreeClass):
        a: int = 1
        b: L0 = L0()

    tree = L1()

    frozen_tree = tree.at[...].apply(freeze)
    assert treelib.tree_flatten(frozen_tree)[0] == []

    mask = tree == tree
    unfrozen_tree = frozen_tree.at[mask].apply(unfreeze, is_leaf=is_frozen)
    assert treelib.tree_flatten(unfrozen_tree)[0] == [1, 2, 3]

    mask = tree > 1
    unfrozen_tree = frozen_tree.at[mask].apply(unfreeze, is_leaf=is_frozen)
    assert treelib.tree_flatten(unfrozen_tree)[0] == [2, 3]

    unfrozen_tree = frozen_tree.at["a"].apply(unfreeze, is_leaf=is_frozen)
    # assert treelib.tree_flatten(unfrozen_tree)[0] == [1]

    # unfrozen_tree = frozen_tree.at["b"].apply(unfreeze, is_leaf=is_frozen)
    # assert treelib.tree_flatten(unfrozen_tree)[0] == [2, 3]


def test_tree_mask_unfreeze():
    @autoinit
    class L0(TreeClass):
        x: int = 2
        y: int = 3

    @leafwise
    @autoinit
    class L1(TreeClass):
        a: int = 1
        b: L0 = L0()

    tree = L1()

    mask = tree == tree
    frozen_tree = tree.at[...].apply(freeze)
    unfrozen_tree = frozen_tree.at[mask].apply(unfreeze, is_leaf=is_frozen)
    assert treelib.tree_flatten(unfrozen_tree)[0] == [1, 2, 3]

    # frozen_tree = tree.at["a"].apply(freeze)
    # unfrozen_tree = frozen_tree.at["a"].apply(unfreeze, is_leaf=is_frozen)
    # assert treelib.tree_flatten(unfrozen_tree)[0] == [1, 2, 3]


def test_wrapper():
    # only apply last wrapper
    assert hash((freeze(1))) == tree_hash(1)

    # lhs = _HashableWrapper(1)
    # # test getter
    # assert lhs.__wrapped__ == 1
    # assert lhs.__wrapped__

    # # comparison with the wrapped object
    # assert lhs != 1
    # # hash of the wrapped object
    # assert hash(lhs) == tree_hash(1)

    # test immutability
    frozen_value = freeze(1)

    with pytest.raises(AttributeError):
        frozen_value.__wrapped__ = 2

    assert freeze(1) == freeze(1)
    assert f"{freeze(1)!r}" == "#1"

    wrapped = freeze(1)

    with pytest.raises(AttributeError):
        delattr(wrapped, "__wrapped__")

    assert wrapped != 1


@pytest.mark.skipif(backend == "default", reason="no array backend installed")
def test_tree_mask_tree_unmask():
    tree = [1, 2, 3.0]
    assert treelib.tree_flatten(tree_mask(tree))[0] == [3.0]
    assert treelib.tree_flatten(tree_unmask(tree_mask(tree)))[0] == [1, 2, 3.0]

    mask_func = lambda x: x < 2
    assert treelib.tree_flatten(tree_mask(tree, mask_func))[0] == [2, 3.0]

    assert freeze(freeze(1)) == freeze(1)

    assert tree_mask({"a": 1}, mask={"a": True}) == {"a": freeze(1)}

    with pytest.raises(ValueError):
        tree_mask({"a": 1}, mask=1.0)

    assert copy.copy(freeze(1)) == freeze(1)

    with pytest.raises(NotImplementedError):
        freeze(1) + 1


@pytest.mark.skipif(backend == "default", reason="no array backend installed")
def test_array_tree_mask_tree_unmask():
    frozen_array = tree_mask(arraylib.ones((5, 5)), mask=lambda _: True)

    assert frozen_array == frozen_array
    assert not (frozen_array == freeze(arraylib.ones((5, 6))))
    # assert not (frozen_array == freeze(arraylib.ones((5, 5)).astype(arraylib.uint8)))
    assert hash(frozen_array) == hash(frozen_array)
