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

import dataclasses as dc
import os
from collections import namedtuple
from typing import Any

import pytest

test_arraylib = os.environ.get("SEPES_TEST_ARRAYLIB", "numpy")
backend = os.environ.get("SEPES_BACKEND", "jax")
from sepes._src.code_build import autoinit, field
from sepes._src.tree_base import TreeClass
from sepes._src.tree_pprint import (
    _table,
    tree_diagram,
    tree_repr,
    tree_str,
    tree_summary,
)
from sepes._src.tree_util import leafwise

if test_arraylib == "jax":
    import jax.numpy as arraylib
elif test_arraylib in ["numpy", "default"]:
    import numpy as arraylib
elif test_arraylib == "torch":
    import torch as arraylib

    arraylib.array = arraylib.tensor
else:
    raise ImportError("no backend installed")


def test_table():
    col1 = ["1\n", "3"]
    col2 = ["2", "4000"]
    assert (
        _table([col1, col2])
        == "┌─┬────┐\n│1│3   │\n│ │    │\n├─┼────┤\n│2│4000│\n└─┴────┘"
    )


def test_func_pp():
    def example(a: int, b=1, *c, d, e=2, **f) -> str:
        ...  # fmt: skip

    assert tree_repr(example) == "example(a, b, *c, d, e, **f)"


@leafwise
@autoinit
class Repr1(TreeClass):
    """A simple tree class for repr testing."""

    a: int = 1
    b: str = "string"
    c: float = 1.0
    d: tuple = "a" * 5
    e: list = None
    f: set = None
    g: dict = None
    h: Any = None
    i: Any = None
    j: Any = None
    k: tuple = field(repr=False, default=(1, 2, 3))
    l: tuple = namedtuple("a", ["b", "c"])(1, 2)
    m: Any = arraylib.ones((5, 5), dtype=arraylib.float32)
    n: Any = arraylib.array(True)
    o: Any = arraylib.array([1, 2.0], dtype=arraylib.complex64)

    def __post_init__(self):
        self.h = arraylib.ones((5, 1), dtype=arraylib.float32)
        self.i = arraylib.ones((1, 6), dtype=arraylib.float32)
        self.j = arraylib.ones((1, 1, 4, 5), dtype=arraylib.float32)
        self.e = [10] * 5
        self.f = {1, 2, 3}
        self.g = {
            "a": "a" * 50,
            "b": "b" * 50,
            "c": arraylib.ones([5, 5], dtype=arraylib.float32),
        }


r1 = Repr1()


@pytest.mark.skipif(backend != "jax", reason="jax is not installed")
def test_repr():
    assert (
        tree_repr(r1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b=string, \n  c=1.0, \n  d=aaaaa, \n  e=[10, 10, 10, 10, 10], \n  f={1, 2, 3}, \n  g=dict(\n    a=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, \n    b=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, \n    c=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00])\n  ), \n  h=f32[5,1](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  i=f32[1,6](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  j=f32[1,1,4,5](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  l=a(b=1, c=2), \n  m=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  n=bool[], \n  o=c64[2]\n)"
    )

    assert (
        tree_repr(r1, depth=1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b=string, \n  c=1.0, \n  d=aaaaa, \n  e=[...], \n  f={...}, \n  g=dict(...), \n  h=f32[5,1](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  i=f32[1,6](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  j=f32[1,1,4,5](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  l=a(...), \n  m=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  n=bool[], \n  o=c64[2]\n)"
    )

    assert tree_repr(r1, depth=0) == "Repr1(...)"


@pytest.mark.skipif(backend != "jax", reason="jax is not installed")
def test_str():
    assert (
        tree_str(r1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b=string, \n  c=1.0, \n  d=aaaaa, \n  e=[10, 10, 10, 10, 10], \n  f={1, 2, 3}, \n  g=dict(\n    a=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, \n    b=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, \n    c=[[1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]]\n  ), \n  h=[[1.] [1.] [1.] [1.] [1.]], \n  i=[[1. 1. 1. 1. 1. 1.]], \n  j=[[[[1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]]]], \n  l=a(b=1, c=2), \n  m=[[1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]], \n  n=True, \n  o=[1.+0.j 2.+0.j]\n)"
    )

    assert (
        tree_str(r1, depth=1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b=string, \n  c=1.0, \n  d=aaaaa, \n  e=[...], \n  f={...}, \n  g=dict(...), \n  h=[[1.] [1.] [1.] [1.] [1.]], \n  i=[[1. 1. 1. 1. 1. 1.]], \n  j=[[[[1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]]]], \n  l=a(...), \n  m=[[1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]], \n  n=True, \n  o=[1.+0.j 2.+0.j]\n)"
    )


@pytest.mark.skipif(backend != "jax", reason="jax is not installed")
def test_tree_summary():
    assert (
        tree_summary(r1, depth=0)
        # trunk-ignore(flake8/E501)
        == "┌────┬─────┬─────┬───────┐\n│Name│Type │Count│Size   │\n├────┼─────┼─────┼───────┤\n│Σ   │Repr1│101  │341.00B│\n└────┴─────┴─────┴───────┘"
    )
    assert (
        tree_summary(r1, depth=1)
        # trunk-ignore(flake8/E501)
        == "┌────┬─────────────────────────┬─────┬───────┐\n│Name│Type                     │Count│Size   │\n├────┼─────────────────────────┼─────┼───────┤\n│.a  │int                      │1    │       │\n├────┼─────────────────────────┼─────┼───────┤\n│.b  │str                      │1    │       │\n├────┼─────────────────────────┼─────┼───────┤\n│.c  │float                    │1    │       │\n├────┼─────────────────────────┼─────┼───────┤\n│.d  │str                      │1    │       │\n├────┼─────────────────────────┼─────┼───────┤\n│.e  │list[int,int,int,int,int]│5    │       │\n├────┼─────────────────────────┼─────┼───────┤\n│.f  │set                      │1    │       │\n├────┼─────────────────────────┼─────┼───────┤\n│.g  │dict                     │27   │100.00B│\n├────┼─────────────────────────┼─────┼───────┤\n│.h  │f32[5,1]                 │5    │20.00B │\n├────┼─────────────────────────┼─────┼───────┤\n│.i  │f32[1,6]                 │6    │24.00B │\n├────┼─────────────────────────┼─────┼───────┤\n│.j  │f32[1,1,4,5]             │20   │80.00B │\n├────┼─────────────────────────┼─────┼───────┤\n│.k  │tuple[int,int,int]       │3    │       │\n├────┼─────────────────────────┼─────┼───────┤\n│.l  │a[int,int]               │2    │       │\n├────┼─────────────────────────┼─────┼───────┤\n│.m  │f32[5,5]                 │25   │100.00B│\n├────┼─────────────────────────┼─────┼───────┤\n│.n  │bool[]                   │1    │1.00B  │\n├────┼─────────────────────────┼─────┼───────┤\n│.o  │c64[2]                   │2    │16.00B │\n├────┼─────────────────────────┼─────┼───────┤\n│Σ   │Repr1                    │101  │341.00B│\n└────┴─────────────────────────┴─────┴───────┘"
    )

    assert (
        tree_summary(r1, depth=2)
        == tree_summary(r1)
        # trunk-ignore(flake8/E501)
        == "┌───────┬────────────┬─────┬───────┐\n│Name   │Type        │Count│Size   │\n├───────┼────────────┼─────┼───────┤\n│.a     │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.b     │str         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.c     │float       │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.d     │str         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[0]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[1]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[2]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[3]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[4]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.f     │set         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.g['a']│str         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.g['b']│str         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.g['c']│f32[5,5]    │25   │100.00B│\n├───────┼────────────┼─────┼───────┤\n│.h     │f32[5,1]    │5    │20.00B │\n├───────┼────────────┼─────┼───────┤\n│.i     │f32[1,6]    │6    │24.00B │\n├───────┼────────────┼─────┼───────┤\n│.j     │f32[1,1,4,5]│20   │80.00B │\n├───────┼────────────┼─────┼───────┤\n│.k[0]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.k[1]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.k[2]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.l.b   │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.l.c   │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.m     │f32[5,5]    │25   │100.00B│\n├───────┼────────────┼─────┼───────┤\n│.n     │bool[]      │1    │1.00B  │\n├───────┼────────────┼─────┼───────┤\n│.o     │c64[2]      │2    │16.00B │\n├───────┼────────────┼─────┼───────┤\n│Σ      │Repr1       │101  │341.00B│\n└───────┴────────────┴─────┴───────┘"
    )


@pytest.mark.skipif(backend != "jax", reason="jax is not installed")
def test_tree_diagram():
    assert tree_diagram(r1, depth=0) == "Repr1(...)"

    # trunk-ignore(flake8/E501)
    out = "Repr1\n├── .a=1\n├── .b=string\n├── .c=1.0\n├── .d=aaaaa\n├── .e=[...]\n├── .f={...}\n├── .g=dict(...)\n├── .h=f32[5,1](μ=1.00, σ=0.00, ∈[1.00,1.00])\n├── .i=f32[1,6](μ=1.00, σ=0.00, ∈[1.00,1.00])\n├── .j=f32[1,1,4,5](μ=1.00, σ=0.00, ∈[1.00,1.00])\n├── .k=(...)\n├── .l=a(...)\n├── .m=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00])\n├── .n=bool[]\n└── .o=c64[2]"

    assert tree_diagram(r1, depth=1) == out


@pytest.mark.skipif(backend != "jax", reason="jax is not installed")
def test_misc():
    x = (1, 2, 3)
    assert tree_repr(x) == tree_str(x) == "(1, 2, 3)"

    def example(a: int, b=1, *c, d, e=2, **f):
        pass

    assert tree_repr(example) == tree_str(example) == "example(a, b, *c, d, e, **f)"

    # example = jax.jit(example)
    # assert (
    #     tree_repr(example) == tree_str(example) == "jit(example(a, b, *c, d, e, **f))"
    # )

    assert (
        tree_repr(arraylib.ones([1, 2], dtype=arraylib.uint16))
        == "ui16[1,2](μ=1.00, σ=0.00, ∈[1,1])"
    )

    # @dc.dataclass
    # class Test:
    #     a: int = 1

    # assert tree_repr(Test()) == tree_str(Test()) == "Test(a=1)"
    # assert tree_repr(Test(), depth=0) == "Test(...)"


@pytest.mark.skipif(backend != "jax", reason="jax is not installed")
def test_extra_tree_diagram():
    @autoinit
    class L0(TreeClass):
        a: int = 1
        b: int = 2

    @autoinit
    class L1(TreeClass):
        c: L0 = L0()
        d: int = 3

    @autoinit
    class L2(TreeClass):
        e: int = 4
        f: L1 = L1()
        g: L0 = L0()
        h: int = 5

    tree = L2()
    # trunk-ignore(flake8/E501)
    out = "L2\n├── .e=4\n├── .f:L1\n│   ├── .c:L0\n│   │   ├── .a=1\n│   │   └── .b=2\n│   └── .d=3\n├── .g:L0\n│   ├── .a=1\n│   └── .b=2\n└── .h=5"

    assert (tree_diagram(tree)) == out

    @autoinit
    class L0(TreeClass):
        a: int = 1

    @autoinit
    class L1(TreeClass):
        b: L0 = L0()

    tree = L1()

    assert tree_diagram(tree) == "L1\n└── .b:L0\n    └── .a=1"


@pytest.mark.skipif(backend != "jax", reason="jax is not installed")
def test_invalid_depth():
    with pytest.raises(TypeError):
        tree_diagram(1, depth="a")
    with pytest.raises(TypeError):
        tree_summary(1, depth="a")


@pytest.mark.skipif(backend != "jax", reason="jax is not installed")
def test_tracer_repr():
    import jax

    @jax.jit
    def f(x):
        out = tree_repr(x)
        assert out == "DynamicJaxprTracer(f32[10,10])"
        out = tree_str(x)
        assert out == "DynamicJaxprTracer(f32[10,10])"
        return x

    f(jax.numpy.ones((10, 10)))


@pytest.mark.skipif(backend != "jax", reason="testing jax specific sharding info")
def test_jax_sharding_tree_summary():
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    x = jax.numpy.ones([4 * 4, 2 * 2])
    mesh = Mesh(devices=np.array(jax.devices()).reshape(4, 2), axis_names=["i", "j"])
    sharding = NamedSharding(mesh=mesh, spec=PartitionSpec("i", "j"))
    x = jax.device_put(x, device=sharding)
    assert (
        tree_summary(x)
        == "┌────┬───────────┬─────┬───────┐\n│Name│Type       │Count│Size   │\n├────┼───────────┼─────┼───────┤\n│Σ   │G:f32[16,4]│64   │256.00B│\n│    │S:f32[4,2] │     │       │\n└────┴───────────┴─────┴───────┘"
    )
