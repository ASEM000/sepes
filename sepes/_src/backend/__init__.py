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

import functools as ft
import os
from importlib.util import find_spec
from typing import Literal, Callable
import logging
from contextlib import contextmanager


@ft.lru_cache(maxsize=None)
def is_package_avaiable(backend: str) -> bool:
    return find_spec(backend) is not None


# by importing the backend modules here, we register the backend implementations
# with the arraylib
if is_package_avaiable("torch"):
    import sepes._src.backend.arraylib.torch
if is_package_avaiable("jax"):
    import sepes._src.backend.arraylib.jax
if is_package_avaiable("numpy"):
    import sepes._src.backend.arraylib.numpy


def optree_backend():
    # no backend is available
    if not is_package_avaiable("optree"):
        raise ImportError("No backend is available. Please install `optree`.")
    from sepes._src.backend.treelib.optree import OpTreeTreeLib

    return OpTreeTreeLib()


def jax_backend():
    if not is_package_avaiable("jax"):
        raise ImportError("`jax` backend requires `jax` to be installed.")
    from sepes._src.backend.treelib.jax import JaxTreeLib

    return JaxTreeLib()


BackendLiteral = Literal["optree", "jax"]  # tree backend
backend: BackendLiteral = os.environ.get("SEPES_BACKEND", "default").lower()
backends_map: dict[BackendLiteral, Callable] = {}
backends_map["jax"] = jax_backend
backends_map["optree"] = optree_backend

if backend == "default":
    # backend promotion in essence is a search for the first available backend
    # in the following order: jax, optree
    # if no backend is available, then the default backend is used
    for backend_name in backends_map:
        if is_package_avaiable(backend_name):
            treelib = backends_map[backend_name]()
            backend = backend_name
            logging.info(f"Successfully set backend to `{backend_name}`")
            break
elif backend == "jax":
    treelib = jax_backend()
    logging.info("Successfully set backend to `jax`")
elif backend == "optree":
    treelib = optree_backend()
    logging.info(f"Successfully set backend to `{backend}`")
else:
    raise ValueError(f"Unknown backend: {backend!r}. available {backends_map.keys()=}")


@contextmanager
def backend_context(backend_name: BackendLiteral):
    """Context manager for switching the tree backend within a context.

    Args:
        backend_name: The name of the backend to switch to. available backends are
            ``optree`` and ``jax``.

    Example:
        Registering a custom tree class with optree backend:

        >>> import sepes as sp
        >>> import optree
        >>> with sp.backend_context("optree"):
        ...     class Tree(sp.TreeClass):
        ...         def __init__(self, a, b):
        ...             self.a = a
        ...             self.b = b
        ...     tree = Tree(1, 2)
        >>> optree.tree_flatten(tree, namespace="sepes")
        ([1, 2], PyTreeSpec(CustomTreeNode(Tree[('a', 'b')], [*, *]), namespace='sepes'))
    """
    global treelib, backend
    old_treelib = treelib
    old_backend = backend
    try:
        treelib = backends_map[backend_name]()
        yield
    finally:
        treelib = old_treelib
        backend = old_backend
