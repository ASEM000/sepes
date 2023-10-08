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


@ft.lru_cache(maxsize=None)
def is_available(backend: str) -> bool:
    return find_spec(backend) is not None


def default_backend():
    # no backend is available
    if not is_available("optree"):
        raise ImportError("No backend is available. Please install `optree`.")

    from sepes._src.backend.arraylib.noarray import NoArray
    from sepes._src.backend.treelib.optree import OpTreeTreeLib

    arraylib = NoArray()
    treelib = OpTreeTreeLib()
    return arraylib, treelib


def jax_backend():
    if not is_available("jax"):
        raise ImportError("`jax` backend requires `jax` to be installed.")

    from sepes._src.backend.arraylib.jax import JaxArray
    from sepes._src.backend.treelib.jax import JaxTreeLib

    arraylib = JaxArray()
    treelib = JaxTreeLib()
    return arraylib, treelib


def numpy_backend():
    if not is_available("optree"):
        raise ImportError("`numpy` backend requires `optree` to be installed.")

    if not is_available("numpy"):
        raise ImportError("`numpy` backend requires `numpy` to be installed.")

    from sepes._src.backend.arraylib.numpy import NumpyArray
    from sepes._src.backend.treelib.optree import OpTreeTreeLib

    arraylib = NumpyArray()
    treelib = OpTreeTreeLib()
    return arraylib, treelib


def torch_backend():
    if not is_available("torch"):
        raise ImportError("`torch` backend requires `torch` to be installed.")
    if not is_available("optree"):
        raise ImportError("`torch` backend requires `optree` to be installed.")

    from sepes._src.backend.arraylib.torch import TorchArray
    from sepes._src.backend.treelib.optree import OpTreeTreeLib

    arraylib = TorchArray()
    treelib = OpTreeTreeLib()
    return arraylib, treelib


BackendLiteral = Literal["default", "jax", "torch", "numpy"]

backend: BackendLiteral = os.environ.get("SEPES_BACKEND", "default").lower()

backends_map: dict[BackendLiteral, Callable] = {}
backends_map["default"] = default_backend
backends_map["jax"] = jax_backend
backends_map["torch"] = torch_backend
backends_map["numpy"] = numpy_backend


if backend == "default":
    # backend promotion in essence is a search for the first available backend
    # in the following order: jax, torch, numpy
    # if no backend is available, then the default backend is used
    for backend_name in ["jax", "torch", "numpy", "default"]:
        if is_available(backend_name):
            arraylib, treelib = backends_map[backend_name]()
            logging.info(f"Successfully set backend to `{backend_name}`")
            break

elif backend == "jax":
    arraylib, treelib = jax_backend()
    logging.info("Successfully set backend to `jax`")

elif backend == "numpy":
    arraylib, treelib = numpy_backend()
    logging.info("Successfully set backend to `numpy`")

elif backend == "torch":
    arraylib, treelib = torch_backend()
    logging.info("Successfully set backend to `torch`")

else:
    raise ValueError(f"Unknown backend: {backend!r}")
