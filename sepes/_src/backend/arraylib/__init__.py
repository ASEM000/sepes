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
"""Backend tools for sepes."""

from __future__ import annotations

import functools as ft
from typing import Callable, NamedTuple


class NoImplError(NamedTuple):
    op: Callable

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"No implementation for {self.op}"
                                  f" with {args=} {kwargs=}")


tobytes = ft.singledispatch(NoImplError("tobytes"))
where = ft.singledispatch(NoImplError("where"))
nbytes = ft.singledispatch(NoImplError("nbytes"))
shape = ft.singledispatch(NoImplError("shape"))
dtype = ft.singledispatch(NoImplError("dtype"))
min = ft.singledispatch(NoImplError("min"))
max = ft.singledispatch(NoImplError("max"))
mean = ft.singledispatch(NoImplError("mean"))
std = ft.singledispatch(NoImplError("std"))
all = ft.singledispatch(NoImplError("all"))
array_equal = ft.singledispatch(NoImplError("array_equal"))
is_floating = ft.singledispatch(NoImplError("is_floating"))
is_integer = ft.singledispatch(NoImplError("is_integer"))
is_inexact = ft.singledispatch(NoImplError("is_inexact"))
is_bool = ft.singledispatch(NoImplError("is_bool"))
ndarrays: list[type] = []
