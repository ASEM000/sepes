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

tobytes = ft.singledispatch(lambda array: ...)
where = ft.singledispatch(lambda condition, x, y: ...)
nbytes = ft.singledispatch(lambda array: ...)
shape = ft.singledispatch(lambda array: ...)
dtype = ft.singledispatch(lambda array: ...)
min = ft.singledispatch(lambda array: ...)
max = ft.singledispatch(lambda array: ...)
mean = ft.singledispatch(lambda array: ...)
std = ft.singledispatch(lambda array: ...)
all = ft.singledispatch(lambda array: ...)
is_floating = ft.singledispatch(lambda array: ...)
is_integer = ft.singledispatch(lambda array: ...)
is_inexact = ft.singledispatch(lambda array: ...)
is_bool = ft.singledispatch(lambda array: ...)


class ArrayLib:
    tobytes = staticmethod(tobytes)
    where = staticmethod(where)
    nbytes = staticmethod(nbytes)
    shape = staticmethod(shape)
    dtype = staticmethod(dtype)
    min = staticmethod(min)
    max = staticmethod(max)
    mean = staticmethod(mean)
    std = staticmethod(std)
    all = staticmethod(all)
    is_floating = staticmethod(is_floating)
    is_integer = staticmethod(is_integer)
    is_inexact = staticmethod(is_inexact)
    is_bool = staticmethod(is_bool)
    ndarrays: tuple[type, ...] = ()
