# Copyright 2024 sepes authors
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

import numpy as np
import mlx.core as mx
import sepes._src.backend.arraylib as arraylib

floating = [mx.float16, mx.float32]
integer = [mx.int8, mx.int16, mx.int32, mx.int64]
complex = [mx.complex64]
inexact = floating + complex
arraylib.tobytes.register(mx.array, lambda x: np.array(x).tobytes())
arraylib.where.register(mx.array, mx.where)
arraylib.nbytes.register(mx.array, lambda x: x.nbytes)
arraylib.shape.register(mx.array, lambda x: x.shape)
arraylib.dtype.register(mx.array, lambda x: x.dtype)
arraylib.min.register(mx.array, mx.min)
arraylib.max.register(mx.array, mx.max)
arraylib.mean.register(mx.array, mx.mean)
arraylib.std.register(mx.array, lambda x: mx.sqrt(mx.var(x)))
arraylib.all.register(mx.array, mx.all)
arraylib.is_floating.register(mx.array, lambda x: np.issubdtype(x.dtype, floating))
arraylib.is_integer.register(mx.array, lambda x: np.issubdtype(x.dtype, integer))
arraylib.is_inexact.register(mx.array, lambda x: np.issubdtype(x.dtype, inexact))
arraylib.is_bool.register(mx.array, lambda x: np.issubdtype(x.dtype, mx.bool_))
arraylib.ndarrays += (mx.array,)
