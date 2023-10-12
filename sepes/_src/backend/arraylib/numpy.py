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

import numpy as np
from numpy import ndarray
import sepes._src.backend.arraylib as arraylib

arraylib.tobytes.register(ndarray, lambda x: np.array(x).tobytes())
arraylib.where.register(ndarray, np.where)
arraylib.nbytes.register(ndarray, lambda x: x.nbytes)
arraylib.shape.register(ndarray, np.shape)
arraylib.dtype.register(ndarray, lambda x: x.dtype)
arraylib.min.register(ndarray, np.min)
arraylib.max.register(ndarray, np.max)
arraylib.mean.register(ndarray, np.mean)
arraylib.std.register(ndarray, np.std)
arraylib.all.register(ndarray, np.all)
arraylib.is_floating.register(ndarray, lambda x: np.issubdtype(x.dtype, np.floating))
arraylib.is_integer.register(ndarray, lambda x: np.issubdtype(x.dtype, np.integer))
arraylib.is_inexact.register(ndarray, lambda x: np.issubdtype(x.dtype, np.inexact))
arraylib.is_bool.register(ndarray, lambda x: np.issubdtype(x.dtype, np.bool_))
arraylib.ndarrays += (ndarray,)
