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
from sepes._src.backend.arraylib.base import ArrayLib

ArrayLib.tobytes.register(ndarray, lambda x: np.array(x).tobytes())
ArrayLib.where.register(ndarray, np.where)
ArrayLib.nbytes.register(ndarray, lambda x: x.nbytes)
ArrayLib.shape.register(ndarray, np.shape)
ArrayLib.dtype.register(ndarray, lambda x: x.dtype)
ArrayLib.min.register(ndarray, np.min)
ArrayLib.max.register(ndarray, np.max)
ArrayLib.mean.register(ndarray, np.mean)
ArrayLib.std.register(ndarray, np.std)
ArrayLib.all.register(ndarray, np.all)
ArrayLib.is_floating.register(ndarray, lambda x: np.issubdtype(x.dtype, np.floating))
ArrayLib.is_integer.register(ndarray, lambda x: np.issubdtype(x.dtype, np.integer))
ArrayLib.is_inexact.register(ndarray, lambda x: np.issubdtype(x.dtype, np.inexact))
ArrayLib.is_bool.register(ndarray, lambda x: np.issubdtype(x.dtype, np.bool_))
ArrayLib.ndarrays += (ndarray,)
