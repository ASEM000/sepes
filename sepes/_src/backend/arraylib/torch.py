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
import torch
from torch import Tensor
import sepes._src.backend.arraylib as arraylib

floatings = [torch.float16, torch.float32, torch.float64]
complexes = [torch.complex32, torch.complex64, torch.complex128]
integers = [torch.int8, torch.int16, torch.int32, torch.int64]

arraylib.tobytes.register(Tensor, lambda x: np.from_dlpack(x).tobytes())
arraylib.where.register(Tensor, torch.where)
arraylib.nbytes.register(Tensor, lambda x: x.nbytes)
arraylib.shape.register(Tensor, lambda x: tuple(x.shape))
arraylib.dtype.register(Tensor, lambda x: x.dtype)
arraylib.min.register(Tensor, torch.min)
arraylib.max.register(Tensor, torch.max)
arraylib.mean.register(Tensor, torch.mean)
arraylib.std.register(Tensor, torch.std)
arraylib.all.register(Tensor, torch.all)
arraylib.is_floating.register(Tensor, lambda x: x.dtype in floatings)
arraylib.is_integer.register(Tensor, lambda x: x.dtype in integers)
arraylib.is_inexact.register(Tensor, lambda x: x.dtype in floatings + complexes)
arraylib.is_bool.register(Tensor, lambda x: x.dtype == torch.bool)
arraylib.ndarrays += (Tensor,)
