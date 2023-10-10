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


from jax import Array
import jax.numpy as jnp
from sepes._src.backend.arraylib.base import ArrayLib

ArrayLib.tobytes.register(Array, lambda x: jnp.array(x).tobytes())
ArrayLib.where.register(Array, jnp.where)
ArrayLib.nbytes.register(Array, lambda x: x.nbytes)
ArrayLib.shape.register(Array, jnp.shape)
ArrayLib.dtype.register(Array, lambda x: x.dtype)
ArrayLib.min.register(Array, jnp.min)
ArrayLib.max.register(Array, jnp.max)
ArrayLib.mean.register(Array, jnp.mean)
ArrayLib.std.register(Array, jnp.std)
ArrayLib.all.register(Array, jnp.all)
ArrayLib.is_floating.register(Array, lambda x: jnp.issubdtype(x.dtype, jnp.floating))
ArrayLib.is_integer.register(Array, lambda x: jnp.issubdtype(x.dtype, jnp.integer))
ArrayLib.is_inexact.register(Array, lambda x: jnp.issubdtype(x.dtype, jnp.inexact))
ArrayLib.is_bool.register(Array, lambda x: jnp.issubdtype(x.dtype, jnp.bool_))
ArrayLib.types += (Array,)
