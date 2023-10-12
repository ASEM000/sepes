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
import sepes._src.backend.arraylib as arraylib

arraylib.tobytes.register(Array, lambda x: jnp.array(x).tobytes())
arraylib.where.register(Array, jnp.where)
arraylib.nbytes.register(Array, lambda x: x.nbytes)
arraylib.shape.register(Array, jnp.shape)
arraylib.dtype.register(Array, lambda x: x.dtype)
arraylib.min.register(Array, jnp.min)
arraylib.max.register(Array, jnp.max)
arraylib.mean.register(Array, jnp.mean)
arraylib.std.register(Array, jnp.std)
arraylib.all.register(Array, jnp.all)
arraylib.is_floating.register(Array, lambda x: jnp.issubdtype(x.dtype, jnp.floating))
arraylib.is_integer.register(Array, lambda x: jnp.issubdtype(x.dtype, jnp.integer))
arraylib.is_inexact.register(Array, lambda x: jnp.issubdtype(x.dtype, jnp.inexact))
arraylib.is_bool.register(Array, lambda x: jnp.issubdtype(x.dtype, jnp.bool_))
arraylib.ndarrays += (Array,)
