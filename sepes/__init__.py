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

from sepes._src.backend import backend_context
from sepes._src.code_build import autoinit, field, fields
from sepes._src.tree_base import TreeClass
from sepes._src.tree_index import at
from sepes._src.tree_mask import is_masked, tree_mask, tree_unmask
from sepes._src.tree_pprint import tree_diagram, tree_repr, tree_str, tree_summary
from sepes._src.tree_util import bcmap, leafwise, value_and_tree

__all__ = [
    # module utils
    "TreeClass",
    # pprint utils
    "tree_diagram",
    "tree_repr",
    "tree_str",
    "tree_summary",
    # masking utils
    "is_masked",
    "tree_unmask",
    "tree_mask",
    # tree utils
    "at",
    "bcmap",
    "value_and_tree",
    # construction utils
    "field",
    "fields",
    "autoinit",
    "leafwise",
    # backend utils
    "backend_context",
]

__version__ = "0.12.0"

at.__module__ = "sepes"
TreeClass.__module__ = "sepes"
