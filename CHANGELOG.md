# Changelog

## V0.11

- Mark full subtrees for replacement.

  ```python
  import sepes
  tree = [1, 2, [3,4]]
  tree_= sp.AtIndexer(tree)[[False,False,True]].set(10)
  assert tree_ == [1, 2, 10]
  ```

  i.e. Inside a mask, marking a _subtree_ mask with single bool leaf, will replace the whole subtree. In this example subtree `[3, 4]` marked with `True` in the mask is an indicator for replacement.

  If the subtree is populated with `True` leaves, then the set value will
  be broadcasted to all subtree leaves.

  ```python
  import sepes
  tree = [1, 2, [3, 4]]
  tree_ = sp.AtIndexer(tree)[[False, False, [True, True]]].set(10)
  assert tree_ == [1, 2, [10, 10]]
  ```

- Do not broadcast path based mask

  ```python
  import sepes as sp
  tree = [1, 2, [3, 4]]
  tree_= sp.AtIndexer(tree)[2].set(10)
  assert tree_ == [1, 2, 10]
  ```

  To broadcast to subtree use `...`

  ```python
  import sepes as sp
  tree = [1, 2, [3, 4]]
  tree_= sp.AtIndexer(tree)[2][...].set(10)
  assert tree_ == [1, 2, [10, 10]]
  ```

- Better lookup errors

  ```python
  import sepes as sp
  tree = {"a": {"b": 1, "c": 2}, "d": 3}
  sp.AtIndexer(tree)["a"]["d"].set(100)
  ```

  ```python
  LookupError: No leaf match is found for where=[a, d]. Available keys are ['a']['b'], ['a']['c'], ['d'].
  Check the following:
      - If where is `str` then check if the key exists as a key or attribute.
      - If where is `int` then check if the index is in range.
      - If where is `re.Pattern` then check if the pattern matches any key.
      - If where is a `tuple` of the above types then check if any of the tuple elements match.
  ```

- Extract subtrees with `pluck`

  ```python
  import sepes as sp
  tree = {"a": 1, "b": [1, 2, 3]}
  indexer = sp.AtIndexer(tree)  # construct an indexer
  # `pluck` returns a list of selected subtrees
  indexer["b"].pluck()
  # [[1, 2, 3]]

  # in comparison, `get` returns same pytree
  indexer["b"].get()
  # {'a': None, 'b': [1, 2, 3]}
  ```

  `pluck` with mask

  ```python
  import sepes as sp
  tree = {"a": 1, "b": [2, 3, 4]}
  mask = {"a": True, "b": [False, True, False]}
  indexer = sp.AtIndexer(tree)
  indexer[mask].pluck()
  # [1, 3]
  ```

  This is equivalent to the following:

  ```python
  [tree["a"], tree["b"][1]]
  ```

  To get the first `n` matches, use `pluck(n)`.

  _A simple application of pluck is to share reference using a descriptor-based approach:_

  ```python
  import sepes as sp
  marker = object()
  class Tie:
      def __set_name__(self, owner, name):
          self.name = name
      def __set__(self, instance, indexer):
          self.where = indexer.where
          where_str = "/".join(map(str, self.where))
          vars(instance)[self.name] = f"Ref:{where_str}"
      def __get__(self, instance, owner):
          (subtree,) = sp.AtIndexer(instance, self.where).pluck(1)
          return subtree

  class Tree(sp.TreeClass):
      shared: Tie = Tie()
      def __init__(self):
          self.lookup = dict(a=marker, b=2)
          self.shared = self.at["lookup"]["a"]
  tree = Tree()
  assert tree.lookup["a"] is tree.shared
  ```

- Revamp the backend mechanism:

  - Rewrite array backend via dispatch to work with `numpy`,`jax`, and `torch` simultaneously. for example the following recognize both `jax` and `torch` entries without backend changes.

  ```python
  import sepes as sp
  import jax.numpy as jnp
  import torch
  tree = [[1, 2], 2, [3, 4], jnp.ones((2, 2)), torch.ones((2, 2))]
  print(sp.tree_repr(tree))
  # [
  #   [1, 2],
  #   2,
  #   [3, 4],
  #   f32[2,2](μ=1.00, σ=0.00, ∈[1.00,1.00]),
  #   torch.f32[2,2](μ=1.00, σ=0.00, ∈[1.00,1.00])
  # ]
  ```

  - Introduce `backend_context` to switch between `jax`/`optree` backend registration and tree utilities. the following example shows how to register with different backends:

  ```python
  import sepes
  import jax
  import optree

  with sepes.backend_context("jax"):
      class JaxTree(sepes.TreeClass):
          def __init__(self):
              self.l1 = 1.0
              self.l2 = 2.0
      print(jax.tree_util.tree_leaves(JaxTree()))

  with sepes.backend_context("optree"):
      class OpTreeTree(sepes.TreeClass):
          def __init__(self):
              self.l1 = 1.0
              self.l2 = 2.0
      print(optree.tree_leaves(OpTreeTree(), namespace="sepes"))
  # [1.0, 2.0]
  # [1.0, 2.0]
  ```

## v0.10.0

- successor of the `jax`-specific `pytreeclass`

- Supports multibackend:

  - `numpy` + `optree` via `export SEPES_BACKEND=numpy` (lightweight option)
  - `jax` via `export SEPES_BACKEND=jax` - The default -
  - `torch` + `optree` via `export SEPES_BACKEND=torch`
  - no array + `optree` via `export SEPES_BACKEND=default`

- drop `callback` option in parallel options in `is_parallel`
- Add parallel processing via `is_parallel` to `.{get,set}`
- `register_excluded_type` to `autoinit` to exclude certain types to be in `field` defaults.
- add `doc` in `field` to add extra documentation for the descriptor `__doc__`
