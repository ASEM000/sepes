# Changelog

## v0.12.1

### Additions

- Add ability to register custom types for masking wrappers.

  Example to define a custom masking wrapper for a specific type.

  ```python
  import sepes as sp
  import jax
  import dataclasses as dc
  @dc.dataclass
  class MyInt:
      value: int
  @dc.dataclass
  class MaskedInt:
      value: MyInt
  # define a rule of how to mask instances of MyInt
  @sp.tree_mask.def_type(MyInt)
  def mask_int(value):
      return MaskedInt(value)
  # define a rule how to unmask the MaskedInt wrapper
  @sp.tree_unmask.def_type(MaskedInt)
  def unmask_int(value):
      return value.value
  tree = [MyInt(1), MyInt(2), {"a": MyInt(3)}]
  masked_tree = sp.tree_mask(tree, cond=lambda _: True)

  masked_tree
  #[MaskedInt(value=MyInt(value=1)), MaskedInt(value=MyInt(value=2)), {'a': MaskedInt(value=MyInt(value=3))}]

  sp.tree_unmask(masked_tree)
  #[MyInt(value=1), MyInt(value=2), {'a': MyInt(value=3)}]

  # `is_masked` recognizes the new masked type
  assert is_masked(masked_tree[0]) is True
  ```

## V0.12

### Deprecations

- Reduce the core API size by removing:
  1.  `tree_graph` (for graphviz)
  2.  `tree_mermaid` (mermaidjs)
  3.  `Partial/partial` -> Use `jax.tree_util.Partial` instead.
  4.  `is_tree_equal` -> Use `bcmap(numpy.testing.*)(pytree1, pytree2)` instead.
  5.  `freeze` -> Use `ft.partial(tree_mask, lambda _: True)` instead.
  6.  `unfreeze` -> Use `tree_unmask` instead.
  7.  `is_nondiff`
  8.  `BaseKey`

### Changes

- `tree_{mask,unmask}` now accepts only callable `cond` argument.

  For masking using pytree boolean mask use the following pattern:

  ```python
  import jax
  import sepes as sp
  import functools as ft
  tree = [[1, 2], 3] # the nested tree
  where = [[True, False], True]  # mask tree[0][1] and tree[1]
  mask = ft.partial(sp.tree_mask, cond=lambda _: True)
  sp.at(tree)[where].apply(mask)  # apply using `at`
  # [[#1, 2], #3]
  # or simply apply to the node directly
  tree = [[mask(1), 2], mask(3)]
  # [[#1, 2], #3]
  ```

- Rename `is_frozen` to `is_masked`

  - frozen could mean non-trainable array, however the masking is not only for arrays but also for other types that will be hidden across jax transformations.

- Rename `AtIndexer` to `at` for shorter syntax.

### Additions

- Add `fill_value` in `at[...].get(fill_value=...)` to add default value for non
  selected leaves. Useful for arrays under `jax.jit` to avoid variable size related errors.
- Add `jax.tree_util.{SequenceKey,GetAttrKey,DictKey}` as valid path keys in `at[...]`.

## V0.11.3

- Raise error if `autoinit` is used with `__init__` method defined.
- Avoid applying `copy.copy` `jax.Array` during flatten/unflatten or `AtIndexer` operations.
- Add `at` as an alias for `AtIndexer` for shorter syntax.
- Deprecate `AtIndexer.__call__` in favor of `value_and_tree` to apply function in a functional manner by copying the input argument.

  ```python
  import sepes as sp
  class Counter(sp.TreeClass):
      def __init__(self, count: int):
          self.count = count
      def increment(self, value):
          self.count += value
          return self.count
  counter = Counter(0)
  # the function follow jax.value_and_grad semantics where the tree is the
  # copied mutated input argument, if the function mutates the input arguments
  sp.value_and_tree(lambda C: C.increment(1))(counter)
  # (1, Counter(count=1))
  ```

- Add sharding info in `tree_summary`, `G` for global, `S` for sharded shape.

  ```python
  import jax
  import sepes as sp
  from jax.sharding import Mesh, NamedSharding as N, PartitionSpec as P
  import numpy as np
  import os
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
  x = jax.numpy.ones([4 * 4, 2 * 2])
  mesh = Mesh(devices=np.array(jax.devices()).reshape(4, 2), axis_names=["i", "j"])
  sharding = N(mesh=mesh, spec=P("i", "j"))
  x = jax.device_put(x, device=sharding)

  print(sp.tree_summary(x))
  ┌────┬───────────┬─────┬───────┐
  │Name│Type       │Count│Size   │
  ├────┼───────────┼─────┼───────┤
  │Σ   │G:f32[16,4]│64   │256.00B│
  │    │S:f32[4,2] │     │       │
  └────┴───────────┴─────┴───────┘
  ```

- Updated docstrings. e.g. How to construct flops counter in `tree_summary` using `jax.jit`

## V0.11.2

- No freezing rule for `jax.Tracer` in `sp.freeze`
- Add pprint rule `jax.Tracer` in `sp.tree_repr`/`sp.tree_str`
- Add no-op warning if user adds `autoinit` to class with `__init__` method.
- Add warning if user add fields in incorrect kind order.
- Add warning if any bases of autoinit has `__init__` method.
- Add `CLASS_VAR` kind in `field` to support class variables in `autoinit`.

## V0.11.1

- `__call__` is added to `AtIndexer` to enable methods that work on copied instance.
  to avoid mutating in-place. _This is useful to write methods in stateful manner, and
  use the `AtIndexer` to operate in a functional manner_. This feature was previously
  enabled only for `TreeClass`, but now it is enabled for any class.

  The following shows how to use `AtIndexer` to call a method that mutates the tree
  in-place, in an out-of-place manner (i.e. execute the method on a copy of the tree)

  ```python
  import sepes as sp
  import jax.tree_util as jtu
  class Counter:
      def __init__(self, count: int):
          self.count = count
      def increment_count(self, count:int) -> int:
          # mutates the tree
          self.count += count
          return self.count
      def __repr__(self) -> str:
          return f"Tree(count={self.count})"
  counter = Counter(0)
  indexer = sp.AtIndexer(counter)
  cur_count, new_counter = indexer["increment_count"](count=1)
  assert counter.count == 0  # did not mutate in-place
  assert cur_count == 1  # the method returned the current count
  assert new_counter.count == 1  # the copied instance where the method was executed
  assert not (counter is new_counter)  # the old and new instance are not the same
  ```

  If the instance is frozen (e.g. `dataclasses.dataclass(frozen=True)`) ,or implements
  a custom `__setattr__`/`__delattr__` then the `__call__` may not work. Register
  a custom mutator/immutator to `AtIndexer` to enable `__call__` to work. For more
  see `AtIndexer.__call__` docstring.

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
