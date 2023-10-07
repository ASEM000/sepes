# Changelog

## V0.Next

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
  import sepes
  tree = [1, 2, [3, 4]]
  tree_= sp.AtIndexer(tree)[0].set(10)
  assert tree_ == [1, 2, 10]
  ```

  To broadcast to subtree use `...`

  ```python
  import sepes
  tree = [1, 2, [3, 4]]
  tree_= sp.AtIndexer(tree)[0][...].set(10)
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

## v0.10.0

- successor of the `jax`-specific `pytreeclass`

- Supports multibackend:

  - `numpy` + `optree` via `export sepes_BACKEND=numpy` (lightweight option)
  - `jax` via `export sepes_BACKEND=jax` - The default -
  - `torch` + `optree` via `export sepes_BACKEND=torch`
  - no array + `optree` via `export sepes_BACKEND=default`

- drop `callback` option in parallel options in `is_parallel`
- Add parallel processing via `is_parallel` to `.{get,set}`
- `register_excluded_type` to `autoinit` to exclude certain types to be in `field` defaults.
- add `doc` in `field` to add extra documentation for the descriptor `__doc__`
