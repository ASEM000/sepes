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
