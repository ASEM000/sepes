# Changelog

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