Visualize, create, and operate on nested trees in the most intuitive way possible.

To select a backend set the environment variable ``SEPES_BACKEND`` to:
    - ``default``: will set the backend among ``jax``, ``torch``, `numpy` in that order 
      if found, otherwise will fallback to ``default``
    - ``jax``: for `jax.numpy` for array and ``jax.tree_util`` for tree util backend.
    - ``torch``: for `torch` for arrays and ``optree`` for tree util backend. by default
      the ``optree`` namespace used is ``sepes``, change it using ``SEPES_NAMESPACE`` environment
      variable.
    - ``numpy``: for ``numpy`` for arrays and ``optree`` for tree util backend.

Installation
------------

Install from pip::

   pip install sepes


.. toctree::
   :caption: User guides
   :maxdepth: 1

   notebooks/common_recipes

.. toctree::
   :caption: Examples
   :maxdepth: 1

   notebooks/build_mini_optimizer_library

.. toctree::
    :caption: API Documentation
    :maxdepth: 1
    
    API/sepes

Apache2.0 License.

Indices
=======

* :ref:`genindex`


