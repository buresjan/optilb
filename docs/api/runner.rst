Runner API
==========

``optilb.runner`` exposes two helpers:

.. code-block:: python

    from optilb.runner import ScaleLevel, run_with_schedule

- ``ScaleLevel`` groups per-method scale settings.
- ``run_with_schedule`` executes an optimiser through successive levels,
  overriding ``max_iter`` with the per-level budget, rescaling BFGS finite
  difference steps, cloning any supplied :class:`optilb.optimizers.EarlyStopper`
  and concatenating history into a single :class:`optilb.OptResult`.

Refer to :doc:`../runner` for a complete example.
