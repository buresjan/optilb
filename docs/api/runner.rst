Runner API
==========

``optilb.runner`` exposes two helpers:

.. code-block:: python

    from optilb.runner import ScaleLevel, run_with_schedule

- ``ScaleLevel`` groups per-method scale settings.
- ``run_with_schedule`` executes an optimiser through successive levels and
  returns an :class:`~optilb.OptResult` containing the combined history.

Refer to :doc:`../runner` for a complete example.
