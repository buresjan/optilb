Latin-Hypercube Sampling (LHS)
=============================

``optilb`` provides a simple function :func:`optilb.sampling.lhs` to generate
Latin-Hypercube samples within a given :class:`optilb.DesignSpace`.

Example usage::

    from optilb import DesignSpace
    from optilb.sampling import lhs

    ds = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    points = lhs(10, ds, seed=42)
    for p in points:
        print(p.x)

The function supports integer bounds, optional scrambling, and centering of
points within each hypercube cell.

