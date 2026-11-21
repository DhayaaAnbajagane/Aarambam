"""Diagnostic products for true-versus-binned IC kernels."""

import numpy as np

from .edges import triangle_valid


__all__ = ['generate_diagnostics']


def _safe_geometric_range(low, high, n):
    if (not np.isfinite(low)) or (not np.isfinite(high)) or high <= low or n <= 0:
        return np.array([], dtype=float)
    return np.geomspace(low, high, int(n))


def _diagnostic_kref(edges, args):
    if args.get('Binned_diagnostic_kref') is not None:
        return float(args['Binned_diagnostic_kref'])
    if args.get('Binned_diagnostic_k3') is not None:
        return float(args['Binned_diagnostic_k3'])

    low  = max(edge[0] for edge in edges)
    high = min(edge[-1] for edge in edges)
    if high <= low:
        all_edges = np.concatenate(edges)
        low       = np.nanmin(all_edges)
        high      = np.nanmax(all_edges)
    return np.sqrt(low * high)


def _ordered_triangle_grid(edges, args, n):
    edges1, edges2, edges3 = edges
    order                  = str(args.get('Binned_diagnostic_triangle_order', 'k1_le_k2_le_k3'))
    kref                   = _diagnostic_kref(edges, args)

    if order == 'k1_le_k2_le_k3':
        xmin    = max(edges1[0] / kref, 0.0)
        xmax    = min(edges1[-1] / kref, 1.0)
        ymin    = max(edges2[0] / kref, 0.0)
        ymax    = min(edges2[-1] / kref, 1.0)
        x       = np.linspace(xmin, xmax, n)
        y       = np.linspace(ymin, ymax, n)
        X, Y    = np.meshgrid(x, y, indexing='ij')
        k1      = X * kref
        k2      = Y * kref
        k3      = np.full_like(k1, kref)
        ordered = (k1 <= k2) & (k2 <= k3)
        xlabel  = 'k1/k3'
        ylabel  = 'k2/k3'

    elif order == 'k1_ge_k2_ge_k3':
        xmin    = max(edges3[0] / kref, 0.0)
        xmax    = min(edges3[-1] / kref, 1.0)
        ymin    = max(edges2[0] / kref, 0.0)
        ymax    = min(edges2[-1] / kref, 1.0)
        x       = np.linspace(xmin, xmax, n)
        y       = np.linspace(ymin, ymax, n)
        X, Y    = np.meshgrid(x, y, indexing='ij')
        k1      = np.full_like(X, kref)
        k2      = Y * kref
        k3      = X * kref
        ordered = (k1 >= k2) & (k2 >= k3)
        xlabel  = 'k3/k1'
        ylabel  = 'k2/k1'

    else:
        raise ValueError(f'Unknown Binned_diagnostic_triangle_order={order!r}.')

    in_range = (
        (k1 >= edges1[0]) & (k1 <= edges1[-1])
        & (k2 >= edges2[0]) & (k2 <= edges2[-1])
        & (k3 >= edges3[0]) & (k3 <= edges3[-1])
    )
    triangle = triangle_valid(k1, k2, k3)

    return {
        'order':    order,
        'kref':     kref,
        'x':        x,
        'y':        y,
        'X':        X,
        'Y':        Y,
        'xlabel':   xlabel,
        'ylabel':   ylabel,
        'k1':       k1,
        'k2':       k2,
        'k3':       k3,
        'ordered':  ordered,
        'triangle': triangle,
        'in_range': in_range,
        'valid':    ordered & triangle & in_range,
        'xlim':     np.array([xmin, xmax], dtype=float),
        'ylim':     np.array([ymin, ymax], dtype=float),
    }


def _errors(true, binned, valid, rel_floor):
    true       = np.where(valid, true, np.nan)
    binned     = np.where(valid, binned, np.nan)
    absolute   = binned - true
    scale      = np.nanpercentile(np.abs(true[valid]), 90) if np.any(valid) else 1.0
    scale      = scale if np.isfinite(scale) and scale > 0 else 1.0
    denominator = np.maximum(np.abs(true), rel_floor * scale)
    fractional = absolute / denominator
    return true, binned, absolute, fractional


def _limit_configuration(name, edges, args, n):
    edges1, edges2, edges3 = edges
    margin                 = float(args.get('Binned_diagnostic_edge_margin', 0.8))
    squeezed_ratio         = float(args.get('Binned_diagnostic_squeezed_ratio', 0.05))
    folded_epsilon         = float(args.get('Binned_diagnostic_folded_epsilon', 1e-3))

    if name == 'equilateral':
        low  = max(edges1[0], edges2[0], edges3[0])
        high = min(edges1[-1], edges2[-1], edges3[-1]) * margin
        k    = _safe_geometric_range(low, high, n)
        return k, k, k, k

    if name == 'squeezed_k1':
        low  = max(edges2[0], edges3[0], edges1[0] / squeezed_ratio)
        high = min(edges2[-1], edges3[-1], edges1[-1] / squeezed_ratio) * margin
        k    = _safe_geometric_range(low, high, n)
        return k, squeezed_ratio * k, k, k

    if name == 'squeezed_k2':
        low  = max(edges1[0], edges3[0], edges2[0] / squeezed_ratio)
        high = min(edges1[-1], edges3[-1], edges2[-1] / squeezed_ratio) * margin
        k    = _safe_geometric_range(low, high, n)
        return k, k, squeezed_ratio * k, k

    if name == 'squeezed_k3':
        low  = max(edges1[0], edges2[0], edges3[0] / squeezed_ratio)
        high = min(edges1[-1], edges2[-1], edges3[-1] / squeezed_ratio) * margin
        k    = _safe_geometric_range(low, high, n)
        return k, k, k, squeezed_ratio * k

    if name == 'folded':
        factor = 2.0 - folded_epsilon
        low    = max(edges1[0], edges2[0], edges3[0] / factor)
        high   = min(edges1[-1], edges2[-1], edges3[-1] / factor) * margin
        k      = _safe_geometric_range(low, high, n)
        return k, k, k, factor * k

    raise ValueError(f'Unknown diagnostic curve={name!r}.')


def generate_diagnostics(kernel_true, kernel_binned, edges, args):
    """Exact and binned kernel on a triangle grid and along the equilateral, squeezed and folded lines.

    Parameters
    ----------
    kernel_true, kernel_binned : callable
        ``f(k1, k2, k3)``: the exact and the interpolated kernel.
    edges : tuple of 3 arrays
        Bin edges of the table.
    args : dict
        Options (all optional): ``Binned_diagnostic_N`` (points per line, 192), ``_rel_floor``,
        ``_kref``, ``_triangle_order``, ``_edge_margin``, ``_squeezed_ratio``, ``_folded_epsilon``.
    """
    n         = max(8, int(args.get('Binned_diagnostic_N', 192)))
    rel_floor = float(args.get('Binned_diagnostic_rel_floor', 1e-3))
    triangle  = _ordered_triangle_grid(edges, args, n)
    true      = np.asarray(kernel_true(triangle['k1'], triangle['k2'], triangle['k3']), dtype=float)
    binned    = np.asarray(kernel_binned(triangle['k1'], triangle['k2'], triangle['k3']), dtype=float)
    true, binned, absolute, fractional = _errors(true, binned, triangle['valid'], rel_floor)

    output = {
        'triangle_x':                triangle['x'],
        'triangle_y':                triangle['y'],
        'triangle_X':                triangle['X'],
        'triangle_Y':                triangle['Y'],
        'triangle_x_label':          np.array(triangle['xlabel']),
        'triangle_y_label':          np.array(triangle['ylabel']),
        'triangle_order':            np.array(triangle['order']),
        'triangle_kref':             np.array(triangle['kref']),
        'triangle_xlim':             triangle['xlim'],
        'triangle_ylim':             triangle['ylim'],
        'triangle_k1':               triangle['k1'],
        'triangle_k2':               triangle['k2'],
        'triangle_k3':               triangle['k3'],
        'triangle_valid':            triangle['valid'].astype(np.int8),
        'triangle_ordered':          triangle['ordered'].astype(np.int8),
        'triangle_triangle_valid':   triangle['triangle'].astype(np.int8),
        'triangle_in_table_range':   triangle['in_range'].astype(np.int8),
        'triangle_K_true':           true,
        'triangle_K_binned':         binned,
        'triangle_K_error':          absolute,
        'triangle_K_fractional_error': fractional,
        'k1_edges':                  np.asarray(edges[0]),
        'k3_edges':                  np.asarray(edges[2]),
    }

    for name in ['equilateral', 'squeezed_k1', 'squeezed_k2', 'squeezed_k3', 'folded']:
        k, k1, k2, k3 = _limit_configuration(name, edges, args, n)

        if k.size == 0:
            for suffix in ['k', 'k1', 'k2', 'k3', 'K_true', 'K_binned', 'K_error', 'K_fractional_error']:
                output[f'{name}_{suffix}'] = np.array([], dtype=float)
            continue

        valid     = triangle_valid(k1, k2, k3)
        true      = np.asarray(kernel_true(k1, k2, k3), dtype=float)
        binned    = np.asarray(kernel_binned(k1, k2, k3), dtype=float)
        true, binned, absolute, fractional = _errors(true, binned, valid, rel_floor)

        output[f'{name}_k']                  = k
        output[f'{name}_k1']                 = k1
        output[f'{name}_k2']                 = k2
        output[f'{name}_k3']                 = k3
        output[f'{name}_K_true']             = true
        output[f'{name}_K_binned']           = binned
        output[f'{name}_K_error']            = absolute
        output[f'{name}_K_fractional_error'] = fractional

    return output
