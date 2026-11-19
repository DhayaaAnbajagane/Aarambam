"""C-table serialization and loading for Aarambam binned kernels."""

import os

import numpy as np

from .edges import centers_from_edges


__all__ = ['load_outputs', 'write_outputs']


PREFIX = 'BinnedKernel'


def _metadata_path(directory):
    return os.path.join(directory, f'{PREFIX}_metadata.txt')


def _parse_metadata(path):
    metadata = {}
    if not os.path.isfile(path):
        return metadata

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields              = line.split()
            metadata[fields[0]] = ' '.join(fields[1:])
    return metadata


def _flat_indices(shape):
    i, j, l = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    return i.ravel(), j.ravel(), l.ravel()


def _flat_centers(edges):
    c1         = centers_from_edges(edges[0])
    c2         = centers_from_edges(edges[1])
    c3         = centers_from_edges(edges[2])
    k1, k2, k3 = np.meshgrid(c1, c2, c3, indexing='ij')
    return k1.ravel(), k2.ravel(), k3.ravel()


def _write_metadata(out, directory, args):
    edges       = out['edges']
    shape       = out['K'].shape
    mode        = out['filter_mode']
    input_kind  = out['hybrid_input_interpolator']
    output_kind = out['hybrid_output_filter']
    ncoef       = int(out['K_coeff'].shape[0])

    with open(_metadata_path(directory), 'w') as handle:
        handle.write(f'template {out.get("template", "unknown")}\n')
        handle.write(f'shape {shape[0]} {shape[1]} {shape[2]}\n')
        handle.write('spacing adaptive_log\n')
        handle.write('adaptive_target kernel\n')
        handle.write('adaptive_strategy roughness\n')
        handle.write(f'adaptive_pilot_N {int(args["Binned_adaptive_pilot_N"])}\n')
        handle.write(f'filter_mode {args["Binned_filter"]}\n')
        handle.write(f'normalized_filter_mode {mode}\n')
        handle.write(f'hybrid_input_interpolator {input_kind}\n')
        handle.write(f'hybrid_output_filter {output_kind}\n')
        handle.write('hybrid_output_leg k3\n')
        handle.write(f'hybrid_ncoef {ncoef}\n')
        handle.write(f'hybrid_coordinate {args["Binned_linear_coordinate"]}\n')
        handle.write('hybrid_coeff_convention global_interp(k1,k2) of k3-local coefficients; F=sum_q Fq*u3^q\n')
        handle.write('value_rule center\n')
        handle.write('kernel_denominator_convention mean\n')
        handle.write('kernel_definition B_over_mean_P1P2_plus_perms\n')
        handle.write(f'enforce_triangle_box {int(bool(args["Binned_enforce_triangle"]))}\n')
        handle.write(f'enforce_triangle_center {int(bool(args["Binned_center_triangle_only"]))}\n')
        handle.write(f'k1_edges_file {PREFIX}_k1_edges.txt\n')
        handle.write(f'k2_edges_file {PREFIX}_k2_edges.txt\n')
        handle.write(f'k3_edges_file {PREFIX}_k3_edges.txt\n')
        handle.write(f'values_file {PREFIX}_values.txt\n')
        if mode == 'hybrid':
            handle.write(f'hybrid_values_file {PREFIX}_hybrid_values.txt\n')

        for axis, edge in enumerate(edges, start=1):
            handle.write(f'k{axis}_min {edge[0]:.17e}\n')
            handle.write(f'k{axis}_max {edge[-1]:.17e}\n')
            handle.write(f'Nk{axis} {len(edge) - 1}\n')


def _write_values_text(out, directory):
    edges          = out['edges']
    shape          = out['K'].shape
    i, j, l        = _flat_indices(shape)
    k1, k2, k3     = _flat_centers(edges)
    valid_center   = out['valid_center'].ravel().astype(int)
    valid_box      = out['valid_box'].ravel().astype(int)
    auxiliary      = out['auxiliary']
    B              = auxiliary['B'].ravel()
    denominator    = auxiliary['denom'].ravel()
    S              = auxiliary['S'].ravel()
    data           = np.column_stack([i, j, l, k1, k2, k3, B, out['K'].ravel(), denominator, S, valid_center, valid_box])
    path           = os.path.join(directory, f'{PREFIX}_values.txt')
    header         = 'Columns:\ni j l k1_center k2_center k3_center B_value K_value denom_value S_value valid_center valid_box'

    np.savetxt(path, data, fmt=['%d', '%d', '%d'] + ['%.17e'] * 7 + ['%d', '%d'], header=header)


def _write_hybrid_text(out, directory):
    if out['filter_mode'] != 'hybrid':
        #Don't leave a hybrid table from an earlier run next to this heaviside one,
        #since the default config points the C code at the hybrid file
        stale = os.path.join(directory, f'{PREFIX}_hybrid_values.txt')
        if os.path.exists(stale): os.remove(stale)
        return

    edges             = out['edges']
    shape             = out['K'].shape
    ncoef             = out['K_coeff'].shape[0]
    i, j, l           = _flat_indices(shape)
    k1, k2, k3        = _flat_centers(edges)
    valid_center      = out['valid_center'].ravel().astype(int)
    valid_box         = out['valid_box'].ravel().astype(int)
    auxiliary_coeff   = out['auxiliary_coefficients']
    coefficient_block = {
        'B':     np.moveaxis(auxiliary_coeff['B'], 0, -1).reshape(-1, ncoef),
        'K':     np.moveaxis(out['K_coeff'], 0, -1).reshape(-1, ncoef),
        'denom': np.moveaxis(auxiliary_coeff['denom'], 0, -1).reshape(-1, ncoef),
        'S':     np.moveaxis(auxiliary_coeff['S'], 0, -1).reshape(-1, ncoef),
    }
    blocks = [coefficient_block[name] for name in ['B', 'K', 'denom', 'S']]
    labels = [f'{label}{q}' for label in ['B', 'K', 'D', 'S'] for q in range(ncoef)]
    data   = np.column_stack([i, j, l, k1, k2, k3, *blocks, valid_center, valid_box])
    path   = os.path.join(directory, f'{PREFIX}_hybrid_values.txt')
    header = f'Columns:\ni j l k1_center k2_center k3_center {" ".join(labels)} valid_center valid_box\nHybrid convention: global interpolation over k1,k2 centers and a cell-local polynomial along k3.'
    formats = ['%d', '%d', '%d'] + ['%.17e'] * (data.shape[1] - 5) + ['%d', '%d']

    np.savetxt(path, data, fmt=formats, header=header)


def write_outputs(out, directory, args):
    """Always write the complete C-readable B/K/D/S table set."""
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)

    for axis, edge in enumerate(out['edges'], start=1):
        np.savetxt(os.path.join(directory, f'{PREFIX}_k{axis}_edges.txt'), edge, fmt='%.17e')

    _write_values_text(out, directory)
    _write_hybrid_text(out, directory)
    _write_metadata(out, directory, args)
    return directory


def _load_edge(directory, axis):
    path = os.path.join(directory, f'{PREFIX}_k{axis}_edges.txt')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing {path!r}.')
    return np.loadtxt(path)


def load_outputs(directory, strict=True):
    """Load C-readable tables written by :func:`write_outputs`."""
    directory = os.path.abspath(directory)
    metadata  = _parse_metadata(_metadata_path(directory))

    try:
        edges = tuple(_load_edge(directory, axis) for axis in [1, 2, 3])
    except FileNotFoundError:
        if strict:
            raise
        return {'ok': False, 'directory': directory, 'prefix': PREFIX}

    shape       = tuple(len(edge) - 1 for edge in edges)
    values_path = os.path.join(directory, f'{PREFIX}_values.txt')

    if not os.path.isfile(values_path):
        if strict:
            raise FileNotFoundError(f'Missing {values_path!r}.')
        return {'ok': False, 'directory': directory, 'prefix': PREFIX}

    table = np.loadtxt(values_path)
    table = table[None, :] if table.ndim == 1 else table

    if table.shape[1] != 12:
        raise ValueError(f'Expected 12 legacy columns in {values_path!r}; found {table.shape[1]}.')

    auxiliary = {
        'B':     table[:, 6].reshape(shape),
        'denom': table[:, 8].reshape(shape),
        'S':     table[:, 9].reshape(shape),
    }
    K            = table[:, 7].reshape(shape)
    valid_center = table[:, 10].reshape(shape).astype(np.int8)
    valid_box    = table[:, 11].reshape(shape).astype(np.int8)
    mode         = metadata.get('normalized_filter_mode', 'hybrid')
    ncoef        = int(metadata.get('hybrid_ncoef', 1))

    if mode == 'hybrid':
        hybrid_path = os.path.join(directory, f'{PREFIX}_hybrid_values.txt')
        if not os.path.isfile(hybrid_path):
            if strict:
                raise FileNotFoundError(f'Missing {hybrid_path!r}.')
            return {'ok': False, 'directory': directory, 'prefix': PREFIX}

        hybrid = np.loadtxt(hybrid_path)
        hybrid = hybrid[None, :] if hybrid.ndim == 1 else hybrid
        expected_columns = 6 + 4 * ncoef + 2
        if hybrid.shape[1] != expected_columns:
            raise ValueError(f'Expected {expected_columns} columns in {hybrid_path!r}; found {hybrid.shape[1]}.')

        coefficients = {}
        for block, name in enumerate(['B', 'K', 'denom', 'S']):
            start              = 6 + block * ncoef
            values             = hybrid[:, start:start + ncoef].reshape(shape + (ncoef,))
            coefficients[name] = np.moveaxis(values, -1, 0)
    else:
        coefficients = {
            'B':     auxiliary['B'][None, ...],
            'K':     K[None, ...],
            'denom': auxiliary['denom'][None, ...],
            'S':     auxiliary['S'][None, ...],
        }

    return {
        'ok':                       True,
        'directory':                directory,
        'prefix':                   PREFIX,
        'metadata':                 metadata,
        'edges':                    edges,
        'K':                        K,
        'K_coeff':                  coefficients['K'],
        'valid_center':             valid_center,
        'valid_box':                valid_box,
        'auxiliary':                {name: auxiliary[name] for name in ['B', 'denom', 'S']},
        'auxiliary_coefficients':   {name: coefficients[name] for name in ['B', 'denom', 'S']},
    }