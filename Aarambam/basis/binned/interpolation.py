"""Separable interpolation tables used by the Aarambam binned IC backend."""

import itertools

import numpy as np
from tqdm.auto import tqdm

from .edges import bin_index, centers_from_edges


__all__ = ['BinnedProductInterpolator', 'filter_mode', 'hybrid_input_kind', 'hybrid_output_kind']


def filter_mode(args):
    """Normalize the public filter aliases into ``heaviside`` or ``hybrid``."""
    mode = str(args.get('Binned_filter', 'hybrid_cubic_k3cubic')).lower()

    if mode in ['heaviside', 'top_hat']:
        return 'heaviside'
    if mode in ['hybrid', 'hybrid_linear', 'hybrid_cubic', 'hybrid_linear_k3linear', 'hybrid_cubic_k3linear', 'hybrid_linear_k3cubic', 'hybrid_cubic_k3cubic']:
        return 'hybrid'

    raise ValueError(f'Unknown Binned_filter={mode!r}. Use heaviside/top_hat or a hybrid_* filter.')


def hybrid_input_kind(args):
    """Return the global interpolation order used on the internal k1/k2 legs."""
    mode = str(args.get('Binned_filter', 'hybrid_cubic_k3cubic')).lower()

    if 'hybrid_cubic' in mode:
        return 'cubic'
    if 'hybrid_linear' in mode:
        return 'linear'

    kind = str(args.get('Binned_hybrid_input_interpolator', 'cubic')).lower()
    if kind not in ['linear', 'cubic']:
        raise ValueError('Binned_hybrid_input_interpolator must be linear or cubic.')
    return kind


def hybrid_output_kind(args):
    """Return the cell-local polynomial order used on the output k3 leg."""
    mode = str(args.get('Binned_filter', 'hybrid_cubic_k3cubic')).lower()

    if mode.endswith('_k3cubic'):
        return 'cubic'
    if mode.endswith('_k3linear'):
        return 'linear'
    if mode in ['hybrid_linear', 'hybrid_cubic']:
        return 'heaviside'

    kind = str(args.get('Binned_hybrid_output_filter', 'cubic')).lower()
    if kind == 'top_hat':
        kind = 'heaviside'
    if kind not in ['heaviside', 'linear', 'cubic']:
        raise ValueError('Binned_hybrid_output_filter must be heaviside, linear or cubic.')
    return kind


def _hybrid_output_order(args):
    return {'heaviside': 0, 'linear': 1, 'cubic': 3}[hybrid_output_kind(args)]


class BinnedProductInterpolator:
    """Build and evaluate C-compatible B/K/D/S product-bin tables."""

    def __init__(self, args, quantities):
        self.args       = args
        self.quantities = dict(quantities)

        required = {'B', 'K', 'denom', 'S'}
        missing  = required.difference(self.quantities)
        if missing:
            raise ValueError(f'Missing required auxiliary quantities: {sorted(missing)}.')

    @staticmethod
    def centers(edges):
        return centers_from_edges(edges)

    def _coord_value(self, k):
        k    = np.asarray(k, dtype=float)
        mode = str(self.args.get('Binned_linear_coordinate', 'log')).lower()

        if mode == 'log':
            return np.log(k)
        if mode in ['k', 'linear']:
            return k
        raise ValueError('Binned_linear_coordinate must be k or log.')

    def _center_for_polynomial(self, lo, hi):
        mode = str(self.args.get('Binned_linear_coordinate', 'log')).lower()
        return np.sqrt(lo * hi) if mode == 'log' else 0.5 * (lo + hi)

    def _u_in_bin(self, x, lo, hi):
        center = self._center_for_polynomial(lo, hi)
        width  = self._coord_value(hi) - self._coord_value(lo)
        out    = np.zeros_like(np.asarray(x, dtype=float))
        np.divide(self._coord_value(x) - self._coord_value(center), width, out=out, where=np.asarray(width) != 0)
        return out

    def point_values(self, k1, k2, k3):
        return {name: np.asarray(function(k1, k2, k3), dtype=float) for name, function in self.quantities.items()}

    @staticmethod
    def _finite_scalar(value, fallback=0.0):
        value = float(np.asarray(value))
        return value if np.isfinite(value) else float(fallback)

    def _safe_point_values(self, k1, k2, k3, fallback):
        values = self.point_values(k1, k2, k3)
        return {name: self._finite_scalar(values[name], fallback=fallback[name]) for name in self.quantities}

    @staticmethod
    def _cubic_coefficients_from_endpoint_slopes(y_lo, y0, y_hi, dy_lo_du, dy_hi_du):
        a      = 0.5
        g_lo   = float(y_lo - y0)
        g_hi   = float(y_hi - y0)
        c2     = (g_lo + g_hi) / (2.0 * a**2)
        dvalue = (g_hi - g_lo) / (2.0 * a)
        c3_lo  = (float(dy_lo_du) - dvalue + 2.0 * a * c2) / (2.0 * a**2)
        c3_hi  = (float(dy_hi_du) - dvalue - 2.0 * a * c2) / (2.0 * a**2)
        c3     = 0.5 * (c3_lo + c3_hi)
        c1     = dvalue - a**2 * c3
        return np.array([c1, c2, c3], dtype=float)

    def _edge_slope_k3(self, edge_index, edges3, k1, k2, fallback):
        nedge = len(edges3)

        if edge_index <= 0:
            ia, ib = 0, 1
        elif edge_index >= nedge - 1:
            ia, ib = nedge - 2, nedge - 1
        else:
            ia, ib = edge_index - 1, edge_index + 1

        ka    = edges3[ia]
        kb    = edges3[ib]
        va    = self._safe_point_values(k1, k2, ka, fallback)
        vb    = self._safe_point_values(k1, k2, kb, fallback)
        width = float(self._coord_value(kb) - self._coord_value(ka))

        if width == 0 or not np.isfinite(width):
            return {name: 0.0 for name in self.quantities}
        return {name: float((vb[name] - va[name]) / width) for name in self.quantities}

    def _hybrid_k3_coefficients(self, l, edges3, k1, k2, center_values):
        output = hybrid_output_kind(self.args)
        lo3    = edges3[l]
        hi3    = edges3[l + 1]

        if output == 'heaviside':
            return {name: np.array([value], dtype=float) for name, value in center_values.items()}

        value_lo = self._safe_point_values(k1, k2, lo3, center_values)
        value_hi = self._safe_point_values(k1, k2, hi3, center_values)

        if output == 'linear':
            return {name: np.array([center_values[name], value_hi[name] - value_lo[name]], dtype=float) for name in self.quantities}

        slope_lo = self._edge_slope_k3(l, edges3, k1, k2, center_values)
        slope_hi = self._edge_slope_k3(l + 1, edges3, k1, k2, center_values)
        width    = float(self._coord_value(hi3) - self._coord_value(lo3))
        width    = width if np.isfinite(width) else 0.0
        output   = {}

        for name in self.quantities:
            coeff      = np.zeros(4, dtype=float)
            coeff[0]   = center_values[name]
            coeff[1:]  = self._cubic_coefficients_from_endpoint_slopes(value_lo[name], center_values[name], value_hi[name], width * slope_lo[name], width * slope_hi[name])
            output[name] = coeff
        return output

    def build(self, edges, valid_center, valid_box, progress=True):
        """Build center values and hybrid k3 coefficients for B, K, D and S."""
        edges1, edges2, edges3 = edges
        cen1                   = self.centers(edges1)
        cen2                   = self.centers(edges2)
        cen3                   = self.centers(edges3)
        shape                  = (len(cen1), len(cen2), len(cen3))
        mode                   = filter_mode(self.args)
        ncoef                  = _hybrid_output_order(self.args) + 1 if mode == 'hybrid' else 1
        values                 = {name: np.zeros(shape, dtype=float) for name in self.quantities}
        coefficients           = {name: np.zeros((ncoef,) + shape, dtype=float) for name in self.quantities}
        enforce_box            = bool(self.args.get('Binned_enforce_triangle', False))
        enforce_center         = bool(self.args.get('Binned_center_triangle_only', False))
        iterator               = itertools.product(range(shape[0]), range(shape[1]))

        if progress:
            iterator = tqdm(iterator, total=shape[0] * shape[1], desc='Binning IC kernel')

        for i, j in iterator:
            for l in range(shape[2]):
                invalid = (enforce_box and not valid_box[i, j, l]) or (enforce_center and not valid_center[i, j, l])

                if invalid:
                    center_values = {name: 0.0 for name in self.quantities}
                    coeff         = {name: np.zeros(ncoef, dtype=float) for name in self.quantities}
                else:
                    point_values  = self.point_values(cen1[i], cen2[j], cen3[l])
                    center_values = {name: self._finite_scalar(value) for name, value in point_values.items()}
                    coeff         = self._hybrid_k3_coefficients(l, edges3, cen1[i], cen2[j], center_values) if mode == 'hybrid' else {name: np.array([center_values[name]], dtype=float) for name in self.quantities}

                for name in self.quantities:
                    values[name][i, j, l]          = center_values[name]
                    coefficients[name][:, i, j, l] = coeff[name]

        return {
            'values':       values,
            'coefficients': coefficients,
            'filter_mode':  mode,
            'input_kind':   hybrid_input_kind(self.args) if mode == 'hybrid' else 'heaviside',
            'output_kind':  hybrid_output_kind(self.args) if mode == 'hybrid' else 'heaviside',
        }

    def _global_weights(self, edges, x, kind):
        x     = np.asarray(x, dtype=float).ravel()
        nodes = self.centers(edges)
        xn    = self._coord_value(nodes)
        xx    = self._coord_value(x)
        n     = len(nodes)
        valid = np.isfinite(xx) & (x >= edges[0]) & (x <= edges[-1])

        if n == 1:
            return np.zeros((1, x.size), dtype=int), np.ones((1, x.size), dtype=float), valid

        if kind == 'cubic' and n >= 4:
            right  = np.searchsorted(xn, xx, side='right')
            base   = np.clip(right - 2, 0, n - 4)
            index  = base[None, :] + np.arange(4)[:, None]
            weight = np.ones((4, x.size), dtype=float)

            for m in range(4):
                xm = xn[index[m]]
                wm = np.ones(x.size, dtype=float)
                for q in range(4):
                    if q == m:
                        continue
                    xq  = xn[index[q]]
                    den = xm - xq
                    wm  = np.where(den != 0, wm * (xx - xq) / den, 0.0)
                weight[m] = wm

            low  = xx <= xn[0]
            high = xx >= xn[-1]
            for mask, endpoint in [(low, 0), (high, n - 1)]:
                if np.any(mask):
                    index[:, mask]  = endpoint
                    weight[:, mask] = 0.0
                    weight[0, mask] = 1.0
            return index.astype(int), weight, valid

        right       = np.searchsorted(xn, xx, side='right')
        i0          = np.clip(right - 1, 0, n - 2)
        i1          = i0 + 1
        denominator = xn[i1] - xn[i0]
        t           = np.zeros_like(xx)
        good        = denominator != 0
        t[good]     = (xx[good] - xn[i0[good]]) / denominator[good]
        t           = np.clip(t, 0.0, 1.0)
        low         = xx <= xn[0]
        high        = xx >= xn[-1]
        i0[low]     = i1[low] = 0
        i0[high]    = i1[high] = n - 1
        t[low | high] = 0.0
        index       = np.vstack([i0, i1]).astype(int)
        weight      = np.vstack([1.0 - t, t])
        weight[:, low | high] = 0.0
        weight[0, low | high] = 1.0
        return index, weight, valid

    def evaluate(self, k1, k2, k3, edges, values, coefficients=None, fill_value=0.0, input_kind=None):
        """Evaluate a generated Heaviside or hybrid table."""
        mode       = filter_mode(self.args)
        k1, k2, k3 = np.broadcast_arrays(k1, k2, k3)
        out_shape  = k1.shape
        x1         = k1.ravel()
        x2         = k2.ravel()
        x3         = k3.ravel()
        out        = np.full(x1.size, fill_value, dtype=float)

        if mode == 'heaviside' or coefficients is None:
            i1, g1 = bin_index(edges[0], x1)
            i2, g2 = bin_index(edges[1], x2)
            i3, g3 = bin_index(edges[2], x3)
            good   = g1 & g2 & g3
            out[good] = values[i1[good], i2[good], i3[good]]
            return np.where(np.isfinite(out), out, fill_value).reshape(out_shape)

        input_kind = hybrid_input_kind(self.args) if input_kind is None else str(input_kind).lower()
        i1, w1, g1 = self._global_weights(edges[0], x1, input_kind)
        i2, w2, g2 = self._global_weights(edges[1], x2, input_kind)
        i3, g3     = bin_index(edges[2], x3)
        good       = g1 & g2 & g3

        if not np.any(good):
            return out.reshape(out_shape)

        u3          = self._u_in_bin(x3[good], edges[2][i3[good]], edges[2][i3[good] + 1])
        porder      = coefficients.shape[0] - 1
        accumulator = np.zeros(np.sum(good), dtype=float)

        for a in range(i1.shape[0]):
            wa = w1[a, good]
            if not np.any(wa != 0):
                continue

            for b in range(i2.shape[0]):
                weight = wa * w2[b, good]
                if not np.any(weight != 0):
                    continue

                coeff = coefficients[:, i1[a, good], i2[b, good], i3[good]]
                value = coeff[0].copy()
                for power in range(1, porder + 1):
                    value += coeff[power] * u3**power
                accumulator += weight * value

        out[np.where(good)[0]] = accumulator
        return np.where(np.isfinite(out), out, fill_value).reshape(out_shape)