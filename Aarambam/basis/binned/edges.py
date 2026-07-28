"""Logarithmic product-bin construction for the Aarambam binned backend."""

import warnings

import numpy as np


__all__ = ['AdaptiveProductGrid', 'bin_index', 'centers_from_edges', 'triangle_possible', 'triangle_valid']


_ADAPTIVE_REL_FLOOR           = 0.05
_ADAPTIVE_MIN_WEIGHT_FRACTION = 0.05


def centers_from_edges(edges):
    """Return geometric bin centers."""
    edges = np.asarray(edges, dtype=float)
    return np.sqrt(edges[:-1] * edges[1:])


def bin_index(edges, x):
    """Return clipped bin indices and an in-range mask."""
    edges = np.asarray(edges, dtype=float)
    x     = np.asarray(x)
    ind   = np.searchsorted(edges, x, side='right') - 1
    ind   = np.where(x == edges[-1], len(edges) - 2, ind)
    good  = (ind >= 0) & (ind < len(edges) - 1)
    ind   = np.clip(ind, 0, len(edges) - 2)
    return ind.astype(int), good


def triangle_valid(k1, k2, k3):
    """Return whether each point satisfies all triangle inequalities."""
    k1, k2, k3 = np.broadcast_arrays(k1, k2, k3)
    positive   = (k1 > 0) & (k2 > 0) & (k3 > 0)
    return positive & (k1 + k2 >= k3) & (k1 + k3 >= k2) & (k2 + k3 >= k1)


def triangle_possible(lo1, hi1, lo2, hi2, lo3, hi3):
    """Return whether a product cell contains at least one valid triangle."""
    positive = min(lo1, lo2, lo3) > 0
    return bool(positive and (hi1 + hi2 >= lo3) and (hi1 + hi3 >= lo2) and (hi2 + hi3 >= lo1))


class AdaptiveProductGrid:
    """Place all three product-grid axes using kernel roughness in log(k)."""

    def __init__(self, target, n_bins, kmin, kmax, pilot_N=256, logger=None):
        self.target  = target
        self.n_bins  = int(n_bins)
        self.kmin    = float(kmin)
        self.kmax    = float(kmax)
        self.pilot_N = int(pilot_N)
        self.logger  = logger
        self._validate()

    def _validate(self):
        if self.n_bins <= 0:
            raise ValueError(f'Binned_Nk must be positive; got {self.n_bins}.')
        if self.kmin <= 0 or self.kmax <= 0:
            raise ValueError('Binned k limits must be positive.')
        if self.kmax <= self.kmin:
            raise ValueError(f'Need Binned_kmax > Binned_kmin; got {self.kmin}, {self.kmax}.')
        if self.pilot_N < 4:
            raise ValueError('Binned_adaptive_pilot_N must be at least 4.')

    def _log(self, message, *values):
        if self.logger is not None:
            self.logger.info(message, *values)

    def _pilot_axis(self):
        return np.geomspace(self.kmin, self.kmax, self.pilot_N)

    @staticmethod
    def _global_scale(values, valid):
        amplitude = np.abs(values[np.isfinite(values) & valid])
        if amplitude.size == 0:
            return 1.0

        scale = np.nanpercentile(amplitude, 90)
        if (not np.isfinite(scale)) or scale <= 0:
            scale = np.nanmax(amplitude)
        return float(scale) if np.isfinite(scale) and scale > 0 else 1.0

    @classmethod
    def _axis_roughness(cls, values, valid, axis):
        v0 = np.take(values, np.arange(values.shape[axis] - 1), axis=axis)
        v1 = np.take(values, np.arange(1, values.shape[axis]), axis=axis)
        m0 = np.take(valid, np.arange(valid.shape[axis] - 1), axis=axis)
        m1 = np.take(valid, np.arange(1, valid.shape[axis]), axis=axis)

        good        = m0 & m1 & np.isfinite(v0) & np.isfinite(v1)
        global_scale = cls._global_scale(values, valid)
        floor       = _ADAPTIVE_REL_FLOOR * global_scale
        delta       = np.abs(v1 - v0)
        local_scale = 0.5 * (np.abs(v0) + np.abs(v1))
        metric      = delta / (local_scale + floor) + delta / (global_scale + floor)
        metric      = np.where(good & np.isfinite(metric), metric, np.nan)
        axes        = tuple(index for index in range(metric.ndim) if index != axis)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            roughness = np.nanmean(metric, axis=axes)

        return np.where(np.isfinite(roughness) & (roughness > 0), roughness, 0.0)

    def _edges_from_roughness(self, pilot_k, roughness):
        x         = np.log(pilot_k)
        weights   = np.asarray(roughness, dtype=float)
        weights   = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        weights  *= np.diff(x)
        positive  = weights[weights > 0]

        if positive.size:
            weights += _ADAPTIVE_MIN_WEIGHT_FRACTION * np.mean(positive)

        if (not np.any(weights > 0)) or (not np.isfinite(np.sum(weights))):
            return np.geomspace(self.kmin, self.kmax, self.n_bins + 1)

        cumulative = np.concatenate([[0.0], np.cumsum(weights)])
        targets    = np.linspace(0.0, cumulative[-1], self.n_bins + 1)
        edge_x     = np.interp(targets, cumulative, x)
        edges      = np.exp(edge_x)
        edges[0]   = self.kmin
        edges[-1]  = self.kmax

        if np.any(np.diff(edges) <= 0):
            return np.geomspace(self.kmin, self.kmax, self.n_bins + 1)
        return edges

    def build(self):
        """Return roughness-adaptive log-coordinate edges for k1, k2 and k3."""
        pilot           = self._pilot_axis()
        k1, k2, k3      = np.meshgrid(pilot, pilot, pilot, indexing='ij')
        valid           = triangle_valid(k1, k2, k3)
        values          = np.asarray(self.target(k1, k2, k3), dtype=float)
        values          = np.where(valid & np.isfinite(values), values, np.nan)
        roughness       = [self._axis_roughness(values, valid, axis) for axis in range(3)]
        edges           = [self._edges_from_roughness(pilot, roughness[axis]) for axis in range(3)]
        edges[1]        = edges[0].copy()

        self._log('Built roughness-adaptive log grid: pilot_N=%d, bins=(%d, %d, %d).', self.pilot_N, self.n_bins, self.n_bins, self.n_bins)
        return tuple(edges)