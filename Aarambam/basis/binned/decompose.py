"""Binned decomposition of a bispectrum template into the tables read by ``2LPTBinned``."""

import itertools
import logging

import numpy as np

from .edges import AdaptiveProductGrid, centers_from_edges, triangle_possible, triangle_valid
from .interpolation import BinnedProductInterpolator, filter_mode, hybrid_input_kind, hybrid_output_kind
from .io import load_outputs, write_outputs


__all__ = ['BasicBinnedDecompose']


class BasicBinnedDecompose:
    """Binned decomposition: tabulates a template's kernel on 3D bins for ``2LPTBinned``.

    Combined with a template by ``Decomposer.go``. The kernel is
    ``K = B / [(P1 P2 + P1 P3 + P2 P3) / 3]`` with ``P(k) = k^(n_s - 4)``. Its values are taken at the
    geometric bin centres, and the bin edges are placed in log k where the kernel varies most. The
    internal legs k1 and k2 share their bins.

    Parameters
    ----------
    n_s : float
        Scalar spectral index.
    Lbox : float
        Box side in Mpc/h.
    Nmax : int
        Largest wavenumber, in units of ``2 pi / Lbox``.
    Binned_Nk : int, optional
        Bins per axis (default 20).
    Binned_kmin, Binned_kmax : float, optional
        k range of the table; default ``2 pi / Lbox`` and ``Nmax`` times that.
    Binned_filter : str, optional
        Interpolation: ``'heaviside'`` (constant per bin) or ``'hybrid_<a>_k3<b>'``, with ``<a>`` the
        interpolation across k1, k2 bins and ``<b>`` the polynomial within a k3 bin (each ``linear`` or
        ``cubic``). Default ``'hybrid_cubic_k3cubic'``.
    Binned_hybrid_input_interpolator, Binned_hybrid_output_filter : str, optional
        ``<a>`` and ``<b>`` when ``Binned_filter`` does not set them (e.g. plain ``'hybrid'``).
    Binned_linear_coordinate : str, optional
        Interpolation coordinate; only ``'log'`` (log k) is supported.
    Binned_enforce_triangle : bool, optional
        Zero the cells that contain no closed triangle (default False).
    Binned_center_triangle_only : bool, optional
        Zero the cells whose centre is not a closed triangle (default False).
    Binned_progress : bool, optional
        Show a progress bar (default True).
    Binned_adaptive_pilot_N : int, optional
        Points per axis of the grid on which the kernel is sampled to place the edges (default 256).
    **kwargs
        Template parameters.

    Raises
    ------
    TypeError
        For an unknown ``Binned_*`` keyword.
    ValueError
        For invalid sizes or interpolation settings.
    """

    DEFAULTS = {
        'Binned_Nk':                        20,
        'Binned_kmin':                      None,
        'Binned_kmax':                      None,
        'Binned_filter':                    'hybrid_cubic_k3cubic',
        'Binned_hybrid_input_interpolator': 'cubic',
        'Binned_hybrid_output_filter':      'cubic',
        'Binned_linear_coordinate':         'log',
        'Binned_enforce_triangle':          False,
        'Binned_center_triangle_only':      False,
        'Binned_progress':                  True,
        'Binned_adaptive_pilot_N':          256,
    }

    def __init__(self, n_s, Lbox, Nmax, **kwargs):
        unknown = sorted(name for name in kwargs if name.startswith('Binned_') and name not in self.DEFAULTS)
        if unknown:
            raise TypeError(f'Unknown binned options: {unknown}.')

        args = dict(n_s = n_s, Lbox = Lbox, Nmax = Nmax)
        args.update(kwargs)

        self.args = args
        self._set_defaults()
        self._validate_settings()
        self._setup_logger()

    def _set_defaults(self):
        for name, value in self.DEFAULTS.items():
            self.args.setdefault(name, value)

    def _validate_settings(self):
        if self.args.get('n_s') is None:
            raise ValueError('n_s is required for BasicBinnedDecompose.')
        if self.args.get('Lbox') is None or float(self.args['Lbox']) <= 0:
            raise ValueError('Lbox must be positive for BasicBinnedDecompose.')
        if self.args.get('Nmax') is None or int(self.args['Nmax']) <= 0:
            raise ValueError('Nmax must be positive for BasicBinnedDecompose.')
        if int(self.args['Binned_Nk']) <= 0:
            raise ValueError('Binned_Nk must be positive.')
        if int(self.args['Binned_adaptive_pilot_N']) < 4:
            raise ValueError('Binned_adaptive_pilot_N must be at least 4.')

        if filter_mode(self.args) == 'hybrid':      #raise on unknown interpolation settings
            hybrid_input_kind(self.args)
            hybrid_output_kind(self.args)

    def _setup_logger(self):
        level       = logging.INFO
        self.logger = logging.getLogger(f'{self.__class__.__module__}.{self.__class__.__name__}')
        self.logger.setLevel(level)
        self.logger.propagate = False

        if not any(getattr(handler, '_aarambam_binned_handler', False) for handler in self.logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)-7s | %(name)s | %(message)s'))
            handler._aarambam_binned_handler = True
            self.logger.addHandler(handler)

        for handler in self.logger.handlers:
            if getattr(handler, '_aarambam_binned_handler', False):
                handler.setLevel(level)

    def _model_name(self):
        for cls in type(self).__mro__[1:]:
            if cls is not BasicBinnedDecompose and cls is not object and not cls.__module__.startswith('Aarambam.basis.binned'):
                return cls.__name__
        return type(self).__name__

    def bispectrum(self, k1, k2, k3):
        """Bispectrum at ``k1, k2, k3``: the template's ``raw_bispectrum`` summed over the six permutations."""
        if not hasattr(self, 'raw_bispectrum'):
            raise NotImplementedError('The model must define bispectrum() or raw_bispectrum().')

        k1, k2, k3 = np.broadcast_arrays(k1, k2, k3)
        momenta    = (k1, k2, k3)
        output     = np.zeros_like(k1, dtype=float)

        for a, b, c in itertools.permutations(range(3)):
            output += self.raw_bispectrum(momenta[a], momenta[b], momenta[c])
        return output

    def shape_function(self, k1, k2, k3):
        """Shape ``S = B (k1 k2 k3)^2`` at ``k1, k2, k3``."""
        return self.bispectrum(k1, k2, k3) * (np.asarray(k1) * np.asarray(k2) * np.asarray(k3))**2

    def primordial_power(self, k):
        """Primordial power ``k^(n_s - 4)`` (unit amplitude) at wavenumbers ``k``."""
        return np.power(np.asarray(k, dtype=float), float(self.args['n_s']) - 4.0)

    def kernel_denominator(self, k1, k2, k3):
        """Denominator of ``kernel``: ``(P1 P2 + P1 P3 + P2 P3) / 3`` at ``k1, k2, k3``."""
        P1 = self.primordial_power(k1)
        P2 = self.primordial_power(k2)
        P3 = self.primordial_power(k3)
        return (P1 * P2 + P1 * P3 + P2 * P3) / 3.0

    def kernel(self, k1, k2, k3):
        """Kernel ``K = B / kernel_denominator`` at ``k1, k2, k3``; 0 where it is not finite."""
        B       = np.asarray(self.bispectrum(k1, k2, k3), dtype=float)
        D       = np.asarray(self.kernel_denominator(k1, k2, k3), dtype=float)
        B, D    = np.broadcast_arrays(B, D)
        output  = np.zeros_like(B, dtype=float)
        eps     = 0.0
        good    = np.isfinite(B) & np.isfinite(D) & (np.abs(D) > eps)
        np.divide(B, D, out=output, where=good)
        return np.where(np.isfinite(output), output, 0.0)

    def _k_limits(self):
        kmin = self.args['Binned_kmin']
        kmax = self.args['Binned_kmax']
        kmin = 2.0 * np.pi / float(self.args['Lbox']) if kmin is None else float(kmin)
        kmax = kmin * int(self.args['Nmax']) if kmax is None else float(kmax)

        if kmin <= 0 or kmax <= 0:
            raise ValueError('Binned k limits must be positive.')
        if kmax <= kmin:
            raise ValueError(f'Need Binned_kmax > Binned_kmin; got {kmin}, {kmax}.')
        return kmin, kmax

    def generate_edges(self):
        """Bin edges of the three axes, placed in log k where the kernel varies most (k1 and k2 equal)."""
        kmin, kmax = self._k_limits()
        builder    = AdaptiveProductGrid(self.kernel, self.args['Binned_Nk'], kmin, kmax, pilot_N=self.args['Binned_adaptive_pilot_N'], logger=self.logger)
        edges      = builder.build()

        for axis, edge in enumerate(edges, start=1):
            self.logger.info('k%d: %d bins over [%.6e, %.6e].', axis, len(edge) - 1, edge[0], edge[-1])
        return edges

    def _triangle_masks(self, edges):
        centers          = tuple(centers_from_edges(edge) for edge in edges)
        C1, C2, C3       = np.meshgrid(*centers, indexing='ij')
        valid_center     = triangle_valid(C1, C2, C3).astype(np.int8)
        valid_box        = np.zeros_like(valid_center, dtype=np.int8)

        for i in range(valid_box.shape[0]):
            for j in range(valid_box.shape[1]):
                for l in range(valid_box.shape[2]):
                    valid_box[i, j, l] = int(triangle_possible(edges[0][i], edges[0][i + 1], edges[1][j], edges[1][j + 1], edges[2][l], edges[2][l + 1]))
        return valid_center, valid_box

    def _quantity_callbacks(self):
        return {'B': self.bispectrum, 'K': self.kernel, 'denom': self.kernel_denominator, 'S': self.shape_function}

    def generate_table(self, edges):
        """Kernel, bispectrum, denominator and shape at the bin centres, with their interpolation
        coefficients, for the three edge arrays ``edges``."""
        valid_center, valid_box = self._triangle_masks(edges)
        interpolator           = BinnedProductInterpolator(self.args, self._quantity_callbacks())
        table                  = interpolator.build(edges, valid_center, valid_box, progress=bool(self.args['Binned_progress']))

        return {
            'edges':                     tuple(np.asarray(edge, dtype=float) for edge in edges),
            'K':                         table['values']['K'],
            'K_coeff':                   table['coefficients']['K'],
            'valid_center':              valid_center,
            'valid_box':                 valid_box,
            'filter_mode':               table['filter_mode'],
            'hybrid_input_interpolator': table['input_kind'],
            'hybrid_output_filter':      table['output_kind'],
            'auxiliary':                 {name: table['values'][name] for name in ['B', 'denom', 'S']},
            'auxiliary_coefficients':    {name: table['coefficients'][name] for name in ['B', 'denom', 'S']},
        }

    def evaluate_binned_quantity(self, name, k1, k2, k3, out=None, fill_value=0.0):
        """Interpolate a tabulated quantity at ``k1, k2, k3``.

        Parameters
        ----------
        name : {'K', 'B', 'denom', 'S'}
            Kernel, bispectrum, kernel denominator or shape.
        k1, k2, k3 : array_like
            Wavenumbers.
        out : dict, optional
            Table from :meth:`go` or :meth:`load_binned_tables`; default the last one made.
        fill_value : float, optional
            Value outside the table (default 0).
        """
        if out is None:
            if not hasattr(self, '_binned_output'):
                raise ValueError('No binned table is attached. Run go() or pass out=... first.')
            out = self._binned_output

        if name == 'K':
            values       = out['K']
            coefficients = out['K_coeff']
        elif name in out['auxiliary']:
            values       = out['auxiliary'][name]
            coefficients = out['auxiliary_coefficients'][name]
        else:
            raise ValueError(f'Unknown binned quantity={name!r}. Use K, B, denom or S.')

        interpolator = BinnedProductInterpolator(self.args, self._quantity_callbacks())
        return interpolator.evaluate(k1, k2, k3, out['edges'], values, coefficients=coefficients, fill_value=fill_value, input_kind=out['hybrid_input_interpolator'])

    def evaluate_binned_kernel(self, k1, k2, k3, out=None, fill_value=0.0):
        """``evaluate_binned_quantity('K', ...)``: the interpolated kernel."""
        return self.evaluate_binned_quantity('K', k1, k2, k3, out=out, fill_value=fill_value)

    def evaluate_binned_shape(self, k1, k2, k3, out=None, fill_value=0.0):
        """``evaluate_binned_quantity('S', ...)``: the interpolated shape."""
        return self.evaluate_binned_quantity('S', k1, k2, k3, out=out, fill_value=fill_value)

    def generate_limit_comparison(self, out):
        """Exact and binned S and K of table ``out`` in the squeezed, equilateral and folded limits."""
        kmin, kmax = self._k_limits()
        kbin       = np.geomspace(kmin, kmax / 2.0, 100)
        kfixed     = np.full_like(kbin, kmin)
        limits     = [(kbin, kbin, kfixed), (kbin, kbin, kbin), (kbin, kbin, 2.0 * kbin)]

        K_original = np.column_stack([self.kernel(k1, k2, k3) for k1, k2, k3 in limits])
        K_apprx    = np.column_stack([self.evaluate_binned_kernel(k1, k2, k3, out=out) for k1, k2, k3 in limits])
        S_original = np.column_stack([self.shape_function(k1, k2, k3) for k1, k2, k3 in limits])
        S_apprx    = np.column_stack([self.evaluate_binned_shape(k1, k2, k3, out=out) for k1, k2, k3 in limits])

        return {
            'K_original': K_original,
            'K_apprx':    K_apprx,
            'S_original': S_original,
            'S_apprx':    S_apprx,
            'kbin':       kbin,
        }

    def go(self):
        """Build the bin edges and tables. Writes nothing; see :meth:`finalize`.

        Returns
        -------
        dict
            ``edges`` (the three edge arrays), ``K`` and ``K_coeff`` (kernel at the bin centres and its
            interpolation coefficients), ``auxiliary`` and ``auxiliary_coefficients`` (the same for ``B``,
            ``denom`` and ``S``), ``valid_center`` and ``valid_box`` (triangle masks), the filter
            metadata, and ``K_original``, ``K_apprx``, ``S_original``, ``S_apprx`` (shape ``(100, 3)``:
            exact and binned values in the squeezed, equilateral and folded limits at the wavenumbers
            ``kbin``).
        """
        self.logger.info('Starting binned decomposition for %s.', self._model_name())
        edges                    = self.generate_edges()
        if len(edges[0]) != len(edges[1]) or not np.array_equal(edges[0], edges[1]):
            raise ValueError('The internal legs k1 and k2 must share their bin edges (generate_edges returned '
                             'different k1 and k2 edges).')
        out                      = self.generate_table(edges)
        out['template']          = self._model_name()
        out['kernel_definition'] = 'B_over_mean_P1P2_plus_perms'
        out.update(self.generate_limit_comparison(out))
        self._binned_output      = out
        self.logger.info('Finished binned decomposition: shape=%s, filter=%s.', out['K'].shape, out['filter_mode'])
        return out

    def finalize(self, out, directory):
        """Write the tables of ``out`` (from :meth:`go`) into ``directory`` (created if needed), replacing
        earlier ``BinnedKernel_*`` files there."""
        write_outputs(out, directory, self.args)
        self.logger.info('Binned kernel tables written to %s.', directory)

    def load_binned_tables(self, directory, strict=True):
        """Read the tables in ``directory`` (written by :meth:`finalize`) for :meth:`evaluate_binned_quantity`.

        Returns the table dict. With ``strict=False`` a missing file returns ``{'ok': False, ...}`` instead
        of raising."""
        loaded = load_outputs(directory, strict=strict)
        if not loaded.get('ok', False):
            return loaded

        metadata = loaded['metadata']
        if metadata.get('filter_mode'):
            self.args['Binned_filter'] = metadata['filter_mode']
        if metadata.get('hybrid_input_interpolator'):
            self.args['Binned_hybrid_input_interpolator'] = metadata['hybrid_input_interpolator']
        if metadata.get('hybrid_output_filter'):
            self.args['Binned_hybrid_output_filter'] = metadata['hybrid_output_filter']
        if metadata.get('hybrid_coordinate'):
            self.args['Binned_linear_coordinate'] = metadata['hybrid_coordinate']

        mode = filter_mode(self.args)
        out  = {
            'edges':                     loaded['edges'],
            'K':                         loaded['K'],
            'K_coeff':                   loaded['K_coeff'],
            'valid_center':              loaded['valid_center'],
            'valid_box':                 loaded['valid_box'],
            'filter_mode':               mode,
            'hybrid_input_interpolator': hybrid_input_kind(self.args) if mode == 'hybrid' else 'heaviside',
            'hybrid_output_filter':      hybrid_output_kind(self.args) if mode == 'hybrid' else 'heaviside',
            'auxiliary':                 loaded['auxiliary'],
            'auxiliary_coefficients':    loaded['auxiliary_coefficients'],
        }
        self._binned_output = out
        return out