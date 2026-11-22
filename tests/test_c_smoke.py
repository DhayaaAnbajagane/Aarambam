"""32^3 smoke tests of the compiled 2LPT executables (marker: mpi).

Needs the executables (installed package, or AARAMBAM_LPT_BIN=<dir with 2LPTBasis, 2LPTBinned>) and mpirun
(or AARAMBAM_MPIRUN). Each case runs on 1 and 2 ranks; ~1 minute in total on one core.
"""
import filecmp, glob, os, subprocess

import numpy as np
import pytest

pytestmark = pytest.mark.mpi

N = 32
CASES = {'basis': ('basis', {}), 'res0': ('resbasis', {'Pk_mode': 0}), 'res1': ('resbasis', {'Pk_mode': 1}),
         'res2': ('resbasis', {'Pk_mode': 2}), 'binned': ('binned', {}),
         'binned_mg0': ('binned', {'BinnedPNGMultiGrid': 0})}
EXE   = {'basis': '2LPTBasis', 'resbasis': '2LPTBasis', 'binned': '2LPTBinned'}
DT    = np.dtype([('ind', 'i4'), ('pot', 'f8')])


def write_config(am, method, out, **extra):
    d = dict(getattr(am.utils, f'DEFAULT_{method.upper()}_INPUTS'))
    d.update(Nmesh=N, Nsample=N, Nmax=N, OutputDir=str(out), GlassTileFac=1, NWriteoutProcesses=1,
             SavePotentialField=1, FFTWWisdomDir='none')      #FFTW_ESTIMATE plans: bit-exact comparisons below
    if method == 'binned':
        d.update(BinnedPNGCacheBlockSize=4)
    d.update(extra)
    path = os.path.join(str(out), 'config')
    open(path, 'w').write(getattr(am.utils, f'make_config_{method}')(**d))
    return path


def run(bins, method, cfg, np_):
    env = dict(os.environ, OMP_NUM_THREADS='1')
    env.setdefault('OMPI_MCA_btl_vader_single_copy_mechanism', 'none')
    return subprocess.run([bins.mpirun, '--oversubscribe', '-np', str(np_), os.path.join(bins.dir, EXE[method]), cfg],
                          capture_output=True, text=True, env=env, timeout=900)


def run_float(lpt_binaries, method, cfg, np_):
    r = run(lpt_binaries, method, cfg, np_)
    if 'built without single-precision FFTW' in r.stdout:
        pytest.skip('executables built without libfftw3f (NGPrecision float)')
    return r


def potential(d, field):
    files = sorted(f for f in glob.glob(os.path.join(d, field + '_potential_*')) if not f.endswith('.npy'))
    return np.concatenate([np.fromfile(f, dtype=DT, offset=16)['pot'] for f in files])


@pytest.fixture(scope='module')
def outputs(am, lpt_binaries, tmp_path_factory):
    out = {}
    for case, (method, extra) in CASES.items():
        for np_ in (1, 2):
            d = tmp_path_factory.mktemp(f'{case}_np{np_}')
            r = run(lpt_binaries, method, write_config(am, method, d, **extra), np_)
            assert r.returncode == 0, r.stdout[-2000:]
            out[case, np_] = str(d)
    return out


def ng_term(d):
    return potential(d, 'Nongauss') - potential(d, 'Gauss')


@pytest.mark.parametrize('case', sorted(CASES))
def test_rank_independent(outputs, case):
    np.testing.assert_array_equal(potential(outputs[case, 1], 'Gauss'), potential(outputs[case, 2], 'Gauss'))
    if case == 'binned':        #multi-grid: the small-grid pairs are summed over the ranks in a rank-dependent order
        a, b = ng_term(outputs[case, 1]), ng_term(outputs[case, 2])
        assert np.abs(a - b).max() <= 1e-12 * np.abs(a).max()
    else:
        np.testing.assert_array_equal(potential(outputs[case, 1], 'Nongauss'), potential(outputs[case, 2], 'Nongauss'))


def test_multigrid_matches_full_grid(outputs):
    """BinnedPNGMultiGrid: the same NG term as every pair on the full grid, to rounding (on 2 ranks it follows from
    test_rank_independent)."""
    np.testing.assert_array_equal(potential(outputs['binned', 1], 'Gauss'), potential(outputs['binned_mg0', 1], 'Gauss'))
    a, b = ng_term(outputs['binned', 1]), ng_term(outputs['binned_mg0', 1])
    assert np.abs(a - b).max() <= 1e-12 * np.abs(b).max()


@pytest.mark.parametrize('precision', ['double', 'float'])
def test_multigrid_medium_grids(am, lpt_binaries, outputs, tmp_path, monkeypatch, precision):
    """The medium-grid path (a band FFT on a smaller grid, all ranks), forced at 32^3 with a small-grid cap of 12."""
    monkeypatch.setenv('AARAMBAM_MG_SMALL_MAX', '12')
    r = run_float(lpt_binaries, 'binned', write_config(am, 'binned', tmp_path, NGPrecision=precision), 2)
    assert r.returncode == 0, r.stdout[-2000:]
    nmed = int(r.stdout.split(' on medium grids')[0].split(',')[-1])
    assert nmed > 0
    a, b = ng_term(str(tmp_path)), ng_term(outputs['binned_mg0', 2])
    assert np.abs(a - b).max() <= (1e-12 if precision == 'double' else 1e-5) * np.abs(b).max()


def test_old_binned_tags_accepted(am, lpt_binaries, outputs, tmp_path):
    """Parameter files from before k2 always took the k1 edges still run (the two tags are ignored)."""
    cfg = write_config(am, 'binned', tmp_path, BinnedPNGMultiGrid=0)
    open(cfg, 'a').write('\nFileWithBinnedK2Edges               unused.txt\nBinnedPNGUseInternalSymmetry        1\n')
    r = run(lpt_binaries, 'binned', cfg, 2)
    assert r.returncode == 0, r.stdout[-2000:]
    np.testing.assert_array_equal(potential(str(tmp_path), 'Nongauss'), potential(outputs['binned_mg0', 2], 'Nongauss'))


def test_resbasis_mode0_is_basis(outputs):
    for f in ('Aarambam_ics', 'Gauss_potential_42', 'Nongauss_potential_42'):
        assert filecmp.cmp(os.path.join(outputs['res0', 1], f), os.path.join(outputs['basis', 1], f), shallow=False)


def test_oscillation_changes_the_field(outputs):
    assert not np.array_equal(potential(outputs['res1', 1], 'Gauss'), potential(outputs['basis', 1], 'Gauss'))


def test_nongaussian_term_present(outputs):
    for case in ('basis', 'binned'):
        g, ng = potential(outputs[case, 1], 'Gauss'), potential(outputs[case, 1], 'Nongauss')
        assert 0 < np.std(ng - g) < 0.1 * np.std(g)     #~1% at Fnl = 100


def test_collate_real_output(am, outputs, tmp_path):
    import shutil
    d = tmp_path / 'c'; shutil.copytree(outputs['binned', 2], d)
    out = am.utils.collate_potential(str(d))
    assert out['Nongauss_potential'].shape == (N, N, N)
    np.testing.assert_array_equal(out['Gauss_potential'].ravel(), potential(outputs['binned', 1], 'Gauss'))


@pytest.mark.parametrize('method', ['basis', 'binned'])
def test_mode_tables_agree(am, lpt_binaries, tmp_path, method):
    """ModeTables 1 (per-mode tables, cached mode fields) reproduces ModeTables 0 to rounding: the two are different
    code paths, so the last bit depends on the compiler. Nmesh 48 > Nsample 32 exercises the out-of-band skipping."""
    pot = {}
    for mt in (0, 1):
        d = tmp_path / f'mt{mt}'; d.mkdir()
        r = run(lpt_binaries, method, write_config(am, method, d, Nmesh=48, ModeTables=mt), 2)
        assert r.returncode == 0, r.stdout[-2000:]
        pot[mt] = {f: potential(str(d), f) for f in ('Gauss', 'Nongauss')}
    np.testing.assert_array_equal(pot[0]['Gauss'], pot[1]['Gauss'])
    a, b = pot[0]['Nongauss'], pot[1]['Nongauss']
    assert np.abs(a - b).max() <= 1e-12 * np.abs(a).max()


def test_fftw_wisdom_file(am, lpt_binaries, tmp_path, monkeypatch):
    """FFTWWisdomDir: the first run measures the FFT plans and saves them in the directory, later runs read them
    back. Runs that read the same file use the same plans (bit-identical); the measuring run (each rank tunes its
    own local plans) and FFTW_ESTIMATE plans agree with them to rounding."""
    wisdom_dir = tmp_path / 'plans' / 'sub'
    monkeypatch.delenv('AARAMBAM_FFTW_WISDOM_DIR', raising=False)
    pot, logs = {}, {}
    for name, tag in (('first', wisdom_dir), ('second', wisdom_dir), ('third', wisdom_dir), ('estimate', 'none')):
        d = tmp_path / name; d.mkdir()
        r = run(lpt_binaries, 'binned', write_config(am, 'binned', d, FFTWWisdomDir=tag), 2)
        assert r.returncode == 0, r.stdout[-2000:]
        pot[name], logs[name] = potential(str(d), 'Nongauss'), r.stdout
    files = list(wisdom_dir.glob('fftw3_wisdom_*.txt'))
    assert len(files) == 1
    wisdom = files[0]
    assert 'FFT plans measured' in logs['first'] and 'saved to ' + str(wisdom) in logs['first']
    assert 'FFT plans read from ' + str(wisdom) in logs['second'] and 'FFT plans read from' in logs['third']
    assert 'FFT plans' not in logs['estimate']
    np.testing.assert_array_equal(pot['second'], pot['third'])          #same plans from the same file
    for name in ('first', 'second'):
        np.testing.assert_allclose(pot[name], pot['estimate'], rtol=0, atol=1e-12 * np.abs(pot['estimate']).max())


# (ranks, Nmesh, Nsample[, measure]): one rank, Nmesh = 3/2 Nsample, measured plans, uneven slabs (40 on 3 ranks), a
# rank without a slab (24 on 7 ranks: slabs of 4)
FFT_CHECK_CASES = [(1, 32, 32), (2, 48, 32), (2, 48, 32, 'measure'), (3, 40, 26), (7, 24, 16)]


@pytest.mark.parametrize('case', FFT_CHECK_CASES, ids=lambda c: '-'.join(map(str, c)))
def test_fft_backends_agree(lpt_binaries, case):
    """2LPTFFTCheck: the band FFT backend (forward, inverse, band -> slab layout) against FFTW3-MPI and against the
    exact transform of plane waves, to 1e-12 of the largest value."""
    exe = os.path.join(lpt_binaries.dir, '2LPTFFTCheck')
    if not os.path.isfile(exe):
        pytest.skip('2LPTFFTCheck not built')
    env = dict(os.environ, OMP_NUM_THREADS='1')
    env.setdefault('OMPI_MCA_btl_vader_single_copy_mechanism', 'none')
    r = subprocess.run([lpt_binaries.mpirun, '--oversubscribe', '-np', str(case[0]), exe] + [str(c) for c in case[1:]],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0 and 'PASS' in r.stdout, r.stdout[-2000:]


@pytest.mark.parametrize('method', ['basis', 'binned'])
def test_fft_backend_band_matches_fftw(am, lpt_binaries, tmp_path, method):
    """FFTBackend band gives the ICs of FFTBackend fftw to rounding, on 2 ranks with Nmesh 48 > Nsample 32."""
    pot = {}
    for backend in ('fftw', 'band'):
        d = tmp_path / backend; d.mkdir()
        r = run(lpt_binaries, method, write_config(am, method, d, Nmesh=48, FFTBackend=backend), 2)
        assert r.returncode == 0, r.stdout[-2000:]
        assert ('FFTBackend band:' in r.stdout) == (backend == 'band')
        pot[backend] = {f: potential(str(d), f) for f in ('Gauss', 'Nongauss')}
    for f in ('Gauss', 'Nongauss'):
        a, b = pot['fftw'][f], pot['band'][f]
        assert np.abs(a - b).max() <= 1e-12 * np.abs(a).max(), f


@pytest.mark.parametrize('method', ['basis', 'binned'])
def test_ng_precision_float_matches_double(am, lpt_binaries, tmp_path, method):
    """NGPrecision float: the NG term (Nongauss - Gauss potential) agrees with NGPrecision double to 1e-5 (measured:
    ~4e-7 binned, ~1e-6 basis), the Gaussian part is unchanged, and the float run is rank independent."""
    ng, gauss = {}, {}
    for prec, np_ in (('double', 2), ('float', 2), ('float', 1)):
        d = tmp_path / f'{prec}{np_}'; d.mkdir()
        r = run_float(lpt_binaries, method, write_config(am, method, d, Nmesh=48, NGPrecision=prec), np_)
        assert r.returncode == 0, r.stdout[-2000:]
        assert ('NGPrecision float:' in r.stdout) == (prec == 'float')
        gauss[prec, np_] = potential(str(d), 'Gauss')
        ng[prec, np_] = potential(str(d), 'Nongauss') - gauss[prec, np_]
    np.testing.assert_array_equal(gauss['double', 2], gauss['float', 2])
    a, b = ng['double', 2], ng['float', 2]
    assert np.abs(a - b).max() <= 1e-5 * np.abs(a).max()
    assert np.sqrt(np.mean((a - b) ** 2) / np.mean(a ** 2)) <= 1e-5
    np.testing.assert_allclose(ng['float', 1], b, rtol=0, atol=1e-6 * np.abs(a).max())


def test_float_wisdom_file(am, lpt_binaries, tmp_path, monkeypatch):
    """NGPrecision float keeps its single-precision plans in fftw3f_wisdom_*.txt next to the double ones."""
    wisdom_dir = tmp_path / 'plans'
    monkeypatch.delenv('AARAMBAM_FFTW_WISDOM_DIR', raising=False)
    logs = []
    for name in ('first', 'second'):
        d = tmp_path / name; d.mkdir()
        r = run_float(lpt_binaries, 'binned', write_config(am, 'binned', d, FFTWWisdomDir=wisdom_dir,
                                                           NGPrecision='float'), 2)
        assert r.returncode == 0, r.stdout[-2000:]
        logs.append(r.stdout)
    assert len(list(wisdom_dir.glob('fftw3f_wisdom_*.txt'))) == 1
    assert 'Single-precision FFT plans measured' in logs[0]
    assert 'Single-precision FFT plans read from' in logs[1]


BAD_CONFIGS = {     #case: (method, config inputs, line appended to the config, expected message)
    'ng_precision':  ('binned', dict(NGPrecision='half'), '', "NGPrecision 'half'"),
    'fft_backend':   ('binned', dict(FFTBackend='cufft'), '', "FFTBackend 'cufft'"),
    'unknown_tag':   ('basis',  {}, 'NotARealTag 1', 'NotARealTag'),
    'kernel_table':  ('binned', dict(BinnedPNGOutputFilterCode=1), '', 'columns'),   #default table has 4 coefficients
    'wisdom_dir':    ('binned', {}, '', 'cannot be created or is not writable'),          #directory set below
    'wisdom_env':    ('binned', dict(FFTWWisdomDir='auto'), '', 'AARAMBAM_FFTW_WISDOM_DIR is longer than'),
    'odd_nmesh':     ('basis',  dict(Nmesh=33), '', 'Nmesh must be even'),
    'nsample':       ('binned', dict(Nsample=48, Nmax=48), '', 'Nsample must be between 1 and Nmesh'),
}


@pytest.mark.parametrize('case', sorted(BAD_CONFIGS))
def test_bad_config_aborts(am, lpt_binaries, tmp_path, monkeypatch, case):
    """A bad config entry stops the run with a message and a non-zero exit code."""
    method, extra, line, message = BAD_CONFIGS[case]
    if case == 'wisdom_dir':        #a file where the directory should be
        (tmp_path / 'a_file').write_text('not a directory')
        extra = dict(FFTWWisdomDir=str(tmp_path / 'a_file' / 'plans'))
    if case == 'wisdom_env':        #longer than the FFTWWisdomDir tag allows: an error, not a shortened path
        monkeypatch.setenv('AARAMBAM_FFTW_WISDOM_DIR', str(tmp_path / ('d' * 600)))
    cfg = write_config(am, method, tmp_path, **extra)
    if line:
        open(cfg, 'a').write('\n' + line + '\n')
    r = run(lpt_binaries, method, cfg, 2)
    assert r.returncode != 0 and message in r.stdout, r.stdout[-2000:]
