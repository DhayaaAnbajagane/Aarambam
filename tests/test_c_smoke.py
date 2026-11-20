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
         'res2': ('resbasis', {'Pk_mode': 2}), 'binned': ('binned', {})}
EXE   = {'basis': '2LPTBasis', 'resbasis': '2LPTBasis', 'binned': '2LPTBinned'}
DT    = np.dtype([('ind', 'i4'), ('pot', 'f8')])


def write_config(am, method, out, **extra):
    d = dict(getattr(am.utils, f'DEFAULT_{method.upper()}_INPUTS'))
    d.update(Nmesh=N, Nsample=N, Nmax=N, OutputDir=str(out), GlassTileFac=1, NWriteoutProcesses=1,
             SavePotentialField=1, FFTWWisdomFile='none')      #FFTW_ESTIMATE plans: bit-exact comparisons below
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


@pytest.mark.parametrize('case', sorted(CASES))
def test_rank_independent(outputs, case):
    for field in ('Gauss', 'Nongauss'):
        np.testing.assert_array_equal(potential(outputs[case, 1], field), potential(outputs[case, 2], field))


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
    """ModeTables 1 (per-mode tables, cached mode fields) reproduces ModeTables 0: bit for bit for the binned
    method, to rounding for the modal one. Nmesh 48 > Nsample 32 exercises the out-of-band skipping."""
    pot = {}
    for mt in (0, 1):
        d = tmp_path / f'mt{mt}'; d.mkdir()
        r = run(lpt_binaries, method, write_config(am, method, d, Nmesh=48, ModeTables=mt), 2)
        assert r.returncode == 0, r.stdout[-2000:]
        pot[mt] = {f: potential(str(d), f) for f in ('Gauss', 'Nongauss')}
    np.testing.assert_array_equal(pot[0]['Gauss'], pot[1]['Gauss'])
    a, b = pot[0]['Nongauss'], pot[1]['Nongauss']
    if method == 'binned':
        np.testing.assert_array_equal(a, b)
    else:
        assert np.abs(a - b).max() <= 1e-12 * np.abs(a).max()


def test_fftw_wisdom_file(am, lpt_binaries, tmp_path, monkeypatch):
    """FFTWWisdomFile 'auto' (here redirected by AARAMBAM_FFTW_WISDOM): the first run measures the FFT plans and
    saves them, later runs read them back. Runs that read the same file use the same plans (bit-identical); the
    measuring run (each rank tunes its own local plans) and FFTW_ESTIMATE plans agree with them to rounding."""
    wisdom = tmp_path / 'plans' / 'wisdom.txt'
    monkeypatch.setenv('AARAMBAM_FFTW_WISDOM', str(wisdom))
    pot, logs = {}, {}
    for name, tag in (('first', 'auto'), ('second', 'auto'), ('third', 'auto'), ('estimate', 'none')):
        d = tmp_path / name; d.mkdir()
        r = run(lpt_binaries, 'binned', write_config(am, 'binned', d, FFTWWisdomFile=tag), 2)
        assert r.returncode == 0, r.stdout[-2000:]
        pot[name], logs[name] = potential(str(d), 'Nongauss'), r.stdout
    assert wisdom.exists()
    assert 'FFT plans measured' in logs['first'] and 'saved to ' + str(wisdom) in logs['first']
    assert 'FFT plans read from ' + str(wisdom) in logs['second'] and 'FFT plans read from' in logs['third']
    assert 'FFT plans' not in logs['estimate']
    np.testing.assert_array_equal(pot['second'], pot['third'])          #same plans from the same file
    for name in ('first', 'second'):
        np.testing.assert_allclose(pot[name], pot['estimate'], rtol=0, atol=1e-12 * np.abs(pot['estimate']).max())


@pytest.mark.parametrize('method', ['basis', 'binned'])
def test_bad_tag_aborts(am, lpt_binaries, tmp_path, method):
    cfg = write_config(am, method, tmp_path)
    open(cfg, 'a').write('\nNotARealTag 1\n')
    r = run(lpt_binaries, method, cfg, 2)
    assert r.returncode != 0 and 'NotARealTag' in r.stdout


def test_mismatched_kernel_table_aborts(am, lpt_binaries, tmp_path):
    cfg = write_config(am, 'binned', tmp_path, BinnedPNGOutputFilterCode=1)   #default table has 4 coefficients
    r = run(lpt_binaries, 'binned', cfg, 1)
    assert r.returncode != 0 and 'columns' in r.stdout
