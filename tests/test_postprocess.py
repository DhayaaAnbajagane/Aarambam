"""collate_potential on synthetic slab files (pure Python)."""
import numpy as np
import pytest

DT = np.dtype([('ind', 'i4'), ('pot', 'f8')])


def write_slab(path, field, nmesh, x0):
    """Same layout as save.c: header (int total_size, int Nmesh, double Box) then (int index, double value)."""
    nx = field.shape[0]
    with open(path, 'wb') as f:
        np.array([0, nmesh], dtype=np.int32).tofile(f)
        np.array([100.0]).tofile(f)
        rec = np.zeros(field.size, dtype=DT)
        rec['ind'] = np.arange(field.size)
        rec['pot'] = field.ravel()
        rec.tofile(f)


@pytest.fixture
def fields():
    rng = np.random.default_rng(3)
    return {t: rng.normal(size=(8, 8, 8)) for t in ('Gauss', 'Nongauss')}


def test_slabs_and_seeds(am, tmp_path, fields):
    for seed in (7, 42):
        for t, a in fields.items():
            for x0, nx in ((0, 3), (3, 5)):                     #uneven slabs, written in any order
                write_slab(tmp_path / f'{t}_potential_{seed}_StartX_{x0:04d}_NumX_{nx}', a[x0:x0+nx] * seed, 8, x0)
    with pytest.raises(ValueError):
        am.utils.collate_potential(str(tmp_path), cleanup=False)
    out = am.utils.collate_potential(str(tmp_path), seed=7)
    np.testing.assert_array_equal(out['Gauss_potential'], fields['Gauss'] * 7)
    left = sorted(p.name for p in tmp_path.glob('*_potential_*'))
    assert left and all('_42_' in n for n in left)


def test_missing_slab(am, tmp_path, fields):
    for t, a in fields.items():
        write_slab(tmp_path / f'{t}_potential_1_StartX_0000_NumX_4', a[:4], 8, 0)
    with pytest.raises(ValueError):
        am.utils.collate_potential(str(tmp_path))
    assert len(list(tmp_path.glob('*_potential_*'))) == 2      #nothing deleted on failure

