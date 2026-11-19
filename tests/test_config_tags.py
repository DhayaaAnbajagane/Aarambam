"""The config templates and the C parameter tables (src/lpt) agree. Pure Python: parses the C sources."""
import os, re

import pytest

from conftest import ROOT

LPT = os.path.join(ROOT, 'src', 'lpt')
METHOD_TABLE = {'basis': 'modal/params_modal.c', 'resbasis': 'modal/params_modal.c', 'binned': 'binned/params_binned.c'}


def c_table(path):
    """{tag: required} from the {"tag", &addr, TYPE, size, required} rows of a C parameter table."""
    rows = re.findall(r'^\s*\{"(\w+)",\s*[^,]+,\s*PARAM_\w+,\s*[^,]+,\s*([01])\}', open(os.path.join(LPT, path)).read(), re.M)
    assert rows, path
    return {t: r == '1' for t, r in rows}


def config_tags(text):
    tags = [l.split()[0] for l in text.splitlines() if l.split() and not l.lstrip().startswith(('%', '#'))]
    assert len(tags) == len(set(tags)), 'duplicate tag in config template'
    return set(tags)


@pytest.mark.parametrize('method', ['basis', 'resbasis', 'binned'])
def test_template_tags_match_c(am, method):
    table = c_table('common/read_param.c')
    table.update(c_table(METHOD_TABLE[method]))
    tags = config_tags(getattr(am.utils, f'make_example_config_{method}')())

    assert tags <= set(table), f'tags the C code would reject: {sorted(tags - set(table))}'
    missing = {t for t, req in table.items() if req} - tags
    assert not missing, f'required C tags the config does not write: {sorted(missing)}'
