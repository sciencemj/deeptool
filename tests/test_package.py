from importlib.metadata import metadata, version

import ood


def test_package_exposes_version():
    assert ood.__version__ == "0.2.0"


def test_distribution_name_is_ood_dl():
    """PyPI 배포명은 ood-dl 이고 import 이름은 ood 로 서로 다르다."""
    assert metadata("ood-dl")["Name"] == "ood-dl"


def test_distribution_version_matches_dunder_version():
    """pyproject 와 __init__.py 사이의 버전 드리프트를 잡는다."""
    assert version("ood-dl") == ood.__version__
