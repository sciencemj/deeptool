from importlib.metadata import metadata, version

import deeptool


def test_package_exposes_version():
    assert deeptool.__version__ == "0.2.0"


def test_distribution_name_is_deeptool():
    """PyPI 배포명은 deeptool 이고 import 이름은 deeptool 로 서로 다르다."""
    assert metadata("deeptool")["Name"] == "deeptool"


def test_distribution_version_matches_dunder_version():
    """pyproject 와 __init__.py 사이의 버전 드리프트를 잡는다."""
    assert version("deeptool") == deeptool.__version__
