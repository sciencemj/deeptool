from importlib.metadata import entry_points, metadata, requires, version

import deeptool


def test_package_exposes_version():
    assert deeptool.__version__ == "0.3.0"


def test_distribution_name_is_deeptool():
    """PyPI 배포명은 deeptool 이고 import 이름은 deeptool 로 서로 다르다."""
    assert metadata("deeptool")["Name"] == "deeptool"


def test_distribution_version_matches_dunder_version():
    """pyproject 와 __init__.py 사이의 버전 드리프트를 잡는다."""
    assert version("deeptool") == deeptool.__version__


def test_dashboard_console_script_is_installed():
    scripts = {
        item.name: item.value
        for item in entry_points(group="console_scripts")
    }

    assert scripts["deeptool-dashboard"] == "deeptool.dashboard:main"


def test_dashboard_dependencies_are_optional():
    requirements = requires("deeptool") or []

    assert any(
        item.startswith("streamlit>=1.37") and "dashboard" in item
        for item in requirements
    )
    assert any(
        item.startswith("altair>=5") and "dashboard" in item
        for item in requirements
    )
