import ast
import pathlib
import re

HANGUL = re.compile(r"[가-힣]")
PACKAGE = pathlib.Path(__file__).parent.parent / "deeptool"


def _docstrings():
    """(파일명, 대상 이름, docstring) 을 모두 수집한다.

    ast.get_docstring 은 인라인 주석을 수집하지 않으므로 주석의 한국어는
    검사 대상에서 자연히 빠진다.
    """
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if doc:
                yield path.name, getattr(node, "name", "<module>"), doc


def test_docstrings_are_english():
    """docstring 은 API 레퍼런스로 공개되므로 영어여야 한다.

    help() 와 IDE 툴팁은 언어 선택권이 없다. 소스에 박힌 언어 하나가
    모든 사용자에게 간다.
    """
    offenders = [
        f"{file}:{name}" for file, name, doc in _docstrings() if HANGUL.search(doc)
    ]

    assert not offenders, "한글이 남은 docstring: " + ", ".join(offenders)


def test_every_public_symbol_has_a_docstring():
    import deeptool

    missing = [
        name
        for name in deeptool.__all__
        if not name.startswith("__") and not (getattr(deeptool, name).__doc__ or "")
    ]

    assert not missing, f"docstring 없는 공개 심볼: {missing}"
