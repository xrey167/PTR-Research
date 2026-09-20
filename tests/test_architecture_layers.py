"""The layer rule from ARCHITECTURE-MASTER, checked instead of asserted.

Section 1 of the master document says "eine Schicht spricht nur die Schicht
darunter an" and "Pods sprechen Storage nur über die PodStorage-Fassade, nie
Backends direkt". These tests check the real tree against those sentences,
and check the checker against a synthetic tree — a rule nobody has seen fail
is not a rule.
"""
import re
import textwrap
from pathlib import Path

import pytest

from neural_pods import architecture

LAYER_HEADING = re.compile(r"^# --- layer (\d+):")


def _layer_headings(source: str) -> list[str]:
    return [m.group(1) for m in map(LAYER_HEADING.match, source.splitlines()) if m]


def _package(tmp_path, files: dict[str, str]):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name, source in files.items():
        (package / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    return package


def test_the_real_tree_obeys_the_documented_layering():
    result = architecture.check()
    assert result["violations"] == []
    assert result["ok"] is True


def test_every_module_is_placed_on_a_layer():
    """A new module has to be placed deliberately; otherwise the manifest
    quietly stops describing the package."""
    unplaced = architecture.modules() - set(architecture.LAYER_OF)
    assert unplaced == set()
    assert set(architecture.LAYER_OF) - architecture.modules() == set()
    assert set(architecture.LAYER_OF.values()) <= set(architecture.LAYERS)


def test_an_upward_import_is_a_violation(tmp_path):
    package = _package(tmp_path, {
        "low": "value = 1\n",
        "high": "from .low import value\n",
        "wrong": "from .high import value\n",     # kern importing pod_arm
    })
    layers = {"low": "kern", "high": "pod_arm", "wrong": "kern"}
    result = architecture.check(package, layer_of=layers, backend_owners={})
    assert result["ok"] is False
    assert {"module": "wrong", "layer": "kern",
            "imports": "high", "target_layer": "pod_arm"} in result["violations"]


def test_downward_and_same_layer_imports_are_fine(tmp_path):
    package = _package(tmp_path, {
        "low": "value = 1\n",
        "peer": "from .low import value\n",
        "high": "from .peer import value\n",
    })
    layers = {"low": "kern", "peer": "kern", "high": "nervensystem"}
    assert architecture.check(package, layer_of=layers,
                              backend_owners={})["ok"] is True


def test_only_the_facade_may_import_a_backend(tmp_path):
    package = _package(tmp_path, {
        "facade": "import lancedb\n",
        "pod": "import lancedb\n",
    })
    layers = {"facade": "storage", "pod": "pod_arm"}
    owners = {"lancedb": {"facade"}}
    result = architecture.check(package, layer_of=layers, backend_owners=owners)
    assert {"module": "pod", "backend": "lancedb",
            "allowed": ["facade"]} in result["violations"]
    assert not any(v.get("module") == "facade" for v in result["violations"])


def test_lancedb_stays_behind_podstorage_in_the_real_tree():
    _runtime, _type_only, backends = architecture.import_graph()
    assert {m for m, imported in backends.items() if "lancedb" in imported} == {"storage"}


def test_type_checking_imports_are_not_runtime_dependencies(tmp_path):
    """`ranking` and `local_search` reference each other only in annotations;
    counting that as a runtime edge would report a cycle that does not exist."""
    package = _package(tmp_path, {
        "a": """
            from typing import TYPE_CHECKING
            if TYPE_CHECKING:
                from .b import Thing
            """,
        "b": "from .a import missing\n",
    })
    layers = {"a": "kern", "b": "kern"}
    result = architecture.check(package, layer_of=layers, backend_owners={})
    assert result["type_only_edges"] == {"a": ["b"]}
    assert not any("import_cycle" in v for v in result["violations"])


def test_a_real_runtime_cycle_is_reported(tmp_path):
    package = _package(tmp_path, {
        "a": "from .b import thing\n",
        "b": "from .a import other\n",
    })
    layers = {"a": "kern", "b": "kern"}
    result = architecture.check(package, layer_of=layers, backend_owners={})
    assert {"import_cycle": ["a", "b"]} in result["violations"]


def test_a_module_listed_but_deleted_is_reported(tmp_path):
    package = _package(tmp_path, {"a": "x = 1\n"})
    result = architecture.check(package, layer_of={"a": "kern", "ghost": "kern"},
                                backend_owners={})
    assert {"module_in_manifest_but_gone": "ghost"} in result["violations"]


@pytest.mark.parametrize("layer", architecture.LAYERS)
def test_no_layer_is_empty(layer):
    assert [m for m, l in architecture.LAYER_OF.items() if l == layer]


# --- the package's public surface ------------------------------------------


def test_everything_in_all_actually_resolves():
    import neural_pods

    missing = [name for name in neural_pods.__all__
               if not hasattr(neural_pods, name)]
    assert missing == []
    assert len(neural_pods.__all__) == len(set(neural_pods.__all__))


def test_re_exports_are_grouped_by_layer():
    """`__init__` lists its imports in layer order, so the public surface can
    be read without executing it. It used to interleave imports with a dozen
    `__all__ +=` statements in whatever order modules were added."""
    import neural_pods

    source = Path(neural_pods.__file__).read_text(encoding="utf-8")
    statements = [line for line in source.splitlines() if line.startswith("__all__")]
    assert statements == ["__all__ = ["]      # one list, no += accretion
    assert _layer_headings(source) == ["1", "2", "4", "5"]


def test_exported_names_come_from_the_layer_they_are_listed_under():
    """A name filed under "layer 2" must really live in a storage module."""
    import neural_pods

    source = Path(neural_pods.__file__).read_text(encoding="utf-8")
    current = None
    claimed: dict[str, str] = {}
    for line in source.splitlines():
        heading = LAYER_HEADING.match(line)
        if heading:
            current = heading.group(1)
        elif line.startswith("from .") and current:
            claimed[line.split()[1].lstrip(".")] = current
    numbers = {name: str(index + 1)
               for index, name in enumerate(architecture.LAYERS)}
    wrong = {module: (claimed_layer, architecture.LAYER_OF.get(module))
             for module, claimed_layer in claimed.items()
             if numbers.get(architecture.LAYER_OF.get(module, ""), "") != claimed_layer}
    assert wrong == {}
