"""The layer model of ARCHITECTURE-MASTER, as data and as a check.

Section 1 of the master document draws a five-layer graph and states the rule
in one line — *"eine Schicht spricht nur die Schicht darunter an"* — and
section 1 adds a second: *"Pods sprechen Storage nur über die
PodStorage-Fassade, nie Backends direkt"*. Both were prose, so nothing
noticed when a module drifted, and the whole point of this codebase is that
claims should be checkable.

This module turns them into a property:

  * `LAYER_OF` places every module in `neural_pods/` on one of the five
    layers. A module may import its own layer and any layer below it; an
    import that points upward is a violation.
  * `BACKEND_OWNERS` names the only modules allowed to import each storage
    or transport backend. Everything else must go through the facade.
  * Imports inside `if TYPE_CHECKING:` are not runtime dependencies and are
    reported separately, so an annotation cannot look like a cycle
    (`ranking` and `local_search` reference each other exactly that way).

`check()` is pure static analysis over the source tree — no imports are
executed — so both the test suite and the gate can run it. A module that is
added without being placed is itself a violation: the placement decision is
made once, deliberately, rather than discovered later.
"""
from __future__ import annotations
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent

#: Bottom to top. A module may depend on its own layer and everything left.
LAYERS: tuple[str, ...] = ("kern", "storage", "haushalt", "nervensystem", "pod_arm")

LAYER_OF: dict[str, str] = {
    # --- 1 KERN: registry/provenance, types, contracts, models, consensus --
    "registry": "kern", "pod_types": "kern", "pod_contract": "kern",
    "pod_builder": "kern", "pod_taxonomy": "kern", "pod_profiles": "kern",
    "execution_manifest": "kern", "model": "kern", "model_pod": "kern",
    "pod_executor": "kern", "data": "kern", "semantics": "kern",
    "lifecycle_transport": "kern", "resource_runtime": "kern",
    "fault_injection": "kern", "raft_backend": "kern", "raft_transport": "kern",
    "placement": "kern", "replication": "kern", "vllm_router": "kern",
    "architecture": "kern",
    # --- 2 STORAGE: the four tiers and everything that indexes into them ---
    "storage": "storage", "pod_cache": "storage", "snapshot_store": "storage",
    "vector_index": "storage", "block_postings": "storage",
    "local_search": "storage", "ranking": "storage",
    "postgres_store": "storage", "registry_lookup_kb": "storage",
    "alias_resolver": "storage", "contextual_retrieval": "storage",
    "spacy_enrichment": "storage",
    # --- 3 HAUSHALT: the donor swarm contract ------------------------------
    "household": "haushalt",
    # --- 4 NERVENSYSTEM: mesh, native protocol, task graph, perception -----
    "mesh": "nervensystem", "mesh_cache": "nervensystem",
    "native_comm": "nervensystem", "taskgraph": "nervensystem",
    "perception": "nervensystem", "pod_protocol": "nervensystem",
    "pod_socket": "nervensystem", "pod_streams": "nervensystem",
    "adaptive_batcher": "nervensystem",
    # --- 5 POD-ARM / DREAM: reflex, dreaming, routing, agentic search ------
    "reflex": "pod_arm", "dream": "pod_arm", "symlink": "pod_arm",
    "dragonfly": "pod_arm", "routing": "pod_arm",
    "semantic_routing": "pod_arm", "routing_harness": "pod_arm",
    "search_agent": "pod_arm", "recursive_search": "pod_arm",
    "recursive_memory": "pod_arm", "ngu_sampling": "pod_arm",
    "continuous_concepts": "pod_arm", "research_tools": "pod_arm",
}

#: Backend -> the only modules allowed to import it directly. Everything else
#: reaches these through PodStorage / MeshEndpoint.
BACKEND_OWNERS: dict[str, set[str]] = {
    "lancedb": {"storage"},
    "paho": {"mesh"},          # mesh_cache imported it for show only
    "psycopg": {"postgres_store", "replication"},
    "qdrant_client": {"routing", "semantic_routing"},
}


def _is_type_checking(node: ast.AST) -> bool:
    test = getattr(node, "test", None)
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _walk_runtime(tree: ast.AST):
    """Every node except the bodies of `if TYPE_CHECKING:` blocks."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child):
                continue
            stack.append(child)


def modules(package: Path = PACKAGE) -> set[str]:
    return {p.stem for p in package.glob("*.py")} - {"__init__"}


def _sibling_targets(node: ast.ImportFrom, known: set[str]) -> set[str]:
    found: set[str] = set()
    if node.level:                                   # from . / from .x
        if node.module:
            root = node.module.split(".")[0]
            if root in known:
                found.add(root)
        else:
            found |= {a.name for a in node.names if a.name in known}
    elif (node.module or "").startswith("neural_pods"):
        parts = node.module.split(".")
        if len(parts) > 1 and parts[1] in known:
            found.add(parts[1])
        found |= {a.name for a in node.names if a.name in known}
    return found


def import_graph(package: Path = PACKAGE, *,
                 backend_owners: dict[str, set[str]] | None = None
                 ) -> tuple[dict[str, set[str]], dict[str, set[str]],
                            dict[str, set[str]]]:
    """(runtime sibling imports, type-only sibling imports, backend imports)."""
    backend_owners = BACKEND_OWNERS if backend_owners is None else backend_owners
    known = modules(package)
    runtime: dict[str, set[str]] = {}
    type_only: dict[str, set[str]] = {}
    backends: dict[str, set[str]] = {}
    for path in sorted(package.glob("*.py")):
        if path.stem == "__init__":
            continue
        tree = ast.parse(path.read_bytes().decode("utf-8-sig"))
        runtime_nodes = set(map(id, _walk_runtime(tree)))
        for node in ast.walk(tree):
            at_runtime = id(node) in runtime_nodes
            if isinstance(node, ast.ImportFrom):
                targets = _sibling_targets(node, known)
                (runtime if at_runtime else type_only).setdefault(
                    path.stem, set()).update(targets)
                root = (node.module or "").split(".")[0]
                if at_runtime and root in backend_owners:
                    backends.setdefault(path.stem, set()).add(root)
            elif isinstance(node, ast.Import) and at_runtime:
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in backend_owners:
                        backends.setdefault(path.stem, set()).add(root)
    return runtime, type_only, backends


def check(package: Path = PACKAGE, *,
          layer_of: dict[str, str] | None = None,
          backend_owners: dict[str, set[str]] | None = None) -> dict:
    """Every way the tree can disagree with the documented architecture.

    The manifests are parameters so the checker itself can be tested against
    a synthetic package: a rule nobody has seen fail is not a rule.
    """
    layer_of = LAYER_OF if layer_of is None else layer_of
    backend_owners = BACKEND_OWNERS if backend_owners is None else backend_owners
    known = modules(package)
    runtime, type_only, backends = import_graph(package,
                                                backend_owners=backend_owners)
    rank = {name: index for index, name in enumerate(LAYERS)}

    unplaced = sorted(known - set(layer_of))
    stale = sorted(set(layer_of) - known)
    unknown_layers = sorted({module for module, layer in layer_of.items()
                             if layer not in rank})

    upward = []
    for module, targets in sorted(runtime.items()):
        if module not in layer_of:
            continue
        for target in sorted(targets):
            if target not in layer_of:
                continue
            if rank[layer_of[target]] > rank[layer_of[module]]:
                upward.append({"module": module, "layer": layer_of[module],
                               "imports": target, "target_layer": layer_of[target]})

    trespass = []
    for module, imported in sorted(backends.items()):
        for backend in sorted(imported):
            if module not in backend_owners[backend]:
                trespass.append({"module": module, "backend": backend,
                                 "allowed": sorted(backend_owners[backend])})

    cycles = sorted({tuple(sorted((module, target)))
                     for module, targets in runtime.items()
                     for target in targets
                     if module in runtime.get(target, set())})

    violations = (upward + trespass
                  + [{"unplaced_module": m} for m in unplaced]
                  + [{"module_in_manifest_but_gone": m} for m in stale]
                  + [{"unknown_layer_for": m} for m in unknown_layers]
                  + [{"import_cycle": list(c)} for c in cycles])
    return {
        "ok": not violations,
        "modules": len(known),
        "layers": {layer: sorted(m for m, l in layer_of.items() if l == layer)
                   for layer in LAYERS},
        "runtime_edges": sum(len(t) for t in runtime.values()),
        "type_only_edges": {m: sorted(t) for m, t in sorted(type_only.items()) if t},
        "violations": violations,
    }


if __name__ == "__main__":
    import json
    import sys
    result = check()
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))
    sys.exit(0 if result["ok"] else 1)
