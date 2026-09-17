from neural_pods.resource_runtime import (
    GPUInfo, HardwareSnapshot, ResourceBudget, ResourceGovernor, ResourceBoundHandler,
    ResourceAdmissionError, RequestTracker,
)


def snapshot():
    return HardwareSnapshot(1.0, 16, 8, 1000, 800,
                            (GPUInfo(0, "test", 1000, 900), GPUInfo(1, "test", 1000, 300)),
                            5000)


def test_budget_reserves_capacity_and_choose_falls_back():
    budget = ResourceBudget.from_snapshot(snapshot(), ram_reserve=.1,
                                          vram_reserve_bytes=100, disk_reserve_bytes=1000)
    governor = ResourceGovernor(budget)
    first = governor.choose(700, preferred="vram")
    assert first and first.tier == "vram" and first.device == 0
    second = governor.choose(700, preferred="vram")
    assert second and second.tier == "ram"
    governor.release(first); governor.release(second)
    assert governor.stats()["used"] == {}


def test_unknown_or_unmeasured_capacity_is_not_admitted():
    budget = ResourceBudget(None, {}, None)
    governor = ResourceGovernor(budget)
    assert governor.choose(1, preferred="ram") is None


def test_request_tracker_lifecycle_phases_and_bounded_records(tmp_path):
    tracker = RequestTracker(max_records=2)
    tracker.start("r1", "t1", "pod", "g1")
    tracker.phase("r1", "queue", 1.5)
    receipt = tracker.finish("r1", success=True, input_tokens=3, output_tokens=5, cache_hit=True)
    assert receipt.success is True and receipt.phases_ms["queue"] == 1.5
    tracker.start("r2", "t2", "pod", "g1"); tracker.finish("r2", success=False, error="x")
    tracker.start("r3", "t3", "pod", "g1"); tracker.finish("r3", success=True)
    stats = tracker.stats()
    assert stats["completed"] == 2 and stats["success"] == 1 and stats["errors"] == 1
    tracker.export_jsonl(tmp_path / "trace.jsonl")
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 2


def test_request_tracker_rejects_duplicate_active_request():
    tracker = RequestTracker()
    tracker.start("r", "t", "pod")
    try:
        tracker.start("r", "t2", "pod")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate active request was accepted")


def test_resource_bound_handler_enforces_and_releases_lease():
    governor = ResourceGovernor(ResourceBudget(ram_bytes=10, vram_bytes={}, disk_bytes=None))
    calls = []
    handler = ResourceBoundHandler(lambda payload, request: calls.append(payload) or payload,
                                   governor, amount_bytes=6, preferred="ram")
    assert handler("ok", None) == "ok"
    assert governor.stats()["used"] == {}
    assert handler.stats() == {"admitted": 1, "rejected": 0}
    blocker = governor.try_acquire("ram", 6)
    assert blocker is not None
    try:
        try:
            handler("no", None)
        except ResourceAdmissionError:
            pass
        else:
            raise AssertionError("over-budget request was admitted")
    finally:
        governor.release(blocker)
    assert calls == ["ok"]


def test_residency_is_generation_bound_and_eviction_releases_capacity():
    governor = ResourceGovernor(ResourceBudget(ram_bytes=100, vram_bytes={}, disk_bytes=None))
    resident = governor.activate("pod-1", "g1", 60, preferred="ram")
    assert resident and resident.generation == "g1"
    assert governor.activate("pod-1", "g1", 60) == resident
    try:
        governor.activate("pod-1", "g2", 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("generation switch bypassed residency barrier")
    try:
        governor.deactivate("pod-1", "g2")
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale generation evicted active residency")
    assert governor.deactivate("pod-1", "g1") is True
    assert governor.stats()["residencies"] == {} and governor.stats()["used"] == {}
