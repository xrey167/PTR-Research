import pytest

from neural_pods.household import Household, Segment
from neural_pods.registry import Registry


@pytest.fixture()
def household():
    return Household(key="household-secret", registry=Registry(":memory:"))


def test_join_and_members(household):
    household.join("pod-1", "household-secret", ram_bytes=8_000_000_000,
                   vram_bytes=24_000_000_000)
    household.join("pod-2", "household-secret", ram_bytes=16_000_000_000,
                   vram_bytes=0)
    members = household.members()
    assert [m["pod_id"] for m in members] == ["pod-1", "pod-2"]
    with pytest.raises(PermissionError, match="key mismatch"):
        household.join("pod-3", "wrong-key", ram_bytes=1, vram_bytes=0)


def test_allocation_requires_full_approval_and_respects_busy(household):
    # 5 GiB VRAM: the first segment fills it, the second overflows to RAM —
    # exercising the vram->ram tier preference chain.
    household.join("donor-1", "household-secret", ram_bytes=16_000_000_000,
                   vram_bytes=5_000_000_000)
    segments = [Segment("reader-adapter", 4_000_000_000),
                Segment("raft-segment", 2_000_000_000)]
    request = household.request_allocation("reader-gen7", segments, ["donor-1"])
    assert request.status == "pending"
    with pytest.raises(PermissionError, match="not fully approved"):
        household.start(request.request_id)
    assert household.approve(request.request_id, "donor-1", "household-secret") == "approved"
    started = household.start(request.request_id)
    tiers = {p["segment"]: p["tier"] for p in started["plan"]}
    assert tiers["reader-adapter"] == "vram"   # fits VRAM, preferred
    assert tiers["raft-segment"] == "ram"      # overflow goes down the hierarchy
    # donor is BUSY now — a second request is refused, not queued silently
    second = household.request_allocation("other-model", segments, ["donor-1"])
    assert second.status == "refused" and second.request_id == "BUSY"


def test_unknown_donor_rejected(household):
    household.join("donor-1", "household-secret", ram_bytes=1, vram_bytes=0)
    with pytest.raises(KeyError, match="unknown donor"):
        household.request_allocation("m", [Segment("s", 1)], ["ghost"])


def test_calibration_records_indication(household):
    household.join("donor-1", "household-secret", ram_bytes=8, vram_bytes=0)
    cal = household.calibrate("donor-1", "reader-gen7", tokens_per_s=12.34)
    assert cal["tokens"] == 8 and cal["tokens_per_s"] == 12.34
    members = household.members()
    assert members[0]["calibrations"]["reader-gen7@donor-1"]["tokens_per_s"] == 12.34


def test_token_cache_session_affinity(household):
    household.join("donor-1", "household-secret", ram_bytes=8, vram_bytes=0)
    household.token_cache_store("session-1", "prefix-abc", 512, "replica-0")
    entry = household.token_cache_lookup("session-1", "prefix-abc")
    assert entry.replica == "replica-0" and entry.token_count == 512
    assert household.saved_tokens == 512  # reuse means saved generation work
    assert household.token_cache_lookup("session-2", "prefix-abc") is None


def test_approvals_are_provenance_events(household):
    household.join("donor-1", "household-secret", ram_bytes=8, vram_bytes=0)
    request = household.request_allocation("m", [Segment("s", 1)], ["donor-1"])
    household.approve(request.request_id, "donor-1", "household-secret")
    events = [e for e in household.registry.events()
              if e["action"] in ("allocation_request", "allocation_approved")]
    kinds = [e["action"] for e in events]
    assert "allocation_request" in kinds and "allocation_approved" in kinds


# --- regression tests for the H2 defects found in the 2026-09-20 review ----


def _started(household, donors, segments, model_ref="m"):
    request = household.request_allocation(model_ref, segments, donors)
    for pod_id in donors:
        household.approve(request.request_id, pod_id, "household-secret")
    household.start(request.request_id)
    return request


def test_release_frees_the_donor_for_the_next_allocation(household):
    """Nothing used to clear `busy`: a donor served one allocation and then
    refused every later one for the lifetime of the process."""
    household.join("donor-1", "household-secret", ram_bytes=10, vram_bytes=0)
    first = _started(household, ["donor-1"], [Segment("s", 1)])
    assert household.stats()["busy_pods"] == 1

    released = household.release(first.request_id, "household-secret")
    assert released["released"] == ["donor-1"]
    assert household.stats()["busy_pods"] == 0 and household.stats()["reservations"] == 0

    second = household.request_allocation("other", [Segment("s", 1)], ["donor-1"])
    assert second.status == "pending"          # not refused any more


def test_release_requires_the_household_key_and_a_started_request(household):
    household.join("donor-1", "household-secret", ram_bytes=10, vram_bytes=0)
    request = _started(household, ["donor-1"], [Segment("s", 1)])
    with pytest.raises(PermissionError, match="key mismatch"):
        household.release(request.request_id, "wrong-key")
    household.release(request.request_id, "household-secret")
    with pytest.raises(PermissionError, match="not started"):
        household.release(request.request_id, "household-secret")
    with pytest.raises(KeyError, match="unknown request"):
        household.release("nope", "household-secret")


def test_approval_requires_the_household_key(household):
    """An approval commits a donor's memory, so a request id and a pod id
    must not be enough to give it."""
    household.join("donor-1", "household-secret", ram_bytes=10, vram_bytes=0)
    request = household.request_allocation("m", [Segment("s", 1)], ["donor-1"])
    with pytest.raises(PermissionError, match="key mismatch"):
        household.approve(request.request_id, "donor-1", "wrong-key")
    assert request.status == "pending"
    assert household.approve(request.request_id, "donor-1", "household-secret") == "approved"


def test_vram_and_ram_budgets_are_counted_separately(household):
    """One shared counter meant a segment placed in VRAM also consumed the
    donor's RAM budget."""
    household.join("donor-1", "household-secret", ram_bytes=100, vram_bytes=100)
    request = _started(household, ["donor-1"],
                       [Segment("a", 100), Segment("b", 100)])
    plan = {entry["segment"]: entry["tier"]
            for entry in household.reservations[request.request_id]["plan"]}
    assert plan == {"a": "vram", "b": "ram"}   # VRAM full, RAM still untouched


def test_tier_preference_is_honoured_strictly(household):
    """A vram-only segment used to fall through into ram regardless."""
    household.join("donor-1", "household-secret", ram_bytes=1000, vram_bytes=1)
    request = household.request_allocation(
        "m", [Segment("vram-only", 100, tier_preference=("vram",))], ["donor-1"])
    household.approve(request.request_id, "donor-1", "household-secret")
    with pytest.raises(RuntimeError, match="no donor fits"):
        household.start(request.request_id)


def test_start_release_and_refusal_are_provenance_events(household):
    household.join("donor-1", "household-secret", ram_bytes=10, vram_bytes=0)
    request = _started(household, ["donor-1"], [Segment("s", 1)])
    household.request_allocation("second", [Segment("s", 1)], ["donor-1"])  # BUSY
    household.release(request.request_id, "household-secret")

    actions = [e["action"] for e in household.registry.events()]
    for expected in ("allocation_request", "allocation_approved",
                     "allocation_started", "allocation_refused",
                     "allocation_released"):
        assert expected in actions, expected
    started = next(e for e in household.registry.events()
                   if e["action"] == "allocation_started")
    assert started["payload"]["plan"][0]["pod"] == "donor-1"


def test_state_is_reconstructible_from_the_event_log(household):
    """The zero-cost rule: a restart loses nothing. Offers come back via
    join(), the reservations come back from the log."""
    household.join("donor-1", "household-secret", ram_bytes=10, vram_bytes=0)
    household.join("donor-2", "household-secret", ram_bytes=10, vram_bytes=0)
    kept = _started(household, ["donor-1"], [Segment("s", 1)], model_ref="kept")
    gone = _started(household, ["donor-2"], [Segment("s", 1)], model_ref="gone")
    household.release(gone.request_id, "household-secret")

    restarted = Household(key="household-secret", registry=household.registry)
    restarted.join("donor-1", "household-secret", ram_bytes=10, vram_bytes=0)
    restarted.join("donor-2", "household-secret", ram_bytes=10, vram_bytes=0)
    restored = restarted.restore_from_events()

    assert restored["restored_requests"] == [kept.request_id]
    assert restored["busy_pods"] == ["donor-1"]
    assert restarted.offers["donor-1"].busy_with == kept.request_id
    assert restarted.offers["donor-2"].busy is False


def test_restored_allocation_can_still_be_released(household):
    """restore_from_events() rebuilt the reservations but not the requests, so
    release() raised KeyError and the restored donors stayed BUSY forever —
    exactly the defect release() exists to prevent."""
    household.join("donor-1", "household-secret", ram_bytes=10, vram_bytes=0)
    request = _started(household, ["donor-1"], [Segment("s", 1)], model_ref="kept")

    restarted = Household(key="household-secret", registry=household.registry)
    restarted.join("donor-1", "household-secret", ram_bytes=10, vram_bytes=0)
    restarted.restore_from_events()

    assert restarted.requests[request.request_id].model_ref == "kept"
    assert restarted.requests[request.request_id].donors == ("donor-1",)
    assert restarted.release(request.request_id, "household-secret")["released"] == ["donor-1"]
    assert restarted.offers["donor-1"].busy is False
    assert restarted.request_allocation("next", [Segment("s", 1)], ["donor-1"]).status == "pending"


def test_concurrent_start_places_the_request_once(household):
    """The status check used to sit outside the lock: two threads both saw
    "approved", both placed, and the second overwrote the first reservation."""
    import threading

    household.join("donor-1", "household-secret", ram_bytes=1000, vram_bytes=0)
    request = household.request_allocation("m", [Segment("s", 1)], ["donor-1"])
    household.approve(request.request_id, "donor-1", "household-secret")

    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def start():
        barrier.wait()
        try:
            household.start(request.request_id)
            outcomes.append("started")
        except PermissionError:
            outcomes.append("refused")

    threads = [threading.Thread(target=start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["refused", "started"]
    assert len(household.reservations) == 1
    household.release(request.request_id, "household-secret")
    assert household.stats()["busy_pods"] == 0
