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
    assert household.approve(request.request_id, "donor-1") == "approved"
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
    household.approve(request.request_id, "donor-1")
    events = [e for e in household.registry.events()
              if e["action"] in ("allocation_request", "allocation_approved")]
    kinds = [e["action"] for e in events]
    assert "allocation_request" in kinds and "allocation_approved" in kinds
