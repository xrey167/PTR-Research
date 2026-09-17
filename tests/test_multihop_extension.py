from research.prepare_multihop_extension import rows


def test_three_hop_extension_is_balanced_and_held_out():
    train, dev, test = rows("train"), rows("dev"), rows("test")
    assert len(train) == 8 and len(dev) == len(test) == 4
    assert {r["assessment"]["hop_count"] for r in train + dev + test} == {3}
    assert not ({r["assessment"]["supplier"] for r in train + dev}
                & {r["assessment"]["supplier"] for r in test})
    assert all(r["pod_type"] == "reasoning" for r in test)
