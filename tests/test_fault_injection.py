from neural_pods.fault_injection import FailAction, FailureInjector, InjectedFailure


def test_failpoint_is_deterministic_and_counted():
    f = FailureInjector({"wal.commit": FailAction(every=2)})
    f.hit("wal.commit")
    try:
        f.hit("wal.commit")
    except InjectedFailure as exc:
        assert "count=2" in str(exc)
    else:
        raise AssertionError("second hit must fail")
    assert f.counts() == {"wal.commit": 2}


def test_delay_failpoint_does_not_raise():
    f = FailureInjector({"transport": FailAction(kind="delay", delay_s=0.0001)})
    f.hit("transport")
    assert f.counts()["transport"] == 1
