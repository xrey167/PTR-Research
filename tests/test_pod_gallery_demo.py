from research.demo_pod_gallery import run


def test_pod_gallery_covers_types_questions_and_revocation():
    result = run()
    assert set(result["pod_types"]) == {"context", "math", "model"}
    assert len(result["questions"]) == 6
    assert {row["selected_pod_type"] for row in result["questions"]} == {"context", "math", "model"}
    assert all(row["active_delta"] > 0 for row in result["questions"])
    assert all(row["revocation_restores_baseline"] for row in result["questions"])
