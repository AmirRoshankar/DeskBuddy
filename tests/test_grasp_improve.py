import numpy as np
import pytest

from deskbuddy.learned import grasp_improve as gi


def test_split_holds_out_whole_cans():
    ids = np.arange(100)
    parts = gi.split(ids)
    assert parts["test"].sum() == 10 and parts["val"].sum() == 10
    assert not (parts["test"] & parts["train"]).any()


def test_wilson_interval_and_auc():
    low, high = gi.wilson(6, 246)
    assert 0.0 < low < 6 / 246 < high < 0.06
    assert gi.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert gi.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert gi.auc([1, 2], [1, 1]) is None


def test_scorer_learns_a_separable_success_rule():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4000, 5)).astype(np.float32)
    grasps = rng.normal(size=(4000, 8)).astype(np.float32)
    labels = grasps[:, 2] > 1.5
    scorer = gi.Scorer(32).fit(contexts, grasps, labels, steps=400, batch=256)
    assert gi.auc(scorer.score(contexts, grasps), labels) > 0.95
    # Broadcast scoring of K draws per context.
    assert scorer.score(contexts[None, :3], np.stack([grasps[:3]] * 4)).shape == (4, 3)


def test_body_baseline_rollout_runs_in_the_toy_physics():
    pytest.importorskip("mujoco")
    from deskbuddy.learned.can_physics import body_rollout

    result = body_rollout(0.066, 0.12222, 0.0, mass=0.02, friction=0.8)
    assert result["grasp"] == "numerical_body_baseline"
    assert set(("pickup", "success")) <= set(result)


def test_legacy_body_radius_outputs_are_moved_onto_the_lid_like_live():
    contexts = np.array([[0.066 / 0.05, 0.066 / 0.05, 2.44, 1.0, 0.0]])
    grasps = np.zeros((1, 8))
    grasps[0, :3] = [0.033 / 0.05, 0.0, 2.4]
    fixed = gi.to_lid(grasps, contexts)
    assert fixed[0, 0] * 0.05 == pytest.approx(gi.LID / 2)
    assert fixed[0, 2] == grasps[0, 2]


def test_generated_rows_sample_around_the_lid_seam():
    rows = gi.generate_rows(5, 4, seed=1)
    radial = np.linalg.norm(rows["targets"][:, :2] * 0.05, axis=1)
    assert np.all(np.abs(radial - gi.LID / 2) <= 0.0031)
    assert rows["contexts"].shape == (20, 5) and rows["params"].shape == (20, 3)
