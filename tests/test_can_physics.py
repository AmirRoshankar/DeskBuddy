import pytest


def test_physics_rollout_includes_gravity_contacts_and_rejects_missed_grasp():
    mujoco = pytest.importorskip("mujoco")
    from deskbuddy.learned.can_physics import rollout, scene_xml

    model = mujoco.MjModel.from_xml_string(scene_xml(0.066, 0.12222))
    assert model.opt.gravity[2] == pytest.approx(-9.81)
    assert model.nq > 10 and model.ngeom > 25
    result = rollout(0.066, 0.12222, [0.065, 0, 0.12022], [0, 1, 0, 0], 0.008)
    assert not result["success"] and not result["pickup"]
    assert result["contacts_max"] > 0
    assert abs(result["final_base_m"][2]) < 0.001
