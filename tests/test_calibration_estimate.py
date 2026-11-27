from types import SimpleNamespace

from deskbuddy.calibration_flow import CalibrationFlow


def test_high_residual_can_continue_without_falsifying_validation():
    flow = CalibrationFlow.__new__(CalibrationFlow)
    calls = []
    flow.bar = SimpleNamespace(step=lambda *args: calls.append(args))
    flow.begin_paper_webcam = lambda: calls.append("paper")
    fit = dict(
        passed=False, fit_rmse_px=11.6, heldout_error_px=[14.0, 15.0], coverage_ok=True
    )
    flow.webcam_assessment(fit)
    assert calls[0][1] == "Use estimated fit"
    assert "not whether your clicks were correct" in calls[0][0]
    calls[0][2]()
    assert flow.webcam_fit["operator_accepted_estimate"]
    assert not flow.webcam_fit["passed"]
    assert not fit.get("operator_accepted_estimate")
    assert calls[-1] == "paper"
