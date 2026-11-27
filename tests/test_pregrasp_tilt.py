import numpy as np

from deskbuddy.can_discard import approach_tilt_limit


def test_pregrasp_allows_observed_26_degree_solution_but_contact_stays_downward():
    contact = 0.12
    assert approach_tilt_limit(contact + 0.02, contact, 25) > 26.6
    assert approach_tilt_limit(contact + 0.04, contact, 25) > 31.7
    assert approach_tilt_limit(contact, contact, 25) == 25
    assert approach_tilt_limit(contact + 0.005, contact, 25) == 25


def test_orientation_tightens_monotonically_and_is_bounded():
    limits = [
        approach_tilt_limit(height, 0.12, 25) for height in np.linspace(0.18, 0.12, 31)
    ]
    assert np.all(np.diff(limits) <= 1e-9)
    assert min(limits) == 25 and max(limits) == 45
