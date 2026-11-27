"""Fit fixed-camera poses and a rigid marker directly from RGB and joints."""

import numpy as np


def fit_rgb_marker(arm, joints, pixels, cameras, heldout=3):
    import cv2
    from scipy.optimize import least_squares

    joints, pixels = np.asarray(joints, float), np.asarray(pixels, float)
    if (
        joints.shape != (len(joints), 5)
        or pixels.shape != (len(joints), 2, 2)
        or len(joints) < 12
    ):
        raise ValueError("Need 12+ paired RGB/joint observations")
    if not np.isfinite(joints).all() or not np.isfinite(pixels[:, 1]).all():
        raise ValueError("Fresh webcam tip coordinates are required at every pose")
    train = len(joints) - heldout
    valid = np.isfinite(pixels).all(axis=2)
    # Auxiliary views with too few samples cannot constrain their own pose.
    if valid[:train, 0].sum() < 6:
        valid[:, 0] = False
    poses, matrices = [], []
    for camera in cameras:
        inverse = np.linalg.inv(np.asarray(camera["T_base_camera"], float))
        poses.extend([*cv2.Rodrigues(inverse[:3, :3])[0].ravel(), *inverse[:3, 3]])
        k = camera["intrinsics"]
        matrices.append(
            np.array([[k["fx"], 0, k["ppx"]], [0, k["fy"], k["ppy"]], [0, 0, 1.0]])
        )
    initial = np.r_[poses, np.zeros(6)]

    # Pan zero is a camera/base yaw gauge; keep it fixed. Fit only the three
    # flex offsets, and one rigid marker offset in the calibrated tool frame.
    def predictions(x, samples):
        offsets = np.r_[0.0, x[12:15], 0.0]
        world = []
        for q in samples:
            frame = arm.frames(q + offsets)[1]
            world.append(frame[:3, 3] + frame[:3, :3] @ x[15:18])
        return np.stack(
            [
                cv2.projectPoints(
                    np.asarray(world), x[j : j + 3], x[j + 3 : j + 6], matrix, None
                )[0].reshape(-1, 2)
                for j, matrix in zip((0, 6), matrices)
            ],
            axis=1,
        )

    def residual(x):
        # Priors limit ambiguities between camera pose, marker offset and joint
        # offsets; a low image residual alone is not a metric accuracy claim.
        delta = predictions(x, joints[:train]) - pixels[:train]
        delta[:, 0] *= 0.25
        return np.r_[
            delta[valid[:train]].ravel(),
            x[12:15] / np.deg2rad(5),
            x[15:18] / 0.05,
            (x[:6] - initial[:6]) * 0.01,
        ]

    low = np.r_[np.full(12, -np.inf), np.full(3, -np.deg2rad(12)), np.full(3, -0.08)]
    high = np.r_[np.full(12, np.inf), np.full(3, np.deg2rad(12)), np.full(3, 0.08)]
    fit = least_squares(
        residual, initial, bounds=(low, high), loss="soft_l1", f_scale=2, max_nfev=250
    )
    errors = np.linalg.norm(predictions(fit.x, joints) - pixels, axis=2)
    errors[~valid] = np.nan

    def serial(values):
        return [
            [None if not np.isfinite(item) else float(item) for item in row]
            for row in values
        ]

    transforms = []
    for j in (0, 6):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(fit.x[j : j + 3])[0]
        pose[:3, 3] = fit.x[j + 3 : j + 6]
        transforms.append(np.linalg.inv(pose).tolist())
    return dict(
        passed=bool(fit.success and errors[train:, 1].max() <= 3),
        train_error_px=serial(errors[:train]),
        heldout_error_px=serial(errors[train:]),
        webcam_heldout_error_px=errors[train:, 1].tolist(),
        auxiliary_samples=int(valid[:, 0].sum()),
        camera_transforms=transforms,
        marker_offset_m=fit.x[15:18].tolist(),
        joint_offset_correction_deg=np.rad2deg(np.r_[0.0, fit.x[12:15], 0.0]).tolist(),
        metric_accuracy_verified=False,
        depth_independent=True,
        note="RGB bootstrap candidate; regularized fit, three held-out poses. Does not prove absolute 3D accuracy.",
    )
