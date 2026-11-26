"""Optional RealSense adapter; importing the offline demo never needs the SDK."""

import numpy as np

from .geometry import median_depth


class RealSense:
    def __init__(self, serial=None):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = self.pipeline.start(config)
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

    def read(self):
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=1000))
        depth, color = frames.get_depth_frame(), frames.get_color_frame()
        if not color:
            raise ValueError("Missing color frame")
        intrinsics = (
            (depth if depth else color).profile.as_video_stream_profile().intrinsics
        )
        color_array = np.asanyarray(color.get_data()).copy()
        return (
            color_array,
            np.asanyarray(depth.get_data()).astype(float) * self.scale
            if depth
            else np.zeros(color_array.shape[:2]),
            intrinsics,
        )

    def point(self, depth, intrinsics, u, v):
        distance = median_depth(depth, u, v)
        return self.deproject(intrinsics, u, v, distance)

    def deproject(self, intrinsics, u, v, distance):
        return np.asarray(
            self.rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], distance)
        )

    def close(self):
        self.pipeline.stop()
