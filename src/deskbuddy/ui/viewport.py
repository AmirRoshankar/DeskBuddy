"""Aspect-preserving camera rendering and reversible pixel coordinates."""

from dataclasses import dataclass

from PIL import Image

from .theme import COLORS


@dataclass(frozen=True)
class CameraViewport:
    width: int
    height: int
    source_width: int = 640
    source_height: int = 480

    @property
    def geometry(self):
        scale = min(self.width / self.source_width, self.height / self.source_height)
        w = max(1, round(self.source_width * scale))
        h = max(1, round(self.source_height * scale))
        return w, h, (self.width - w) // 2, (self.height - h) // 2

    def to_camera(self, x, y):
        w, h, left, top = self.geometry
        if not left <= x < left + w or not top <= y < top + h:
            return None
        return int((x - left) * self.source_width / w), int(
            (y - top) * self.source_height / h
        )

    def from_camera(self, u, v):
        w, h, left, top = self.geometry
        return left + u * w / self.source_width, top + v * h / self.source_height

    def render(self, image):
        w, h, left, top = self.geometry
        frame = Image.new("RGB", (self.width, self.height), COLORS.surface)
        frame.paste(image.resize((w, h), Image.Resampling.BILINEAR), (left, top))
        return frame
