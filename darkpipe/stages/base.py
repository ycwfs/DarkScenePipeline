"""Stage interfaces for the composable pipeline."""
from dataclasses import dataclass, field


class FrameStage:
    """Frame-in/frame-out stage (enhancement, super-resolution). BGR uint8 lists."""
    name: str = "stage"
    whole_video: bool = False  # True: stage must see the entire frame list at once

    def load(self, device: str) -> None:
        raise NotImplementedError

    def __call__(self, frames: list) -> list:
        raise NotImplementedError

    def close(self) -> None:
        pass


@dataclass
class RecognitionEvent:
    frame_index: int
    timestamp: float
    label: str
    confidence: float
    topk: list = field(default_factory=list)  # [(label, prob), ...] top-3
    model: str = ""
    window: int = 0

    def to_dict(self):
        return dict(frame_index=self.frame_index, timestamp=round(self.timestamp, 3),
                    label=self.label, confidence=round(self.confidence, 4),
                    topk=[[l, round(p, 4)] for l, p in self.topk],
                    model=self.model, window=self.window)


class Recognizer:
    """Sliding-window action recognizer. push() every processed frame; returns an event
    every `stride` frames once the window is full."""
    name: str = "recognizer"
    window: int = 16
    stride: int = 8

    def load(self, device: str) -> None:
        raise NotImplementedError

    def push(self, frame_bgr, frame_index: int, timestamp: float):
        raise NotImplementedError

    def close(self) -> None:
        pass
