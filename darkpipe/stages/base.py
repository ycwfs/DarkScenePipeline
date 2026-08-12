"""Stage interfaces for the composable pipeline."""
from dataclasses import dataclass, field


class FrameStage:
    """Frame-in/frame-out stage (enhancement, super-resolution). BGR uint8 lists."""
    name: str = "stage"
    whole_video: bool = False  # True: stage must see the entire frame list at once
    # True: run AFTER recognition, i.e. this stage only changes the picture people look at.
    # Recognition takes the enhanced, pre-SR frame, and the behaviour head was trained on
    # exactly that; a stage that alters colour or scale before it would shift the input
    # distribution away from training for no gain, since recognition resizes to 224 and
    # normalises anyway. Declared as a flag rather than inferred from the stage name --
    # the name-prefix version of this split silently dropped a newly added stage once.
    post_recognition: bool = False

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
    every `stride` frames once the window is full.

    `labels` is per-instance, not global: the ARID recognizers speak CLASSES while the
    behavior recognizers speak BEHAVIORS (and the open-vocabulary one accepts any list)."""
    name: str = "recognizer"
    window: int = 16
    stride: int = 8
    labels: list = []

    def load(self, device: str) -> None:
        raise NotImplementedError

    def push(self, frame_bgr, frame_index: int, timestamp: float):
        raise NotImplementedError

    def reset(self) -> None:
        """Forget the current stream (buffer + stride phase), keeping the loaded weights.

        For processing several independent clips through one loaded recognizer: without it
        the tail of clip N leaks into the first window of clip N+1."""
        pass

    def close(self) -> None:
        pass
