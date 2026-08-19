"""What comes out of VideoWriter has to be a file the customer can actually open.

Every mp4 this project delivers -- behaviour clips, the offline whole-video output, the
recorded stream -- goes through this one class, and for a long time it wrote MPEG-4 Part 2:
`cv2.VideoWriter` was asked for "mp4v" first and that always succeeds, so the "avc1" branch
behind it was dead code, and this OpenCV build has no H.264 encoder to fall back to anyway.
Nothing failed. ffplay and VLC played the results on the dev box. The clips only turned out
to be unopenable once they reached a browser and 微信 on the far side of a delivery.

The lesson is that "the writer returned without raising" proves nothing, so these tests read
the finished files back and assert on what the container says. The first test is the one that
matters; the rest pin the ways a raw pipe can go wrong that a cv2 handle could not.
"""
import os
import shutil
import struct
import subprocess

import cv2
import numpy as np
import pytest

from darkpipe.media import VideoWriter

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="no ffmpeg binary; the writer falls back to cv2")


def fourcc(path):
    """The four-char code the container declares, e.g. 'h264' vs the old 'FMP4'."""
    cap = cv2.VideoCapture(path)
    try:
        assert cap.isOpened(), f"cannot even open {path}"
        return struct.pack("<i", int(cap.get(cv2.CAP_PROP_FOURCC))).decode("ascii", "replace")
    finally:
        cap.release()


def decode(path):
    """-> (frames actually decoded, declared fps). Reads the file the way a player would."""
    cap = cv2.VideoCapture(path)
    try:
        assert cap.isOpened(), f"cannot open {path}"
        fps, n = cap.get(cv2.CAP_PROP_FPS), 0
        while cap.read()[0]:
            n += 1
        return n, fps
    finally:
        cap.release()


def frames(n, w=64, h=48):
    """Moving content -- a constant image compresses to nothing and hides frame loss."""
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, :, i % 3] = 40 + (i * 7) % 200
        f[(i * 3) % h: ((i * 3) % h) + 4, :] = 255
        out.append(f)
    return out


def write(path, fps, imgs):
    w = VideoWriter(str(path), fps)
    for f in imgs:
        w.write(f)
    w.close()
    return w


def test_the_output_is_h264_not_mpeg4(tmp_path):
    """The whole bug in one assertion: 'FMP4'/'mp4v' here means undeliverable files."""
    p = tmp_path / "clip.mp4"
    write(p, 25.0, frames(30))
    assert fourcc(str(p)).lower() in ("h264", "avc1"), \
        "writer produced a codec browsers and 微信 refuse to play"


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="no ffprobe")
def test_the_pixel_format_is_the_universally_playable_one(tmp_path):
    """yuv444/yuvj420 decode here and fail on hardware players; be boring on purpose."""
    p = tmp_path / "clip.mp4"
    write(p, 25.0, frames(20))
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    assert out == "yuv420p"


def test_every_frame_written_survives_to_the_file(tmp_path):
    """`count` is reported as frames_written in the metadata; it must not outrun the file."""
    p = tmp_path / "clip.mp4"
    w = write(p, 20.0, frames(57))
    n, fps = decode(str(p))
    assert w.count == 57
    assert n == 57, f"wrote 57 frames, {n} came back"
    assert abs(fps - 20.0) < 0.01


def test_odd_dimensions_still_produce_a_playable_file(tmp_path):
    """yuv420p halves chroma in both axes, so an odd width/height cannot be encoded as-is.
    SR and the label bar are perfectly capable of producing one."""
    p = tmp_path / "odd.mp4"
    write(p, 15.0, frames(10, w=65, h=49))
    n, _ = decode(str(p))
    assert n == 10
    assert fourcc(str(p)).lower() in ("h264", "avc1")


def test_the_file_is_finished_when_close_returns(tmp_path):
    """clips.py moves the mp4 into place on the next line after close(). If close() returned
    while ffmpeg was still writing the moov atom, the delivered clip would have no index."""
    p, moved = tmp_path / "stage.mp4", tmp_path / "final.mp4"
    write(p, 24.0, frames(40))
    shutil.move(str(p), str(moved))
    assert decode(str(moved))[0] == 40


def test_a_writer_that_never_saw_a_frame_closes_quietly(tmp_path):
    """Clips are abandoned before the first frame often enough that this is a real path."""
    p = tmp_path / "empty.mp4"
    w = VideoWriter(str(p), 25.0)
    w.close()
    assert w.count == 0
    assert not os.path.exists(p), "an empty file would be delivered as a broken clip"


def test_closing_twice_is_harmless(tmp_path):
    """The clip recorder's error path closes the writer, then the shutdown path closes it."""
    p = tmp_path / "twice.mp4"
    w = write(p, 25.0, frames(5))
    w.close()
    assert decode(str(p))[0] == 5


def test_a_frame_of_the_wrong_size_does_not_corrupt_the_rest(tmp_path):
    """A raw pipe has no framing: a short frame shifts every byte after it and the tail of
    the video decodes as garbage. cv2 used to drop such frames silently instead."""
    p = tmp_path / "mixed.mp4"
    w = VideoWriter(str(p), 25.0)
    for f in frames(10):
        w.write(f)
    w.write(np.zeros((30, 40, 3), np.uint8))       # different size, mid-stream
    for f in frames(10):
        w.write(f)
    w.close()
    n, _ = decode(str(p))
    assert w.count == 21
    assert n == 21, f"stream desynced: {n} of 21 frames decoded"


def test_a_frame_of_the_wrong_dtype_is_refused_loudly(tmp_path):
    """float32 frames are 4x the bytes; silently piping them produces an hour of noise."""
    p = tmp_path / "bad.mp4"
    w = VideoWriter(str(p), 25.0)
    w.write(frames(1)[0])
    with pytest.raises(RuntimeError, match="bgr24"):
        w.write(np.zeros((48, 64, 3), np.float32))
    w.close()


def test_it_is_smaller_than_the_mpeg4_it_replaces(tmp_path):
    """Not a micro-benchmark -- these files are uploaded to HDFS and pulled back over the
    site network, and the old encoder was writing several times the bytes for worse
    compatibility. If this ever inverts, something has gone wrong with the encoder settings."""
    imgs = frames(60, w=320, h=240)
    new = tmp_path / "new.mp4"
    write(new, 25.0, imgs)
    old = tmp_path / "old.mp4"
    vw = cv2.VideoWriter(str(old), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    assert vw.isOpened()
    for f in imgs:
        vw.write(f)
    vw.release()
    assert os.path.getsize(new) < os.path.getsize(old)
