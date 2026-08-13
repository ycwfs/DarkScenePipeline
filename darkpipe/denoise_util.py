"""Re-export so clips.py can denoise without importing the stage machinery.

The clip writer needs only the per-frame function; going through darkpipe.stages would pull
in the whole stage factory (and its lazy backend imports) on a thread that has no use for
any of it.
"""
from .stages.denoise import MODES, denoise_frame  # noqa: F401
