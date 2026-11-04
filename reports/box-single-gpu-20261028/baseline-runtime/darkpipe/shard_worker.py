"""Private offline shard entry point; receives the complete PipelineConfig as JSON."""
import json
import sys

from .config import PipelineConfig, validate
from .pipeline import run_offline


def main(path):
    with open(path, encoding="utf-8") as f:
        cfg = PipelineConfig(**json.load(f))
    if cfg.mode != "offline" or cfg.gpus or cfg.clip_dir:
        raise ValueError("worker requires offline mode, no recursive shards, and no clips")
    validate(cfg)
    run_offline(cfg)
    if cfg.events_json and cfg.recognize == "off":
        with open(cfg.events_json, "w", encoding="utf-8") as f:
            json.dump([], f)


if __name__ == "__main__":
    main(sys.argv[1])
