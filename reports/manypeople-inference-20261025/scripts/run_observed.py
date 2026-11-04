"""Run the released operator unchanged, recording batch progress externally."""
import json
import os
from pathlib import Path
import sys
import time


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main() -> None:
    sys.path.insert(0, '/app')
    import main as operator
    from darkpipe.person.runtime import PersonEngine

    report = Path(os.environ['BATCH_REPORT'])
    original_process = PersonEngine.process
    original_one = operator.run_offline_one
    active = {}
    timings = {}

    def process(self, raw, index, stamp, epoch, fps, prepared=None):
        if index == 0:
            timings['start'] = time.monotonic()
        result = original_process(self, raw, index, stamp, epoch, fps, prepared=prepared)
        record = result[2]
        # Only serialize scalar observations; the model inputs and pictures are unchanged.
        if index % 250 == 0:
            progress = dict(active, frames=index + 1, source_seconds=stamp,
                elapsed=time.monotonic() - timings['start'], people=record['people'],
                raw_luma=record['raw_luma'], events=self.events, updated_at=time.time())
            save(report / 'progress.json', progress)
            with (report / 'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(progress, ensure_ascii=False) + '\n')
            print(f'[progress] {Path(self.cfg.input).name}: frame={index+1} '
                  f'source={stamp:.1f}s people={record["people"]} events={self.events}', flush=True)
        return result

    def one(args, src, session, workdir, stem, gpus, verbose=True):
        active.clear()
        active.update(file=src, session=session, stem=stem, started_at=time.time())
        save(report / 'progress.json', dict(active, frames=0, status='loading'))
        result = original_one(args, src, session, workdir, stem, gpus, verbose=verbose)
        with (report / 'completed-items.jsonl').open('a') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        save(report / 'progress.json', dict(active, status='finished',
            frames=result['stats']['frames'], events=result['events_total']))
        return result

    PersonEngine.process = process
    operator.run_offline_one = one
    save(report / 'effective-arguments.json', vars(operator.build_parser().parse_args()))
    operator.main()


if __name__ == '__main__':
    main()
