"""Run the real packaged main; observe output metadata without changing any pixels."""
import json
import os
from pathlib import Path
import sys
import time


def main():
    sys.path.insert(0,os.environ['BOX_RUNTIME'])
    import main as operator
    from darkpipe.person import service
    original=service.Output.__init__
    output=Path(os.environ['DARKPIPE_AUDIT_DIR']);output.mkdir(parents=True,exist_ok=True)
    with (output/'frames.jsonl').open('w',buffering=1) as logfile:
        def instrument(self,engine,callback,writer=None):
            def observed(shown,clean,record):
                row={k:v for k,v in record.items() if k not in ('tracks','keys')}
                row.update(origin=engine.origin,output_monotonic=time.monotonic(),
                    rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
                    engine=engine.snapshot() if record['frame']%100==0 else None)
                logfile.write(json.dumps(row)+'\n')
                callback(shown,clean,record)
            original(self,engine,observed,writer)
        service.Output.__init__=instrument
        operator.main()


if __name__=='__main__':
    main()
