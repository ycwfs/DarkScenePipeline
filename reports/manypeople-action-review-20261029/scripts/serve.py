"""Serve the review directory on loopback, including byte ranges for MP4 seeking."""
from __future__ import annotations

import argparse
import functools
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]


class ReviewHandler(SimpleHTTPRequestHandler):
    def send_head(self) -> BinaryIO | None:
        self.remaining = 0
        root = Path(self.directory).resolve()
        path = Path(self.translate_path(self.path)).resolve()
        if not path.is_relative_to(root):
            self.send_error(403, 'Outside review directory')
            return None
        if path.is_dir():
            path = (path/'index.html').resolve()
            if not path.is_relative_to(root):
                self.send_error(403, 'Outside review directory')
                return None
        if not path.is_file():
            self.send_error(404, 'File not found')
            return None
        try:
            stream = path.open('rb')
        except OSError:
            self.send_error(404, 'File unavailable')
            return None
        size = os.fstat(stream.fileno()).st_size
        start, end = 0, size-1
        value = self.headers.get('Range') if self.command == 'GET' else None
        # If-Range validation is unsupported: return the full representation.
        if self.headers.get('If-Range'):
            value = None
        if value:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', value.strip())
            valid = bool(match and any(match.groups()) and size)
            if valid:
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), size-1) if right else size-1
                else:
                    suffix = int(right)
                    valid = suffix > 0
                    start = max(0, size-suffix)
                valid = valid and 0 <= start <= end < size
            if not valid:
                stream.close()
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return None
        self.remaining = max(0, end-start+1)
        self.send_response(206 if value else 200)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(self.remaining))
        self.send_header('Cache-Control', 'no-cache')
        if value:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        stream.seek(start)
        return stream

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        try:
            while self.remaining:
                block = source.read(min(self.remaining, 256*1024))
                if not block:
                    break
                outputfile.write(block)
                self.remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers cancel the previous request when seeking/changing clips.
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    handler = functools.partial(ReviewHandler, directory=str(ROOT))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Review URL: http://127.0.0.1:{server.server_port}/index.html', flush=True)
        print('Forward this port in VS Code, then open it in your local Chrome/Edge browser.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
