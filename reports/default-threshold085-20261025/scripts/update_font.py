"""Keep every old bundled glyph and add the user-selected action name."""
import hashlib
import json
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
REL = Path('platform/dark-demo-v2/darkpipe/assets/NotoSansCJK-subset.ttf')
SOURCE = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = TTFont(OUT / 'before' / REL)
    old = set(before.getBestCmap())
    font = TTFont(SOURCE, fontNumber=0, recalcTimestamp=False)
    options = subset.Options()
    options.recalc_timestamp = False
    task = subset.Subsetter(options=options)
    task.populate(unicodes=old | set(map(ord, '跑动')))
    task.subset(font)
    font.save(ROOT / REL)
    after = TTFont(ROOT / REL)
    assert old <= set(after.getBestCmap())
    assert all(ord(ch) in after.getBestCmap() for ch in '跑动')
    record = dict(source=str(SOURCE), source_sha256=sha(SOURCE), font_index=0,
                  original_sha256=sha(OUT / 'before' / REL),
                  output_sha256=sha(ROOT / REL), old_codepoints=len(old),
                  new_codepoints=len(after.getBestCmap()), added_text='跑动',
                  original_glyphs_preserved=True)
    (OUT / 'font-verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
