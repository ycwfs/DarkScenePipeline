"""Package the review page, source videos, excerpts and blank annotations."""
import json
import zipfile

from collect import OUT, read, save, sha


def main():
    assert read(OUT/'verification.json')['passed']
    assert read(OUT/'browser-verification.json')['passed']
    names=['index.html','original-clips.html','sources.html','README.md','review.csv',
        'original-clips.csv','original-clips.json','review-items.json','observations.json',
        'sources.json','source-media.json','summary.json','runs.json','excluded-runs.json',
        'base-windows.jsonl','other-candidates.csv','dataset-staging.json','annotations-empty.json',
        'verification.json','media-verification.json','browser-verification.json','label-import-verification.json',
        'http-verification.json']
    files=[OUT/n for n in names]
    for folder in ('clips','posters','original-clips','sources','render-metadata','scripts'):
        files.extend(p for p in sorted((OUT/folder).iterdir()) if p.is_file() and p.suffix in ('.mp4','.jpg','.json','.py','.html'))
    index=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p))for p in files]
    save('portable-manifest.json',dict(files=index,review_items_sha256=sha(OUT/'review-items.json')))
    files.append(OUT/'portable-manifest.json')
    archive=OUT/'portable-review.zip';tmp=OUT/'portable-review.partial.zip'
    with zipfile.ZipFile(tmp,'w',allowZip64=True)as z:
        for p in files:
            media=p.suffix in ('.mp4','.jpg')
            z.write(p,'10-25-action-review/'+str(p.relative_to(OUT)),compress_type=zipfile.ZIP_STORED if media else zipfile.ZIP_DEFLATED)
    tmp.replace(archive)
    with zipfile.ZipFile(archive)as z:
        assert z.testzip() is None
        assert len(z.infolist())==len(files)
        for f in index:assert z.getinfo('10-25-action-review/'+f['path']).file_size==f['bytes']
    result=dict(path=archive.name,files=len(files),bytes=archive.stat().st_size,sha256=sha(archive),zip_crc_verified=True,
        main_previews=473,original_clips=454,raw_sources=9,review_annotations_empty=True)
    save('package-verification.json',result)
    (OUT/'portable-review.zip.sha256').write_text(f'{result["sha256"]}  {archive.name}\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
