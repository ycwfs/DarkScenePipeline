# Platform operator packaging (developer notes)

This directory packages DarkScenePipeline as two operators for the video-parsing data-centre
platform, per `docs/第三方算法开封接入规范1.docx`. The submitted artefacts are two zips; the
Chinese `README.md` inside each is the 算子使用说明文件 the spec asks for. **This file is not
part of the submission** — it is the how-to for whoever builds and re-submits.

```
platform/
├── op_dark_behavior/         处理算子       -> 事件_行为识别_暗光场景行为识别.zip
├── op_dark_behavior_eval/    测试验证算子   -> 事件_行为识别_暗光场景行为识别测试验证.zip
│     each contains: main.py, suanzi.json, README.md (中文), zip_name
├── oputil.py                 shared helpers, copied into both zips
├── Dockerfile                one image for both operators (runtime + weights, no source)
├── build_image.sh            docker build (+ --save to export a tarball)
├── make_operator_zip.sh      builds the flat zips
├── validate_suanzi.py        spec conformance checker
└── dryrun_test.py            runs an operator with the argv the framework would build
```

## Design decisions worth knowing

**The operators are thin adapters, not a second pipeline.** `op_dark_behavior/main.py` builds a
`PipelineConfig`, calls `validate()` (`darkpipe/config.py`) and then `run_offline()`
(`darkpipe/pipeline.py`) — or `run_offline_sharded()` (`darkpipe/shard.py`) for multiple GPUs.
That is the exact code path the `darkpipe` CLI uses, so operator and CLI cannot diverge in
behaviour or performance.

**`darkpipe/` ships inside the zip, not inside the image.** The image holds the conda env and
the weights only. `main.py` puts its own directory first on `sys.path`, so the copy in the zip
always wins. One source of truth; no image/zip version skew. The cost is a ~200 KB zip instead
of a ~10 KB one, which is nothing.

**The zip doesn't get the whole `darkpipe/` tree, only the reachable part.** The platform
surface only exposes off/retinexformer + off/bicubic + off/behavior (same constraint as
`build_image.sh`'s `REQUIRED` weights), so `cidnet`/`realrestorer`/`lightsr`/`catanet`/`xclip`
stage files, their vendored architectures, and `cli.py`/`server.py` (the interactive CLI's
`serve` subcommand, never called by an operator) can't be reached through either `main.py`.
`make_operator_zip.sh` excludes exactly those paths from the `rsync` into the zip. The source
tree itself is untouched — `compare/` and the interactive CLI still need the full backend set —
this is a packaging-time trim, not a deletion. When the platform surface changes, update the
`--exclude` list alongside `build_image.sh`'s `REQUIRED` array.

**`cfg.stats`.** `run_offline` / `run_offline_sharded` publish `frames/seconds/fps/…` on the
config object. The operator reports those as `fps`/`seconds` and its own wall-clock as
`wall_seconds`. Without this the adapter's timer would wrap model loading and charge a one-off
5–6 s startup to the frame rate — that is exactly how an early version reported 14.7 fps for a
run the pipeline itself measured at 20.4 fps.

**Output paths are treated as opaque.** The framework picks them and does not guarantee an
extension. The video output is encoded to a temp `.mp4` and renamed onto the requested path,
because cv2 selects the container from the extension and fails outright on a bare path.

## Build the image

Docker with the nvidia runtime is required (`docker run --rm --gpus all ubuntu nvidia-smi`
should list the GPUs).

```bash
bash platform/build_image.sh                 # -> darkpipe-operator:0.1.0
bash platform/build_image.sh --save          # also writes darkpipe-operator-0.1.0.tar.gz
```

`build_image.sh` stages its own build context containing only the weights it needs, so the
10 GB `ckpts/` directory and the 163 GB `compare/` tree are never sent to the daemon. The
platform surface only exposes off/retinexformer + off/bicubic + off/behavior, so the only
weights required are `NTIRE.pth` and `videomamba_t_behavior_32f.pth`.

Network notes for this box: github.com times out, so the Miniforge installer and the
mamba-ssm / causal-conv1d wheels all need the `GH_PREFIX` mirror; PyPI, conda-forge and the
CUDA base image are slow but reachable.

```bash
GH_PREFIX=https://ghfast.top/ \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
bash platform/build_image.sh
```

**Two things the build cannot verify, and one it must not pretend to.** `docker build` gets no
GPU, so `import mamba_ssm` fails there ("0 active drivers") — the in-image self-check therefore
imports the CPU-safe packages for real and only reads the *versions* of `mamba_ssm` /
`causal_conv1d`. Their actual usability is established by the `docker run --gpus all` step
below, not asserted in the Dockerfile. Both bugs this exposed were invisible at build time:
triton JIT-compiles a C helper on first CUDA use, so the image needs `gcc` **and** `libc6-dev`
(`--no-install-recommends gcc` alone leaves out `crti.o` and the link fails), and triton's
cache defaults to `$HOME/.triton`, which is why `TRITON_CACHE_DIR` is pinned to `/tmp`.

**Why Miniforge and not Miniconda.** The first build failed at `conda create` with
`TermsOfServiceNotAcceptedError`: Miniconda's `defaults` channel (repo.anaconda.com) now
requires accepting Anaconda's terms, and those terms require a paid licence for organisations
over 200 people. Accepting them in a Dockerfile would hand that obligation to whoever runs the
image at the data centre. Miniforge resolves everything from conda-forge instead, which has no
such condition. `CONDA_CHANNEL=` overrides the channel URL if conda-forge itself needs a mirror.

## Build the zips

```bash
bash platform/make_operator_zip.sh                    # both, into platform/dist/
bash platform/make_operator_zip.sh op_dark_behavior   # just one
```

It validates `suanzi.json` before zipping and refuses to produce a non-conforming package. The
zip is flat — `suanzi.json`, `README.md`, `main.py` sit at the root with no wrapper directory,
which the spec requires and which is the single easiest thing to get wrong.

Renaming a package (if the platform's category dictionary differs from 事件/行为识别): edit
`op_*/zip_name` and re-run. Nothing else references the name.

## Verify before submitting

```bash
# 1. manifests conform to the spec
.venv/bin/python platform/validate_suanzi.py platform/op_*/suanzi.json

# 2. the processing operator, driven by argv built from its own suanzi.json
.venv/bin/python platform/dryrun_test.py --video /path/to/clip.mp4 --set ckpt_dir=./ckpts
.venv/bin/python platform/dryrun_test.py --video /path/to/clip.mp4 --set ckpt_dir=./ckpts \
    --set sr_scale=3 --set gpu_ids=0,1,2,3

# 3. the evaluation operator over a labelled manifest
.venv/bin/python platform/dryrun_test.py --op platform/op_dark_behavior_eval \
    --video /path/to/manifest.csv --set ckpt_dir=./ckpts --set max_clips=20

# 4. the pipeline itself still passes, including the manifest/code drift guards
.venv/bin/python -m pytest tests/ -q

# 5. the built zip, run inside the image the way the framework will run it
unzip -q platform/dist/事件_行为识别_暗光场景行为识别.zip -d /tmp/pkg
docker run --rm --gpus all \
    -v /tmp/pkg:/opt/darkpipe/op:ro -v /path/to/clip.mp4:/data/in.mp4:ro -v /tmp/out:/out \
    -w /opt/darkpipe/op darkpipe-operator:0.1.0 \
    /opt/conda/envs/darkpipe/bin/python -u main.py \
      --video_path /data/in.mp4 --enhance retinexformer --sr bicubic --sr_scale 2 \
      --recognize behavior --reco_span_sec 1.0 --max_frames 900 --label_bar true \
      --gpu_ids 0 --enhance_chunk 4 \
      --output_video /out/fw/output_video.mp4 --events_json /out/fw/events_json.json \
      --summary_json /out/fw/summary_json.json
```

`dryrun_test.py` deliberately points every `outputPath` into a directory that does not exist
yet, which is what proves the operator creates its own parent directories as the spec demands.

Step 5 mounts the package **read-only** and omits `--ckpt_dir` on purpose: that is what proves
the framework's own copy of the code runs against the weights baked into the image, with no
write access to the source and no runtime download. `tests/test_platform_operators.py` covers
the failure mode nothing else would catch — `suanzi.json` and `main.py` drifting apart, which
surfaces as an unrecognised-flag crash inside a scheduler rather than at review time.

## Measured numbers (single RTX 3090, fresh process per row, 900-frame clips)

| input     | config                                       | GPUs | fps  | wall    |
| --------- | -------------------------------------------- | ---- | ---- | ------- |
| 640×480  | retinexformer + bicubic ×2 + behavior       | 1    | 24.3 | 42.6 s  |
| 1280×720 | retinexformer + bicubic ×2 + behavior       | 1    | 8.0  | 118.1 s |
| 1280×720 | same                                         | 4    | 11.2 | 80.2 s  |
| 1280×720 | retinexformer + sr off + behavior            | 4    | 17.7 | 51.0 s  |
| 640×480  | same as row 1,**inside the container** | 1    | 22.9 | 45.7 s  |

The container costs about 6% of throughput against the host on identical work (22.9 vs 24.3 fps,
same 55 events), which is ordinary overhead and still well clear of the 15 fps target.

`enhance_chunk` defaults to 4 in the operators (the library default is 32). Warm in-process
measurement at 640×480: chunk 4 → 27.3 fps / 2.38 GiB reserved, chunk 8 → 27.9 fps / 4.70 GiB.
Throughput is flat in this range while memory is linear, so 4 is the better operating point and
halves what the operator has to declare in `suanzi.json`. Worth considering for the library
default too — that is a separate change and has not been made.

## Open question for the platform team

**Resolved for `video_path`:** the processing operator's video input never uses `hdfs://`.
Production input is real-time streams — 国标(GB28181, converted to a pullable URL by the
gateway)/`rtsp://`/`http(s)://` (flv, hls) — plus container-local paths for testing; all of
those OpenCV opens directly, so `fetch_input`'s `hdfs://` branch is never exercised for
`video_path`. `op_dark_behavior/main.py`'s `--video_path` help text, `suanzi.json` description
and README no longer mention `hdfs://` as an accepted form.

**Still open for `op_dark_behavior_eval`'s `dataset_manifest`:** that parameter still goes
through the same `fetch_input`, and whether its `hdfs://` branch is ever exercised (vs. the
framework materialising the manifest to a local path before invoking the operator) has not been
confirmed with the platform team. The branch stays in `oputil.py` — shared by both operators —
until that's clarified; it must not be assumed dead there.
