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

**`outputPath` alone doesn't get results out of production.** The spec's own `outputPath`
mechanism only guarantees the file exists inside the container at exit; nothing in it describes
the framework persisting that file anywhere a human can fetch it afterwards. So every operator
takes two destinations — `local_output_dir` (a path inside the container, in practice an NFS
share the platform mounted) and `hdfs_output_dir` — and `oputil.deliver()` copies each
`outputPath` file to whichever are filled, both under the **same** `<timestamp>_<pid>` run
subdirectory so the two sides line up.

Either may be blank; **both blank is refused before any GPU work starts**, because that
configuration produces results nobody can retrieve, and discovering that after a multi-minute
run wastes the run. Note the spec's rule here: a `default` field *makes a parameter mandatory
and non-empty*, so "may be left blank" means the manifest must not give it a default — which is
why neither destination has one.

`upload_outputs()` mirrors `fetch_input()`'s scheme-sniffing: `hdfs://` goes through
`hdfs dfs -put`/`hadoop fs -put` (fails closed with a clear error if neither client is on
`PATH`); anything else is an already-reachable local/mounted directory and gets a
`shutil.copy2`. That non-hdfs branch is not a hypothetical — it is both how `local_output_dir`
works and what makes `dryrun_test.py` and step 5 exercisable on a box with no HDFS at all.

**Every event is also appended to `<clip_dir>/<session>/events.jsonl`.** The clip sidecars
only describe behaviours a clip was cut for, and `/events` (SSE) is unreachable whenever the
platform does not publish `serve_port` — which an operator cannot declare, the spec's
`container` block having no port field any more than it has a volume one. Without the log,
that deployment shape has no durable record of `other`, or of anything `clip_skip_labels`
filters. Verified by skipping every label the test clip produces: 0 clips written, 31 events
still logged. It is uploaded to HDFS once at shutdown rather than per-event, being append-only.

**The serve operator stages clips locally before moving them to `clip_dir`.** Same reasoning
pointed at a live path: `clip_dir` is normally NFS, and encoding frame-by-frame across it puts
network latency on the writer thread, where one stall longer than the bounded queue turns into
dropped frames. It also means a browsing user never sees a half-written mp4 growing in the
folder for the length of an incident. The move is one sequential copy, off the critical path.

## Build the image

Docker with the nvidia runtime is required (`docker run --rm --gpus all ubuntu nvidia-smi`
should list the GPUs).

```bash
bash platform/build_image.sh                 # -> darkpipe-operator:0.2.0
bash platform/build_image.sh --save          # also writes darkpipe-operator-0.2.0.tar.gz
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
.venv/bin/python platform/dryrun_test.py --video /path/to/clip.mp4 --set ckpt_dir=./ckpts \
    --set local_output_dir=/tmp/darkpipe_nfs --set hdfs_output_dir=
.venv/bin/python platform/dryrun_test.py --video /path/to/clip.mp4 --set ckpt_dir=./ckpts \
    --set sr_scale=3 --set gpu_ids=0,1,2,3 \
    --set local_output_dir=/tmp/darkpipe_nfs --set hdfs_output_dir=/tmp/darkpipe_dryrun_out

# 3. the evaluation operator over a labelled manifest
.venv/bin/python platform/dryrun_test.py --op platform/op_dark_behavior_eval \
    --video /path/to/manifest.csv --set ckpt_dir=./ckpts --set max_clips=20 \
    --set local_output_dir=/tmp/darkpipe_nfs --set hdfs_output_dir=

# 4. the pipeline itself still passes, including the manifest/code drift guards
.venv/bin/python -m pytest tests/ -q

# 5. the built zip, run inside the image the way the framework will run it
unzip -q platform/dist/事件_行为识别_暗光场景行为识别.zip -d /tmp/pkg
docker run --rm --gpus all \
    -v /tmp/pkg:/opt/darkpipe/op:ro -v /path/to/clip.mp4:/data/in.mp4:ro -v /tmp/out:/out \
    -v /tmp/hdfs_out:/hdfs_out -v /tmp/nfs_out:/nfs_out \
    -w /opt/darkpipe/op darkpipe-operator:0.2.0 \
    /opt/conda/envs/darkpipe/bin/python -u main.py \
      --video_path /data/in.mp4 --enhance retinexformer --sr bicubic --sr_scale 2 \
      --recognize behavior --reco_span_sec 1.0 --max_frames 900 --label_bar true \
      --gpu_ids 0 --enhance_chunk 4 \
      --local_output_dir /nfs_out --hdfs_output_dir /hdfs_out \
      --output_video /out/fw/output_video.mp4 --events_json /out/fw/events_json.json \
      --summary_json /out/fw/summary_json.json

# 6. the serve operator: same image, but the proof is the three live streams and the clips,
#    not an output file. run_seconds turns the persistent service into a finite check.
unzip -q platform/dist/事件_行为识别_暗光场景行为识别实时服务.zip -d /tmp/pkg_serve
docker run -d --name darkserve-test --gpus all -p 8099:8000 \
    -v /tmp/pkg_serve:/opt/darkpipe/op:ro -v /path/to/clip.mp4:/data/in.mp4:ro \
    -v /tmp/nfs_clips:/opt/darkpipe/clips -v /tmp/hdfs_land:/hdfs_land -v /tmp/out:/out \
    -w /opt/darkpipe/op darkpipe-operator:0.2.0 \
    /opt/conda/envs/darkpipe/bin/python -u main.py \
      --video_path /data/in.mp4 --enhance retinexformer --sr bicubic --sr_scale 2 \
      --recognize behavior --reco_span_sec 1.0 --label_bar true --gpu_ids 0 \
      --ckpt_dir /opt/darkpipe/ckpts --serve_port 8000 --jpeg_quality 85 \
      --max_stream_fps 15 --clip_dir /opt/darkpipe/clips --clip_pre_sec 2 \
      --clip_post_sec 2 --clip_max_sec 10 --clip_skip_labels other \
      --hdfs_output_dir /hdfs_land --run_seconds 45 --session_json /out/session_json.json
sleep 30
curl -s http://localhost:8099/health          # from OUTSIDE: proves the port mapping too
timeout 4 curl -sN http://localhost:8099/events
timeout 3 curl -sN http://localhost:8099/stream -o /tmp/mj.bin   # count `--frame` boundaries
docker wait darkserve-test && docker rm -f darkserve-test
find /tmp/nfs_clips /tmp/hdfs_land -name '*.mp4'   # same tree on both sides
```

Step 6's real assertion is on the clips, and the cheap check is the wrong one: that the files
exist says nothing about whether they are *watchable*. Decode each one and compare its playback
length (`frames / header-fps`) against the `duration_seconds` in its sidecar — they have to
match. That comparison is what caught the first version writing a 30 s incident as a 56 s video
(see `darkpipe/clips.py`'s writer loop for why estimating a header fps could not fix it).

`--hdfs_output_dir /hdfs_out` above is a plain mounted path, not `hdfs://` — it exercises
`upload_outputs()`'s local-copy branch, which is all this box (no HDFS cluster reachable) can
prove. The `hdfs://` branch (`hdfs dfs -put`/`hadoop fs -put`) is structurally identical to
`fetch_input()`'s existing, longstanding `hdfs://` download branch — neither has ever been
exercised end-to-end in this environment, for the same reason (no cluster, no client in the
image). Confirm `/tmp/hdfs_out/<timestamp>_<pid>/` contains all three uploaded files after the
run above, which is the same proof for the upload side that the rest of step 5 already gives for
the outputPath side.

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

## Always-on camera input: `op_dark_behavior_serve`, and `run_serve.sh`

`op_dark_behavior` accepts real-time stream URLs (GB28181/rtsp/http(s) flv,hls) as its
`video_path` input, but it is still an **offline batch job**: `--max_frames` must be a positive
integer for a live source, and `run_offline()` only writes `output_video`/`events_json`/
`summary_json` after its read loop ends (`darkpipe/pipeline.py`) — a camera that never drops
means the loop never ends and **nothing is ever written**, not even partial output. Removing the
`max_frames` bound does not turn this operator into a continuous monitor; it turns it into a
hung process with no output, ever.

The continuous-monitoring code is `darkpipe/server.py` (`darkpipe --mode serve`): a persistent
capture thread (auto-reconnects on stream drops with exponential backoff) plus a persistent
process thread, serving live results over HTTP — `/stream` (MJPEG), `/events` (SSE), `/health`,
`/config` — and, with `clip_dir` set, cutting an mp4 per non-`other` behaviour via
`darkpipe/clips.py`. It ships two ways.

**As an operator: `op_dark_behavior_serve`.** This document previously argued a persistent
service "structurally cannot" be an operator, on the grounds that the spec describes
`outputPath` as a file the framework collects after the container exits. That was too strong,
and the spec text is what corrects it: `outputs` is explicitly "可为空数组", and the file-passing
requirement ("至少有一个文件") is scoped to operators participating in 编排 — a standalone
service chains to nothing. The operator therefore declares one small `session_json` output
(endpoints, clip directories, final counters), **written at startup and rewritten at shutdown**
so a container that gets killed still leaves a non-empty file behind. `run_seconds` bounds the
run when an orchestrator needs the container to terminate; left at 0 it runs until stopped.

Two consequences worth keeping in mind: the platform has to map `serve_port` for the three
streams to be reachable, and `clip_dir` has to be mounted (NFS or host path) or the clips die
with the container — which is why `hdfs_output_dir` exists alongside it, and why it is optional
rather than required here (unlike the batch operators, where HDFS is the *only* way results
survive, a mounted `clip_dir` already is one).

**Standalone: `platform/run_serve.sh`.** Same image, same code, no manifest — for running it
directly rather than through the platform.

`platform/run_serve.sh` runs it as a standalone long-lived container, reusing the same
`darkpipe-operator:0.2.0` image (fastapi/uvicorn were added to the Dockerfile for this — they
cost a few MB against a multi-GB image) but mounting `darkpipe/` read-only from the repo instead
of baking source into the image, the same way step 5 above mounts the zip:

```bash
bash platform/run_serve.sh rtsp://<gb28181-gateway-url>
bash platform/run_serve.sh http://cam.local/live.flv --reco-span-sec 1.0
PORT=8080 GPU=1 bash platform/run_serve.sh rtsp://...
```

No camera available: `capture_loop` auto-loops local files, so a clip works as a stand-in to
verify the service itself:

```bash
bash platform/run_serve.sh /path/to/clip.mp4
curl http://localhost:8000/health
curl -N http://localhost:8000/events
```

`run_serve.sh` itself goes into no zip — it is a separate deployment started and managed
directly, with `docker run --restart unless-stopped` for resilience across container/host
restarts on top of `capture_loop`'s own stream-level reconnect. The submitted form of the same
capability is `op_dark_behavior_serve`; use whichever matches how the box is operated.

Note the packaging consequence: `make_operator_zip.sh`'s exclude list is **per operator**, not a
global constant. `server.py`/`clips.py` are dead weight in the two batch zips and are the entire
point of the serve zip, so the script keys the exclusion off the operator name.

### Live output formats, and why the image version had to move

`0.1.0` had no `ffmpeg` binary (OpenCV bundles the libs, not the CLI), so MJPEG was the only
thing the service could emit. `0.2.0` adds `ffmpeg` in its own Dockerfile layer — placed after
the `gcc` layer for the same reason that one sits after the pip installs: editing it must not
invalidate 13 GB of dependencies. It costs ~250 MB against a 17 GB image.

That is why the tag moved rather than being rebuilt in place. The serve operator can now
*require* ffmpeg, and an operator that requires it must not claim to run on an image without
it — a stale `0.1.0` would otherwise fail at startup for reasons nothing in the manifest
explains. All three `suanzi.json` files, `build_image.sh` and `run_serve.sh` point at `0.2.0`.

`darkpipe/streams.py` feeds every format from the **JPEG slot**, not from raw frames: those
bytes already exist for `/stream`, so an extra format costs a remux rather than a second
encode, and the pipe carries ~30 KB per frame instead of ~1 MB. ffmpeg infers the resolution
from the MJPEG stream, so nothing in that module has to know the frame size.

Three things worth knowing before changing it:
- **HLS latency is structural** (~3–6 s even at 1 s segments) — it is offered for browsers that
  need it, and is the wrong choice for the live demo. FLV is the low-latency one that also
  matches what the GB28181 gateways serve on the *input* side.
- **`/live.flv` is one encoder per viewer.** A shared encoder would be cheaper but requires
  holding the FLV header and replaying from the last keyframe for clients that join mid-stream.
  `max_flv_clients` bounds the cost instead; `rtmp_push_url` is the answer for many viewers —
  push once, let a real media server fan out.
- **Missing ffmpeg fails at startup, not at first request.** `validate()` checks for the binary
  whenever a non-mjpeg format is selected. A service that starts, reports healthy, and only
  breaks when someone opens the FLV URL during a demonstration is the failure worth preventing.

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
