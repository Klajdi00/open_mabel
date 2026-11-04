"""
MABEL Data Studio — FastAPI app.

One ASGI app serves the static web/ frontend, the website rig assets, the JSON
catalog/meta, the dense binary bundles, per-frame video + depth, lidar/scene
clouds, EDL projects, validation/preview, materialization and sample-gen jobs,
plus a WS playback stream. All blocking h5py work runs in the threadpool so the
event loop never stalls.
"""
import asyncio
import json
import os

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import (catalog, config as CFG, edl as EDL, episode, jobs, materialize,
               projects, scene, video)


def create_app():
    app = FastAPI(title="MABEL Data Studio")

    # Dev tool: never let the browser serve a stale page/module. Forces a fresh
    # fetch of HTML/JS/CSS on every load so code + catalog are always current.
    @app.middleware("http")
    async def _no_cache(request, call_next):
        resp = await call_next(request)
        path = request.url.path
        # Keep the app shell + JSON API always-fresh for live editing/dev — BUT
        # never clobber an endpoint that opted into caching (immutable per-frame
        # media: camera JPEGs, depth PNGs, lidar/point-cloud bins). Otherwise the
        # browser re-fetches every frame on each playback step / loop, which
        # floods the server and shows up as huge video latency.
        cacheable = (path == "/" or path.endswith((".html", ".js", ".css"))
                     or path.startswith("/api"))
        if cacheable and "cache-control" not in resp.headers:
            resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

    # ---- catalog -----------------------------------------------------------
    @app.get("/api/catalog")
    async def get_catalog():
        return await run_in_threadpool(catalog.full_catalog)

    @app.get("/api/datasets")
    async def get_datasets():
        return await run_in_threadpool(catalog.list_datasets)

    @app.get("/api/ds/{dataset}/episodes")
    async def get_episodes(dataset: str):
        return await run_in_threadpool(catalog.list_episodes, dataset)

    def _resolve(dataset, ep):
        try:
            return catalog.episode_path(dataset, ep)
        except FileNotFoundError:
            raise HTTPException(404, f"episode not found: {dataset}/{ep}")

    @app.get("/api/ds/{dataset}/ep/{ep}/meta")
    async def get_meta(dataset: str, ep: str):
        p = await run_in_threadpool(_resolve, dataset, ep)
        return await run_in_threadpool(episode.meta, p)

    # ---- dense numeric bundle (timeline + plots) ---------------------------
    @app.get("/api/ds/{dataset}/ep/{ep}/arrays")
    async def get_arrays(dataset: str, ep: str):
        p = await run_in_threadpool(_resolve, dataset, ep)
        blob = await run_in_threadpool(episode.dense_bundle, p)
        return Response(blob, media_type="application/octet-stream")

    @app.get("/api/ds/{dataset}/ep/{ep}/frame/{i}")
    async def get_frame(dataset: str, ep: str, i: int):
        p = await run_in_threadpool(_resolve, dataset, ep)
        state, _ = await run_in_threadpool(episode.frame_state, p, i)
        return state

    # ---- video -------------------------------------------------------------
    @app.get("/api/ds/{dataset}/ep/{ep}/cameras")
    async def get_cameras(dataset: str, ep: str):
        p = await run_in_threadpool(_resolve, dataset, ep)
        return await run_in_threadpool(video.cameras, p)

    @app.get("/api/ds/{dataset}/ep/{ep}/cam/{name}/frame/{i}.jpg")
    async def get_cam_frame(dataset: str, ep: str, name: str, i: int):
        p = await run_in_threadpool(_resolve, dataset, ep)
        b = await run_in_threadpool(video.frame_jpeg, p, name, i)
        if b is None:
            raise HTTPException(404, "no such camera frame")
        return Response(b, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/ds/{dataset}/ep/{ep}/cam/{name}/depth/{i}.png")
    async def get_depth(dataset: str, ep: str, name: str, i: int, colorize: int = 1):
        p = await run_in_threadpool(_resolve, dataset, ep)
        b = await run_in_threadpool(video.depth_png, p, name, i, bool(colorize))
        if b is None:
            raise HTTPException(404, "no depth for camera")
        return Response(b, media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/ds/{dataset}/ep/{ep}/cam/{name}.mp4")
    async def get_cam_mp4(dataset: str, ep: str, name: str):
        p = await run_in_threadpool(_resolve, dataset, ep)
        m = await run_in_threadpool(episode.meta, p)
        fps = m.get("fps") or 15.0
        ep_file = os.path.basename(p)
        out = await run_in_threadpool(video.ensure_mp4, p, dataset, ep_file, name, fps)
        if not out:
            raise HTTPException(404, "transcode failed / no camera")
        return FileResponse(out, media_type="video/mp4")

    # ---- lidar + scene -----------------------------------------------------
    _MEDIA_CACHE = {"Cache-Control": "max-age=3600"}   # immutable per-frame media → browser-cacheable

    @app.get("/api/ds/{dataset}/ep/{ep}/lidar.bin")
    async def get_lidar(dataset: str, ep: str, start: int = None, end: int = None):
        p = await run_in_threadpool(_resolve, dataset, ep)
        b = await run_in_threadpool(episode.lidar_bin, p, start, end)
        return Response(b, media_type="application/octet-stream", headers=_MEDIA_CACHE)

    @app.get("/api/ds/{dataset}/ep/{ep}/lidar/cloud/{i}.bin")
    async def get_lidar_cloud(dataset: str, ep: str, i: int):
        p = await run_in_threadpool(_resolve, dataset, ep)
        b = await run_in_threadpool(scene.lidar_cloud_frame, p, i)
        return Response(b, media_type="application/octet-stream", headers=_MEDIA_CACHE)

    @app.get("/api/ds/{dataset}/ep/{ep}/scene.bin")
    async def get_scene(dataset: str, ep: str, start: int = None, end: int = None,
                        stride: int = 2, voxel: float = 0.05):
        p = await run_in_threadpool(_resolve, dataset, ep)
        b = await run_in_threadpool(scene.accumulated_scene, p, start, end, stride, voxel)
        return Response(b, media_type="application/octet-stream", headers=_MEDIA_CACHE)

    @app.get("/api/ds/{dataset}/ep/{ep}/occupancy.json")
    async def get_occupancy(dataset: str, ep: str):
        p = await run_in_threadpool(_resolve, dataset, ep)
        out = await run_in_threadpool(scene.occupancy_map, p)
        if out is None:
            raise HTTPException(404, "no occupancy (needs lidar + chassis pose)")
        return out

    @app.get("/api/ds/{dataset}/ep/{ep}/cam/{name}/cloud/{i}.bin")
    async def get_cam_cloud(dataset: str, ep: str, name: str, i: int,
                            stride: int = 4, max_points: int = 60000):
        from . import camcloud
        p = await run_in_threadpool(_resolve, dataset, ep)
        b = await run_in_threadpool(camcloud.camera_cloud, p, name, i, stride, max_points)
        return Response(b, media_type="application/octet-stream", headers=_MEDIA_CACHE)

    @app.get("/api/ds/{dataset}/ep/{ep}/recon.bin")
    async def get_recon(dataset: str, ep: str, source: str = "head",
                        kind: str = "cloud", frames: int = 8,
                        ps: int = 0, voxel: float = 0.02):
        """Fused SLAM-style reconstruction from one camera (head/base): a dense
        point CLOUD (0x05) or a coloured MESH (0x06). ``ps`` = pixel stride (0 →
        server default), ``voxel`` = cloud downsample size in metres."""
        from . import camcloud
        p = await run_in_threadpool(_resolve, dataset, ep)
        b = await run_in_threadpool(camcloud.reconstruction, p, source, kind, frames, ps, voxel)
        return Response(b, media_type="application/octet-stream", headers=_MEDIA_CACHE)

    @app.get("/api/ds/{dataset}/ep/{ep}/quality.json")
    async def get_quality(dataset: str, ep: str):
        """Temporal-integrity + defect report for one episode."""
        from . import quality
        p = await run_in_threadpool(_resolve, dataset, ep)
        out = await run_in_threadpool(quality.analyze, p)
        return JSONResponse(out, headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/ds/{dataset}/ep/{ep}/align.json")
    async def get_align(dataset: str, ep: str, cam: str = None):
        """Robust camera(pixel)↔proprio lag for the Align-to-pixels button."""
        from . import quality
        p = await run_in_threadpool(_resolve, dataset, ep)
        out = await run_in_threadpool(quality.align_to_pixels, p, cam)
        return JSONResponse(out, headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/ds/{dataset}/ep/{ep}/frequency.json")
    async def get_frequency(dataset: str, ep: str):
        """Per-channel effective-Hz tracks for the frequency heat."""
        from . import quality
        p = await run_in_threadpool(_resolve, dataset, ep)
        out = await run_in_threadpool(quality.frequency_tracks, p)
        return JSONResponse(out, headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/embodiment/{robot}")
    async def get_embodiment(robot: str):
        """Embodiment registry (§1.3): actuators/DOF, cameras, ActionSpec, limits."""
        from . import embodiment
        return await run_in_threadpool(embodiment.registry)

    @app.get("/api/ds/{dataset}/quality")
    async def get_corpus_quality(dataset: str):
        """Whole-dataset quality: per-episode scores (worst-first) + a joint score."""
        from . import quality
        return await run_in_threadpool(quality.corpus_quality, dataset)

    @app.get("/api/ds/{dataset}/ep/{ep}/calibration")
    async def get_calibration(dataset: str, ep: str):
        from . import camcloud
        p = await run_in_threadpool(_resolve, dataset, ep)
        return await run_in_threadpool(camcloud.cameras_calibration, p)

    @app.get("/api/ds/{dataset}/ep/{ep}/extrinsics/{i}")
    async def get_extrinsics(dataset: str, ep: str, i: int):
        from . import camcloud
        p = await run_in_threadpool(_resolve, dataset, ep)
        return await run_in_threadpool(camcloud.cameras_extrinsics, p, i)

    @app.get("/api/ds/{dataset}/ep/{ep}/extrinsics_all")
    async def get_extrinsics_all(dataset: str, ep: str):
        from . import camcloud
        p = await run_in_threadpool(_resolve, dataset, ep)
        out = await run_in_threadpool(camcloud.cameras_extrinsics_all, p)
        return JSONResponse(out, headers={"Cache-Control": "max-age=3600"})

    # ---- EDL projects ------------------------------------------------------
    @app.get("/api/projects")
    async def get_projects():
        return await run_in_threadpool(projects.list_projects)

    @app.get("/api/projects/{name}")
    async def get_project(name: str):
        try:
            return await run_in_threadpool(projects.get, name)
        except FileNotFoundError:
            raise HTTPException(404, "no such project")

    @app.put("/api/projects/{name}")
    async def put_project(name: str, request: Request):
        body = await request.json()
        return await run_in_threadpool(projects.save, name, body)

    @app.delete("/api/projects/{name}")
    async def del_project(name: str):
        ok = await run_in_threadpool(projects.delete, name)
        return {"deleted": ok}

    @app.post("/api/edl/validate")
    async def post_validate(request: Request):
        body = await request.json()
        return await run_in_threadpool(EDL.validate, body)

    @app.post("/api/edl/preview")
    async def post_preview(request: Request):
        body = await request.json()
        clean, _ = await run_in_threadpool(EDL.normalize, body)
        return await run_in_threadpool(EDL.preview_map, clean)

    # ---- save / materialize (job) -----------------------------------------
    @app.post("/api/save")
    async def post_save(request: Request):
        body = await request.json()
        edl = body.get("edl", body)
        out_dataset = body.get("out_dataset") or body.get("dataset") or "edited"
        task = body.get("task")
        jid = jobs.create("materialize")
        jobs.run(jid, lambda prog: materialize.materialize(edl, out_dataset, task, prog))
        return {"job": jid}

    @app.post("/api/export/episodes")
    async def post_export_episodes(request: Request):
        """Batch PER-EPISODE export: each item = one clip's single-segment EDL,
        materialized individually to out_dataset/out_file (overwrite = idempotent).
        One source demo → one processed episode; no cross-episode merge."""
        body = await request.json()
        out_dataset = body.get("out_dataset") or "processed"
        task = body.get("task")
        items = body.get("items") or []

        def work(prog):
            results = []
            total = max(1, len(items))
            for i, it in enumerate(items):
                prog(i, total, 0)
                res = materialize.materialize(it.get("edl"), out_dataset, task,
                                              out_file=it.get("out_file"))
                results.append({"source": it.get("source"), "out_file": res["file"],
                                "num_frames": res["num_frames"]})
            prog(total, total, 0)
            return {"dataset": out_dataset, "results": results, "count": len(results)}

        jid = jobs.create("export-episodes")
        jobs.run(jid, work)
        return {"job": jid}

    @app.post("/api/export/lerobot/dataset")
    async def post_export_lerobot_dataset(request: Request):
        """Aggregate an ENTIRE processed dataset folder into one LeRobot v2.1
        dataset (the finished-folder export). Every episode_*.h5 becomes a LeRobot
        episode."""
        from . import lerobot_export
        import glob
        body = await request.json()
        dataset = body.get("dataset")
        repo_id = body.get("repo_id") or f"mabel/{dataset}"
        h5s = sorted(glob.glob(os.path.join(CFG.DATASET_ROOT, dataset, "episode_*.h5")))
        if not h5s:
            raise HTTPException(400, f"no episodes in dataset '{dataset}'")
        out_root = os.path.join(CFG.WORKSPACE, "lerobot_exports")
        os.makedirs(out_root, exist_ok=True)

        def work(prog):
            return lerobot_export.to_lerobot_dataset(h5s, out_root, repo_id, prog)

        jid = jobs.create("export-lerobot-dataset")
        jobs.run(jid, work)
        return {"job": jid}

    @app.post("/api/export/lerobot")
    async def post_export_lerobot(request: Request):
        """Materialize the curated EDL, then convert it to a LeRobot v2.1 dataset."""
        from . import lerobot_export
        body = await request.json()
        edl = body.get("edl", body)
        out_dataset = body.get("out_dataset") or "curated"
        task = body.get("task")

        def work(prog):
            res = materialize.materialize(edl, out_dataset, task, prog)
            h5 = os.path.join(CFG.DATASET_ROOT, res["dataset"], res["file"])
            out_root = os.path.join(CFG.WORKSPACE, "lerobot_exports")
            os.makedirs(out_root, exist_ok=True)
            lr = lerobot_export.to_lerobot(h5, out_root, repo_id=f"mabel/{out_dataset}")
            return {**res, "lerobot": lr}

        jid = jobs.create("export-lerobot")
        jobs.run(jid, work)
        return {"job": jid}

    @app.post("/api/dev/generate-sample")
    async def post_gen(request: Request):
        body = await request.json()
        from . import sampledata
        name = body.get("name", "studio_sample")
        n = int(body.get("episodes", 3))
        fr = int(body.get("frames", 120))
        jid = jobs.create("sample")
        jobs.run(jid, lambda prog: sampledata.generate(name, n, fr, progress=prog))
        return {"job": jid}

    @app.get("/api/jobs/{jid}")
    async def get_job(jid: str):
        j = jobs.get(jid)
        if not j:
            raise HTTPException(404, "no such job")
        return j

    # ---- WS playback -------------------------------------------------------
    @app.websocket("/ws/state/{dataset}/{ep}")
    async def ws_state(ws: WebSocket, dataset: str, ep: str):
        await ws.accept()
        try:
            p = await run_in_threadpool(_resolve, dataset, ep)
        except HTTPException:
            await ws.close()
            return
        state = {"frame": 0, "fps": 15.0, "speed": 1.0, "playing": False}
        _, num = await run_in_threadpool(episode.frame_state, p, 0)

        async def send_frame(i):
            st, _ = await run_in_threadpool(episode.frame_state, p, i)
            await ws.send_text(json.dumps(st))

        async def receiver():
            while True:
                msg = json.loads(await ws.receive_text())
                cmd = msg.get("cmd")
                if cmd == "seek":
                    state["frame"] = max(0, min(int(msg.get("frame", 0)), num - 1))
                    state["playing"] = False
                    await send_frame(state["frame"])
                elif cmd == "play":
                    state["fps"] = float(msg.get("fps", state["fps"]))
                    state["speed"] = float(msg.get("speed", state["speed"]))
                    state["playing"] = True
                elif cmd == "pause":
                    state["playing"] = False

        async def ticker():
            while True:
                if state["playing"]:
                    state["frame"] += 1
                    if state["frame"] >= num:
                        state["frame"] = num - 1
                        state["playing"] = False
                    await send_frame(state["frame"])
                await asyncio.sleep(1.0 / max(1.0, state["fps"] * state["speed"]))

        try:
            await asyncio.gather(receiver(), ticker())
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    @app.get("/api/trajectories")
    async def trajectories(tf: str = "left_palm", stride: int = 2, dataset: str = None):
        """Every episode's TF path (for the dataset-distribution overlay). Pass
        ?dataset= to limit to one dataset."""
        def work():
            out = []
            dsets = [{"name": dataset}] if dataset else catalog.list_datasets()
            for ds in dsets:
                for ep in catalog.list_episodes(ds["name"]):
                    if ep.get("error"):
                        continue
                    try:
                        p = catalog.episode_path(ds["name"], ep["file"])
                        pts = episode.trajectory(p, tf, stride)
                    except Exception:
                        pts = []
                    if pts:
                        out.append({"dataset": ds["name"], "episode": ep["file"],
                                    "task": ep.get("task", ""), "n": len(pts), "points": pts})
            return out
        return await run_in_threadpool(work)

    @app.get("/api/health")
    async def health():
        return {"ok": True, "dataset_root": CFG.DATASET_ROOT}

    # Warm the MuJoCo model in the background at boot (its first compile costs
    # ~30s; after that it's cached to a .mjb on disk). Doing it here means camera
    # extrinsics/clouds are ready by the time the user opens an episode, instead
    # of stalling the first 3D-camera interaction. Guarded: no-op without mujoco.
    @app.on_event("startup")
    async def _warm_model():
        import threading
        def _warm():
            try:
                from . import camcloud
                camcloud._model()
            except Exception:
                pass
        threading.Thread(target=_warm, daemon=True).start()

    # ---- static ------------------------------------------------------------
    if os.path.isdir(CFG.WEBSITE_ASSETS):
        app.mount("/website-assets", StaticFiles(directory=CFG.WEBSITE_ASSETS),
                  name="website-assets")
    app.mount("/", StaticFiles(directory=CFG.WEB_DIR, html=True), name="web")
    return app


app = create_app()
