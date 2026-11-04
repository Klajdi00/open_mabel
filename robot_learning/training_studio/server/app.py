"""
MABEL Trainer Studio — FastAPI app.

Serves the static web/ frontend and the trainer API: dataset registry, arch
registry + reactive arch-spec (the animated diagram), the modular feature
selection, launch/pause/retrain, live metrics/logs over WebSocket, and on-demand
eval + MuJoCo deploy. Mirrors the Data Curation studio's structure/conventions.
"""
import asyncio
import json
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import (arch_registry as AR, auth, compute, config as CFG,
               dataset_registry as DR, embodiment as EMB, eval as EV,
               feature_select as FS, jobs, lerobot_dataset as L,
               metrics_store as MS, trainer as TR)


def _resolve_selection(body):
    return FS.resolve(body.get("scope", "whole_body"),
                      groups=body.get("groups"), cameras=body.get("cameras"),
                      action_scope=body.get("action_scope"),
                      action_groups=body.get("action_groups"))


def create_app():
    MS.init_db()
    app = FastAPI(title="MABEL Trainer Studio")

    @app.on_event("startup")
    async def _reconcile_jobs():
        # detached trainings survive a studio restart — re-attach to live ones and
        # mark dead ones interrupted, so the GUI can pick them back up.
        try:
            await run_in_threadpool(MS.reconcile, TR._LOGS_DIR)
            # re-attach the system-metrics poller to any still-running detached job
            # so GPU/CPU keeps streaming after a studio restart (Mac + CUDA alike).
            for r in await run_in_threadpool(MS.list_runs):
                if r.get("status") in ("running", "preparing", "queued"):
                    TR.ensure_sysmon(r["run_id"])
        except Exception:
            pass

    @app.middleware("http")
    async def _no_cache(request, call_next):
        resp = await call_next(request)
        path = request.url.path
        cacheable = (path == "/" or path.endswith((".html", ".js", ".css"))
                     or path.startswith("/api"))
        if cacheable and "cache-control" not in resp.headers:
            resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

    # ── environment / rig ────────────────────────────────────────────────────
    @app.get("/api/health")
    async def health():
        return {"ok": True, "port": CFG.PORT}

    @app.get("/api/compute")
    async def get_compute():
        return await run_in_threadpool(compute.describe)

    @app.get("/api/embodiment")
    async def get_embodiment():
        return EMB.registry()

    # ── datasets ─────────────────────────────────────────────────────────────
    @app.get("/api/datasets")
    async def get_datasets():
        return await run_in_threadpool(DR.list_datasets)

    @app.post("/api/selection")
    async def post_selection(request: Request):
        body = await request.json()
        return _resolve_selection(body)

    @app.post("/api/datasets/import")
    async def import_dataset(request: Request):
        body = await request.json()
        path = body.get("path", "")
        if not path:
            raise HTTPException(400, "no path given")
        try:
            return await run_in_threadpool(DR.import_dataset, path)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/ds/{dataset}/ep/{ep}/arrays")
    async def ds_arrays(dataset: str, ep: int):
        root = await run_in_threadpool(DR.find, dataset)
        arrs = await run_in_threadpool(L.read_episode_arrays, root, ep)
        # sanitise NaN/Inf → 0 so the preview of a NaN-corrupted dataset (e.g. the
        # stress set) still serialises to JSON instead of 500-ing.
        return {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist()
                for k, v in arrs.items()}

    @app.get("/api/ds/{dataset}/ep/{ep}/cam/{cam}/frame/{i}.jpg")
    async def ds_frame(dataset: str, ep: int, cam: str, i: int):
        # cached whole-episode decode → the first frame warms the clip, the rest
        # are instant (no more O(n²) re-decode-from-zero per frame)
        root = await run_in_threadpool(DR.find, dataset)
        jpg = await run_in_threadpool(L.read_video_frame_jpeg, root, cam, ep, i)
        if jpg is None:
            raise HTTPException(404, "no frame")
        return Response(jpg, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    # ── architecture registry + reactive diagram ─────────────────────────────
    @app.get("/api/arch/schema")
    async def arch_schema():
        return AR.schema()

    @app.post("/api/arch/spec")
    async def arch_spec(request: Request):
        body = await request.json()
        arch = body["arch"]
        sel = _resolve_selection(body.get("selection", body))
        hp = {**AR.defaults(arch), **body.get("hp", {})}
        spec = AR.arch_spec(arch, hp, sel)
        issues = AR.validate(arch, hp, sel)
        return {"spec": spec, "issues": issues, "selection": sel}

    @app.post("/api/arch/compile")
    async def arch_compile(request: Request):
        # COMPILE the real LeRobot nn.Module → the truthful diagram graph.
        from . import nn_compiler as NC
        body = await request.json()
        arch = body["arch"]
        sel = _resolve_selection(body.get("selection", body))
        hp = {**AR.defaults(arch), **body.get("hp", {})}
        return await run_in_threadpool(NC.compile_spec, arch, hp, sel)

    @app.post("/api/arch/validate")
    async def arch_validate(request: Request):
        body = await request.json()
        arch = body["arch"]
        sel = _resolve_selection(body.get("selection", body))
        hp = {**AR.defaults(arch), **body.get("hp", {})}
        return {"issues": AR.validate(arch, hp, sel), "selection": sel}

    # ── training lifecycle ───────────────────────────────────────────────────
    def _client_ip(request):
        fwd = request.headers.get("x-forwarded-for", "")
        return (fwd.split(",")[0].strip() if fwd
                else (request.client.host if request.client else "?"))

    def _require_auth(request):
        """EVERY mutating training action requires a valid operator session
        (styled username+passkey popup → signed cookie). Checked SERVER-side so a
        job cannot be started OR killed without the operator credentials."""
        if not auth.verify_session(auth.request_session(request)):
            raise HTTPException(401, "operator login required — sign in with your "
                                     "username + passkey to run training actions")

    @app.get("/api/whoami")
    async def whoami(request: Request):
        user = auth.verify_session(auth.request_session(request))
        return {"authed": bool(user), "user": user or "",
                "server": "MABEL Trainer Studio"}

    @app.post("/api/login")
    async def login(request: Request):
        now = time.time()
        ip = _client_ip(request)
        if auth.throttled(ip, now):
            return JSONResponse({"error": "too many attempts — wait a few minutes"},
                                status_code=429)
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not auth.check_credentials(body.get("username"), body.get("passkey")):
            auth.note_fail(ip, now)
            return JSONResponse({"error": "invalid username or passkey"},
                                status_code=401)
        auth.note_ok(ip)
        token = auth.make_session(CFG.TRAIN_USER, now)
        resp = JSONResponse({"ok": True, "user": CFG.TRAIN_USER, "token": token})
        secure = (request.headers.get("x-forwarded-proto", request.url.scheme)
                  == "https")
        resp.set_cookie(auth.COOKIE, token, max_age=auth.TTL_SECONDS,
                        httponly=True, samesite="lax", secure=secure, path="/")
        return resp

    @app.post("/api/logout")
    async def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(auth.COOKIE, path="/")
        return resp

    @app.post("/api/train/start")
    async def train_start(request: Request):
        _require_auth(request)
        body = await request.json()
        arch = body.get("arch", "ACT")
        sel = _resolve_selection(body.get("selection", body))
        datasets = body.get("datasets") or []
        if not datasets:
            raise HTTPException(400, "no datasets selected")
        run_id = TR.new_run_id()
        hp = {**AR.defaults(arch), **body.get("hp", {})}
        issues = AR.validate(arch, hp, sel)
        if any(i["level"] == "error" for i in issues):
            raise HTTPException(400, {"error": "validation failed", "issues": issues})
        MS.register_run(run_id, name=body.get("name") or f"{arch}-run",
                        arch=arch, policy_type=AR.ARCHES[arch]["policy_type"],
                        scope=sel["scope"], dataset_id=",".join(datasets), status="queued")
        params = {"run_id": run_id, "arch": arch, "hp": body.get("hp", {}),
                  "selection": sel, "datasets": datasets,
                  "split": tuple(body.get("split", [0.8, 0.1, 0.1])),
                  "seed": body.get("seed", 1000), "device": body.get("device", "auto"),
                  "name": body.get("name"),
                  "resume": body.get("resume", False),
                  "pretrained_ckpt": body.get("pretrained_ckpt")}
        jid = jobs.create("train")
        jobs.run(jid, lambda prog: TR.start_training(params, prog))
        return {"run_id": run_id, "job": jid}

    @app.post("/api/train/{run_id}/pause")
    async def train_pause(run_id: str, request: Request):
        _require_auth(request)
        return {"paused": await run_in_threadpool(TR.pause, run_id)}

    @app.post("/api/train/{run_id}/stop")
    async def train_stop(run_id: str, request: Request):
        _require_auth(request)
        return {"stopped": await run_in_threadpool(TR.stop, run_id)}

    @app.post("/api/train/{run_id}/resume")
    async def train_resume(run_id: str, request: Request):
        """TRUE resume — same run, optimizer state + step counter continue from
        the checkpoint; only the remaining steps are trained."""
        _require_auth(request)
        body = await request.json()
        try:
            return await run_in_threadpool(
                TR.resume_run, run_id, body.get("steps"), body.get("checkpoint"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/train/retrain")
    async def train_retrain(request: Request):
        """Retrain from a checkpoint (default: same dataset as the source run)."""
        _require_auth(request)
        body = await request.json()
        src = MS.get_run(body["run_id"]) if body.get("run_id") else None
        meta = json.loads(src["meta_json"]) if src and src.get("meta_json") else {}
        arch = body.get("arch") or (src and src["arch"]) or "ACT"
        sel = _resolve_selection(body.get("selection") or meta.get("selection", {"scope": "whole_body"}))
        datasets = body.get("datasets") or (src["dataset_id"].split(",") if src else [])
        ckpt = body.get("pretrained_ckpt") or TR.latest_checkpoint(body.get("run_id", ""))
        run_id = TR.new_run_id()
        MS.register_run(run_id, name=(body.get("name") or f"{arch}-retrain"),
                        arch=arch, policy_type=AR.ARCHES[arch]["policy_type"],
                        scope=sel["scope"], dataset_id=",".join(datasets), status="queued")
        params = {"run_id": run_id, "arch": arch, "hp": body.get("hp", meta.get("hp", {})),
                  "selection": sel, "datasets": datasets,
                  "split": tuple(body.get("split", [0.8, 0.1, 0.1])),
                  "seed": body.get("seed", 1000), "device": body.get("device", "auto"),
                  "name": body.get("name"), "pretrained_ckpt": ckpt, "resume": False}
        jid = jobs.create("train")
        jobs.run(jid, lambda prog: TR.start_training(params, prog))
        return {"run_id": run_id, "job": jid, "from_checkpoint": ckpt}

    @app.get("/api/train/runs")
    async def train_runs():
        def work():
            runs = MS.list_runs()
            for r in runs:
                p = MS.run_progress(r["run_id"])
                r["step"], r["epoch"] = p["step"], p["epoch"]
                if r.get("created_at") and r.get("updated_at"):
                    r["trained_hours"] = round((r["updated_at"] - r["created_at"]) / 3600.0, 3)
            return runs
        return {"runs": await run_in_threadpool(work)}

    @app.get("/api/train/{run_id}/logs")
    async def train_logs(run_id: str):
        # all persisted terminal + training logs (re-viewable after stop/fail)
        return {"logs": await run_in_threadpool(MS.get_logs, run_id)}

    @app.get("/api/train/{run_id}")
    async def train_run(run_id: str):
        run = await run_in_threadpool(MS.get_run, run_id)
        if not run:
            raise HTTPException(404, "no such run")
        run["metric_keys"] = await run_in_threadpool(MS.metric_keys, run_id)
        run["checkpoints"] = await run_in_threadpool(MS.get_checkpoints, run_id)
        return run

    @app.get("/api/train/{run_id}/series")
    async def train_series(run_id: str, key: str, mode: str = "train"):
        return {"series": await run_in_threadpool(MS.get_series, run_id, key, mode)}

    @app.get("/api/jobs/{jid}")
    async def get_job(jid: str):
        j = jobs.get(jid)
        if not j:
            raise HTTPException(404, "no such job")
        return j

    # ── community queue (owner side) ──────────────────────────────────────────
    # The owner token stays here on the server; the review page (community.html)
    # only ever talks to these local endpoints, so the token never hits a browser.
    def _community(method, path, body=None):
        import urllib.request as U
        if not CFG.COMMUNITY_OWNER_TOKEN or not CFG.COMMUNITY_API:
            raise HTTPException(503, "community queue disabled (set MABEL_COMMUNITY_API and MABEL_COMMUNITY_TOKEN)")
        req = U.Request(CFG.COMMUNITY_API + path, method=method,
                        headers={"X-Owner-Token": CFG.COMMUNITY_OWNER_TOKEN,
                                 "Content-Type": "application/json"})
        data = json.dumps(body).encode() if body is not None else None
        try:
            with U.urlopen(req, data=data, timeout=15) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            raise HTTPException(502, f"community queue unreachable: {e}")

    def _report(jid_c, status, run_id=""):
        try:
            _community("POST", f"/api/community/jobs/{jid_c}/status",
                       {"status": status, "run_id": run_id})
        except Exception:
            pass  # status reporting is best-effort; the run itself is what matters

    @app.get("/api/community/pending")
    async def community_pending():
        if not CFG.COMMUNITY_OWNER_TOKEN or not CFG.COMMUNITY_API:
            return {"enabled": False, "jobs": []}
        data = await run_in_threadpool(_community, "GET",
                                       "/api/community/admin/jobs?status=pending&limit=100")
        return {"enabled": True, "jobs": data.get("jobs", [])}

    def _launch_community_job(job):
        """Map a community proposal → a real training run (whole-body, all cams)."""
        arch = job["arch"]
        sel = _resolve_selection({"scope": "whole_body"})
        datasets = [job["dataset"]]
        hp = {**AR.defaults(arch), **(job.get("hp") or {})}
        issues = AR.validate(arch, hp, sel)
        if any(i["level"] == "error" for i in issues):
            raise HTTPException(400, {"error": "validation failed", "issues": issues})
        run_id = TR.new_run_id()
        name = f"community-{job.get('submitter') or 'anon'}-{job['id']}"
        MS.register_run(run_id, name=name, arch=arch,
                        policy_type=AR.ARCHES[arch]["policy_type"],
                        scope=sel["scope"], dataset_id=",".join(datasets), status="queued")
        params = {"run_id": run_id, "arch": arch, "hp": job.get("hp", {}),
                  "selection": sel, "datasets": datasets, "split": (0.8, 0.1, 0.1),
                  "seed": 1000, "device": "auto", "name": name,
                  "resume": False, "pretrained_ckpt": None}

        def _fn(prog):
            try:
                res = TR.start_training(params, prog)
                _report(job["id"], "done", run_id)
                return res
            except Exception:
                _report(job["id"], "failed", run_id)
                raise
        jid = jobs.create("train")
        jobs.run(jid, _fn)
        return run_id, jid

    @app.post("/api/community/{jid}/approve")
    async def community_approve(jid: int, request: Request):
        _require_auth(request)
        data = await run_in_threadpool(_community, "GET",
                                       "/api/community/admin/jobs?status=pending&limit=100")
        job = next((j for j in data.get("jobs", []) if j["id"] == jid), None)
        if not job:
            raise HTTPException(404, "no such pending job")
        run_id, job_id = _launch_community_job(job)
        await run_in_threadpool(_report, jid, "running", run_id)
        return {"run_id": run_id, "job": job_id}

    @app.post("/api/community/{jid}/reject")
    async def community_reject(jid: int, request: Request):
        _require_auth(request)
        await run_in_threadpool(_community, "POST",
                                f"/api/community/jobs/{jid}/status", {"status": "rejected"})
        return {"id": jid, "status": "rejected"}

    # ── eval + deploy (jobs) ─────────────────────────────────────────────────
    @app.post("/api/eval/{run_id}")
    async def post_eval(run_id: str, request: Request):
        _require_auth(request)
        body = await request.json()
        run = MS.get_run(run_id)
        if not run:
            raise HTTPException(404, "no such run")
        meta = json.loads(run.get("meta_json") or "{}")
        prep = meta.get("prep", {})
        ckpt = body.get("checkpoint") or TR.latest_checkpoint(run_id)
        if not ckpt:
            raise HTTPException(400, "no checkpoint")
        episode = int(body.get("episode", (meta.get("split", {}).get("test") or [0])[0]))
        device = compute.resolve_device(body.get("device", "auto"))
        jid = jobs.create("eval")
        jobs.run(jid, lambda prog: EV.run_open_loop(
            ckpt, prep.get("v21"), prep.get("v30"), episode, device=device,
            max_frames=body.get("max_frames", 200), progress=prog))
        return {"job": jid}

    @app.post("/api/deploy/{run_id}")
    async def post_deploy(run_id: str, request: Request):
        _require_auth(request)
        from . import deploy as DP
        body = await request.json()
        run = MS.get_run(run_id)
        if not run:
            raise HTTPException(404, "no such run")
        meta = json.loads(run.get("meta_json") or "{}")
        prep = meta.get("prep", {})
        sel = meta.get("selection") or _resolve_selection({"scope": "whole_body"})
        ckpt = body.get("checkpoint") or TR.latest_checkpoint(run_id)
        if not ckpt:
            raise HTTPException(400, "no checkpoint")
        device = compute.resolve_device(body.get("device", "auto"))
        jid = jobs.create("deploy")
        jobs.run(jid, lambda prog: DP.run(
            ckpt, prep.get("v30"), sel, device=device,
            control_steps=int(body.get("control_steps", 120)), progress=prog))
        return {"job": jid}

    # ── live metrics WebSocket ───────────────────────────────────────────────
    @app.websocket("/ws/train/{run_id}")
    async def ws_train(ws: WebSocket, run_id: str):
        await ws.accept()
        last_m, last_l = 0, 0
        try:
            while True:
                ev = await run_in_threadpool(MS.events_since, run_id, last_m, last_l)
                last_m, last_l = ev["last_metric_id"], ev["last_log_id"]
                run = await run_in_threadpool(MS.get_run, run_id)
                await ws.send_text(json.dumps({"metrics": ev["metrics"], "logs": ev["logs"],
                                               "checkpoints": ev["checkpoints"],
                                               "status": run.get("status") if run else None}))
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, Exception):
            pass

    # ── experiment tree (runstore: robot_learning/{training,eval}) ──────────────────
    # A durable, arch/dataset-grouped view of every run + its checkpoints, plots
    # and evals — the GUI's "Experiments" browser reads these. Backed by
    # experiment.ExperimentManager over server/runstore.py.
    from . import experiment as EXP, runstore as RS
    from fastapi.responses import FileResponse
    _mgr = EXP.ExperimentManager()

    @app.get("/api/experiments")
    async def api_experiments():
        # grouped by arch → [{run summary}], newest first
        return JSONResponse({"groups": await run_in_threadpool(_mgr.group_by_arch)})

    @app.get("/api/experiments/{run_id}")
    async def api_experiment(run_id: str):
        d = await run_in_threadpool(_mgr.get_run, run_id)
        if d is None:
            raise HTTPException(404, "no such run")
        return JSONResponse(d)

    @app.get("/api/experiments/{run_id}/plot/{name}")
    async def api_experiment_plot(run_id: str, name: str):
        run = RS.RunRegistry().find(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        p = os.path.join(run.layout.plots_dir, os.path.basename(name))
        if not os.path.exists(p):
            raise HTTPException(404, "no such plot")
        return FileResponse(p, media_type="image/png")

    @app.post("/api/experiments/train")
    async def api_experiment_train(request: Request):
        _require_auth(request)
        body = await request.json()
        arch = body["arch"]; ds_id = body["dataset"]
        root = await run_in_threadpool(DR.find, ds_id)
        try:
            return JSONResponse(await run_in_threadpool(
                lambda: _mgr.train(arch, ds_id, root, repo_id=body.get("repo_id", f"mabel/{RS.slug(ds_id)}"),
                                   hp=body.get("hp"), device=body.get("device", "cuda"),
                                   task=body.get("task"), notes=body.get("notes"), detach=True)))
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.post("/api/experiments/{run_id}/resume")
    async def api_experiment_resume(run_id: str, request: Request):
        _require_auth(request)
        body = await request.json() if await request.body() else {}
        try:
            return JSONResponse(await run_in_threadpool(
                lambda: _mgr.resume(run_id, steps=body.get("steps"),
                                    checkpoint=body.get("checkpoint"), detach=True)))
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.post("/api/experiments/{run_id}/eval")
    async def api_experiment_eval(run_id: str, request: Request):
        _require_auth(request)
        body = await request.json() if await request.body() else {}
        try:
            return JSONResponse(await run_in_threadpool(
                lambda: _mgr.evaluate(run_id, checkpoint=body.get("checkpoint"),
                                      kind=body.get("kind", "open_loop"),
                                      episodes=int(body.get("episodes", 1)))))
        except Exception as e:
            raise HTTPException(400, str(e))

    # ── static ───────────────────────────────────────────────────────────────
    if os.path.isdir(CFG.WEBSITE_ASSETS):
        app.mount("/website-assets", StaticFiles(directory=CFG.WEBSITE_ASSETS),
                  name="website-assets")
    app.mount("/", StaticFiles(directory=CFG.WEB_DIR, html=True), name="web")
    return app


app = create_app()
