"""Orchestrator agent: runs the agents in order, decides what happens next
from each result, handles rejections and errors, and logs every action.

The log is written to offchain_store/<CASE>/AGENT_LOG.json after every action
(it is NOT registered on the ledger yet).
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from agents.base import CaseContext, Skip, utc_now
from agents.steps import (ExplanationAgent, FinalResultAgent, IntakeAgent,
                          PredictionAgent, PreprocessingAgent, ProvenanceAgent,
                          ValidationAgent)
from ledger.factory import get_ledger
from storage.offchain import next_case_id, store_json


class Orchestrator:
    def __init__(self):
        self.final_agent = FinalResultAgent()
        self.steps = [IntakeAgent(), PreprocessingAgent(), PredictionAgent(),
                      ValidationAgent(), ExplanationAgent(), self.final_agent]
        self.provenance = ProvenanceAgent()

    # ---------------- public ----------------
    def run_case(self, image_path, case_id=None, root=None, ledger=None, on_event=None) -> dict:
        ledger = ledger or get_ledger()
        case_id = case_id or next_case_id(root)
        work = Path(tempfile.mkdtemp())
        ctx = CaseContext(case_id=case_id, image_path=Path(image_path), root=root,
                          ledger=ledger, work=work, on_event=on_event)
        try:
            for agent in self.steps:
                if ctx.failure:
                    break
                self._run_agent(ctx, agent)

            if ctx.failure:
                # error handling: still record the failure in the provenance chain
                # (only if the ledger works and at least the original is registered)
                if (not ctx.failure.get("ledger") and ctx.parent is not None
                        and ctx.failure["agent"] != self.final_agent.name):
                    self._run_agent(ctx, self.final_agent)
            else:
                self._register_case(ctx)
        finally:
            shutil.rmtree(work, ignore_errors=True)

        try:
            artifacts = ledger.history(case_id)
        except Exception:
            artifacts = list(ctx.arts.values())

        result = ctx.final or {
            "case_id": case_id, "predicted_class": None, "confidence": None,
            "status": "ERROR", "status_fr": "Erreur de traitement",
            "reasons": [f"agent_error:{ctx.failure['agent']}"] if ctx.failure else [],
            "model_version": ctx.model_version,
            "disclaimer": "Prototype pédagogique, pas un diagnostic médical.",
        }
        return {"case_id": case_id, "ok": ctx.failure is None, "result": result,
                "artifacts": artifacts, "agent_log": ctx.log, "failure": ctx.failure}

    # ---------------- internals ----------------
    def _emit(self, ctx, event, **data):
        if ctx.on_event:
            try:
                ctx.on_event({"event": event, "case_id": ctx.case_id, **data})
            except Exception:
                pass

    def _log(self, ctx, agent, action, status, started, t0, detail=None):
        entry = {"seq": len(ctx.log) + 1, "agent": agent, "action": action,
                 "status": status, "started_at": started,
                 "duration_s": round(time.perf_counter() - t0, 3),
                 "detail": detail or {}}
        ctx.log.append(entry)
        try:  # a logging problem must never break the pipeline
            store_json(ctx.case_id, {"case_id": ctx.case_id, "entries": ctx.log},
                       "AGENT_LOG", ctx.root)
        except OSError:
            pass
        self._emit(ctx, "log", entry=entry)

    def _fail(self, ctx, agent, action, exc, started, t0, ledger_failure=False):
        info = {"agent": agent, "type": type(exc).__name__, "message": str(exc)[:300],
                "ledger": ledger_failure}
        if ctx.failure is None:
            ctx.failure = info
        self._log(ctx, agent, action, "error", started, t0,
                  {"error": info["type"], "message": info["message"]})

    def _run_agent(self, ctx, agent):
        self._emit(ctx, "agent_started", agent=agent.name)
        started, t0 = utc_now(), time.perf_counter()
        try:
            specs, detail = agent.run(ctx)
        except Skip as s:
            self._log(ctx, agent.name, agent.action, "skipped", started, t0, {"reason": s.reason})
            return
        except Exception as exc:
            self._fail(ctx, agent.name, agent.action, exc, started, t0)
            return
        self._log(ctx, agent.name, agent.action, "ok", started, t0, detail)

        for spec in specs:
            self._emit(ctx, "agent_started", agent=self.provenance.name)
            started, t0 = utc_now(), time.perf_counter()
            action = f"hash and register {spec.artifact_type}"
            try:
                d = self.provenance.register(ctx, spec)
            except Exception as exc:
                self._fail(ctx, self.provenance.name, action, exc, started, t0,
                           ledger_failure=True)
                return
            self._log(ctx, self.provenance.name, action, "ok", started, t0, d)

    def _register_case(self, ctx):
        started, t0 = utc_now(), time.perf_counter()
        action = "register case summary"
        try:
            d = self.provenance.register_case(ctx)
        except Exception as exc:
            self._fail(ctx, self.provenance.name, action, exc, started, t0, ledger_failure=True)
            return
        if d is None:
            self._log(ctx, self.provenance.name, action, "skipped", started, t0,
                      {"reason": "no prediction or ledger without case summary"})
        else:
            self._log(ctx, self.provenance.name, action, "ok", started, t0, d)
