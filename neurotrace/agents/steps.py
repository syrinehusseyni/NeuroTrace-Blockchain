"""One agent per pipeline step. Each agent does ONE job and reports to the
orchestrator; hashing + ledger registration is done by the ProvenanceAgent."""
from __future__ import annotations

import config
from agents.base import Agent, ArtifactSpec, CaseContext, Skip
from ai.explain import explain
from ai.predict import predict
from hashing.artifacts import create_artifact
from hashing.sha256 import sha256_file
from preprocessing.preprocess import preprocess_image
from storage.offchain import resolve, store_file, store_json
from validation.validate import REJECTED, validate


class IntakeAgent(Agent):
    name, action = "IntakeAgent", "store original image off-chain"

    def run(self, ctx: CaseContext):
        ref = store_file(ctx.case_id, ctx.image_path, "ORIGINAL_IMAGE", ctx.root)
        ctx.stored_original = resolve(ref, ctx.root)
        return [ArtifactSpec("ORIGINAL_IMAGE", ref)], {"storage_ref": ref}


class PreprocessingAgent(Agent):
    name, action = "PreprocessingAgent", "validate and preprocess the image"

    def run(self, ctx: CaseContext):
        pre, model_img = preprocess_image(ctx.stored_original, ctx.case_id, out_dir=ctx.work)
        ctx.pre, ctx.model_img = pre, model_img
        detail = {"valid": pre["valid"], "errors": pre["errors"],
                  "quality_flags": pre["quality_flags"]}
        if not pre["valid"]:
            detail["outcome"] = "image_rejected"
            return [], detail
        ref = store_file(ctx.case_id, pre["preprocessed_path"], "PREPROCESSED_IMAGE", ctx.root)
        return [ArtifactSpec("PREPROCESSED_IMAGE", ref)], detail


class PredictionAgent(Agent):
    name, action = "PredictionAgent", "classify the MRI image"

    def run(self, ctx: CaseContext):
        if not ctx.pre or not ctx.pre["valid"]:
            raise Skip("image invalid")
        pred = predict(ctx.model_img)
        ctx.pred, ctx.model_version = pred, pred["model_version"]
        ref = store_json(ctx.case_id, pred, "PREDICTION", ctx.root)
        return ([ArtifactSpec("PREDICTION", ref, ctx.model_version)],
                {"predicted_class": pred["predicted_class"],
                 "confidence": round(pred["confidence"], 4),
                 "model_version": ctx.model_version})


class ValidationAgent(Agent):
    name, action = "ValidationAgent", "apply the validation rules"

    def run(self, ctx: CaseContext):
        val = validate(ctx.pre, ctx.pred)
        val["preprocessing"] = {k: ctx.pre.get(k) for k in
                                ("source_file", "original", "operations",
                                 "quality", "quality_flags", "errors")}
        ctx.val = val
        ref = store_json(ctx.case_id, val, "VALIDATION", ctx.root)
        return ([ArtifactSpec("VALIDATION", ref, ctx.model_version)],
                {"status": val["status"], "reasons": val["reasons"]})


class ExplanationAgent(Agent):
    name, action = "ExplanationAgent", "generate the Grad-CAM explanation"

    def run(self, ctx: CaseContext):
        if not ctx.pred:
            raise Skip("no prediction")
        if ctx.val["status"] == REJECTED:
            raise Skip("prediction rejected by validation")
        exp = explain(ctx.model_img, ctx.pred, ctx.case_id, ctx.work)
        ref = store_file(ctx.case_id, exp["overlay_path"], "EXPLANATION", ctx.root)
        return [ArtifactSpec("EXPLANATION", ref, ctx.model_version)], {"method": exp["method"]}


class FinalResultAgent(Agent):
    name, action = "FinalResultAgent", "assemble the final result"

    def run(self, ctx: CaseContext):
        pred, val, fail = ctx.pred, ctx.val, ctx.failure
        if fail:
            status, status_fr = "ERROR", "Erreur de traitement"
            reasons = [f"agent_error:{fail['agent']}"]
        else:
            status, status_fr, reasons = val["status"], val["status_fr"], val["reasons"]
        final = {
            "case_id": ctx.case_id,
            "predicted_class": pred["predicted_class"] if pred else None,
            "confidence": pred["confidence"] if pred else None,
            "status": status, "status_fr": status_fr,
            "reasons": reasons, "model_version": ctx.model_version,
            "disclaimer": "Prototype pédagogique, pas un diagnostic médical.",
        }
        ctx.final = final
        ref = store_json(ctx.case_id, final, "FINAL_RESULT", ctx.root)
        return [ArtifactSpec("FINAL_RESULT", ref, ctx.model_version)], {"status": status}


class ProvenanceAgent:
    """Hashes each artifact (SHA-256) and registers it on the ledger."""
    name = "ProvenanceAgent"

    def register(self, ctx: CaseContext, spec: ArtifactSpec) -> dict:
        art = ctx.ledger.record(create_artifact(
            ctx.case_id, spec.artifact_type, spec.storage_ref,
            ctx.parent, spec.model_version, ctx.root))
        ctx.parent = art["artifact_id"]
        ctx.arts[spec.artifact_type] = art
        return {"artifact_id": art["artifact_id"], "sha256": art["sha256"],
                "parent": art.get("parent_artifact_id")}

    def register_case(self, ctx: CaseContext) -> dict | None:
        fn = getattr(ctx.ledger, "register_case", None)
        if not fn or not ctx.pred:
            return None
        fn(case_id=ctx.case_id,
           mri_hash=ctx.arts["ORIGINAL_IMAGE"]["sha256"],
           prediction_hash=ctx.arts["PREDICTION"]["sha256"],
           model_hash=sha256_file(config.CHECKPOINT_PATH),
           predicted_class=ctx.pred["predicted_class"],
           confidence=ctx.pred["confidence"],
           model_version=ctx.model_version,
           status=ctx.val["status"])
        return {"status": ctx.val["status"]}
