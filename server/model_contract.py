"""Validation for portable user-provided thoth-model/v1 TorchScript classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_SCHEMA = "thoth-model/v1"


class ModelContractError(ValueError):
    pass


def _positive(value: object, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelContractError(f"{field} must be a positive integer") from exc
    if result < 1:
        raise ModelContractError(f"{field} must be a positive integer")
    return result


def _names(value: object) -> list[str]:
    if isinstance(value, str):
        value = value.split(",")
    return [" ".join(str(item).split()) for item in value if str(item).strip()] if isinstance(value, (list, tuple)) else []


def normalize_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != MODEL_SCHEMA:
        raise ModelContractError(f"metadata.schema must be {MODEL_SCHEMA}")
    name, version = str(value.get("name") or "").strip(), str(value.get("version") or "").strip()
    if not name or not version:
        raise ModelContractError("model name and version are required")
    raw_inputs = value.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ModelContractError("inputs must be a non-empty ordered list")
    inputs, seen = [], set()
    for index, item in enumerate(raw_inputs):
        if not isinstance(item, dict):
            raise ModelContractError(f"inputs[{index}] must be an object")
        sensor = str(item.get("sensor") or "").lower()
        if sensor not in {"radar", "csi"} or sensor in seen:
            raise ModelContractError("v1 accepts radar and/or csi at most once")
        seen.add(sensor)
        representation = str(item.get("representation") or "").lower()
        allowed = {"raw_adc", "fft_power"} if sensor == "radar" else {"iq", "magnitude_phase"}
        if representation not in allowed:
            raise ModelContractError(f"unsupported {sensor} representation")
        count_key = "frames" if sensor == "radar" else "samples"
        shape = item.get("shape") or item.get("expected_shape")
        if not isinstance(shape, list) or not shape:
            raise ModelContractError(f"inputs[{index}].shape is required")
        fit = str(item.get("fit") or item.get("padding_truncation") or "left_pad_latest")
        if fit not in {"left_pad_latest", "right_pad_earliest"}:
            raise ModelContractError("unsupported padding/truncation policy")
        normalization = item.get("normalization") or {"kind": "none"}
        if isinstance(normalization, str):
            normalization = {"kind": normalization}
        kind = str(normalization.get("kind") or "none") if isinstance(normalization, dict) else "invalid"
        if kind not in {"none", "zscore", "minmax"}:
            raise ModelContractError("normalization kind must be none, zscore, or minmax")
        normalized = {"sensor": sensor, "representation": representation, count_key: _positive(item.get(count_key), count_key), "shape": [_positive(part, "shape") for part in shape], "fit": fit, "normalization": {**normalization, "kind": kind}}
        if sensor == "csi":
            normalized["receivers"] = item.get("receivers", item.get("receiver_selection", []))
            normalized["subcarriers"] = item.get("subcarriers", item.get("subcarrier_selection", []))
        inputs.append(normalized)
    output = value.get("output") if isinstance(value.get("output"), dict) else {"kind": value.get("output_kind")}
    kind, path = str(output.get("kind") or ""), output.get("path") or []
    if kind not in {"logits", "probabilities"} or not isinstance(path, list) or any(not isinstance(selector, (str, int)) for selector in path):
        raise ModelContractError("output must define logits/probabilities and a valid selector path")
    return {"schema": MODEL_SCHEMA, "name": name, "version": version, "inputs": inputs, "output": {"kind": kind, "path": path}, "class_names": _names(value.get("class_names") or value.get("labels"))}


def validate_torchscript(path: Path, metadata: object) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ModelContractError("CPU PyTorch is unavailable on the validation server") from exc
    normalized = normalize_metadata(metadata)
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        tensors = [torch.zeros(tuple(item["shape"]), dtype=torch.float32) for item in normalized["inputs"]]
        with torch.inference_mode():
            output = model(*tensors)
        for selector in normalized["output"]["path"]:
            output = output[selector]
        values = output.detach().cpu().float().reshape(-1)
    except Exception as exc:
        raise ModelContractError(f"TorchScript dry run failed: {exc}") from exc
    if values.numel() < 1 or not bool(torch.isfinite(values).all()):
        raise ModelContractError("selected output must contain at least one finite class value")
    embedded = []
    for attribute in ("class_names", "labels"):
        try:
            embedded = _names(getattr(model, attribute))
        except Exception:
            embedded = []
        if embedded:
            break
    names = embedded or normalized["class_names"]
    if not names:
        raise ModelContractError("class_names are required when the archive has no embedded labels")
    if values.numel() == 1 and len(names) == 2:
        normalized["binary_output"] = True
    elif len(names) != values.numel():
        raise ModelContractError(f"class_names has {len(names)} entries but output has {values.numel()}")
    normalized["class_names"], normalized["output_dimension"] = names, int(values.numel())
    return normalized
