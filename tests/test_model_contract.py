import pytest

torch = pytest.importorskip("torch", reason="CPU PyTorch not installed (optional dependency)")

from server.model_contract import ModelContractError, normalize_metadata, validate_torchscript


class Classifier(torch.nn.Module):
    def forward(self, radar, csi):
        return (torch.stack((radar.sum(), csi.sum())),)


def contract(schema="whispy-model/v1"):
    return {
        "schema": schema,
        "name": "portable",
        "version": "2",
        "inputs": [
            {"sensor": "radar", "representation": "raw_adc", "frames": 10, "shape": [4], "fit": "left_pad_latest", "normalization": {"kind": "none"}},
            {"sensor": "csi", "representation": "iq", "samples": 3, "shape": [6], "receivers": [0], "subcarriers": [1, 2], "fit": "right_pad_earliest", "normalization": {"kind": "zscore", "mean": 0, "std": 1}},
        ],
        "output": {"kind": "logits", "path": [0]},
        "class_names": ["still", "moving"],
    }


def test_complete_v1_metadata_and_dry_run(tmp_path):
    artifact = tmp_path / "model.pt"
    torch.jit.save(torch.jit.trace(Classifier(), (torch.zeros(4), torch.zeros(6))), str(artifact))
    result = validate_torchscript(artifact, contract())
    assert result["output_dimension"] == 2
    assert [item["sensor"] for item in result["inputs"]] == ["radar", "csi"]


def test_minute_execution_and_e2_representations():
    meta = contract()
    meta["inputs"] = [
        {"sensor": "radar", "representation": "e2_maps", "frames": 50, "shape": [1, 50, 2, 24, 24], "fit": "left_pad_latest", "normalization": {"kind": "none"}},
        {"sensor": "csi", "representation": "e2_grid", "samples": 128, "shape": [1, 128, 52], "fit": "left_pad_latest", "normalization": {"kind": "none"}},
    ]
    meta["execution"] = "minute"
    meta["aggregation"] = {"kind": "top2", "threshold": 0.7}
    result = normalize_metadata(meta)
    assert result["execution"] == "minute"
    assert result["aggregation"] == {"kind": "top2", "threshold": 0.7}
    assert [item["representation"] for item in result["inputs"]] == ["e2_maps", "e2_grid"]


def test_legacy_thoth_model_schema_accepted():
    """Pre-rename ``thoth-model/v1`` metadata validates and normalizes to
    the canonical ``whispy-model/v1`` schema."""
    result = normalize_metadata(contract(schema="thoth-model/v1"))
    assert result["schema"] == "whispy-model/v1"


def test_rejects_duplicate_inputs_and_wrong_classes(tmp_path):
    duplicate = contract()
    duplicate["inputs"] = [duplicate["inputs"][0], duplicate["inputs"][0]]
    with pytest.raises(ModelContractError, match="at most once"):
        normalize_metadata(duplicate)

    artifact = tmp_path / "model.pth"
    torch.jit.save(torch.jit.trace(Classifier(), (torch.zeros(4), torch.zeros(6))), str(artifact))
    bad = contract()
    bad["class_names"] = ["only-one"]
    with pytest.raises(ModelContractError, match="class_names"):
        validate_torchscript(artifact, bad)
