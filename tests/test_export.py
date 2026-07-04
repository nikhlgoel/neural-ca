import numpy as np
import onnxruntime as ort
import torch

from neural_ca.config import Config, DataConfig, ModelConfig
from neural_ca.eval.export import DeltaCore, export_onnx
from neural_ca.model.nca import NCA
from neural_ca.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = NCA(channels=16, hidden=32)
    cfg = Config(model=ModelConfig(channels=16, hidden=32), data=DataConfig(target="heart"))
    path = tmp_path / "m.pt"
    save_checkpoint(model, cfg, path)
    loaded = load_checkpoint(path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_onnx_export_matches_torch(tmp_path) -> None:
    model = NCA(channels=16, hidden=32)
    out = tmp_path / "m.onnx"
    export_onnx(model, out, size=24)

    state = torch.randn(1, 16, 24, 24)
    with torch.no_grad():
        torch_delta = DeltaCore(model).eval()(state).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_delta = sess.run(["delta"], {"state": state.numpy()})[0]
    assert np.abs(torch_delta - onnx_delta).max() < 1e-4
