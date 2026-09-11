"""Export real model architectures with random weights and compare imported IR."""

from collections.abc import Iterator
from pathlib import Path

import onnx
import pytest
import torch
import torchvision.models
import transformers

import niro
from tests.helpers.onnx import assert_niro_matches_onnx


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18",
        "mobilenet_v2",
        "squeezenet1_1",
        "shufflenet_v2_x0_5",
        pytest.param(
            "efficientnet_b0",
            marks=pytest.mark.xfail(
                strict=True,
                raises=pytest.RaisesExc(
                    TypeError, match="^mul operands and result must have the same type$"
                ),
                reason="ONNX Mul broadcasting is not supported by native Niro Mul",
            ),
        ),
    ],
)
def test_import_torchvision_model(model_name: str, tmp_path: Path) -> None:
    model = torchvision.models.get_model(model_name, weights=None).eval()
    _assert_model_import(model, torch.randn(1, 3, 64, 64), tmp_path)


@pytest.mark.parametrize("model_name", ["bert", "gpt2"])
@pytest.mark.xfail(
    strict=True,
    raises=pytest.RaisesExc(
        TypeError, match="^matmul operands must be rank-two tensors$"
    ),
    reason="Transformer attention requires ONNX MatMul with rank greater than two",
)
def test_import_transformers_model(model_name: str, tmp_path: Path) -> None:
    model = _bert() if model_name == "bert" else _gpt2()
    model.eval()
    model.set_attn_implementation("eager")
    _assert_model_import(model, torch.arange(8).unsqueeze(0), tmp_path)


@pytest.fixture(autouse=True, scope="module")
def torch_settings() -> Iterator[None]:
    """Keep CPU exports bounded and restore the caller's threads and random state."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            yield
    finally:
        torch.set_num_threads(threads)


def _assert_model_import(
    model: torch.nn.Module, inputs: torch.Tensor, root: Path
) -> None:
    model_path = root / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (inputs,),
            model_path,
            dynamo=True,
            opset_version=18,
            external_data=False,
            verbose=False,
        )
    original = onnx.shape_inference.infer_shapes(
        onnx.load(model_path), strict_mode=True
    )
    onnx.checker.check_model(original, full_check=True)

    module = niro.from_onnx(original)

    assert_niro_matches_onnx(module, original.graph)


def _bert() -> transformers.BertModel:
    return transformers.BertModel(
        transformers.BertConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            return_dict=False,
        )
    )


def _gpt2() -> transformers.GPT2Model:
    return transformers.GPT2Model(
        transformers.GPT2Config(
            vocab_size=128,
            n_embd=32,
            n_layer=1,
            n_head=4,
            n_positions=32,
            bos_token_id=0,
            eos_token_id=1,
            use_cache=False,
            return_dict=False,
        )
    )
