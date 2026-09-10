"""Verify independently generated IR without filtering rejected examples."""

import pickle

import hypothesis
import pytest
from hypothesis import strategies as st

from niro import ir, verify

from ..strategies import ir as ir_strategies


def assert_verified_without_mutation(module: ir.Module) -> None:
    # Pickle preserves CFG cycles and NaNs, which defeat deepcopy/equality checks.
    before = pickle.dumps(module)
    assert verify.module(module) is module
    assert pickle.dumps(module) == before
    assert verify.module(module) is module
    assert pickle.dumps(module) == before


@hypothesis.given(ir_strategies.modules())
def test_generated_modules_verify(module: ir.Module) -> None:
    assert_verified_without_mutation(module)
    hypothesis.event(f"functions={len(module.functions)}")
    hypothesis.event(f"globals={len(module.globals)}")
    for kind in {
        type(op).__name__
        for function in module.functions
        if function.body is not None
        for op in ir.iter_ops(function.body)
    }:
        hypothesis.event(f"operation={kind}")


@pytest.mark.parametrize("external", [False, True])
@hypothesis.given(data=st.data())
def test_generated_standalone_functions_verify(
    external: bool, data: st.DataObject
) -> None:
    function = data.draw(ir_strategies.functions(external=external))
    assert_verified_without_mutation(ir.Module(functions=[function]))
    if function.body is not None:
        hypothesis.event(f"blocks={len(function.body.blocks)}")
