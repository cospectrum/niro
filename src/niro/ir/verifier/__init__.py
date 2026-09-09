"""Verification of completed Niro IR modules.

`ir.verify(module)` returns an `ir.VerifiedModule`, the input type required by
`niro.to_mlir`. `ModuleBuilder.verify()` provides the same check for a builder,
and `niro.from_onnx` verifies its result automatically.

The returned object is the original module. Verify it again after making changes.
"""

from niro.ir.verifier.program import verify

__all__ = ["verify"]
