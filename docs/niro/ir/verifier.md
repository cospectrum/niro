`ir.verify(module)` returns an `ir.VerifiedModule`, the input type required by
`niro.to_mlir`. `ModuleBuilder.verify()` provides the same check for a builder,
and `niro.from_onnx` verifies its result automatically.

The returned object is the original module. Verify it again after making changes.

::: niro.ir.verifier
    options:
      filters: public
      heading_level: 1
      show_root_full_path: true
      show_root_heading: true
