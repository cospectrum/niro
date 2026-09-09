def test_exports_ir() -> None:
    import niro
    from niro import ir

    assert niro.ir is ir


def test_ir_exports_operations() -> None:
    from niro import ir

    _ = (
        ir.Add,
        ir.Call,
        ir.Const,
        ir.If,
        ir.MatMul,
        ir.Mul,
        ir.Return,
        ir.Transpose,
        ir.UnknownOp,
        ir.Yield,
    )


def test_ir_exports_program() -> None:
    from niro import ir

    _ = (
        ir.Block,
        ir.Function,
        ir.FunctionType,
        ir.Module,
        ir.VerifiedModule,
        ir.Region,
    )


def test_builder_and_verifier_have_separate_namespaces() -> None:
    import niro
    from niro import builder, ir, verifier

    assert niro.builder is builder
    assert niro.verifier is verifier
    module = builder.ModuleBuilder()
    assert verifier.verify(module.raw) is module.raw
    assert module.verify() is module.raw
    for name in ("ModuleBuilder", "FunctionBuilder", "BlockBuilder", "verify"):
        assert not hasattr(ir, name)
        assert not hasattr(niro, name)


def test_exports_from_onnx() -> None:
    import niro
    from niro import onnx

    assert niro.from_onnx is onnx.from_onnx


def test_exports_to_mlir() -> None:
    import niro
    from niro import mlir

    assert niro.to_mlir is mlir.to_mlir


def test_exports_mlir_output() -> None:
    import niro
    from niro import mlir

    assert niro.format_mlir is mlir.format_mlir
    assert niro.write_mlir is mlir.write_mlir
