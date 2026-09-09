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
        ir.BlockBuilder,
        ir.Function,
        ir.FunctionBuilder,
        ir.FunctionType,
        ir.Module,
        ir.VerifiedModule,
        ir.ModuleBuilder,
        ir.Region,
    )


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
