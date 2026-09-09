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
    from niro import from_onnx
    from niro.onnx import from_onnx as onnx_from_onnx

    assert from_onnx is onnx_from_onnx
    assert niro.from_onnx is onnx_from_onnx


def test_exports_to_mlir() -> None:
    import niro
    from niro import to_mlir
    from niro.mlir import to_mlir as mlir_to_mlir

    assert to_mlir is mlir_to_mlir
    assert niro.to_mlir is mlir_to_mlir


def test_exports_mlir_output() -> None:
    import niro
    from niro import format_mlir, write_mlir
    from niro.mlir import format_mlir as mlir_format_mlir
    from niro.mlir import write_mlir as mlir_write_mlir

    assert format_mlir is mlir_format_mlir
    assert write_mlir is mlir_write_mlir
    assert niro.format_mlir is mlir_format_mlir
    assert niro.write_mlir is mlir_write_mlir
