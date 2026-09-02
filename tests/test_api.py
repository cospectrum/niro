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
        ir.Region,
    )


def test_exports_import_onnx() -> None:
    import niro
    from niro import import_onnx
    from niro.onnx import import_onnx as onnx_import_onnx

    assert import_onnx is onnx_import_onnx
    assert niro.import_onnx is onnx_import_onnx


def test_exports_mlir() -> None:
    import niro
    from niro import export_mlir
    from niro.mlir import export_mlir as mlir_export_mlir

    assert export_mlir is mlir_export_mlir
    assert niro.export_mlir is mlir_export_mlir


def test_exports_mlir_output() -> None:
    import niro
    from niro import format_mlir, write_mlir
    from niro.mlir import format_mlir as mlir_format_mlir
    from niro.mlir import write_mlir as mlir_write_mlir

    assert format_mlir is mlir_format_mlir
    assert write_mlir is mlir_write_mlir
    assert niro.format_mlir is mlir_format_mlir
    assert niro.write_mlir is mlir_write_mlir
