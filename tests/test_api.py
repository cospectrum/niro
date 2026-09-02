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
    from niro.frontends.onnx import import_onnx as frontend_import_onnx

    assert import_onnx is frontend_import_onnx
    assert niro.import_onnx is frontend_import_onnx


def test_exports_lower_to_mlir() -> None:
    import niro
    from niro import lower_to_mlir
    from niro.backends.mlir import lower_to_mlir as backend_lower_to_mlir

    assert lower_to_mlir is backend_lower_to_mlir
    assert niro.lower_to_mlir is backend_lower_to_mlir


def test_exports_mlir_output() -> None:
    import niro
    from niro import format_mlir, write_mlir
    from niro.backends.mlir import format_mlir as backend_format_mlir
    from niro.backends.mlir import write_mlir as backend_write_mlir

    assert format_mlir is backend_format_mlir
    assert write_mlir is backend_write_mlir
    assert niro.format_mlir is backend_format_mlir
    assert niro.write_mlir is backend_write_mlir
