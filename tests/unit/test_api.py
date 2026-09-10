def test_exports_ir() -> None:
    import niro
    from niro import ir

    assert niro.ir is ir


def test_exports_rewrite() -> None:
    import niro
    from niro import rewrite

    assert niro.rewrite is rewrite


def test_index_has_a_separate_namespace() -> None:
    import niro
    from niro import index, ir

    assert niro.index is index
    for name in index.__all__:
        assert hasattr(index, name)
        assert not hasattr(ir, name)


def test_ir_exports_operations() -> None:
    from niro import ir

    _ = (
        ir.Add,
        ir.Branch,
        ir.Call,
        ir.CondBranch,
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


def test_builder_and_verify_have_separate_namespaces() -> None:
    import niro
    from niro import builder, ir, verify

    assert niro.builder is builder
    assert niro.verify is verify
    module = builder.ModuleBuilder()
    assert verify.module(module.raw) is module.raw
    assert module.verify() is module.raw
    for name in ("ModuleBuilder", "FunctionBuilder", "BlockBuilder", "verify"):
        assert not hasattr(ir, name)
        if name != "verify":
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


def test_pass_interface_accepts_an_indexed_verified_module_pass() -> None:
    import niro
    from niro import index, ir, passes, verify

    pass_: passes.Pass = passes.noop
    module = verify.module(ir.Module())
    indexed = index.index_module(module)
    assert pass_(indexed) is indexed
    assert indexed.module is module
    assert niro.passes is passes
    assert passes.__all__ == ["Pass", "noop"]
