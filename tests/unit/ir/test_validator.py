from niro import ir


def test_validate_returns_same_module() -> None:
    module = ir.Module(
        functions=[ir.Function("external", ir.FunctionType((), ()))],
        globals=[ir.Global("value", ir.ScalarType.I32, 1)],
    )

    assert ir.validate(module) is module
