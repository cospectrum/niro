import pytest

from niro import ir


def test_validate_returns_same_module() -> None:
    module = ir.Module(
        functions=[ir.Function("external", ir.FunctionType((), ()))],
        globals=[ir.Global("value", ir.ScalarType.I32, 1)],
    )

    assert ir.validate(module) is module


def test_global_validation_is_explicit() -> None:
    module = ir.Module(globals=[ir.Global("", ir.ScalarType.I32, 1)])

    with pytest.raises(ValueError, match="global name cannot be empty"):
        ir.validate(module)


@pytest.mark.parametrize("branch", ["then_region", "else_region"])
def test_validate_visits_nested_regions_in_later_functions_and_blocks(
    branch: str,
) -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    invalid = ir.UnknownOp("", (), ())
    nested = ir.If((), condition, ir.Region(), ir.Region())
    getattr(nested, branch).blocks.append(ir.Block(operations=[invalid]))
    outer = ir.If(
        (), condition, ir.Region([ir.Block(operations=[nested])]), ir.Region()
    )
    module = ir.Module(
        functions=[
            ir.Function("external", ir.FunctionType((), ())),
            ir.Function(
                "main",
                ir.FunctionType((), ()),
                ir.Region(
                    [
                        ir.Block(),
                        ir.Block(operations=[outer]),
                    ]
                ),
            ),
        ]
    )

    with pytest.raises(ValueError, match="UnknownOp name cannot be empty"):
        ir.validate(module)


def test_validate_checks_duplicate_operation_results() -> None:
    value = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
    operation = ir.Call("callee", (), (value, value))
    module = ir.Module(
        functions=[
            ir.Function(
                "main",
                ir.FunctionType((), ()),
                ir.Region([ir.Block(operations=[operation])]),
            )
        ]
    )

    with pytest.raises(ValueError, match="operation should produce unique values"):
        ir.validate(module)
