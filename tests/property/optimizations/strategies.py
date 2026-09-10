"""Bounded verified modules for structural properties shared by optimization passes."""

from collections.abc import Sequence

import hypothesis
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from niro import ir, verify


@st.composite
def mixed_modules(draw: DrawFn) -> ir.VerifiedModule:
    """Generate mixed tensor programs; calls may be recursive or external.

    Signatures are declared before bodies so calls can target any definition.
    The first function stays a leaf, providing an eligible inlining target even
    when other functions form cycles. Branch nesting is limited to two levels.
    These modules need not terminate and are intended for validity checks only.
    """
    shape = draw(
        st.one_of(
            st.none(),
            st.lists(st.one_of(st.none(), st.integers(0, 3)), max_size=3).map(tuple),
        )
    )
    tensor = ir.TensorType(
        draw(
            st.sampled_from(
                [
                    ir.ScalarType.I32,
                    ir.ScalarType.I64,
                    ir.ScalarType.F32,
                    ir.ScalarType.F64,
                ]
            )
        ),
        shape,
    )
    functions = [
        ir.Function(
            f"f{index}",
            ir.FunctionType(
                (tensor, ir.ScalarType.BOOL), (tensor,) * draw(st.integers(0, 2))
            ),
            attributes={"tag": index},
        )
        for index in range(draw(st.integers(2, 4)))
    ]
    external = ir.Function("external", ir.FunctionType((tensor,), (tensor,)))
    for index, function in enumerate(functions):
        populate(draw, function, tensor, [] if index == 0 else functions, external)
    hypothesis.event(f"rank={'unknown' if shape is None else len(shape)}")
    return verify.module(
        ir.Module(
            functions=list(draw(st.permutations([*functions, external]))),
            globals=[
                ir.Global("global", ir.ScalarType.I32, draw(st.integers(-10, 10)))
            ],
            attributes={"tag": "generated"},
        )
    )


def populate(
    draw: DrawFn,
    function: ir.Function,
    tensor: ir.TensorType,
    callees: Sequence[ir.Function],
    external: ir.Function,
) -> None:
    """Fill a fresh declaration with typed operations, captures and fresh IDs."""
    next_id = draw(st.integers(0, 10))

    def fresh(type_: ir.Type) -> ir.Value:
        nonlocal next_id
        result = ir.Value(ir.ValueId(next_id), type_)
        next_id += draw(st.integers(1, 3))
        return result

    argument, condition = fresh(tensor), fresh(ir.ScalarType.BOOL)

    def body(operand: ir.Value, depth: int) -> tuple[list[ir.Op], ir.Value]:
        operations: list[ir.Op] = []
        choices = ["add", "mul", "transpose", "external", "unknown", "global", "const"]
        if (
            tensor.shape is not None
            and len(tensor.shape) == 2
            and tensor.shape[0] == tensor.shape[1]
        ):
            choices.append("matmul")
        if callees:
            choices.append("call")
        if depth < 2:
            choices.append("if")
        for kind in draw(st.lists(st.sampled_from(choices), min_size=1, max_size=3)):
            result = fresh(tensor)
            if kind in ("add", "mul"):
                constructor = ir.Add if kind == "add" else ir.Mul
                operations.append(constructor(result, operand, operand))
            elif kind == "matmul":
                operations.append(ir.MatMul(result, operand, operand))
            elif kind == "transpose":
                rank = (
                    tensor.rank if tensor.rank is not None else draw(st.integers(0, 3))
                )
                permutation = tuple(draw(st.permutations(tuple(range(rank)))))
                inverse = tuple(permutation.index(axis) for axis in range(rank))
                intermediate = fresh(
                    ir.infer.transpose_result_type(tensor, permutation)
                )
                operations.extend(
                    [
                        ir.Transpose(intermediate, operand, permutation),
                        ir.Transpose(result, intermediate, inverse),
                    ]
                )
            elif kind == "external":
                operations.append(ir.Call(external.name, (operand,), (result,)))
            elif kind == "call":
                callee = draw(st.sampled_from(callees))
                results = tuple(fresh(type_) for type_ in callee.type.outputs)
                operations.append(ir.Call(callee.name, (operand, condition), results))
                if results:
                    operand = results[-1]
                continue
            elif kind == "if":
                branches = []
                for _ in range(2):
                    branch_ops, returned = body(operand, depth + 1)
                    branches.append(
                        ir.Region(
                            [ir.Block(operations=[*branch_ops, ir.Yield((returned,))])]
                        )
                    )
                operations.append(ir.If((result,), condition, *branches))
            elif kind in ("global", "const"):
                scalar = fresh(ir.ScalarType.I32)
                op = (
                    ir.GetGlobal("global", scalar)
                    if kind == "global"
                    else ir.Const(scalar, draw(st.integers(-10, 10)))
                )
                operations.extend(
                    [op, ir.UnknownOp("mix", (operand, scalar), (result,))]
                )
            else:
                operations.append(ir.UnknownOp("unknown", (operand,), (result,)))
            operand = result
        return operations, operand

    operations, returned = body(argument, 0)
    # Exercise inlining on every module, even when generated call edges form cycles.
    if callees:
        leaf = callees[0]
        results = tuple(fresh(type_) for type_ in leaf.type.outputs)
        operations.append(ir.Call(leaf.name, (returned, condition), results))
        if results:
            returned = results[-1]
    operations.append(ir.Return((returned,) * len(function.type.outputs)))
    function.body = ir.Region([ir.Block((argument, condition), operations)])
