"""Lower Niro IR to high-level MLIR using xDSL."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import cast, overload

from xdsl.dialects import arith, builtin, func, ml_program, scf, tensor
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue

from niro import ir
from niro.builder import ENTRY_POINT_ATTR


@dataclass(slots=True)
class Ctx:
    block: Block
    values: dict[ir.ValueId, SSAValue]
    globals: list[Operation]
    function_name: str


def lower_to_mlir(module: ir.Module) -> builtin.ModuleOp:
    """Lower a Niro module to a verified, high-level MLIR module."""
    entry_point = module.attributes.get(ENTRY_POINT_ATTR)
    if entry_point is None:
        raise ValueError("Niro module must have an entry point")
    if not isinstance(entry_point, str):
        raise TypeError("Niro entry point must be a string")
    if not any(
        function.name == entry_point and function.body is not None
        for function in module.functions
    ):
        raise ValueError(f"invalid Niro entry point: {entry_point!r}")

    globals_: list[Operation] = []
    functions = [
        _lower_function(function, entry_point, globals_)
        for function in module.functions
    ]
    attributes = _lower_attributes(module.attributes, exclude={ENTRY_POINT_ATTR})
    result = builtin.ModuleOp([*globals_, *functions], attributes=attributes)
    result.verify()
    return result


def _lower_function(
    function: ir.Function,
    entry_point: str,
    globals_: list[Operation],
) -> func.FuncOp:
    inputs = [_lower_type(value_type) for value_type in function.type.inputs]
    outputs = [_lower_type(value_type) for value_type in function.type.outputs]
    if function.body is None:
        result = func.FuncOp.external(function.name, inputs, outputs)
        result.attributes.update(_lower_attributes(function.attributes))
        return result
    (niro_block,) = function.body.blocks
    block = Block(arg_types=inputs)
    ctx = Ctx(
        block=block,
        values={
            argument.id: block_argument
            for argument, block_argument in zip(
                function.arguments, block.args, strict=True
            )
        },
        globals=globals_,
        function_name=function.name,
    )
    _lower_operations(ctx, niro_block.operations)
    result = func.FuncOp(
        function.name,
        (inputs, outputs),
        Region(block),
        visibility="public" if function.name == entry_point else "private",
    )
    result.attributes.update(_lower_attributes(function.attributes))
    if function.name == entry_point:
        result.attributes["niro.entry_point"] = builtin.UnitAttr()
    return result


def _lower_operations(ctx: Ctx, operations: list[ir.Op]) -> None:
    for operation in operations:
        _lower_operation(ctx, operation)


def _lower_operation(ctx: Ctx, operation: ir.Op) -> None:
    match operation:
        case ir.Const():
            _lower_const(ctx, operation)
        case ir.Add():
            _lower_arithmetic(ctx, operation, arith.AddiOp, arith.AddfOp)
        case ir.Mul():
            _lower_arithmetic(ctx, operation, arith.MuliOp, arith.MulfOp)
        case ir.MatMul():
            _lower_matmul(ctx, operation)
        case ir.Transpose():
            _lower_transpose(ctx, operation)
        case ir.Call():
            lowered = func.CallOp(
                operation.callee,
                [_lookup_value(ctx, value) for value in operation.arguments],
                [_lower_type(value.type) for value in operation.results],
            )
            ctx.block.add_op(lowered)
            _record_results(ctx, operation.results, lowered.results)
        case ir.Return():
            ctx.block.add_op(
                func.ReturnOp(
                    *(_lookup_value(ctx, value) for value in operation.operands)
                )
            )
        case ir.Yield():
            ctx.block.add_op(
                scf.YieldOp(
                    *(_lookup_value(ctx, value) for value in operation.operands)
                )
            )
        case ir.If():
            lowered = scf.IfOp(
                _lookup_value(ctx, operation.condition),
                [_lower_type(value.type) for value in operation.results],
                _lower_region(ctx, operation.then_region),
                _lower_region(ctx, operation.else_region),
            )
            ctx.block.add_op(lowered)
            _record_results(ctx, operation.results, lowered.results)
        case ir.UnknownOp():
            raise NotImplementedError(
                f"cannot lower unknown operation to MLIR: {operation.name}"
            )


def _lower_region(ctx: Ctx, region: ir.Region) -> Region:
    (niro_block,) = region.blocks
    block = Block()
    nested = Ctx(block, dict(ctx.values), ctx.globals, ctx.function_name)
    _lower_operations(nested, niro_block.operations)
    return Region(block)


def _lower_const(ctx: Ctx, operation: ir.Const) -> None:
    if isinstance(operation.result.type, ir.TensorType):
        data = cast(bytes, operation.value)
        tensor_type = cast(
            builtin.TensorType[builtin.AnyDenseElement],
            _lower_type(operation.result.type),
        )
        value = builtin.DenseIntOrFPElementsAttr(tensor_type, builtin.BytesAttr(data))
        symbol = f"__niro_{ctx.function_name}_{int(operation.result.id)}"
        ctx.globals.append(
            ml_program.GlobalOp(
                builtin.StringAttr(symbol),
                tensor_type,
                None,
                value,
                builtin.StringAttr("private"),
            )
        )
        lowered = ml_program.GlobalLoadConstantOp(
            builtin.SymbolRefAttr(symbol), tensor_type
        )
    else:
        scalar_type = operation.result.type
        lowered = arith.ConstantOp(_lower_scalar_literal(operation.value, scalar_type))
    ctx.block.add_op(lowered)
    ctx.values[operation.result.id] = lowered.results[0]


def _lower_arithmetic(
    ctx: Ctx,
    operation: ir.Add | ir.Mul,
    integer_op: type[arith.AddiOp | arith.MuliOp],
    float_op: type[arith.AddfOp | arith.MulfOp],
) -> None:
    scalar_type = _element_type(operation.result.type)
    op_type = (
        float_op
        if scalar_type in (ir.ScalarType.F32, ir.ScalarType.F64)
        else integer_op
    )
    lowered = op_type(
        _lookup_value(ctx, operation.lhs),
        _lookup_value(ctx, operation.rhs),
        result_type=_lower_type(operation.result.type),
    )
    ctx.block.add_op(lowered)
    ctx.values[operation.result.id] = lowered.result


def _lower_transpose(ctx: Ctx, operation: ir.Transpose) -> None:
    result_type = cast(ir.TensorType, operation.result.type)
    _require_static_shape(result_type, "transpose")
    lowered_type = _lower_type(result_type)
    empty = tensor.EmptyOp([], lowered_type)
    permutation = builtin.DenseArrayBase.from_list(builtin.i64, operation.permutation)
    lowered = linalg.TransposeOp(
        _lookup_value(ctx, operation.operand),
        empty.tensor,
        permutation,
        lowered_type,
    )
    ctx.block.add_ops([empty, lowered])
    ctx.values[operation.result.id] = lowered.results[0]


def _lower_matmul(ctx: Ctx, operation: ir.MatMul) -> None:
    lhs_type = cast(ir.TensorType, operation.lhs.type)
    rhs_type = cast(ir.TensorType, operation.rhs.type)
    result_type = cast(ir.TensorType, operation.result.type)
    _require_static_shape(lhs_type, "matmul")
    _require_static_shape(rhs_type, "matmul")
    _require_static_shape(result_type, "matmul")
    lowered_type = _lower_type(result_type)
    zero_value: int | float = (
        0.0 if result_type.element_type in (ir.ScalarType.F32, ir.ScalarType.F64) else 0
    )
    zero = arith.ConstantOp(_lower_scalar_literal(zero_value, result_type.element_type))
    empty = tensor.EmptyOp([], lowered_type)
    fill = linalg.FillOp(
        inputs=[zero.result], outputs=[empty.tensor], res=[lowered_type]
    )
    lowered = linalg.MatmulOp(
        inputs=[
            _lookup_value(ctx, operation.lhs),
            _lookup_value(ctx, operation.rhs),
        ],
        outputs=[fill.results[0]],
        res=[lowered_type],
    )
    ctx.block.add_ops([zero, empty, fill, lowered])
    ctx.values[operation.result.id] = lowered.results[0]


@overload
def _lower_type(
    value_type: ir.ScalarType,
) -> builtin.AnyDenseElement: ...


@overload
def _lower_type(
    value_type: ir.TensorType,
) -> (
    builtin.TensorType[builtin.AnyDenseElement]
    | builtin.UnrankedTensorType[builtin.AnyDenseElement]
): ...


def _lower_type(value_type: ir.Type) -> Attribute:
    if isinstance(value_type, ir.TensorType):
        element_type = _lower_scalar_type(value_type.element_type)
        if value_type.shape is None:
            return builtin.UnrankedTensorType(element_type)
        dimensions = [
            builtin.DYNAMIC_INDEX if dimension is None else dimension
            for dimension in value_type.shape
        ]
        return builtin.TensorType(element_type, dimensions)
    return _lower_scalar_type(value_type)


def _lower_scalar_type(value_type: ir.ScalarType) -> builtin.AnyDenseElement:
    match value_type:
        case ir.ScalarType.BOOL:
            return builtin.i1
        case ir.ScalarType.I32:
            return builtin.i32
        case ir.ScalarType.I64:
            return builtin.i64
        case ir.ScalarType.F32:
            return builtin.f32
        case ir.ScalarType.F64:
            return builtin.f64


def _lower_scalar_literal(
    value: ir.Literal,
    value_type: ir.ScalarType,
) -> builtin.IntegerAttr | builtin.FloatAttr:
    match value_type:
        case ir.ScalarType.BOOL:
            return builtin.BoolAttr.from_bool(cast(bool, value))
        case ir.ScalarType.I32:
            return builtin.IntegerAttr(cast(int, value), builtin.i32)
        case ir.ScalarType.I64:
            return builtin.IntegerAttr(cast(int, value), builtin.i64)
        case ir.ScalarType.F32:
            return builtin.FloatAttr(cast(float, value), builtin.f32)
        case ir.ScalarType.F64:
            return builtin.FloatAttr(cast(float, value), builtin.f64)


def _lower_attributes(
    attributes: dict[str, ir.Attribute],
    *,
    exclude: Collection[str] = (),
) -> dict[str, Attribute]:
    return {
        name if "." in name else f"niro.{name}": _lower_attribute(value)
        for name, value in attributes.items()
        if name not in exclude
    }


def _lower_attribute(value: ir.Attribute) -> Attribute:
    if value is None:
        return builtin.UnitAttr()
    if isinstance(value, bool):
        return builtin.BoolAttr.from_bool(value)
    if isinstance(value, int):
        return builtin.IntegerAttr(value, builtin.i64)
    if isinstance(value, float):
        return builtin.FloatAttr(value, builtin.f64)
    if isinstance(value, str):
        return builtin.StringAttr(value)
    if isinstance(value, bytes):
        return builtin.BytesAttr(value)
    return builtin.ArrayAttr(_lower_attribute(element) for element in value)


def _lookup_value(ctx: Ctx, value: ir.Value) -> SSAValue:
    return ctx.values[value.id]


def _record_results(
    ctx: Ctx,
    niro_values: tuple[ir.Value, ...],
    mlir_values: tuple[SSAValue, ...],
) -> None:
    ctx.values.update(
        (niro_value.id, mlir_value)
        for niro_value, mlir_value in zip(niro_values, mlir_values, strict=True)
    )


def _element_type(value_type: ir.Type) -> ir.ScalarType:
    if isinstance(value_type, ir.TensorType):
        return value_type.element_type
    return value_type


def _require_static_shape(value_type: ir.TensorType, operation: str) -> None:
    if value_type.shape is None or any(dim is None for dim in value_type.shape):
        raise NotImplementedError(f"{operation} requires a static ranked tensor")
