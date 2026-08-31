"""Lower Niro IR to high-level MLIR using xDSL."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import cast

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
    attributes = _attributes(module.attributes, exclude={ENTRY_POINT_ATTR})
    result = builtin.ModuleOp([*globals_, *functions], attributes=attributes)
    result.verify()
    return result


def _lower_function(
    function: ir.Function,
    entry_point: str,
    globals_: list[Operation],
) -> func.FuncOp:
    inputs = [_type(value_type) for value_type in function.type.inputs]
    outputs = [_type(value_type) for value_type in function.type.outputs]
    if function.body is None:
        result = func.FuncOp.external(function.name, inputs, outputs)
        result.attributes.update(_attributes(function.attributes))
        return result
    if len(function.body.blocks) != 1:
        raise ValueError(f"function {function.name!r} must have one block")

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
    _lower_operations(ctx, function.body.blocks[0].operations)
    result = func.FuncOp(
        function.name,
        (inputs, outputs),
        Region(block),
        visibility="public" if function.name == entry_point else "private",
    )
    result.attributes.update(_attributes(function.attributes))
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
                [_value(ctx, value) for value in operation.arguments],
                [_type(value.type) for value in operation.results],
            )
            ctx.block.add_op(lowered)
            _record_results(ctx, operation.results, lowered.results)
        case ir.Return():
            ctx.block.add_op(
                func.ReturnOp(*(_value(ctx, value) for value in operation.operands))
            )
        case ir.Yield():
            ctx.block.add_op(
                scf.YieldOp(*(_value(ctx, value) for value in operation.operands))
            )
        case ir.If():
            lowered = scf.IfOp(
                _value(ctx, operation.condition),
                [_type(value.type) for value in operation.results],
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
    if len(region.blocks) != 1 or region.blocks[0].arguments:
        raise ValueError("structured regions must have one block and no arguments")
    block = Block()
    nested = Ctx(block, dict(ctx.values), ctx.globals, ctx.function_name)
    _lower_operations(nested, region.blocks[0].operations)
    return Region(block)


def _lower_const(ctx: Ctx, operation: ir.Const) -> None:
    result_type = _type(operation.result.type)
    if isinstance(operation.result.type, ir.TensorType):
        if not isinstance(operation.value, bytes):
            raise TypeError("tensor constant data must be bytes")
        _require_static_tensor(operation.result.type, "tensor constant")
        expected_size = _tensor_byte_size(operation.result.type)
        if len(operation.value) != expected_size:
            raise ValueError(
                f"tensor constant has {len(operation.value)} bytes, "
                f"expected {expected_size}"
            )
        assert isinstance(result_type, builtin.TensorType)
        value = builtin.DenseIntOrFPElementsAttr(
            result_type, builtin.BytesAttr(operation.value)
        )
        symbol = f"__niro_{ctx.function_name}_{int(operation.result.id)}"
        ctx.globals.append(
            ml_program.GlobalOp(
                builtin.StringAttr(symbol),
                result_type,
                None,
                value,
                builtin.StringAttr("private"),
            )
        )
        lowered = ml_program.GlobalLoadConstantOp(
            builtin.SymbolRefAttr(symbol), result_type
        )
    else:
        lowered = arith.ConstantOp(_scalar_attribute(operation.value, result_type))
    ctx.block.add_op(lowered)
    ctx.values[operation.result.id] = lowered.results[0]


def _lower_arithmetic(
    ctx: Ctx,
    operation: ir.Add | ir.Mul,
    integer_op: type[arith.AddiOp | arith.MuliOp],
    float_op: type[arith.AddfOp | arith.MulfOp],
) -> None:
    scalar_type = _element_type(operation.result.type)
    if scalar_type is ir.ScalarType.BOOL:
        raise TypeError("boolean arithmetic is not supported")
    op_type = (
        float_op
        if scalar_type in (ir.ScalarType.F32, ir.ScalarType.F64)
        else integer_op
    )
    lowered = op_type(
        _value(ctx, operation.lhs),
        _value(ctx, operation.rhs),
        result_type=_type(operation.result.type),
    )
    ctx.block.add_op(lowered)
    ctx.values[operation.result.id] = lowered.result


def _lower_transpose(ctx: Ctx, operation: ir.Transpose) -> None:
    operand_type = _require_static_tensor(operation.operand.type, "transpose")
    result_type = _require_static_tensor(operation.result.type, "transpose")
    assert operand_type.shape is not None
    if sorted(operation.permutation) != list(range(len(operand_type.shape))):
        raise ValueError("transpose permutation must contain every dimension once")
    lowered_type = _type(result_type)
    empty = tensor.EmptyOp([], lowered_type)
    permutation = builtin.DenseArrayBase.from_list(
        builtin.i64, operation.permutation
    )
    lowered = linalg.TransposeOp(
        _value(ctx, operation.operand),
        empty.tensor,
        permutation,
        lowered_type,
    )
    ctx.block.add_ops([empty, lowered])
    ctx.values[operation.result.id] = lowered.results[0]


def _lower_matmul(ctx: Ctx, operation: ir.MatMul) -> None:
    lhs_type = _require_matrix(operation.lhs.type, "matmul lhs")
    rhs_type = _require_matrix(operation.rhs.type, "matmul rhs")
    result_type = _require_matrix(operation.result.type, "matmul result")
    if not (
        lhs_type.element_type
        is rhs_type.element_type
        is result_type.element_type
    ):
        raise TypeError("matmul element types must match")
    lowered_type = _type(result_type)
    scalar_type = _type(result_type.element_type)
    zero_value: int | float = (
        0.0
        if result_type.element_type in (ir.ScalarType.F32, ir.ScalarType.F64)
        else 0
    )
    zero = arith.ConstantOp(_scalar_attribute(zero_value, scalar_type))
    empty = tensor.EmptyOp([], lowered_type)
    fill = linalg.FillOp(
        inputs=[zero.result], outputs=[empty.tensor], res=[lowered_type]
    )
    lowered = linalg.MatmulOp(
        inputs=[_value(ctx, operation.lhs), _value(ctx, operation.rhs)],
        outputs=[fill.results[0]],
        res=[lowered_type],
    )
    ctx.block.add_ops([zero, empty, fill, lowered])
    ctx.values[operation.result.id] = lowered.results[0]


def _type(value_type: ir.Type) -> Attribute:
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
        case ir.TensorType(element_type, None):
            return builtin.UnrankedTensorType(_type(element_type))
        case ir.TensorType(element_type, shape):
            assert shape is not None
            dimensions = [
                builtin.DYNAMIC_INDEX if dimension is None else dimension
                for dimension in shape
            ]
            return builtin.TensorType(_type(element_type), dimensions)


def _scalar_attribute(
    value: ir.Literal,
    value_type: Attribute,
) -> builtin.IntegerAttr | builtin.FloatAttr:
    if value_type == builtin.i1:
        if not isinstance(value, bool):
            raise TypeError("boolean constant must contain a bool")
        return builtin.BoolAttr.from_bool(value)
    if value_type in (builtin.i32, builtin.i64):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("integer constant must contain an int")
        assert isinstance(value_type, builtin.IntegerType)
        return builtin.IntegerAttr(value, value_type)
    if isinstance(value_type, (builtin.Float32Type, builtin.Float64Type)):
        if not isinstance(value, float):
            raise TypeError("floating-point constant must contain a float")
        return builtin.FloatAttr(value, value_type)
    raise TypeError("constant must have a scalar type")


def _attributes(
    attributes: dict[str, ir.Attribute],
    *,
    exclude: Collection[str] = (),
) -> dict[str, Attribute]:
    return {
        name if "." in name else f"niro.{name}": _attribute(value)
        for name, value in attributes.items()
        if name not in exclude
    }


def _attribute(value: ir.Attribute) -> Attribute:
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
    return builtin.ArrayAttr(_attribute(element) for element in value)


def _value(ctx: Ctx, value: ir.Value) -> SSAValue:
    try:
        return ctx.values[value.id]
    except KeyError:
        raise ValueError(f"Niro value {int(value.id)} is not defined") from None


def _record_results(
    ctx: Ctx,
    niro_values: tuple[ir.Value, ...],
    mlir_values: tuple[SSAValue, ...],
) -> None:
    if len(niro_values) != len(mlir_values):
        raise ValueError("operation result count changed during MLIR lowering")
    ctx.values.update(
        (niro_value.id, mlir_value)
        for niro_value, mlir_value in zip(
            niro_values, mlir_values, strict=True
        )
    )


def _element_type(value_type: ir.Type) -> ir.ScalarType:
    if isinstance(value_type, ir.TensorType):
        return value_type.element_type
    return value_type


def _require_static_tensor(value_type: ir.Type, operation: str) -> ir.TensorType:
    if not isinstance(value_type, ir.TensorType):
        raise TypeError(f"{operation} requires a tensor")
    if value_type.shape is None or any(dim is None for dim in value_type.shape):
        raise NotImplementedError(f"{operation} requires a static ranked tensor")
    return value_type


def _require_matrix(value_type: ir.Type, operation: str) -> ir.TensorType:
    tensor_type = _require_static_tensor(value_type, operation)
    assert tensor_type.shape is not None
    if len(tensor_type.shape) != 2:
        raise TypeError(f"{operation} requires a rank-two tensor")
    return tensor_type


def _tensor_byte_size(value_type: ir.TensorType) -> int:
    assert value_type.shape is not None
    widths = {
        ir.ScalarType.BOOL: 1,
        ir.ScalarType.I32: 4,
        ir.ScalarType.I64: 8,
        ir.ScalarType.F32: 4,
        ir.ScalarType.F64: 8,
    }
    shape = cast(tuple[int, ...], value_type.shape)
    return math.prod(shape) * widths[value_type.element_type]
