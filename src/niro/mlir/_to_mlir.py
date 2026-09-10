"""Lower Niro IR to high-level MLIR using xDSL.

``_lower_*`` functions return lowered values without mutating caller-owned state.
``_emit_*`` functions append operations and update explicitly passed value tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_never, cast, overload

from xdsl.dialect_interfaces.op_asm import OpAsmDialectInterface
from xdsl.dialects import arith, builtin, cf, func, ml_program, scf, tensor
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue

from niro import ir
from niro.ir import VerifiedModule

ValueTable = dict[ir.ValueId, SSAValue]
"""Mutable mapping from function-local Niro value IDs to lowered MLIR SSA values."""


def to_mlir(niro_module: VerifiedModule) -> builtin.ModuleOp:
    """Convert verified Niro IR to a verified, high-level MLIR module.

    Lower branches to the CF dialect. Multiblock If arms use scf.execute_region
    inside scf.if, preserving their control flow and yielded results.
    Raise NotImplementedError for unknown operations.
    """
    lowered_functions = [
        _lower_function(function) for function in niro_module.functions
    ]
    declared_globals = [_lower_global(global_) for global_ in niro_module.globals]
    generated_globals = [
        global_operation
        for function_globals, _ in lowered_functions
        for global_operation in function_globals
    ]
    functions = [function for _, function in lowered_functions]
    result = builtin.ModuleOp(
        [*declared_globals, *generated_globals, *functions],
        attributes=_lower_attributes(niro_module.attributes),
    )
    result.verify()
    return result


def _lower_function(
    function: ir.Function,
) -> tuple[tuple[Operation, ...], func.FuncOp]:
    """Return generated constant globals and the lowered function.

    Precreate CFG blocks and lower definitions before uses. External declarations
    have no generated globals.
    """
    inputs = [_lower_type(value_type) for value_type in function.type.inputs]
    outputs = [_lower_type(value_type) for value_type in function.type.outputs]
    if function.body is None:
        result = func.FuncOp.external(function.name, inputs, outputs)
        result.attributes.update(_lower_attributes(function.attributes))
        return (), result
    generated_globals: list[Operation] = []
    body = _emit_cfg(function.body, {}, generated_globals, function.name)
    result = func.FuncOp(function.name, (inputs, outputs), body)
    result.attributes.update(_lower_attributes(function.attributes))
    return tuple(generated_globals), result


def _emit_operations(
    block: Block,
    values: ValueTable,
    generated_globals: list[Operation],
    function_name: str,
    operations: list[ir.Op],
    blocks: Mapping[ir.Block, Block],
) -> None:
    """Append lowered operations in order, updating values and generated globals."""
    for operation in operations:
        _emit_operation(
            block,
            values,
            generated_globals,
            function_name,
            operation,
            blocks,
        )


def _emit_operation(
    block: Block,
    values: ValueTable,
    generated_globals: list[Operation],
    function_name: str,
    operation: ir.Op,
    blocks: Mapping[ir.Block, Block],
) -> None:
    """Append an operation's lowering and record its results and any globals.

    Operands must already be bound; opaque operations cannot be lowered.
    """
    match operation:
        case ir.Const():
            _emit_const(
                block,
                values,
                generated_globals,
                function_name,
                operation,
            )
        case ir.GetGlobal():
            lowered = ml_program.GlobalLoadConstantOp(
                builtin.SymbolRefAttr(operation.name),
                _lower_type(operation.result.type),
            )
            block.add_op(lowered)
            values[operation.result.id] = lowered.result
        case ir.Add():
            _emit_arithmetic(block, values, operation, arith.AddiOp, arith.AddfOp)
        case ir.Mul():
            _emit_arithmetic(block, values, operation, arith.MuliOp, arith.MulfOp)
        case ir.MatMul():
            _emit_matmul(block, values, operation)
        case ir.Transpose():
            _emit_transpose(block, values, operation)
        case ir.Call():
            lowered = func.CallOp(
                operation.callee,
                [_lookup_value(values, value) for value in operation.arguments],
                [_lower_type(value.type) for value in operation.results],
            )
            block.add_op(lowered)
            _bind_results(values, operation.results, lowered.results)
        case ir.Branch():
            block.add_op(
                cf.BranchOp(
                    blocks[operation.target],
                    *(_lookup_value(values, value) for value in operation.arguments),
                )
            )
        case ir.CondBranch():
            block.add_op(
                cf.ConditionalBranchOp(
                    _lookup_value(values, operation.condition),
                    blocks[operation.true_target],
                    [
                        _lookup_value(values, value)
                        for value in operation.true_arguments
                    ],
                    blocks[operation.false_target],
                    [
                        _lookup_value(values, value)
                        for value in operation.false_arguments
                    ],
                )
            )
        case ir.Return():
            block.add_op(
                func.ReturnOp(
                    *(_lookup_value(values, value) for value in operation.operands)
                )
            )
        case ir.Yield():
            block.add_op(
                scf.YieldOp(
                    *(_lookup_value(values, value) for value in operation.operands)
                )
            )
        case ir.If():
            lowered = scf.IfOp(
                _lookup_value(values, operation.condition),
                [_lower_type(value.type) for value in operation.results],
                _emit_region(
                    operation.then_region,
                    operation.results,
                    values,
                    generated_globals,
                    function_name,
                ),
                _emit_region(
                    operation.else_region,
                    operation.results,
                    values,
                    generated_globals,
                    function_name,
                ),
            )
            block.add_op(lowered)
            _bind_results(values, operation.results, lowered.results)
        case ir.UnknownOp():
            raise NotImplementedError(
                f"cannot lower unknown operation to MLIR: {operation.name}"
            )
        case _ as unreachable:
            assert_never(unreachable)


def _emit_cfg(
    region: ir.Region,
    visible_values: ValueTable,
    generated_globals: list[Operation],
    function_name: str,
) -> Region:
    """Lower a CFG in reverse postorder while preserving source block layout.

    Precreate blocks and arguments for forward edges and loop backedges. Emit
    dominating definitions before their uses. Copy enclosing value bindings so
    region-local values do not escape; share the generated constant globals.
    """
    blocks = {
        source: Block(arg_types=[_lower_type(v.type) for v in source.arguments])
        for source in region.blocks
    }
    values = dict(visible_values)
    for source, target in blocks.items():
        _bind_results(values, source.arguments, tuple(target.args))
    visited: set[ir.Block] = set()
    postorder: list[ir.Block] = []
    pending = [(region.blocks[0], False)]
    while pending:
        source, exiting = pending.pop()
        if exiting:
            postorder.append(source)
            continue
        if source in visited:
            continue
        visited.add(source)
        pending.append((source, True))
        pending.extend(
            (target, False) for target in ir.get_successors(source.operations[-1])
        )
    for source in reversed(postorder):
        _emit_operations(
            blocks[source],
            values,
            generated_globals,
            function_name,
            source.operations,
            blocks,
        )
    return Region(list(blocks.values()))


def _emit_region(
    region: ir.Region,
    results: tuple[ir.Value, ...],
    visible_values: ValueTable,
    generated_globals: list[Operation],
    function_name: str,
) -> Region:
    """Lower an If arm, wrapping multiblock CFGs in scf.execute_region.

    scf.if requires single-block arms. An execute region preserves arbitrary
    internal branches and yields its results to the enclosing arm.
    """
    lowered = _emit_cfg(region, visible_values, generated_globals, function_name)
    if len(region.blocks) == 1:
        return lowered
    execute = scf.ExecuteRegionOp([_lower_type(v.type) for v in results], lowered)
    return Region(Block([execute, scf.YieldOp(*execute.results)]))


def _emit_const(
    block: Block,
    values: ValueTable,
    generated_globals: list[Operation],
    function_name: str,
    operation: ir.Const,
) -> None:
    """Append a constant load and bind its result.

    Tensor literals add a private resource-backed global; scalars use an
    arithmetic constant. Literal types must already have been verified.
    """
    if isinstance(operation.result.type, ir.TensorType):
        data = cast(bytes, operation.literal)
        tensor_type = cast(
            builtin.TensorType[builtin.AnyDenseElement],
            _lower_type(operation.result.type),
        )
        symbol = f"__niro_{function_name}_{int(operation.result.id)}"
        value = _dense_resource(symbol, tensor_type, data)
        generated_globals.append(
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
        lowered = arith.ConstantOp(
            _lower_scalar_literal(operation.literal, scalar_type)
        )
    block.add_op(lowered)
    values[operation.result.id] = lowered.results[0]


def _lower_global(global_: ir.Global) -> ml_program.GlobalOp:
    """Return a private tensor global, registering its initializer resource."""
    if not isinstance(global_.type, ir.TensorType):
        raise TypeError("MLIR globals currently require tensor types")
    data = cast(bytes, global_.initializer)
    tensor_type = cast(
        builtin.TensorType[builtin.AnyDenseElement], _lower_type(global_.type)
    )
    value = _dense_resource(global_.name, tensor_type, data)
    lowered = ml_program.GlobalOp(
        builtin.StringAttr(global_.name),
        tensor_type,
        None,
        value,
        builtin.StringAttr("private"),
    )
    lowered.attributes.update(_lower_attributes(global_.attributes))
    return lowered


def _dense_resource(
    name: str,
    tensor_type: builtin.TensorType[builtin.AnyDenseElement],
    data: bytes,
) -> builtin.DenseResourceAttr:
    """Register packed tensor bytes with the builtin dialect and return an attribute."""
    resources = builtin.Builtin.get_interface(OpAsmDialectInterface)
    assert resources is not None
    handle = resources.declare_resource(name)
    resources.parse_resource(handle, f"0x{data.hex().upper()}")
    return builtin.DenseResourceAttr.from_params(handle, tensor_type)


def _emit_arithmetic(
    block: Block,
    values: ValueTable,
    operation: ir.Add | ir.Mul,
    integer_op: type[arith.AddiOp | arith.MuliOp],
    float_op: type[arith.AddfOp | arith.MulfOp],
) -> None:
    """Append integer or float arithmetic by element type and bind the result."""
    scalar_type = _element_type(operation.result.type)
    op_type = (
        float_op
        if scalar_type in (ir.ScalarType.F32, ir.ScalarType.F64)
        else integer_op
    )
    lowered = op_type(
        _lookup_value(values, operation.lhs),
        _lookup_value(values, operation.rhs),
        result_type=_lower_type(operation.result.type),
    )
    block.add_op(lowered)
    values[operation.result.id] = lowered.result


def _emit_transpose(
    block: Block,
    values: ValueTable,
    operation: ir.Transpose,
) -> None:
    """Append an empty tensor and transpose, binding a statically shaped result."""
    result_type = cast(ir.TensorType, operation.result.type)
    _require_static_shape(result_type, "transpose")
    lowered_type = _lower_type(result_type)
    empty = tensor.EmptyOp([], lowered_type)
    permutation = builtin.DenseArrayBase.from_list(builtin.i64, operation.permutation)
    lowered = linalg.TransposeOp(
        _lookup_value(values, operation.operand),
        empty.tensor,
        permutation,
        lowered_type,
    )
    block.add_ops([empty, lowered])
    values[operation.result.id] = lowered.results[0]


def _emit_matmul(
    block: Block,
    values: ValueTable,
    operation: ir.MatMul,
) -> None:
    """Append a zero-filled output tensor and matrix product, then bind the result.

    Operand and result shapes must be static and ranked.
    """
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
            _lookup_value(values, operation.lhs),
            _lookup_value(values, operation.rhs),
        ],
        outputs=[fill.results[0]],
        res=[lowered_type],
    )
    block.add_ops([zero, empty, fill, lowered])
    values[operation.result.id] = lowered.results[0]


@overload
def _lower_type(
    value_type: ir.ScalarType,
) -> builtin.AnyDenseElement:
    """Return the builtin MLIR type for a scalar."""


@overload
def _lower_type(
    value_type: ir.TensorType,
) -> (
    builtin.TensorType[builtin.AnyDenseElement]
    | builtin.UnrankedTensorType[builtin.AnyDenseElement]
):
    """Return an MLIR type, preserving unranked tensors and dynamic dimensions."""


def _lower_type(value_type: ir.Type) -> Attribute:
    """Return an MLIR type, preserving unranked tensors and dynamic dimensions."""
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
    """Return the builtin MLIR type corresponding to a Niro scalar type."""
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
    """Return an MLIR scalar attribute for a literal verified against its type."""
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
    attributes: ir.Attributes,
) -> dict[str, Attribute]:
    """Return lowered attributes, prefixing unqualified names with `niro.`."""
    return {
        name if "." in name else f"niro.{name}": _lower_attribute(value)
        for name, value in attributes.items()
    }


def _lower_attribute(value: ir.AttributeValue) -> Attribute:
    """Return an MLIR attribute, recursively lowering sequences to arrays."""
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


def _lookup_value(values: ValueTable, value: ir.Value) -> SSAValue:
    """Return the previously bound MLIR SSA value for a Niro value ID."""
    return values[value.id]


def _bind_results(
    values: ValueTable,
    niro_values: tuple[ir.Value, ...],
    mlir_values: tuple[SSAValue, ...],
) -> None:
    """Add corresponding result bindings; both result tuples must have equal length."""
    values.update(
        (niro_value.id, mlir_value)
        for niro_value, mlir_value in zip(niro_values, mlir_values, strict=True)
    )


def _element_type(value_type: ir.Type) -> ir.ScalarType:
    """Return a tensor element type or the scalar type itself."""
    if isinstance(value_type, ir.TensorType):
        return value_type.element_type
    return value_type


def _require_static_shape(value_type: ir.TensorType, operation: str) -> None:
    """Raise `NotImplementedError` unless every tensor dimension is known."""
    if value_type.shape is None or any(dim is None for dim in value_type.shape):
        raise NotImplementedError(f"{operation} requires a static ranked tensor")
