"""Convenient construction API for Niro IR."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from typing import Final, cast

from niro import ir

ENTRY_POINT_ATTR: Final = "entry_point"

type CallTarget = FunctionBuilder | ir.Function | str


class ModuleBuilder:
    """Construct a Niro module one function at a time.

    The resulting ir.Module is available as builder.ir.
    """

    def __init__(self) -> None:
        self.ir = ir.Module()
        self._functions: dict[str, ir.Function] = {}

    def func(
        self,
        name: str,
        arg_types: Sequence[ir.Type] = (),
        ret_types: Sequence[ir.Type] = (),
    ) -> FunctionBuilder:
        """Create a defined function with one empty entry block."""
        return FunctionBuilder(self, name, arg_types, ret_types)

    def extern(
        self,
        name: str,
        arg_types: Sequence[ir.Type] = (),
        ret_types: Sequence[ir.Type] = (),
    ) -> ir.Function:
        """Declare a function whose implementation is outside this module."""
        function = ir.Function(
            id=ir.FuncId(len(self.ir.functions)),
            name=name,
            type=ir.FunctionType(tuple(arg_types), tuple(ret_types)),
        )
        self._register_function(function)
        return function

    def set_entry_point(self, function: FunctionBuilder | ir.Function | str) -> None:
        """Select the function used as the program entry point."""
        resolved = self.resolve_function(function)
        if resolved.body is None:
            raise ValueError("entry point must be a defined function")
        self.ir.attributes[ENTRY_POINT_ATTR] = resolved.name

    def resolve_function(self, function: CallTarget) -> ir.Function:
        """Resolve a builder, function, or symbol name in this module."""
        match function:
            case FunctionBuilder():
                resolved = function.function
            case ir.Function():
                resolved = function
            case str():
                try:
                    return self._functions[function]
                except KeyError:
                    raise ValueError(f"unknown function: {function!r}") from None

        if self._functions.get(resolved.name) is not resolved:
            raise ValueError(
                f"function {resolved.name!r} does not belong to this module"
            )
        return resolved

    def _register_function(self, function: ir.Function) -> None:
        if function.name in self._functions:
            raise ValueError(f"duplicate function: {function.name!r}")
        self.ir.functions.append(function)
        self._functions[function.name] = function


class FunctionBuilder:
    """Construct a function and its body region.

    Create one with ModuleBuilder.func().
    The resulting ir.Function is available as builder.function.
    """

    def __init__(
        self,
        module: ModuleBuilder,
        name: str,
        arg_types: Sequence[ir.Type],
        ret_types: Sequence[ir.Type],
    ) -> None:
        self._module = module
        self._operations: list[ir.Op] = []
        self._values: list[ir.Value] = []
        region = ir.Region()
        self.function = ir.Function(
            id=ir.FuncId(len(module.ir.functions)),
            name=name,
            type=ir.FunctionType(tuple(arg_types), tuple(ret_types)),
            body=region,
        )
        self.body = RegionBuilder(self, region)
        self.entry = self.body.block(arg_types)
        module._register_function(self.function)

    @property
    def args(self) -> tuple[ir.Value, ...]:
        return self.entry.args

    def region(self) -> RegionBuilder:
        """Create a detached region owned by this function."""
        return RegionBuilder(self, ir.Region())

    def _new_values(self, value_types: Sequence[ir.Type]) -> tuple[ir.Value, ...]:
        return tuple(
            ir.Value(ir.ValueId(len(self._values) + index), value_type)
            for index, value_type in enumerate(value_types)
        )

    def _new_op_id(self) -> ir.OpId:
        return ir.OpId(len(self._operations))

    def _commit_values(self, values: Sequence[ir.Value]) -> None:
        self._values.extend(values)

    def _commit_operation(self, operation: ir.Op) -> None:
        self._operations.append(operation)

    def _validate_owned_value(self, value: ir.Value) -> None:
        index = int(value.id)
        if index >= len(self._values) or self._values[index] is not value:
            raise ValueError("value does not belong to this function")

    def _validate_call_signature(
        self,
        function: ir.Function,
        arguments: tuple[ir.Value, ...],
    ) -> None:
        for argument in arguments:
            self._validate_owned_value(argument)
        actual = tuple(argument.type for argument in arguments)
        if actual != function.type.inputs:
            raise TypeError(
                f"call argument types {actual!r} do not match {function.type.inputs!r}"
            )


class RegionBuilder:
    """Construct blocks in a region."""

    def __init__(self, function: FunctionBuilder, region: ir.Region) -> None:
        self._function = function
        self.region = region
        self.blocks: list[BlockBuilder] = []

    def block(self, arg_types: Sequence[ir.Type] = ()) -> BlockBuilder:
        """Append a block with arguments of the given types."""
        arguments = self._function._new_values(arg_types)
        block = ir.Block(arguments=arguments)
        builder = BlockBuilder(self._function, self, block)
        self._function._commit_values(arguments)
        self.region.blocks.append(block)
        self.blocks.append(builder)
        return builder


class IfBuilder:
    """Construct the regions of an if operation."""

    def __init__(
        self,
        operation: ir.If,
        then_region: RegionBuilder,
        else_region: RegionBuilder,
    ) -> None:
        self.operation = operation
        self.then_region = then_region
        self.else_region = else_region

    @property
    def results(self) -> tuple[ir.Value, ...]:
        return self.operation.results


class BlockBuilder:
    """Construct operations in one block."""

    def __init__(
        self,
        function: FunctionBuilder,
        region: RegionBuilder,
        block: ir.Block,
    ) -> None:
        self._function = function
        self.region = region
        self.block = block

    @property
    def args(self) -> tuple[ir.Value, ...]:
        return self.block.arguments

    def constant(self, value: ir.Literal, result_type: ir.Type) -> ir.Value:
        return self._append_result(
            result_type,
            lambda op_id, result: ir.Const(id=op_id, result=result, literal=value),
        )

    def bool(self, value: builtins.bool) -> ir.Value:
        return self.constant(value, ir.ScalarType.BOOL)

    def i32(self, value: int) -> ir.Value:
        return self.constant(value, ir.ScalarType.I32)

    def i64(self, value: int) -> ir.Value:
        return self.constant(value, ir.ScalarType.I64)

    def f32(self, value: float) -> ir.Value:
        return self.constant(value, ir.ScalarType.F32)

    def f64(self, value: float) -> ir.Value:
        return self.constant(value, ir.ScalarType.F64)

    def tensor(self, data: bytes, result_type: ir.TensorType) -> ir.Value:
        """Create a tensor constant from contiguous little-endian data."""
        return self.constant(data, result_type)

    def add(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return self._append_result(
            lhs.type,
            lambda op_id, result: ir.Add(id=op_id, result=result, lhs=lhs, rhs=rhs),
            operands=(lhs, rhs),
        )

    def mul(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return self._append_result(
            lhs.type,
            lambda op_id, result: ir.Mul(id=op_id, result=result, lhs=lhs, rhs=rhs),
            operands=(lhs, rhs),
        )

    def matmul(
        self,
        lhs: ir.Value,
        rhs: ir.Value,
    ) -> ir.Value:
        result_type = ir.matmul_result_type(lhs.type, rhs.type)
        return self._append_result(
            result_type,
            lambda op_id, result: ir.MatMul(id=op_id, result=result, lhs=lhs, rhs=rhs),
            operands=(lhs, rhs),
        )

    def transpose(
        self,
        operand: ir.Value,
        permutation: Sequence[int],
    ) -> ir.Value:
        normalized = tuple(permutation)
        result_type = ir.transpose_result_type(operand.type, normalized)
        return self._append_result(
            result_type,
            lambda op_id, result: ir.Transpose(
                id=op_id,
                result=result,
                operand=operand,
                permutation=normalized,
            ),
            operands=(operand,),
        )

    def unknown(
        self,
        name: str,
        operands: Sequence[ir.Value] = (),
        result_types: Sequence[ir.Type] = (),
        attributes: Mapping[str, ir.Attribute] | None = None,
    ) -> tuple[ir.Value, ...]:
        """Create an operation whose semantics are not known to Niro."""
        normalized_operands = tuple(operands)
        return self._append_results(
            result_types,
            lambda op_id, results: ir.UnknownOp(
                id=op_id,
                name=name,
                operands=normalized_operands,
                results=results,
                attributes=dict(attributes or {}),
            ),
            operands=normalized_operands,
        )

    def if_(
        self,
        condition: ir.Value,
        result_types: Sequence[ir.Type] = (),
    ) -> IfBuilder:
        """Append an if operation and return builders for its two regions."""
        then_region = self._function.region()
        else_region = self._function.region()
        self._append_results(
            result_types,
            lambda op_id, values: ir.If(
                id=op_id,
                results=values,
                condition=condition,
                then_region=then_region.region,
                else_region=else_region.region,
            ),
            operands=(condition,),
        )
        operation = cast(ir.If, self.block.operations[-1])
        return IfBuilder(operation, then_region, else_region)

    def call(
        self,
        callee: CallTarget,
        arguments: ir.Value | Sequence[ir.Value] = (),
    ) -> ir.Value | tuple[ir.Value, ...] | None:
        target = self._function._module.resolve_function(callee)
        match arguments:
            case ir.Value():
                normalized = (arguments,)
            case _:
                normalized = tuple(arguments)
        self._function._validate_call_signature(target, normalized)
        results = self._append_results(
            target.type.outputs,
            lambda op_id, values: ir.Call(
                id=op_id,
                callee=target.name,
                arguments=normalized,
                results=values,
            ),
            operands=normalized,
        )
        if not results:
            return None
        if len(results) == 1:
            return results[0]
        return results

    def return_(self, *operands: ir.Value) -> None:
        expected = self._function.function.type.outputs
        actual = tuple(value.type for value in operands)
        if actual != expected:
            raise TypeError(f"return types {actual!r} do not match {expected!r}")
        self._append_operation(
            lambda op_id: ir.Return(id=op_id, operands=operands),
            operands=operands,
        )

    def yield_(self, *operands: ir.Value) -> None:
        self._append_operation(
            lambda op_id: ir.Yield(id=op_id, operands=operands),
            operands=operands,
        )

    def _append_result(
        self,
        result_type: ir.Type,
        make_operation: Callable[[ir.OpId, ir.Value], ir.Op],
        operands: Sequence[ir.Value] = (),
    ) -> ir.Value:
        (result,) = self._append_results(
            (result_type,),
            lambda op_id, results: make_operation(op_id, results[0]),
            operands,
        )
        return result

    def _append_results(
        self,
        result_types: Sequence[ir.Type],
        make_operation: Callable[[ir.OpId, tuple[ir.Value, ...]], ir.Op],
        operands: Sequence[ir.Value] = (),
    ) -> tuple[ir.Value, ...]:
        if self._has_terminator():
            raise ValueError("cannot add an operation after a block terminator")
        for operand in operands:
            self._validate_owned_value(operand)
        results = self._function._new_values(result_types)
        operation = make_operation(self._function._new_op_id(), results)
        self._function._commit_values(results)
        self._function._commit_operation(operation)
        self.block.operations.append(operation)
        return results

    def _append_operation(
        self,
        make_operation: Callable[[ir.OpId], ir.Op],
        operands: Sequence[ir.Value] = (),
    ) -> None:
        if self._has_terminator():
            raise ValueError("cannot add an operation after a block terminator")
        for operand in operands:
            self._validate_owned_value(operand)
        operation = make_operation(self._function._new_op_id())
        self._function._commit_operation(operation)
        self.block.operations.append(operation)

    def _has_terminator(self) -> builtins.bool:
        if not self.block.operations:
            return False
        match self.block.operations[-1]:
            case ir.Return() | ir.Yield():
                return True
            case _:
                return False

    def _validate_owned_value(self, value: ir.Value) -> None:
        self._function._validate_owned_value(value)
