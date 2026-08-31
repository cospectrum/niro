"""Convenient construction API for Niro IR."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from typing import Final

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
        builder = FunctionBuilder(self, name, arg_types, ret_types)
        self._add_function(builder.function)
        return builder

    def extern(
        self,
        name: str,
        arg_types: Sequence[ir.Type] = (),
        ret_types: Sequence[ir.Type] = (),
    ) -> ir.Function:
        """Declare a function whose implementation is outside this module."""
        function = ir.Function(
            name,
            ir.FunctionType(tuple(arg_types), tuple(ret_types)),
        )
        self._add_function(function)
        return function

    def entry_point(self, function: FunctionBuilder | ir.Function | str) -> None:
        """Select the function used as the program entry point."""
        resolved = self.resolve(function)
        if resolved.body is None:
            raise ValueError("entry point must be a defined function")
        self.ir.attributes[ENTRY_POINT_ATTR] = resolved.name

    def resolve(self, function: CallTarget) -> ir.Function:
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

    def _add_function(self, function: ir.Function) -> None:
        if function.name in self._functions:
            raise ValueError(f"duplicate function: {function.name!r}")
        self.ir.functions.append(function)
        self._functions[function.name] = function


class FunctionBuilder:
    """Build operations in the entry block of one function.

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
        self._values = [
            ir.Value(ir.ValueId(index), value_type)
            for index, value_type in enumerate(arg_types)
        ]
        self._block = ir.Block(arguments=tuple(self._values))
        self.function = ir.Function(
            name,
            ir.FunctionType(tuple(arg_types), tuple(ret_types)),
            ir.Region([self._block]),
        )

    @property
    def args(self) -> tuple[ir.Value, ...]:
        return self.function.arguments

    def constant(self, value: ir.Literal, result_type: ir.Type) -> ir.Value:
        return self._append_result(
            result_type,
            lambda result: ir.Const(result, value),
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
            lambda result: ir.Add(result, lhs, rhs),
            operands=(lhs, rhs),
        )

    def mul(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return self._append_result(
            lhs.type,
            lambda result: ir.Mul(result, lhs, rhs),
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
            lambda result: ir.MatMul(result, lhs, rhs),
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
            lambda result: ir.Transpose(result, operand, normalized),
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
            lambda results: ir.UnknownOp(
                name=name,
                operands=normalized_operands,
                results=results,
                attributes=dict(attributes or {}),
            ),
            operands=normalized_operands,
        )

    def call(
        self,
        callee: CallTarget,
        arguments: ir.Value | Sequence[ir.Value] = (),
    ) -> ir.Value | tuple[ir.Value, ...] | None:
        target = self._module.resolve(callee)
        match arguments:
            case ir.Value():
                normalized = (arguments,)
            case _:
                normalized = tuple(arguments)
        self._require_signature(target, normalized)
        results = tuple(
            self._new_value(result_type) for result_type in target.type.outputs
        )
        self._append(
            ir.Call(target.name, normalized, results),
            operands=normalized,
        )
        if not results:
            return None
        if len(results) == 1:
            return results[0]
        return results

    def return_(self, *operands: ir.Value) -> None:
        expected = self.function.type.outputs
        actual = tuple(value.type for value in operands)
        if actual != expected:
            raise TypeError(f"return types {actual!r} do not match {expected!r}")
        self._append(ir.Return(operands), operands=operands)

    def _new_value(self, value_type: ir.Type) -> ir.Value:
        if self._is_terminated():
            raise ValueError("cannot add an operation after return")
        value = ir.Value(ir.ValueId(len(self._values)), value_type)
        self._values.append(value)
        return value

    def _append_result(
        self,
        result_type: ir.Type,
        make_operation: Callable[[ir.Value], ir.Op],
        operands: Sequence[ir.Value] = (),
    ) -> ir.Value:
        (result,) = self._append_results(
            (result_type,),
            lambda results: make_operation(results[0]),
            operands,
        )
        return result

    def _append_results(
        self,
        result_types: Sequence[ir.Type],
        make_operation: Callable[[tuple[ir.Value, ...]], ir.Op],
        operands: Sequence[ir.Value] = (),
    ) -> tuple[ir.Value, ...]:
        if self._is_terminated():
            raise ValueError("cannot add an operation after return")
        for operand in operands:
            self._require_owned(operand)
        results = tuple(
            ir.Value(ir.ValueId(len(self._values) + index), result_type)
            for index, result_type in enumerate(result_types)
        )
        operation = make_operation(results)
        self._values.extend(results)
        self._block.operations.append(operation)
        return results

    def _append(self, operation: ir.Op, operands: Sequence[ir.Value] = ()) -> None:
        if self._is_terminated():
            raise ValueError("cannot add an operation after return")
        for operand in operands:
            self._require_owned(operand)
        self._block.operations.append(operation)

    def _is_terminated(self) -> builtins.bool:
        if not self._block.operations:
            return False
        match self._block.operations[-1]:
            case ir.Return():
                return True
            case _:
                return False

    def _require_owned(self, value: ir.Value) -> None:
        index = int(value.id)
        if index < 0 or index >= len(self._values):
            raise ValueError("value does not belong to this function")
        if self._values[index] is not value:
            raise ValueError("value does not belong to this function")

    def _require_signature(
        self,
        function: ir.Function,
        arguments: tuple[ir.Value, ...],
    ) -> None:
        for argument in arguments:
            self._require_owned(argument)
        actual = tuple(argument.type for argument in arguments)
        if actual != function.type.inputs:
            raise TypeError(
                f"call argument types {actual!r} do not match {function.type.inputs!r}"
            )
