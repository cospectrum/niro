"""Convenient construction API for Niro IR."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import Final

from niro import ir

ENTRY_POINT_ATTR: Final = "entry_point"

type CallTarget = FunctionBuilder | ir.Function | str


class ModuleBuilder:
    """Build a module and maintain its function symbol table.

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
        self._require_unique_function_name(name)
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
        self._require_unique_function_name(name)
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

    def _require_unique_function_name(self, name: str) -> None:
        if not name:
            raise ValueError("function name cannot be empty")
        if name in self._functions:
            raise ValueError(f"duplicate function: {name!r}")

    def _add_function(self, function: ir.Function) -> None:
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
        result = self._new_value(result_type)
        self._append(ir.Const(result, value))
        return result

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
        if not isinstance(data, bytes):
            raise TypeError("tensor data must be bytes")
        return self.constant(data, result_type)

    def add(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        self._require_same_type("add", lhs, rhs)
        result = self._new_value(lhs.type)
        self._append(ir.Add(result, lhs, rhs), operands=(lhs, rhs))
        return result

    def mul(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        self._require_same_type("mul", lhs, rhs)
        result = self._new_value(lhs.type)
        self._append(ir.Mul(result, lhs, rhs), operands=(lhs, rhs))
        return result

    def matmul(
        self,
        lhs: ir.Value,
        rhs: ir.Value,
        result_type: ir.Type,
    ) -> ir.Value:
        self._require_owned(lhs)
        self._require_owned(rhs)
        result = self._new_value(result_type)
        self._append(ir.MatMul(result, lhs, rhs), operands=(lhs, rhs))
        return result

    def transpose(
        self,
        operand: ir.Value,
        permutation: Sequence[int],
    ) -> ir.Value:
        self._require_owned(operand)
        normalized = tuple(permutation)
        match operand.type:
            case ir.TensorType(_, None):
                result_type = operand.type
            case ir.TensorType(element_type, shape) if shape is not None:
                if sorted(normalized) != list(range(len(shape))):
                    raise ValueError(
                        "transpose permutation must contain every dimension once"
                    )
                result_type = ir.TensorType(
                    element_type,
                    tuple(shape[index] for index in normalized),
                )
            case _:
                raise TypeError("transpose operand must be a tensor")

        result = self._new_value(result_type)
        self._append(
            ir.Transpose(result, operand, normalized),
            operands=(operand,),
        )
        return result

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

    def _require_same_type(
        self,
        operation: str,
        lhs: ir.Value,
        rhs: ir.Value,
    ) -> None:
        self._require_owned(lhs)
        self._require_owned(rhs)
        if lhs.type != rhs.type:
            raise TypeError(f"{operation} operands must have the same type")

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
