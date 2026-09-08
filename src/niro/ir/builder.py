"""Builders for constructing Niro IR programs.

Start from [`ModuleBuilder`][niro.ir.builder.ModuleBuilder]; the nested builders
are obtained from it rather than constructed directly.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence

from niro.ir.data import AttributeName, AttributeValue, Literal
from niro.ir.infer import matmul_result_type, transpose_result_type
from niro.ir.ops import (
    Add,
    Call,
    Const,
    GetGlobal,
    If,
    MatMul,
    Mul,
    Op,
    Return,
    Transpose,
    UnknownOp,
    Yield,
)
from niro.ir.program import (
    Block,
    Function,
    FunctionType,
    Global,
    Module,
    Region,
    SymbolName,
)
from niro.ir.types import ScalarType, TensorType, Type
from niro.ir.values import Value, ValueId

type CallTarget = FunctionBuilder | Function | SymbolName
GlobalTarget = Global | SymbolName


class Builder[T]:
    """Base class for builders of a single IR object.

    Attributes:
        inner: The IR object under construction.
    """

    inner: T


class Ctx:
    def __init__(self, module: Module, function: Function) -> None:
        self._module = module
        self.function = function
        self._next_value_id = 0

    def new_value(self, type: Type) -> Value:
        value = Value(ValueId(self._next_value_id), type)
        self._next_value_id += 1
        return value

    def resolve_function(self, target: CallTarget) -> Function:
        name = (
            target.inner.name
            if isinstance(target, FunctionBuilder)
            else (target.name if isinstance(target, Function) else target)
        )
        for function in self._module.functions:
            if function.name == name:
                return function
        raise ValueError(f"unknown function: {name!r}")

    def resolve_global(self, target: GlobalTarget) -> Global:
        name = target.name if isinstance(target, Global) else target
        for global_ in self._module.globals:
            if global_.name == name:
                return global_
        raise ValueError(f"unknown global: {name!r}")


class ModuleBuilder(Builder[Module]):
    """Builder for [`niro.ir.Module`][]."""

    def __init__(self) -> None:
        self.inner: Module = Module()
        """The [`niro.ir.Module`][] under construction."""

    def function(
        self,
        *,
        name: SymbolName,
        type: FunctionType,
        input_names: Sequence[str | None] | None = None,
        output_names: Sequence[str | None] | None = None,
        attributes: Mapping[AttributeName, AttributeValue] | None = None,
    ) -> FunctionBuilder:
        """Declare a function and return its builder."""
        self._require_available_symbol(name)
        fn = Function(
            name=name,
            type=type,
            input_names=None if input_names is None else tuple(input_names),
            output_names=None if output_names is None else tuple(output_names),
            attributes=dict(attributes or {}),
        )
        self.inner.functions.append(fn)
        return FunctionBuilder(Ctx(self.inner, fn), fn)

    def global_(self, name: SymbolName, type: Type, initializer: Literal) -> Global:
        """Declare and return an initialized global."""
        self._require_available_symbol(name)
        global_ = Global(name, type, initializer)
        self.inner.globals.append(global_)
        return global_

    def _require_available_symbol(self, name: SymbolName) -> None:
        names = {item.name for item in [*self.inner.globals, *self.inner.functions]}
        if name in names:
            raise ValueError("module symbol names must be unique")


class FunctionBuilder(Builder[Function]):
    """Builder for [`niro.ir.Function`][].

    Attributes:
        inner: The [`niro.ir.Function`][] under construction.
    """

    def __init__(
        self,
        ctx: Ctx,
        function: Function,
    ) -> None:
        self._ctx = ctx
        self.inner: Function = function

    def region(self) -> RegionBuilder:
        """Create and return the function body region."""
        if self.inner.body:
            raise ValueError("function already has a body")
        body = Region()
        self.inner.body = body
        return RegionBuilder(self._ctx, body)


class RegionBuilder(Builder[Region]):
    """Builder for [`niro.ir.Region`][].

    Attributes:
        inner: The [`niro.ir.Region`][] under construction.
    """

    def __init__(self, ctx: Ctx, region: Region) -> None:
        self._ctx = ctx
        self.inner: Region = region

    def first_block(self) -> BlockBuilder:
        """Append the first block, deriving function input arguments."""
        if self.inner.blocks:
            raise ValueError("region already has a first block")
        arg_types = self._function_input_types if self._is_function_body else ()
        return self.block(arg_types)

    def block(self, arg_types: Sequence[Type] = ()) -> BlockBuilder:
        """Append a block with arguments of the given types."""
        if self.inner.blocks:
            raise ValueError("multiple blocks per region are not supported")
        if self._is_function_body and tuple(arg_types) != self._function_input_types:
            raise TypeError(
                "entry block argument types must match function input types"
            )
        args = tuple(self._ctx.new_value(type) for type in arg_types)
        block = Block(arguments=args)
        builder = BlockBuilder(self._ctx, block)
        self.inner.blocks.append(block)
        return builder

    @property
    def _is_function_body(self) -> bool:
        return self.inner is self._ctx.function.body

    @property
    def _function_input_types(self) -> tuple[Type, ...]:
        return self._ctx.function.type.inputs


class BlockBuilder(Builder[Block]):
    """Builder for [`niro.ir.Block`][].

    Attributes:
        inner: The [`niro.ir.Block`][] under construction.
    """

    def __init__(
        self,
        ctx: Ctx,
        block: Block,
    ) -> None:
        self._ctx = ctx
        self.inner: Block = block

    def _append_operation[OpT: Op](
        self,
        result_types: Sequence[Type],
        create_op: Callable[[tuple[Value, ...]], OpT],
    ) -> OpT:
        if self.inner.operations and self.inner.operations[-1].is_terminator():
            raise ValueError("cannot append an operation after a block terminator")
        results = tuple(self._ctx.new_value(type) for type in result_types)
        op = create_op(results)
        self.inner.operations.append(op)
        return op

    def const(self, literal: Literal, type: Type) -> Value:
        """Append a constant and return its result."""

        def create(results: tuple[Value, ...]) -> Const:
            (result,) = results
            return Const(result=result, literal=literal)

        op = self._append_operation([type], create)
        return op.result

    def get_global(self, global_: GlobalTarget) -> Value:
        """Append a global load and return its result."""
        resolved = self._ctx.resolve_global(global_)

        def create(results: tuple[Value, ...]) -> GetGlobal:
            (result,) = results
            return GetGlobal(name=resolved.name, result=result)

        op = self._append_operation([resolved.type], create)
        return op.result

    def bool(self, value: builtins.bool) -> Value:
        """Append a boolean constant and return its result."""
        return self.const(value, ScalarType.BOOL)

    def i32(self, value: int) -> Value:
        """Append an I32 constant and return its result."""
        return self.const(value, ScalarType.I32)

    def i64(self, value: int) -> Value:
        """Append an I64 constant and return its result."""
        return self.const(value, ScalarType.I64)

    def f32(self, value: float) -> Value:
        """Append an F32 constant and return its result."""
        return self.const(value, ScalarType.F32)

    def f64(self, value: float) -> Value:
        """Append an F64 constant and return its result."""
        return self.const(value, ScalarType.F64)

    def tensor(self, data: bytes, type: TensorType) -> Value:
        """Append a tensor constant and return its result."""
        return self.const(data, type)

    def add(self, lhs: Value, rhs: Value) -> Value:
        """Append an addition and return its result."""

        def create(results: tuple[Value, ...]) -> Add:
            (result,) = results
            return Add(result=result, lhs=lhs, rhs=rhs)

        op = self._append_operation([lhs.type], create)
        return op.result

    def mul(self, lhs: Value, rhs: Value) -> Value:
        """Append a multiplication and return its result."""

        def create(results: tuple[Value, ...]) -> Mul:
            (result,) = results
            return Mul(result=result, lhs=lhs, rhs=rhs)

        op = self._append_operation([lhs.type], create)
        return op.result

    def matmul(self, lhs: Value, rhs: Value) -> Value:
        """Append a matrix multiplication and return its result."""
        type = matmul_result_type(lhs.type, rhs.type)

        def create(results: tuple[Value, ...]) -> MatMul:
            (result,) = results
            return MatMul(result=result, lhs=lhs, rhs=rhs)

        op = self._append_operation([type], create)
        return op.result

    def transpose(
        self,
        operand: Value,
        permutation: Sequence[int],
    ) -> Value:
        """Append a transpose and return its result."""
        permutation = tuple(permutation)
        type = transpose_result_type(operand.type, permutation)

        def create(results: tuple[Value, ...]) -> Transpose:
            (result,) = results
            return Transpose(
                result=result,
                operand=operand,
                permutation=permutation,
            )

        op = self._append_operation([type], create)
        return op.result

    def unknown_op(
        self,
        name: str,
        operands: Sequence[Value] = (),
        result_types: Sequence[Type] = (),
        attributes: Mapping[AttributeName, AttributeValue] | None = None,
    ) -> tuple[Value, ...]:
        """Append an unknown operation and return its results."""
        operands = tuple(operands)

        def create(results: tuple[Value, ...]) -> UnknownOp:
            return UnknownOp(
                name=name,
                operands=operands,
                results=results,
                attributes=dict(attributes or {}),
            )

        op = self._append_operation(result_types, create)
        return op.results

    def if_(
        self,
        condition: Value,
        result_types: Sequence[Type] = (),
    ) -> IfBuilder:
        """Append a conditional and return its region builders."""
        then_region = RegionBuilder(self._ctx, Region())
        else_region = RegionBuilder(self._ctx, Region())

        def create(results: tuple[Value, ...]) -> If:
            return If(
                results=results,
                condition=condition,
                then_region=then_region.inner,
                else_region=else_region.inner,
            )

        op = self._append_operation(result_types, create)
        return IfBuilder(op, then_region, else_region)

    def call(
        self,
        callee: CallTarget,
        arguments: Sequence[Value] = (),
    ) -> tuple[Value, ...]:
        """Append a function call and return its results."""
        function = self._ctx.resolve_function(callee)
        actual_types = tuple(argument.type for argument in arguments)
        if actual_types != function.type.inputs:
            raise TypeError(
                f"call argument types {actual_types!r} do not match "
                f"{function.type.inputs!r}"
            )

        def create(results: tuple[Value, ...]) -> Call:
            return Call(
                callee=function.name,
                arguments=tuple(arguments),
                results=results,
            )

        op = self._append_operation(function.type.outputs, create)
        return op.results

    def return_(self, *operands: Value) -> None:
        """Terminate the block by returning values from the function."""

        def create(results: tuple[Value, ...]) -> Return:
            assert not results
            return Return(operands=operands)

        self._append_operation([], create)

    def yield_(self, *operands: Value) -> None:
        """Terminate the block by yielding values from a nested region."""

        def create(results: tuple[Value, ...]) -> Yield:
            assert not results
            return Yield(operands=operands)

        self._append_operation([], create)


class IfBuilder(Builder[If]):
    """Builder for the regions of [`niro.ir.If`][].

    Attributes:
        inner: The [`niro.ir.If`][] under construction.
        then_region: The [`niro.ir.builder.RegionBuilder`][] for the taken branch.
        else_region: The [`niro.ir.builder.RegionBuilder`][] for the other branch.
    """

    def __init__(
        self,
        if_: If,
        then_region: RegionBuilder,
        else_region: RegionBuilder,
    ) -> None:
        self.inner: If = if_
        self.then_region: RegionBuilder = then_region
        self.else_region: RegionBuilder = else_region
