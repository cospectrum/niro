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
    VerifiedModule,
)
from niro.ir.types import ScalarType, TensorType, Type
from niro.ir.values import Value, ValueId
from niro.ir.verifier import verify


class Builder[T]:
    """Base class for builders of a single IR object.

    Attributes:
        raw: The IR object under construction.
    """

    raw: T


class ModuleCtx:
    def __init__(self, module: Module) -> None:
        self._module = module

    def resolve_function(self, name: SymbolName) -> Function | None:
        for function in self._module.functions:
            if function.name == name:
                return function
        return None

    def resolve_global(self, name: SymbolName) -> Global | None:
        for global_ in self._module.globals:
            if global_.name == name:
                return global_
        return None


class FunctionCtx(ModuleCtx):
    def __init__(self, module: Module, function: Function) -> None:
        super().__init__(module)
        self.function = function
        self._next_value_id = 0

    def new_value(self, type: Type) -> Value:
        value = Value(ValueId(self._next_value_id), type)
        self._next_value_id += 1
        return value


class ModuleBuilder(Builder[Module]):
    """Builder for [`niro.ir.Module`][]."""

    def __init__(self) -> None:
        self.raw: Module = Module()
        """The [`niro.ir.Module`][] under construction."""

    def verify(self) -> VerifiedModule:
        """Verify and return the module under construction."""
        return verify(self.raw)

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
        fn = Function(
            name=name,
            type=type,
            input_names=None if input_names is None else tuple(input_names),
            output_names=None if output_names is None else tuple(output_names),
            attributes=dict(attributes or {}),
        )
        self.raw.functions.append(fn)
        return FunctionBuilder(FunctionCtx(self.raw, fn), fn)

    def global_(self, name: SymbolName, type: Type, initializer: Literal) -> Global:
        """Declare and return an initialized global."""
        global_ = Global(name, type, initializer)
        self.raw.globals.append(global_)
        return global_


class FunctionBuilder(Builder[Function]):
    """Builder for [`niro.ir.Function`][].

    Attributes:
        raw: The [`niro.ir.Function`][] under construction.
    """

    def __init__(
        self,
        ctx: FunctionCtx,
        function: Function,
    ) -> None:
        self._ctx = ctx
        self.raw: Function = function

    def region(self) -> FunctionRegionBuilder:
        """Create and return the function body region."""
        if self.raw.body:
            raise ValueError("function already has a body")
        body = Region()
        self.raw.body = body
        return FunctionRegionBuilder(self._ctx, body)


class FunctionRegionBuilder(Builder[Region]):
    """Builder for a function body region, with entry-block argument inference.

    Attributes:
        raw: The [`niro.ir.Region`][] under construction.
    """

    def __init__(self, ctx: FunctionCtx, region: Region) -> None:
        self._ctx = ctx
        self.raw: Region = region

    def block(self, arg_types: Sequence[Type] = ()) -> BlockBuilder:
        """Append a block with arguments of the given types."""
        args = tuple(self._ctx.new_value(type) for type in arg_types)
        block = Block(arguments=args)
        builder = BlockBuilder(self._ctx, block)
        self.raw.blocks.append(block)
        return builder

    def first_block(self) -> BlockBuilder:
        """Append the first block, deriving function input arguments."""
        if self.raw.blocks:
            raise ValueError("function region already has a first block")
        return self.block(self._ctx.function.type.inputs)


class IfRegionBuilder(Builder[Region]):
    """Builder for an If branch containing a single argument-free block.

    Attributes:
        raw: The [`niro.ir.Region`][] under construction.
    """

    def __init__(self, ctx: FunctionCtx, region: Region) -> None:
        self._ctx = ctx
        self.raw: Region = region

    def block(self) -> BlockBuilder:
        """Create the branch's only block, without arguments."""
        if self.raw.blocks:
            raise ValueError("if region already has a block")
        block = Block()
        builder = BlockBuilder(self._ctx, block)
        self.raw.blocks.append(block)
        return builder


class BlockBuilder(Builder[Block]):
    """Builder for [`niro.ir.Block`][].

    Attributes:
        raw: The [`niro.ir.Block`][] under construction.
    """

    def __init__(
        self,
        ctx: FunctionCtx,
        block: Block,
    ) -> None:
        self._ctx = ctx
        self.raw: Block = block

    def _append_operation[OpT: Op](
        self,
        result_types: Sequence[Type],
        create_op: Callable[[tuple[Value, ...]], OpT],
    ) -> OpT:
        results = tuple(self._ctx.new_value(type) for type in result_types)
        op = create_op(results)
        self.raw.operations.append(op)
        return op

    def const(self, literal: Literal, type: Type) -> Value:
        """Append a constant and return its result."""

        def create(results: tuple[Value, ...]) -> Const:
            (result,) = results
            return Const(result=result, literal=literal)

        op = self._append_operation([type], create)
        return op.result

    def get_global(
        self, global_: Global | SymbolName, *, type: Type | None = None
    ) -> Value:
        """Append a global load, inferring its type from the module when possible.

        Supply ``type`` for an undeclared global. If declared, an explicit type
        must match the declaration.
        """
        name = global_.name if isinstance(global_, Global) else global_
        resolved = self._ctx.resolve_global(name)
        if type is None:
            type = resolved.type if resolved is not None else None

        if resolved and type != resolved.type:
            raise TypeError("global result type does not match its declaration")

        if type is None:
            raise ValueError(f"type is required for unknown global: {name!r}")

        def create(results: tuple[Value, ...]) -> GetGlobal:
            (result,) = results
            return GetGlobal(name=name, result=result)

        op = self._append_operation([type], create)
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
        then_region = IfRegionBuilder(self._ctx, Region())
        else_region = IfRegionBuilder(self._ctx, Region())

        def create(results: tuple[Value, ...]) -> If:
            return If(
                results=results,
                condition=condition,
                then_region=then_region.raw,
                else_region=else_region.raw,
            )

        op = self._append_operation(result_types, create)
        return IfBuilder(op, then_region, else_region)

    def call(
        self,
        callee: FunctionBuilder | Function | SymbolName,
        arguments: Sequence[Value] = (),
        *,
        result_types: Sequence[Type] | None = None,
    ) -> tuple[Value, ...]:
        """Append a call, inferring result types from the module when possible.

        Supply ``result_types`` for an undeclared function, including ``()`` for
        no results. If declared, explicit types must match its output types.
        """
        name = (
            callee.raw.name
            if isinstance(callee, FunctionBuilder)
            else (callee.name if isinstance(callee, Function) else callee)
        )
        function = self._ctx.resolve_function(name)
        if result_types is None:
            result_types = function.type.outputs if function is not None else None

        if function and tuple(result_types or ()) != function.type.outputs:
            raise TypeError("call result types do not match its declaration")

        if result_types is None:
            raise ValueError(
                f"result_types are required for unknown function: {name!r}"
            )

        def create(results: tuple[Value, ...]) -> Call:
            return Call(
                callee=name,
                arguments=tuple(arguments),
                results=results,
            )

        op = self._append_operation(result_types, create)
        return op.results

    def return_(self, *operands: Value) -> None:
        """Terminate the block by returning values from the function."""

        def create(results: tuple[Value, ...]) -> Return:
            return Return(operands=operands)

        self._append_operation([], create)

    def yield_(self, *operands: Value) -> None:
        """Terminate the block by yielding values from a nested region."""

        def create(results: tuple[Value, ...]) -> Yield:
            return Yield(operands=operands)

        self._append_operation([], create)


class IfBuilder(Builder[If]):
    """Builder for the regions of [`niro.ir.If`][].

    Attributes:
        raw: The [`niro.ir.If`][] under construction.
        then_region: The [`niro.ir.builder.IfRegionBuilder`][] for the taken branch.
        else_region: The [`niro.ir.builder.IfRegionBuilder`][] for the other branch.
    """

    def __init__(
        self,
        if_: If,
        then_region: IfRegionBuilder,
        else_region: IfRegionBuilder,
    ) -> None:
        self.raw: If = if_
        self.then_region: IfRegionBuilder = then_region
        self.else_region: IfRegionBuilder = else_region
