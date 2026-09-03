from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence

from niro import ir

type CallTarget = FunctionBuilder | ir.Function | ir.SymbolName
type GlobalTarget = ir.Global | ir.SymbolName


class Ctx:
    def __init__(self, module: ir.Module, function: ir.Function) -> None:
        self._module = module
        self._function = function
        self._next_value_id = 0

    def new_value(self, type: ir.Type) -> ir.Value:
        value = ir.Value(ir.ValueId(self._next_value_id), type)
        self._next_value_id += 1
        return value

    def resolve_function(self, target: CallTarget) -> ir.Function:
        name = (
            target.ir.name
            if isinstance(target, FunctionBuilder)
            else (target.name if isinstance(target, ir.Function) else target)
        )
        for function in self._module.functions:
            if function.name == name:
                return function
        raise ValueError(f"unknown function: {name!r}")

    def resolve_global(self, target: GlobalTarget) -> ir.Global:
        name = target.name if isinstance(target, ir.Global) else target
        for global_ in self._module.globals:
            if global_.name == name:
                return global_
        raise ValueError(f"unknown global: {name!r}")


class ModuleBuilder:
    def __init__(self) -> None:
        self.ir = ir.Module()

    def function(
        self,
        *,
        name: ir.SymbolName,
        type: ir.FunctionType,
        input_names: Sequence[str | None] | None = None,
        output_names: Sequence[str | None] | None = None,
        attributes: Mapping[ir.AttributeName, ir.AttributeValue] | None = None,
    ) -> FunctionBuilder:
        self._require_available_symbol(name)
        fn = ir.Function(
            name=name,
            type=type,
            input_names=None if input_names is None else tuple(input_names),
            output_names=None if output_names is None else tuple(output_names),
            attributes=dict(attributes or {}),
        )
        self.ir.functions.append(fn)
        return FunctionBuilder(Ctx(self.ir, fn), fn)

    def global_(
        self, name: ir.SymbolName, type: ir.Type, initializer: ir.Literal
    ) -> ir.Global:
        self._require_available_symbol(name)
        global_ = ir.Global(name, type, initializer)
        self.ir.globals.append(global_)
        return global_

    def _require_available_symbol(self, name: ir.SymbolName) -> None:
        names = {item.name for item in [*self.ir.globals, *self.ir.functions]}
        if name in names:
            raise ValueError("module symbol names must be unique")


class FunctionBuilder:
    def __init__(
        self,
        ctx: Ctx,
        function: ir.Function,
    ) -> None:
        self._ctx = ctx
        self.ir = function

    def region(self) -> RegionBuilder:
        if self.ir.body:
            raise ValueError("function already has a body")
        body = ir.Region()
        self.ir.body = body
        return RegionBuilder(self._ctx, body)


class RegionBuilder:
    def __init__(self, ctx: Ctx, region: ir.Region) -> None:
        self._ctx = ctx
        self.ir = region

    def first_block(self) -> BlockBuilder:
        if self.ir.blocks:
            raise ValueError("region already has blocks")
        return self.block()

    def block(self, arg_types: Sequence[ir.Type] = ()) -> BlockBuilder:
        """Append a block with arguments of the given types."""
        if self.ir.blocks:
            raise ValueError("multiple blocks per region are not supported")
        args = tuple(self._ctx.new_value(type) for type in arg_types)
        block = ir.Block(arguments=args)
        builder = BlockBuilder(self._ctx, block)
        self.ir.blocks.append(block)
        return builder


class BlockBuilder:
    def __init__(
        self,
        ctx: Ctx,
        block: ir.Block,
    ) -> None:
        self._ctx = ctx
        self.ir = block

    def _append_operation[Op: ir.Op](
        self,
        result_types: Sequence[ir.Type],
        create_op: Callable[[tuple[ir.Value, ...]], Op],
    ) -> Op:
        if self.ir.operations and self.ir.operations[-1].is_terminator():
            raise ValueError("cannot append an operation after a block terminator")
        results = tuple(self._ctx.new_value(type) for type in result_types)
        op = create_op(results)
        self.ir.operations.append(op)
        return op

    def const(self, literal: ir.Literal, type: ir.Type) -> ir.Value:
        def create(results: tuple[ir.Value, ...]) -> ir.Const:
            (result,) = results
            return ir.Const(result=result, literal=literal)

        op = self._append_operation([type], create)
        return op.result

    def get_global(self, global_: GlobalTarget) -> ir.Value:
        resolved = self._ctx.resolve_global(global_)

        def create(results: tuple[ir.Value, ...]) -> ir.GetGlobal:
            (result,) = results
            return ir.GetGlobal(name=resolved.name, result=result)

        op = self._append_operation([resolved.type], create)
        return op.result

    def bool(self, value: builtins.bool) -> ir.Value:
        return self.const(value, ir.ScalarType.BOOL)

    def i32(self, value: int) -> ir.Value:
        return self.const(value, ir.ScalarType.I32)

    def i64(self, value: int) -> ir.Value:
        return self.const(value, ir.ScalarType.I64)

    def f32(self, value: float) -> ir.Value:
        return self.const(value, ir.ScalarType.F32)

    def f64(self, value: float) -> ir.Value:
        return self.const(value, ir.ScalarType.F64)

    def tensor(self, data: bytes, type: ir.TensorType) -> ir.Value:
        return self.const(data, type)

    def add(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        def create(results: tuple[ir.Value, ...]) -> ir.Add:
            (result,) = results
            return ir.Add(result=result, lhs=lhs, rhs=rhs)

        op = self._append_operation([lhs.type], create)
        return op.result

    def mul(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        def create(results: tuple[ir.Value, ...]) -> ir.Mul:
            (result,) = results
            return ir.Mul(result=result, lhs=lhs, rhs=rhs)

        op = self._append_operation([lhs.type], create)
        return op.result

    def matmul(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        type = ir.matmul_result_type(lhs.type, rhs.type)

        def create(results: tuple[ir.Value, ...]) -> ir.MatMul:
            (result,) = results
            return ir.MatMul(result=result, lhs=lhs, rhs=rhs)

        op = self._append_operation([type], create)
        return op.result

    def transpose(
        self,
        operand: ir.Value,
        permutation: Sequence[int],
    ) -> ir.Value:
        permutation = tuple(permutation)
        type = ir.transpose_result_type(operand.type, permutation)

        def create(results: tuple[ir.Value, ...]) -> ir.Transpose:
            (result,) = results
            return ir.Transpose(
                result=result,
                operand=operand,
                permutation=permutation,
            )

        op = self._append_operation([type], create)
        return op.result

    def unknown_op(
        self,
        name: str,
        operands: Sequence[ir.Value] = (),
        result_types: Sequence[ir.Type] = (),
        attributes: Mapping[ir.AttributeName, ir.AttributeValue] | None = None,
    ) -> tuple[ir.Value, ...]:
        operands = tuple(operands)

        def create(results: tuple[ir.Value, ...]) -> ir.UnknownOp:
            return ir.UnknownOp(
                name=name,
                operands=operands,
                results=results,
                attributes=dict(attributes or {}),
            )

        op = self._append_operation(result_types, create)
        return op.results

    def if_(
        self,
        condition: ir.Value,
        result_types: Sequence[ir.Type] = (),
    ) -> IfBuilder:
        then_region = RegionBuilder(self._ctx, ir.Region())
        else_region = RegionBuilder(self._ctx, ir.Region())

        def create(results: tuple[ir.Value, ...]) -> ir.If:
            return ir.If(
                results=results,
                condition=condition,
                then_region=then_region.ir,
                else_region=else_region.ir,
            )

        op = self._append_operation(result_types, create)
        return IfBuilder(op, then_region, else_region)

    def call(
        self,
        callee: CallTarget,
        arguments: Sequence[ir.Value] = (),
    ) -> tuple[ir.Value, ...]:
        function = self._ctx.resolve_function(callee)
        actual_types = tuple(argument.type for argument in arguments)
        if actual_types != function.type.inputs:
            raise TypeError(
                f"call argument types {actual_types!r} do not match "
                f"{function.type.inputs!r}"
            )

        def create(results: tuple[ir.Value, ...]) -> ir.Call:
            return ir.Call(
                callee=function.name,
                arguments=tuple(arguments),
                results=results,
            )

        op = self._append_operation(function.type.outputs, create)
        return op.results

    def return_(self, *operands: ir.Value) -> None:
        def create(results: tuple[ir.Value, ...]) -> ir.Return:
            assert not results
            return ir.Return(operands=operands)

        self._append_operation([], create)

    def yield_(self, *operands: ir.Value) -> None:
        def create(results: tuple[ir.Value, ...]) -> ir.Yield:
            assert not results
            return ir.Yield(operands=operands)

        self._append_operation([], create)


class IfBuilder:
    def __init__(
        self,
        if_: ir.If,
        then_region: RegionBuilder,
        else_region: RegionBuilder,
    ) -> None:
        self.ir = if_
        self.then_region = then_region
        self.else_region = else_region
