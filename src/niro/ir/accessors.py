"""IR accessors and reference records, re-exported in [`niro.ir`][].

Use [`get_definition`][niro.ir.get_definition] to locate a value's defining
operation result or block argument, [`iter_uses`][niro.ir.iter_uses] to find
its operand slots, and [`get_parent_block`][niro.ir.get_parent_block] to locate
an operation's immediate containing block. These searches include nested regions.
[`iter_blocks`][niro.ir.iter_blocks], [`iter_ops`][niro.ir.iter_ops], and
[`iter_defined_values`][niro.ir.iter_defined_values] provide depth-first traversal.

A [`Definition`][niro.ir.Definition] is either an
[`OpResult`][niro.ir.OpResult] or a [`BlockArgument`][niro.ir.BlockArgument].
A [`Use`][niro.ir.Use] identifies one operand slot: `add(x, x)` has two uses of
`x`, with the same operation as `owner` and operand indices 0 and 1.
Owners are references to existing IR objects; the records do not copy values.
Records refer to positions in the current IR; repeat the query after edits that
change those positions.

Examples:
    Locate definitions, uses, and the containing block of an addition:

    ```python
    from niro import ir

    x = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    y = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    add = ir.Add(y, x, x)
    block = ir.Block(arguments=(x,), operations=[add, ir.Return((y,))])
    function = ir.Function(
        "double",
        ir.FunctionType((x.type,), (y.type,)),
        ir.Region([block]),
    )

    assert ir.get_definition(function, x.id) == ir.BlockArgument(block, 0)
    assert ir.get_definition(function, y.id) == ir.OpResult(add, 0)
    assert list(ir.iter_uses(function, x.id)) == [ir.Use(add, 0), ir.Use(add, 1)]
    assert ir.get_parent_block(function, add) is block
    ```
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import assert_never

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
from niro.ir.program import Block, Function, Region
from niro.ir.values import Value, ValueId

__all__ = [
    "BlockArgument",
    "Definition",
    "OpResult",
    "Use",
    "get_definition",
    "get_operands",
    "get_parent_block",
    "get_regions",
    "get_results",
    "iter_blocks",
    "iter_defined_values",
    "iter_ops",
    "iter_uses",
]


@dataclass(frozen=True, slots=True)
class OpResult:
    """A reference to one result of an existing operation.

    Attributes:
        owner: Operation defining the value.
        result_index: Zero-based index into [`get_results`][niro.ir.get_results].
    """

    owner: Op
    result_index: int


@dataclass(frozen=True, slots=True)
class BlockArgument:
    """A reference to one argument of an existing block.

    Attributes:
        owner: Block defining the value.
        argument_index: Zero-based index into the owner's arguments.
    """

    owner: Block
    argument_index: int


type Definition = OpResult | BlockArgument
"""The operation result or block argument defining an SSA value."""


@dataclass(frozen=True, slots=True)
class Use:
    """A reference to one operand slot using an SSA value.

    An operation using the same value twice has two uses with different indices.
    The owner owns the operand slot, not the referenced value.

    Attributes:
        owner: Operation containing the operand slot.
        operand_index: Zero-based index into [`get_operands`][niro.ir.get_operands].
    """

    owner: Op
    operand_index: int


def get_definition(function: Function, value_id: ValueId) -> Definition | None:
    """Find a value's definition anywhere in a function, including nested regions.

    Match by function-local ID. Assume definition IDs are unique, as required by
    the IR; this lookup does not verify the function. Each call scans the body.

    Args:
        function: Function to search.
        value_id: ID of the value whose definition is requested.

    Returns:
        A reference to the defining result or block argument, or None if absent
        or the function has no body. Owners are the original IR objects.
    """
    if function.body is None:
        return None
    for block in iter_blocks(function.body):
        for index, argument in enumerate(block.arguments):
            if argument.id == value_id:
                return BlockArgument(block, index)
        for op in block.operations:
            for index, result in enumerate(get_results(op)):
                if result.id == value_id:
                    return OpResult(op, index)
    return None


def iter_uses(function: Function, value_id: ValueId) -> Iterator[Use]:
    """Yield every operand slot referencing a function-local value ID.

    Scan operations in [`iter_ops`][niro.ir.iter_ops] order, then operands in
    index order. Include nested captures and repeated operands. No definition
    lookup or verification is performed; each call scans the body.

    Args:
        function: Function to search.
        value_id: ID of the referenced value.

    Yields:
        References to operand slots on the original operations. Yield nothing
        when no operands match or the function has no body.
    """
    if function.body is None:
        return
    for op in iter_ops(function.body):
        for index, operand in enumerate(get_operands(op)):
            if operand.id == value_id:
                yield Use(op, index)


def get_parent_block(function: Function, op: Op) -> Block | None:
    """Find an operation's immediate containing block, including nested regions.

    Match the operation by object identity, not structural equality. Each call
    scans the body; no ownership links or cached index are maintained.

    Args:
        function: Function to search.
        op: Operation object to locate.

    Returns:
        The original containing block, or None if the operation is absent or
        the function has no body.
    """
    if function.body is None:
        return None
    for block in iter_blocks(function.body):
        if any(candidate is op for candidate in block.operations):
            return block
    return None


def iter_blocks(region: Region) -> Iterator[Block]:
    """Yield blocks depth-first, before blocks in their operations' nested regions.

    Preserve block and operation order; visit an [`If`][niro.ir.ops.If]'s then
    region before its else region. Include empty blocks without copying them.
    A region with no blocks yields nothing.
    """
    for block in region.blocks:
        yield block
        for op in block.operations:
            for nested_region in get_regions(op):
                yield from iter_blocks(nested_region)


def iter_defined_values(region: Region) -> Iterator[Value]:
    """Yield block arguments and operation results, visiting nested regions depth-first."""
    for block in region.blocks:
        yield from block.arguments
        for op in block.operations:
            yield from get_results(op)
            for nested_region in get_regions(op):
                yield from iter_defined_values(nested_region)


def iter_ops(region: Region) -> Iterator[Op]:
    """Yield operations depth-first, visiting parents before their nested regions.

    Preserve block and operation order; visit an [`If`][niro.ir.ops.If]'s then
    region before its else region. Yield the original operations without copying.
    """
    for block in region.blocks:
        for op in block.operations:
            yield op
            for nested_region in get_regions(op):
                yield from iter_ops(nested_region)


def get_operands(op: Op) -> tuple[Value, ...]:
    """Return immediate SSA operands in their defined order, without copying values."""
    match op:
        case Const() | GetGlobal():
            return ()
        case Transpose(operand=operand):
            return (operand,)
        case Add(lhs=lhs, rhs=rhs) | Mul(lhs=lhs, rhs=rhs) | MatMul(lhs=lhs, rhs=rhs):
            return lhs, rhs
        case Call(arguments=arguments):
            return arguments
        case Return(operands=operands) | Yield(operands=operands):
            return operands
        case If(condition=condition):
            return (condition,)
        case UnknownOp(operands=operands):
            return operands

        case _ as unreachable:
            assert_never(unreachable)


def get_results(op: Op) -> tuple[Value, ...]:
    """Return immediate SSA results in their defined order, without copying values."""
    match op:
        case (
            Const(result=result)
            | GetGlobal(result=result)
            | Transpose(result=result)
            | Add(result=result)
            | Mul(result=result)
            | MatMul(result=result)
        ):
            return (result,)
        case Call(results=results) | If(results=results) | UnknownOp(results=results):
            return results
        case Return() | Yield():
            return ()
        case _ as unreachable:
            assert_never(unreachable)


def get_regions(op: Op) -> tuple[Region, ...]:
    """Return immediate regions without copying them; If orders then before else."""
    match op:
        case If(then_region=then_region, else_region=else_region):
            return then_region, else_region
        case (
            Const()
            | GetGlobal()
            | Transpose()
            | Add()
            | Mul()
            | MatMul()
            | Call()
            | Return()
            | Yield()
            | UnknownOp()
        ):
            return ()
        case _ as unreachable:
            assert_never(unreachable)
