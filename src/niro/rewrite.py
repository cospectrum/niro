"""Functional helpers for editing SSA IR.

Edits return new IR without mutating their inputs. Unchanged objects and metadata
may be shared: treat both versions as immutable, even while the IR schema still
contains mutable containers. Operation targets are matched by identity and
insertion points refer to the input version; reacquire references after edits.

Operation edits check unique definitions, operand references, and replacement
types. Passes remain responsible for dominance, region scope, terminators, and
semantic equivalence (including effects). Verify the completed module with
[`niro.verify.module`][]. Traversal and cloning also accept incomplete fragments.

Examples:
    Remove an identity transpose and redirect its consumers:

    ```python
    from niro import ir, rewrite

    tensor = ir.TensorType(ir.ScalarType.F32, (2, 3))
    x = ir.Value(ir.ValueId(0), tensor)
    y = ir.Value(ir.ValueId(1), tensor)
    transpose = ir.Transpose(y, x, (0, 1))
    block = ir.Block(arguments=(x,), operations=[transpose, ir.Return((y,))])
    function = ir.Function(
        "identity", ir.FunctionType((tensor,), (tensor,)), ir.Region([block])
    )
    optimized = rewrite.replace_ops(
        function, (transpose,), (), replacements={y.id: x}
    )
    assert optimized.first_block.operations == [ir.Return((x,))]
    assert len(block.operations) == 2
    ```
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import assert_never

from niro import ir
from niro.ir import Block, Function, Op, Region, Type, Use, Value, ValueId

__all__ = [
    "InsertPoint",
    "ValueSupply",
    "clone_region",
    "erase_ops",
    "fresh_value",
    "insert_ops",
    "map_blocks",
    "map_regions",
    "move_ops",
    "replace_ops",
    "replace_uses",
    "value_supply",
    "with_operands",
]


@dataclass(frozen=True, slots=True)
class InsertPoint:
    """A gap in an input block, before operations[index] or at the end.

    Attributes:
        block: Block in the function version being edited.
        index: Position from zero through the number of operations, inclusive.
    """

    block: Block
    index: int

    def __post_init__(self) -> None:
        if not 0 <= self.index <= len(self.block.operations):
            raise ValueError("insertion index is outside the block")


@dataclass(frozen=True, slots=True)
class ValueSupply:
    """An immutable source of fresh function-local value IDs.

    Attributes:
        next_id: Next unused ID. Thread the returned supply through allocations;
            reusing an old supply would allocate the same IDs again.
    """

    next_id: int = 0

    def __post_init__(self) -> None:
        if self.next_id < 0:
            raise ValueError("next value ID must be nonnegative")


def value_supply(function: Function) -> ValueSupply:
    """Create a supply above every definition ID, including nested regions."""
    values = () if function.body is None else ir.iter_defined_values(function.body)
    return ValueSupply(1 + max((value.id for value in values), default=-1))


def fresh_value(supply: ValueSupply, type: Type) -> tuple[Value, ValueSupply]:
    """Return a fresh typed value and the advanced supply, without changing either input."""
    return ir.Value(ir.ValueId(supply.next_id), type), ValueSupply(supply.next_id + 1)


def with_operands(op: Op, operands: Sequence[Value]) -> Op:
    """Replace immediate operands in [`niro.ir.get_operands`][] order.

    Require the same arity. Definitions, nested regions, and other fields are
    preserved. This low-level helper permits type changes for coordinated edits.
    """
    operands = tuple(operands)
    if len(operands) != len(ir.get_operands(op)):
        raise ValueError("operand arity does not match")
    if operands == ir.get_operands(op):
        return op
    match op:
        case ir.Const() | ir.GetGlobal():
            return op
        case ir.Transpose():
            return dataclasses.replace(op, operand=operands[0])
        case ir.Add() | ir.Mul() | ir.MatMul():
            return dataclasses.replace(op, lhs=operands[0], rhs=operands[1])
        case ir.Call():
            return dataclasses.replace(op, arguments=operands)
        case ir.Return() | ir.Yield() | ir.UnknownOp():
            return dataclasses.replace(op, operands=operands)
        case ir.If():
            return dataclasses.replace(op, condition=operands[0])
        case _ as unreachable:
            assert_never(unreachable)


def map_regions(op: Op, transform: Callable[[Region], Region]) -> Op:
    """Transform immediate regions in accessor order; recursion is explicit.

    The callback must not mutate its input. Operations without regions are
    returned unchanged. No SSA verification is performed.
    """
    regions = tuple(transform(region) for region in ir.get_regions(op))
    match op:
        case ir.If():
            if regions[0] is op.then_region and regions[1] is op.else_region:
                return op
            return dataclasses.replace(
                op, then_region=regions[0], else_region=regions[1]
            )
        case (
            ir.Const()
            | ir.GetGlobal()
            | ir.Transpose()
            | ir.Add()
            | ir.Mul()
            | ir.MatMul()
            | ir.Call()
            | ir.Return()
            | ir.Yield()
            | ir.UnknownOp()
        ):
            return op
        case _ as unreachable:
            assert_never(unreachable)


def map_blocks(region: Region, transform: Callable[[Block], Block]) -> Region:
    """Transform immediate blocks in order; recursion is explicit.

    The callback must not mutate its input. No SSA verification is performed.
    """
    return dataclasses.replace(
        region, blocks=[transform(block) for block in region.blocks]
    )


def replace_uses(
    function: Function,
    mapping: Mapping[ValueId, Value],
    *,
    where: Callable[[Use], bool] | None = None,
) -> Function:
    """Substitute operands throughout a function, leaving definitions unchanged.

    Substitutions are simultaneous: with a->b and b->c, original uses of a
    become b, not c. Include nested captures and repeated operand slots.
    Mapping keys must be defined in the input and replacements must have the
    same types. Replacement values must be defined in the resulting function.

    Args:
        function: Input function.
        mapping: Old function-local value IDs to replacement values.
        where: Optional filter called for matching uses on original operations.

    Returns:
        A function with selected operand references replaced.

    Raises:
        ValueError: A mapping or resulting SSA reference is invalid.
    """
    _check_mapping(function, mapping)
    if function.body is None or not mapping:
        return function
    result = dataclasses.replace(
        function, body=_substitute(function.body, mapping, where)
    )
    _check_references(result, mapping)
    return result


def replace_ops(
    function: Function,
    old_ops: Sequence[Op],
    new_ops: Sequence[Op],
    *,
    at: InsertPoint | None = None,
    replacements: Mapping[ValueId, Value] | None = None,
) -> Function:
    """Remove selected operations, insert a sequence, and redirect value uses.

    Resolve every target against the input function by identity. Targets may
    be nonadjacent or in different blocks, but cannot include both an operation
    and one of its descendants. Removing an operation also removes its regions.
    The insertion block must survive. New operations are inserted verbatim,
    then simultaneous substitutions apply to all surviving and inserted uses.

    Args:
        function: Input function.
        old_ops: Operations to remove, each listed once.
        new_ops: Operations to insert in the supplied order.
        at: Gap in the original function. Required for nonempty new_ops; the
            operation after the gap may itself be removed.
        replacements: Old value IDs to replacement values of the same type.
            Omit when new operations preserve the needed result IDs, or when
            removed results have no surviving uses.

    Returns:
        The edited function. Inputs are untouched, including on failure.

    Raises:
        ValueError: Targets overlap or are absent, an insertion point is invalid,
            or the result has duplicate definitions or invalid value references.
    """
    mapping = {} if replacements is None else replacements
    _check_mapping(function, mapping)
    removed = {id(op) for op in old_ops}
    if len(removed) != len(old_ops):
        raise ValueError("operation targets must be distinct")
    occurrences = collections.Counter(
        id(op) for op in (() if function.body is None else ir.iter_ops(function.body))
    )
    for op in old_ops:
        if occurrences[id(op)] == 0:
            raise ValueError("operation target is not in the input function")
        if occurrences[id(op)] != 1:
            raise ValueError("operation target occurs more than once in the input")
        for region in ir.get_regions(op):
            if any(id(child) in removed for child in ir.iter_ops(region)):
                raise ValueError("operation targets overlap")
            if at is not None and any(
                block is at.block for block in ir.iter_blocks(region)
            ):
                raise ValueError("insertion block is inside a removed operation")
    if new_ops and at is None:
        raise ValueError("inserting operations requires an insertion point")
    if at is not None:
        count = (
            0
            if function.body is None
            else sum(block is at.block for block in ir.iter_blocks(function.body))
        )
        if count == 0:
            raise ValueError("insertion block is not in the input function")
        if count != 1:
            raise ValueError("insertion block occurs more than once in the input")
        if not 0 <= at.index <= len(at.block.operations):
            raise ValueError("insertion index is outside the block")
    if function.body is None:
        return function

    def edit_block(block: ir.Block) -> ir.Block:
        operations: list[ir.Op] = []
        for index, op in enumerate(block.operations):
            if at is not None and block is at.block and index == at.index:
                operations.extend(new_ops)
            if id(op) not in removed:
                operations.append(map_regions(op, edit_region))
        if at is not None and block is at.block and at.index == len(block.operations):
            operations.extend(new_ops)
        return dataclasses.replace(block, operations=operations)

    def edit_region(region: ir.Region) -> ir.Region:
        return map_blocks(region, edit_block)

    body = edit_region(function.body)
    if mapping:
        body = _substitute(body, mapping)
    result = dataclasses.replace(function, body=body)
    _check_references(result, mapping)
    return result


def insert_ops(function: Function, ops: Sequence[Op], *, at: InsertPoint) -> Function:
    """Insert operations at an input gap; see [`replace_ops`][niro.rewrite.replace_ops]."""
    return replace_ops(function, (), ops, at=at)


def erase_ops(function: Function, ops: Sequence[Op]) -> Function:
    """Erase operations with no surviving uses of their definitions.

    Uses inside the erased subgraph are allowed. The caller must establish that
    discarding the operations' effects is legal.
    """
    return replace_ops(function, ops, ())


def move_ops(function: Function, ops: Sequence[Op], *, to: InsertPoint) -> Function:
    """Move operations in supplied order, preserving IDs and owned regions.

    The destination is a gap in the input version and cannot be inside any
    moved operation. The caller must establish dominance, scope, and effect
    ordering at the destination.
    """
    return replace_ops(function, ops, ops, at=to)


def clone_region(
    region: Region,
    supply: ValueSupply,
    *,
    captures: Mapping[ValueId, Value] | None = None,
) -> tuple[Region, dict[ValueId, Value], ValueSupply]:
    """Copy a region with fresh IDs for every definition, including nested ones.

    Args:
        region: Fragment to copy. Its definition IDs must be unique.
        supply: Fresh IDs from the destination function. Must not collide with
            captured values; use [`value_supply`][niro.rewrite.value_supply].
        captures: Optional substitutions for values defined outside the region.
            Unmapped captures are preserved. Mapped captures must retain types.
            Unused keys are ignored; keys naming region definitions are rejected.

    Returns:
        The copied region, a map from every original definition ID to its fresh
        value, and the advanced supply. The map excludes captured values.

    Raises:
        ValueError: Definitions repeat, a capture key names a region definition, types
            differ, or allocated IDs collide with captured values.
    """
    captures = {} if captures is None else captures
    definitions = _definitions(region)
    free = {
        operand.id: operand
        for op in ir.iter_ops(region)
        for operand in ir.get_operands(op)
        if operand.id not in definitions
    }
    for key, value in captures.items():
        if key in definitions:
            raise ValueError("capture key names a region definition")
        if key in free and free[key].type != value.type:
            raise ValueError("replacement types do not match")
    captured_ids = {captures.get(key, value).id for key, value in free.items()}
    renamed: dict[ir.ValueId, ir.Value] = {}
    for key, value in definitions.items():
        fresh, supply = fresh_value(supply, value.type)
        if fresh.id in captured_ids:
            raise ValueError("fresh value ID collides with a captured value")
        renamed[key] = fresh
    mapping = dict(captures) | renamed

    def clone_block(block: ir.Block) -> ir.Block:
        operations = []
        for op in block.operations:
            copied = with_operands(
                op, [mapping.get(v.id, v) for v in ir.get_operands(op)]
            )
            copied = _with_results(
                copied, tuple(renamed[v.id] for v in ir.get_results(op))
            )
            operations.append(map_regions(copied, clone))
        return dataclasses.replace(
            block,
            arguments=tuple(renamed[v.id] for v in block.arguments),
            operations=operations,
        )

    def clone(body: ir.Region) -> ir.Region:
        return map_blocks(body, clone_block)

    return clone(region), renamed, supply


def _with_results(op: ir.Op, results: tuple[ir.Value, ...]) -> ir.Op:
    match op:
        case (
            ir.Const()
            | ir.GetGlobal()
            | ir.Transpose()
            | ir.Add()
            | ir.Mul()
            | ir.MatMul()
        ):
            return dataclasses.replace(op, result=results[0])
        case ir.Call() | ir.If() | ir.UnknownOp():
            return dataclasses.replace(op, results=results)
        case ir.Return() | ir.Yield():
            return op
        case _ as unreachable:
            assert_never(unreachable)


def _substitute(
    region: ir.Region,
    mapping: Mapping[ir.ValueId, ir.Value],
    where: Callable[[ir.Use], bool] | None = None,
) -> ir.Region:
    def transform(block: ir.Block) -> ir.Block:
        operations = []
        for op in block.operations:
            operands = tuple(
                mapping[value.id]
                if value.id in mapping and (where is None or where(ir.Use(op, index)))
                else value
                for index, value in enumerate(ir.get_operands(op))
            )
            updated = with_operands(op, operands)
            operations.append(
                map_regions(updated, lambda body: _substitute(body, mapping, where))
            )
        return dataclasses.replace(block, operations=operations)

    return map_blocks(region, transform)


def _definitions(region: ir.Region) -> dict[ir.ValueId, ir.Value]:
    definitions: dict[ir.ValueId, ir.Value] = {}
    for value in ir.iter_defined_values(region):
        if value.id in definitions:
            raise ValueError(f"duplicate definition for value {value.id}")
        definitions[value.id] = value
    return definitions


def _check_mapping(
    function: ir.Function, mapping: Mapping[ir.ValueId, ir.Value]
) -> None:
    if not mapping:
        return
    definitions = {} if function.body is None else _definitions(function.body)
    for key, value in mapping.items():
        if key not in definitions:
            raise ValueError(
                f"replacement source {key} is not defined in the input function"
            )
        if definitions[key].type != value.type:
            raise ValueError("replacement types do not match")


def _check_references(
    function: ir.Function, mapping: Mapping[ir.ValueId, ir.Value]
) -> None:
    if function.body is None:
        return
    definitions = _definitions(function.body)
    operands = (
        value for op in ir.iter_ops(function.body) for value in ir.get_operands(op)
    )
    for value in itertools.chain(operands, mapping.values()):
        if value.id not in definitions:
            raise ValueError(f"value {value.id} has a dangling use or replacement")
        if definitions[value.id].type != value.type:
            raise ValueError(
                f"operand type disagrees with definition of value {value.id}"
            )
