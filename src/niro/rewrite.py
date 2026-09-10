"""Functional helpers for editing SSA IR.

Edits return new IR without mutating their inputs. Unchanged objects and metadata
may be shared: treat both versions as immutable, even while the IR schema still
contains mutable containers. Operation targets are matched by identity and
insertion points refer to the input version; reacquire references after edits.
Cloning advances the supplied [`ValueSupply`][niro.ir.ValueSupply] on success;
the source IR is unchanged.

Operation edits check unique definitions, operand references, and replacement
types. Passes remain responsible for dominance, region scope, terminators, and
semantic equivalence (including effects). Verify the completed module with
[`niro.verify.module`][]. Traversal and cloning also accept incomplete fragments.

Examples start with MLIR-like sketches of the IR, followed by executable Python.
The sketches use Niro operation names and numeric SSA IDs matching the Python
examples.

Examples:
    Remove an identity transpose and redirect its consumers:

    ```text
    Before                              After
    %1 = transpose %0, [0, 1]           return %0
    return %1

    replacements: {%1: %0}
    ```

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
    optimized = rewrite.replace_op(
        function, transpose, (), replacements={y.id: x}
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
from niro.ir import Block, Function, Op, Region, Use, Value, ValueId, ValueSupply

__all__ = [
    "InsertPoint",
    "after",
    "before",
    "clone_region",
    "erase_ops",
    "insert_ops",
    "map_blocks",
    "map_regions",
    "move_ops",
    "replace_op",
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


def before(function: Function, op: Op) -> InsertPoint:
    """Locate the gap immediately before an operation, including nested regions.

    Match by identity and require exactly one occurrence in the input function.
    The returned point refers to that function version; reacquire it after edits.
    No SSA verification is performed.

    Args:
        function: Function to search.
        op: Existing operation whose position is requested.

    Returns:
        An insertion point in the operation's containing block.

    Raises:
        ValueError: The operation is absent or occurs more than once.

    Examples:
        Locate gaps around a constant without computing its block index:

        In this complete program, `constant` is the operation defining `%0`.
        The comments mark the insertion points returned by the helpers:

        ```text
        module {
          func @answer() -> i32 {
            // Gap 0: before(function, constant)
            %0 = const 42 : i32
            // Gap 1: after(function, constant)
            return %0
          }
        }
        ```

        ```python
        from niro import ir, rewrite

        result = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
        constant = ir.Const(result, 42)
        block = ir.Block(operations=[constant, ir.Return((result,))])
        function = ir.Function(
            "answer", ir.FunctionType((), (result.type,)), ir.Region([block])
        )
        assert rewrite.before(function, constant) == rewrite.InsertPoint(block, 0)
        assert rewrite.after(function, constant) == rewrite.InsertPoint(block, 1)
        ```
    """
    point: InsertPoint | None = None
    blocks = () if function.body is None else ir.iter_blocks(function.body)
    for block in blocks:
        for index, candidate in enumerate(block.operations):
            if candidate is not op:
                continue
            if point is not None:
                raise ValueError("operation target occurs more than once in the input")
            point = InsertPoint(block, index)
    if point is None:
        raise ValueError("operation target is not in the input function")
    return point


def after(function: Function, op: Op) -> InsertPoint:
    """Locate the gap immediately after an operation, including at a block's end.

    Use the identity lookup and input-version semantics of
    [`before`][niro.rewrite.before]. A point after a terminator is representable;
    callers remain responsible for valid operation ordering when editing.

    Args:
        function: Function to search.
        op: Existing operation whose position is requested.

    Returns:
        An insertion point in the operation's containing block.

    Raises:
        ValueError: The operation is absent or occurs more than once.
    """
    point = before(function, op)
    return InsertPoint(point.block, point.index + 1)


def value_supply(function: Function) -> ValueSupply:
    """Create a supply above every definition ID, including nested regions.

    Examples:
        Start allocating after an existing function's highest value ID:

        ```python
        from niro import ir, rewrite

        x = ir.Value(ir.ValueId(7), ir.ScalarType.I32)
        function = ir.Function(
            "identity", ir.FunctionType((x.type,), (x.type,)),
            ir.Region([ir.Block((x,), [ir.Return((x,))])]),
        )
        supply = rewrite.value_supply(function)
        assert supply.next_id == 8
        fresh = supply.fresh(ir.ScalarType.I32)
        assert fresh.id == 8
        ```
    """
    values = () if function.body is None else ir.iter_defined_values(function.body)
    return ir.ValueSupply(1 + max((value.id for value in values), default=-1))


def with_operands(op: Op, operands: Sequence[Value]) -> Op:
    """Replace immediate operands in [`niro.ir.get_operands`][] order.

    Require the same arity. Definitions, nested regions, and other fields are
    preserved. This low-level helper permits type changes for coordinated edits.

    Examples:
        Swap an addition's operands while preserving its result:

        ```text
        Before                              After
        %2 = add %0, %1 : i32               %2 = add %1, %0 : i32
        ```

        ```python
        from niro import ir, rewrite

        x, y, result = (
            ir.Value(ir.ValueId(i), ir.ScalarType.I32) for i in range(3)
        )
        original = ir.Add(result, x, y)
        swapped = rewrite.with_operands(original, (y, x))
        assert swapped == ir.Add(result, y, x)
        assert original == ir.Add(result, x, y)
        ```
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

    Examples:
        Complete both empty branches of an If with Yield terminators:

        ```text
        Before (unfinished)                 After
        if %0 {                             if %0 {
                                              yield
        } else {                            } else {
                                              yield
        }                                   }
        ```

        ```python
        import dataclasses
        from niro import ir, rewrite

        condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
        branch = ir.If(
            (), condition, ir.Region([ir.Block()]), ir.Region([ir.Block()])
        )

        def finish(region: ir.Region) -> ir.Region:
            return rewrite.map_blocks(
                region,
                lambda block: dataclasses.replace(
                    block, operations=[*block.operations, ir.Yield()]
                ),
            )

        completed = rewrite.map_regions(branch, finish)
        assert isinstance(completed, ir.If)
        assert completed.then_region.blocks[0].operations == [ir.Yield()]
        assert completed.else_region.blocks[0].operations == [ir.Yield()]
        assert branch.then_region.blocks[0].operations == []
        ```
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

    Examples:
        Complete an unfinished function body with a Return terminator:

        ```text
        Before (unfinished)                 After
        %0 = const 42 : i32                 %0 = const 42 : i32
                                            return %0
        ```

        ```python
        import dataclasses
        from niro import ir, rewrite

        result = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
        constant = ir.Const(result, 42)
        body = ir.Region([ir.Block(operations=[constant])])
        completed = rewrite.map_blocks(
            body,
            lambda block: dataclasses.replace(
                block, operations=[*block.operations, ir.Return((result,))]
            ),
        )
        assert completed.blocks[0].operations == [constant, ir.Return((result,))]
        assert body.blocks[0].operations == [constant]
        ```
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

    Examples:
        Redirect uses of a duplicate constant. The definitions remain in place;
        use [`erase_ops`][niro.rewrite.erase_ops] to remove the unused one later:

        ```text
        Before                              After
        %0 = const 5 : i32                  %0 = const 5 : i32
        %1 = const 5 : i32                  %1 = const 5 : i32
        %2 = add %1, %1 : i32               %2 = add %0, %0 : i32
        return %2                           return %2

        mapping: {%1: %0}

        With where selecting the addition's operand_index == 0:
        %2 = add %1, %1 : i32         ->     %2 = add %0, %1 : i32
        ```

        ```python
        from niro import ir, rewrite

        a, b, result = (
            ir.Value(ir.ValueId(i), ir.ScalarType.I32) for i in range(3)
        )
        add = ir.Add(result, b, b)
        block = ir.Block(operations=[
            ir.Const(a, 5), ir.Const(b, 5), add, ir.Return((result,))
        ])
        function = ir.Function(
            "sum", ir.FunctionType((), (result.type,)), ir.Region([block])
        )
        updated = rewrite.replace_uses(function, {b.id: a})
        assert updated.first_block is not None
        assert updated.first_block.operations[2] == ir.Add(result, a, a)
        assert block.operations[2] is add

        # Alternatively, replace only the left operand of this addition.
        selected = rewrite.replace_uses(
            function, {b.id: a},
            where=lambda use: use.owner is add and use.operand_index == 0,
        )
        assert selected.first_block is not None
        assert selected.first_block.operations[2] == ir.Add(result, a, b)
        ```
    """
    _check_mapping(function, mapping)
    if function.body is None or not mapping:
        return function
    result = dataclasses.replace(
        function, body=_substitute(function.body, mapping, where)
    )
    _check_references(result, mapping)
    return result


def replace_op(
    function: Function,
    old_op: Op,
    new_ops: Sequence[Op],
    *,
    replacements: Mapping[ValueId, Value] | None = None,
) -> Function:
    """Replace one operation with a sequence at its original position.

    Locate the target by identity with [`before`][niro.rewrite.before], then
    delegate to [`replace_ops`][niro.rewrite.replace_ops]. Result IDs may be
    preserved or uses redirected with replacements. An empty sequence removes
    the operation. All replacement and validation semantics are unchanged.

    Args:
        function: Input function.
        old_op: Operation to replace.
        new_ops: Operations to insert in the supplied order.
        replacements: Old value IDs to replacement values of the same type.

    Returns:
        The edited function. Inputs are untouched, including on failure.

    Raises:
        ValueError: The target is absent or ambiguous, or the edit violates
            the requirements of [`replace_ops`][niro.rewrite.replace_ops].

    Examples:
        Fold an addition while retaining its result ID and the input constants:

        ```text
        Before                              After
        %0 = const 2 : i32                  %0 = const 2 : i32
        %1 = const 3 : i32                  %1 = const 3 : i32
        %2 = add %0, %1 : i32               %2 = const 5 : i32
        return %2                           return %2
        ```

        ```python
        from niro import ir, rewrite

        a, b, result = (
            ir.Value(ir.ValueId(i), ir.ScalarType.I32) for i in range(3)
        )
        left, right = ir.Const(a, 2), ir.Const(b, 3)
        add = ir.Add(result, a, b)
        block = ir.Block(operations=[left, right, add, ir.Return((result,))])
        function = ir.Function(
            "five", ir.FunctionType((), (result.type,)), ir.Region([block])
        )
        folded = ir.Const(result, 5)
        updated = rewrite.replace_op(function, add, (folded,))
        assert updated.first_block is not None
        assert updated.first_block.operations == [
            left, right, folded, ir.Return((result,))
        ]
        assert block.operations[2] is add
        ```
    """
    return replace_ops(
        function,
        (old_op,),
        new_ops,
        at=before(function, old_op),
        replacements=replacements,
    )


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

    Examples:
        Fold two constants and their addition into one constant. Preserving the
        result ID keeps the Return valid without a replacement value map:

        ```text
        Before                              After
        %0 = const 2 : i32                  %2 = const 5 : i32
        %1 = const 3 : i32                  return %2
        %2 = add %0, %1 : i32
        return %2
        ```

        ```python
        from niro import ir, rewrite

        a, b, result = (
            ir.Value(ir.ValueId(i), ir.ScalarType.I32) for i in range(3)
        )
        left, right = ir.Const(a, 2), ir.Const(b, 3)
        add = ir.Add(result, a, b)
        block = ir.Block(operations=[left, right, add, ir.Return((result,))])
        function = ir.Function(
            "five", ir.FunctionType((), (result.type,)), ir.Region([block])
        )
        folded = ir.Const(result, 5)
        updated = rewrite.replace_ops(
            function, (left, right, add), (folded,),
            at=rewrite.InsertPoint(block, 0),
        )
        assert updated.first_block is not None
        assert updated.first_block.operations == [folded, ir.Return((result,))]
        assert len(block.operations) == 4
        ```
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
    """Insert operations at an input gap; see [`replace_ops`][niro.rewrite.replace_ops].

    Examples:
        Insert a constant before the Return. Insertion alone does not change
        existing operands, so the new result is initially unused:

        ```text
        Before                              After
        return                              %0 = const 42 : i32
                                            return
        ```

        ```python
        from niro import ir, rewrite

        block = ir.Block(operations=[ir.Return()])
        function = ir.Function("empty", ir.FunctionType((), ()), ir.Region([block]))
        supply = rewrite.value_supply(function)
        result = supply.fresh(ir.ScalarType.I32)
        constant = ir.Const(result, 42)
        updated = rewrite.insert_ops(
            function, (constant,), at=rewrite.InsertPoint(block, 0)
        )
        assert updated.first_block is not None
        assert updated.first_block.operations == [constant, ir.Return()]
        assert block.operations == [ir.Return()]
        ```
    """
    return replace_ops(function, (), ops, at=at)


def erase_ops(function: Function, ops: Sequence[Op]) -> Function:
    """Erase operations with no surviving uses of their definitions.

    Uses inside the erased subgraph are allowed. The caller must establish that
    discarding the operations' effects is legal.

    Examples:
        Remove an unused calculation together with its constant. The addition
        uses the constant, but both disappear in the same edit:

        ```text
        Before                              After
        %0 = const 3 : i32                  return
        %1 = add %0, %0 : i32
        return
        ```

        ```python
        from niro import ir, rewrite

        a = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
        b = ir.Value(ir.ValueId(1), ir.ScalarType.I32)
        constant, add = ir.Const(a, 3), ir.Add(b, a, a)
        block = ir.Block(operations=[constant, add, ir.Return()])
        function = ir.Function("empty", ir.FunctionType((), ()), ir.Region([block]))
        updated = rewrite.erase_ops(function, (constant, add))
        assert updated.first_block is not None
        assert updated.first_block.operations == [ir.Return()]
        assert len(block.operations) == 3
        ```
    """
    return replace_ops(function, ops, ())


def move_ops(function: Function, ops: Sequence[Op], *, to: InsertPoint) -> Function:
    """Move operations in supplied order, preserving IDs and owned regions.

    The destination is a gap in the input version and cannot be inside any
    moved operation. The caller must establish dominance, scope, and effect
    ordering at the destination.

    Examples:
        Reorder two independent constants, keeping both before their consumer.
        Index 2 denotes the gap before Add in the original block:

        ```text
        Before                              After
        %0 = const 2 : i32                  %1 = const 3 : i32
        %1 = const 3 : i32                  %0 = const 2 : i32
        %2 = add %0, %1 : i32               %2 = add %0, %1 : i32
        return %2                           return %2
        ```

        ```python
        from niro import ir, rewrite

        a, b, result = (
            ir.Value(ir.ValueId(i), ir.ScalarType.I32) for i in range(3)
        )
        first, second = ir.Const(a, 2), ir.Const(b, 3)
        add, ret = ir.Add(result, a, b), ir.Return((result,))
        block = ir.Block(operations=[first, second, add, ret])
        function = ir.Function(
            "sum", ir.FunctionType((), (result.type,)), ir.Region([block])
        )
        updated = rewrite.move_ops(
            function, (first,), to=rewrite.InsertPoint(block, 2)
        )
        assert updated.first_block is not None
        assert updated.first_block.operations == [second, first, add, ret]
        assert block.operations == [first, second, add, ret]
        ```
    """
    return replace_ops(function, ops, ops, at=to)


def clone_region(
    region: Region,
    supply: ValueSupply,
    *,
    captures: Mapping[ValueId, Value] | None = None,
) -> tuple[Region, dict[ValueId, Value]]:
    """Copy a region with fresh IDs for every definition, including nested ones.

    Args:
        region: Fragment to copy. Its definition IDs must be unique.
        supply: Fresh IDs from the destination function. Must not collide with
            captured values; use [`value_supply`][niro.rewrite.value_supply].
            Advanced on success and left unchanged on failure.
        captures: Optional substitutions for values defined outside the region.
            Unmapped captures are preserved. Mapped captures must retain types.
            Unused keys are ignored; keys naming region definitions are rejected.

    Returns:
        The copied region and a map from every original definition ID to its
        fresh value. The map excludes captured values.

    Raises:
        ValueError: Definitions repeat, a capture key names a region definition, types
            differ, or allocated IDs collide with captured values.

    Examples:
        Copy a branch that captures x, replacing that capture with a destination
        function's argument. Local definitions get fresh IDs; Yield is remapped:

        ```text
        Source branch (unchanged)           Copied branch
        %1 = add %0, %0 : i32               %11 = add %10, %10 : i32
        yield %1                            yield %11

        captures:        {%0: %10}
        values (return): {%1: %11}
        supply.next_id:  11 -> 12
        ```

        ```python
        from niro import ir, rewrite

        x = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
        local = ir.Value(ir.ValueId(1), ir.ScalarType.I32)
        branch = ir.Region([ir.Block(operations=[
            ir.Add(local, x, x), ir.Yield((local,))
        ])])
        argument = ir.Value(ir.ValueId(10), ir.ScalarType.I32)
        destination = ir.Function(
            "destination", ir.FunctionType((argument.type,), ()),
            ir.Region([ir.Block((argument,), [ir.Return()])]),
        )
        supply = rewrite.value_supply(destination)
        copied, values = rewrite.clone_region(
            branch, supply, captures={x.id: argument}
        )
        fresh_local = values[local.id]
        assert fresh_local.id == 11
        assert copied.blocks[0].operations == [
            ir.Add(fresh_local, argument, argument), ir.Yield((fresh_local,))
        ]
        assert x.id not in values  # The returned map contains definitions only.
        assert supply.next_id == 12
        assert branch.blocks[0].operations[0] == ir.Add(local, x, x)
        ```
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
    allocated = supply.clone()
    renamed: dict[ir.ValueId, ir.Value] = {}
    for key, value in definitions.items():
        fresh = allocated.fresh(value.type)
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

    copied = clone(region)
    supply.next_id = allocated.next_id
    return copied, renamed


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
