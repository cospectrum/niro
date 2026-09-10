"""Reusable draws for every Niro IR node, constructed without the builder.

Use ``modules()`` or ``functions()`` with ``@given``, or compose ``draw_*``
functions inside another Hypothesis composite strategy. Draws return ordinary
IR; verification belongs in the consuming test and must not filter examples.

The full language includes opaque operations, dynamic tensors, external and
recursive calls, and potentially nonterminating CFGs. Validity does not imply
backend support or that a generated function can safely be interpreted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from niro import ir


@dataclass(frozen=True)
class Bounds:
    """Bound generation work, independently of a program's execution behavior.

    max_steps limits optional operation selections across an entire function,
    including nested arms. Operand materializations and terminators are extra,
    bounded by max_arity and the number of blocks/arms. max_blocks bounds the
    function CFG; If arms always have one block. max_depth bounds nested Ifs.
    Tensor ranks and dimensions are at most three and four, respectively.
    """

    max_steps: int = 12
    max_blocks: int = 4
    max_depth: int = 2
    max_arity: int = 3

    def __post_init__(self) -> None:
        """Reject bounds that cannot describe a finite function body."""
        if min(self.max_steps, self.max_depth, self.max_arity) < 0:
            raise ValueError("generation bounds must be nonnegative")
        if self.max_blocks < 1:
            raise ValueError("max_blocks must be positive")


DEFAULT_BOUNDS = Bounds()
"""Immutable default generation bounds."""


@dataclass
class Context:
    """Function-wide allocation, operation budget, and read-only module symbols.

    Use a fresh context per function. Nested scopes share this context to keep
    IDs unique and consume the same budget. Existing regions require a supply
    whose next ID is fresh relative to all definitions and captures.
    """

    bounds: Bounds = DEFAULT_BOUNDS
    functions: Mapping[str, ir.FunctionType] = field(default_factory=dict)
    globals: Mapping[str, ir.Type] = field(default_factory=dict)
    supply: ir.ValueSupply = field(default_factory=ir.ValueSupply)
    remaining_steps: int = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the shared budget for this function."""
        self.remaining_steps = self.bounds.max_steps


@dataclass
class Scope:
    """A block under construction and its currently visible SSA values.

    values includes captures, block arguments, and results emitted so far.
    Node draws may emit operand definitions here but return their own node;
    call emit to append that node and expose its results to subsequent draws.
    """

    block: ir.Block
    values: list[ir.Value]


def emit(scope: Scope, operation: ir.Op) -> None:
    """Append an operation and expose only its immediate results in this scope."""
    scope.block.operations.append(operation)
    scope.values.extend(ir.get_results(operation))


def scalar_types() -> SearchStrategy[ir.ScalarType]:
    """Generate any supported scalar type."""
    return st.sampled_from(tuple(ir.ScalarType))


def scalar_literals(type_: ir.ScalarType) -> SearchStrategy[bool | int | float]:
    """Generate representable scalar literals, including nonfinite floats."""
    match type_:
        case ir.ScalarType.BOOL:
            return st.booleans()
        case ir.ScalarType.I32:
            return st.integers(min_value=-(2**31), max_value=2**31 - 1)
        case ir.ScalarType.I64:
            return st.integers(min_value=-(2**63), max_value=2**63 - 1)
        case ir.ScalarType.F32:
            return st.floats(width=32)
        case ir.ScalarType.F64:
            return st.floats(width=64)


def draw_shape(draw: DrawFn, *, static: bool = False) -> ir.Shape | None:
    """Draw rank zero through three, unknown rank, and dimensions zero to four."""
    if not static and draw(st.booleans()):
        return None
    dimension = st.integers(0, 4)
    dimensions = dimension if static else st.one_of(st.none(), dimension)
    return tuple(draw(st.lists(dimensions, max_size=3)))


def draw_tensor_type(draw: DrawFn, *, static: bool = False) -> ir.TensorType:
    """Draw a tensor type, optionally requiring a literal-compatible shape."""
    return ir.TensorType(draw(scalar_types()), draw_shape(draw, static=static))


def draw_type(draw: DrawFn, *, static: bool = False) -> ir.Type:
    """Draw a scalar or tensor type; static restricts only tensor shapes."""
    if draw(st.booleans()):
        return draw_tensor_type(draw, static=static)
    return draw(scalar_types())


def draw_literal(draw: DrawFn, type_: ir.Type) -> ir.Literal:
    """Draw a matching literal; tensor types must have a fully static shape."""
    if not isinstance(type_, ir.TensorType):
        return draw(scalar_literals(type_))
    assert type_.shape is not None
    assert all(dimension is not None for dimension in type_.shape)
    size = (
        math.prod(d for d in type_.shape if d is not None)
        * type_.element_type.byte_width
    )
    return draw(st.binary(min_size=size, max_size=size))


def draw_attributes(draw: DrawFn) -> ir.Attributes:
    """Draw bounded metadata containing every attribute value variant."""
    leaves = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(),
        st.text(max_size=12),
        st.binary(max_size=16),
    )
    values = st.recursive(
        leaves, lambda children: st.lists(children, max_size=3).map(tuple), max_leaves=8
    )
    return draw(st.dictionaries(st.text(max_size=12), values, max_size=3))


def draw_value(
    draw: DrawFn, supply: ir.ValueSupply, type_: ir.Type | None = None
) -> ir.Value:
    """Allocate a fresh value of the supplied or a drawn type; do not define it."""
    return supply.fresh(draw_type(draw) if type_ is None else type_)


def draw_function_type(draw: DrawFn, *, max_arity: int = 3) -> ir.FunctionType:
    """Draw independent input and output signatures of bounded arity."""
    inputs = tuple(draw_type(draw) for _ in range(draw(st.integers(0, max_arity))))
    outputs = tuple(draw_type(draw) for _ in range(draw(st.integers(0, max_arity))))
    return ir.FunctionType(inputs, outputs)


def draw_global(draw: DrawFn, *, name: str | None = None) -> ir.Global:
    """Draw an initialized scalar or static tensor global with metadata."""
    type_ = draw_type(draw, static=True)
    return ir.Global(
        draw(st.text(min_size=1, max_size=12)) if name is None else name,
        type_,
        draw_literal(draw, type_),
        draw_attributes(draw),
    )


def draw_const(
    draw: DrawFn, context: Context, *, type_: ir.Type | None = None
) -> ir.Const:
    """Draw a constant; an explicitly supplied tensor type must be static."""
    type_ = draw_type(draw, static=True) if type_ is None else type_
    return ir.Const(draw_value(draw, context.supply, type_), draw_literal(draw, type_))


def draw_operand(
    draw: DrawFn, context: Context, scope: Scope, type_: ir.Type
) -> ir.Value:
    """Reuse a visible value or emit a matching constant/opaque source.

    Dynamic and unranked tensors cannot be constants, so missing values of
    those types are materialized by a zero-operand UnknownOp. These supporting
    definitions do not consume the optional operation budget.
    """
    candidates = [value for value in scope.values if value.type == type_]
    if candidates and draw(st.booleans()):
        return draw(st.sampled_from(candidates))
    if isinstance(type_, ir.TensorType) and (
        type_.shape is None or None in type_.shape
    ):
        value = draw_value(draw, context.supply, type_)
        emit(scope, ir.UnknownOp("test.source", (), (value,)))
    else:
        operation = draw_const(draw, context, type_=type_)
        emit(scope, operation)
        value = operation.result
    return value


def _draw_operands(
    draw: DrawFn,
    context: Context,
    scope: Scope,
    types: Sequence[ir.Type],
) -> tuple[ir.Value, ...]:
    """Resolve an ordered type signature to visible or newly materialized values."""
    return tuple(draw_operand(draw, context, scope, type_) for type_ in types)


def draw_get_global(draw: DrawFn, context: Context) -> ir.GetGlobal:
    """Draw a load of a declared global; the context must contain a global."""
    name = draw(st.sampled_from(tuple(context.globals)))
    return ir.GetGlobal(name, draw_value(draw, context.supply, context.globals[name]))


def _draw_binary_operands(
    draw: DrawFn,
    context: Context,
    scope: Scope,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Draw a fresh result and two equal-typed numeric operands."""
    numeric = tuple(type_ for type_ in ir.ScalarType if type_ is not ir.ScalarType.BOOL)
    candidates = [
        value.type
        for value in scope.values
        if (
            value.type.element_type
            if isinstance(value.type, ir.TensorType)
            else value.type
        )
        is not ir.ScalarType.BOOL
    ]
    if candidates and draw(st.booleans()):
        type_ = draw(st.sampled_from(candidates))
    else:
        element = draw(st.sampled_from(numeric))
        type_ = (
            ir.TensorType(element, draw_shape(draw)) if draw(st.booleans()) else element
        )
    lhs = draw_operand(draw, context, scope, type_)
    rhs = draw_operand(draw, context, scope, type_)
    return draw_value(draw, context.supply, type_), lhs, rhs


def draw_add(draw: DrawFn, context: Context, scope: Scope) -> ir.Add:
    """Draw numeric addition, emitting any missing operand definitions."""
    return ir.Add(*_draw_binary_operands(draw, context, scope))


def draw_mul(draw: DrawFn, context: Context, scope: Scope) -> ir.Mul:
    """Draw numeric multiplication, emitting any missing operand definitions."""
    return ir.Mul(*_draw_binary_operands(draw, context, scope))


def draw_matmul(draw: DrawFn, context: Context, scope: Scope) -> ir.MatMul:
    """Draw compatible rank-two operands without calling production inference."""
    dimension = st.one_of(st.none(), st.integers(0, 4))
    candidates = [
        value.type
        for value in scope.values
        if isinstance(value.type, ir.TensorType) and value.type.rank == 2
    ]
    lhs_type = (
        draw(st.sampled_from(candidates))
        if candidates and draw(st.booleans())
        else ir.TensorType(draw(scalar_types()), (draw(dimension), draw(dimension)))
    )
    assert lhs_type.shape is not None
    inner = lhs_type.shape[1]
    rhs_inner = (
        draw(dimension) if inner is None else draw(st.sampled_from((inner, None)))
    )
    rhs_type = ir.TensorType(lhs_type.element_type, (rhs_inner, draw(dimension)))
    assert rhs_type.shape is not None
    lhs = draw_operand(draw, context, scope, lhs_type)
    rhs = draw_operand(draw, context, scope, rhs_type)
    result_type = ir.TensorType(
        lhs_type.element_type, (lhs_type.shape[0], rhs_type.shape[1])
    )
    return ir.MatMul(draw_value(draw, context.supply, result_type), lhs, rhs)


def draw_transpose(draw: DrawFn, context: Context, scope: Scope) -> ir.Transpose:
    """Draw a tensor permutation and derive its shape independently of inference."""
    candidates = [
        value.type for value in scope.values if isinstance(value.type, ir.TensorType)
    ]
    type_ = (
        draw(st.sampled_from(candidates))
        if candidates and draw(st.booleans())
        else draw_tensor_type(draw)
    )
    rank = draw(st.integers(0, 3)) if type_.shape is None else len(type_.shape)
    permutation = tuple(draw(st.permutations(tuple(range(rank)))))
    shape = (
        None
        if type_.shape is None
        else tuple(type_.shape[axis] for axis in permutation)
    )
    operand = draw_operand(draw, context, scope, type_)
    result = draw_value(draw, context.supply, ir.TensorType(type_.element_type, shape))
    return ir.Transpose(result, operand, permutation)


def draw_call(draw: DrawFn, context: Context, scope: Scope) -> ir.Call:
    """Draw a call; context signatures may include external/recursive callees."""
    name = draw(st.sampled_from(tuple(context.functions)))
    signature = context.functions[name]
    arguments = _draw_operands(draw, context, scope, signature.inputs)
    results = tuple(
        draw_value(draw, context.supply, type_) for type_ in signature.outputs
    )
    return ir.Call(name, arguments, results)


def draw_unknown_op(draw: DrawFn, context: Context, scope: Scope) -> ir.UnknownOp:
    """Draw an opaque operation with visible operands and fresh arbitrary results."""
    operands = (
        tuple(
            draw(
                st.lists(
                    st.sampled_from(scope.values), max_size=context.bounds.max_arity
                )
            )
        )
        if scope.values
        else ()
    )
    count = draw(st.integers(0, context.bounds.max_arity))
    results = tuple(draw_value(draw, context.supply) for _ in range(count))
    return ir.UnknownOp(
        draw(st.text(min_size=1, max_size=12)), operands, results, draw_attributes(draw)
    )


def draw_return(
    draw: DrawFn,
    context: Context,
    scope: Scope,
    output_types: Sequence[ir.Type],
) -> ir.Return:
    """Draw a function return, materializing missing output values in its block."""
    return ir.Return(_draw_operands(draw, context, scope, output_types))


def draw_yield(
    draw: DrawFn,
    context: Context,
    scope: Scope,
    output_types: Sequence[ir.Type],
) -> ir.Yield:
    """Draw an If-arm yield, materializing missing output values in its block."""
    return ir.Yield(_draw_operands(draw, context, scope, output_types))


def draw_branch(
    draw: DrawFn, context: Context, scope: Scope, target: ir.Block
) -> ir.Branch:
    """Draw a branch matching its target's arguments; caller establishes ownership."""
    arguments = _draw_operands(
        draw, context, scope, tuple(v.type for v in target.arguments)
    )
    return ir.Branch(target, arguments)


def draw_cond_branch(
    draw: DrawFn,
    context: Context,
    scope: Scope,
    true_target: ir.Block,
    false_target: ir.Block,
) -> ir.CondBranch:
    """Draw a boolean branch with arguments matching each destination block."""
    condition = draw_operand(draw, context, scope, ir.ScalarType.BOOL)
    true_arguments = _draw_operands(
        draw, context, scope, tuple(v.type for v in true_target.arguments)
    )
    false_arguments = _draw_operands(
        draw, context, scope, tuple(v.type for v in false_target.arguments)
    )
    return ir.CondBranch(
        condition, true_target, false_target, true_arguments, false_arguments
    )


def draw_if(draw: DrawFn, context: Context, scope: Scope, *, depth: int = 0) -> ir.If:
    """Draw two isolated, argument-free arms capturing the current outer scope."""
    assert depth < context.bounds.max_depth
    condition = draw_operand(draw, context, scope, ir.ScalarType.BOOL)
    count = draw(st.integers(0, context.bounds.max_arity))
    results = tuple(draw_value(draw, context.supply) for _ in range(count))
    types = tuple(value.type for value in results)
    arms = [
        draw_if_region(draw, context, scope.values, types, depth=depth + 1)
        for _ in range(2)
    ]
    return ir.If(results, condition, arms[0], arms[1])


def draw_operation(
    draw: DrawFn, context: Context, scope: Scope, *, depth: int = 0
) -> ir.Op:
    """Draw any eligible nonterminator; emit its prerequisites but return the node."""
    kinds = ["const", "add", "mul", "matmul", "transpose", "unknown"]
    if context.globals:
        kinds.append("get_global")
    if context.functions:
        kinds.append("call")
    if depth < context.bounds.max_depth:
        kinds.append("if")
    match draw(st.sampled_from(kinds)):
        case "const":
            return draw_const(draw, context)
        case "add":
            return draw_add(draw, context, scope)
        case "mul":
            return draw_mul(draw, context, scope)
        case "matmul":
            return draw_matmul(draw, context, scope)
        case "transpose":
            return draw_transpose(draw, context, scope)
        case "unknown":
            return draw_unknown_op(draw, context, scope)
        case "get_global":
            return draw_get_global(draw, context)
        case "call":
            return draw_call(draw, context, scope)
        case "if":
            return draw_if(draw, context, scope, depth=depth)
    raise AssertionError("unhandled operation draw")


def draw_block(
    draw: DrawFn,
    context: Context,
    *,
    block: ir.Block | None = None,
    visible: Sequence[ir.Value] = (),
    output_types: Sequence[ir.Type] = (),
    successors: Sequence[ir.Block] = (),
    yield_: bool = False,
    depth: int = 0,
) -> ir.Block:
    """Fill an empty block with operations and its context-appropriate terminator.

    A supplied block retains its preallocated arguments and identity. visible
    must dominate this block or be captured from an enclosing region. At most
    two successors are permitted, and only for function blocks. If arms must
    be argument-free and use Yield. Mutates the supplied block and context.
    """
    block = ir.Block() if block is None else block
    assert not block.operations
    assert len(successors) <= 2
    assert not yield_ or (not successors and not block.arguments)
    scope = Scope(block, [*visible, *block.arguments])
    steps = draw(st.integers(0, context.remaining_steps))
    for _ in range(steps):
        if context.remaining_steps == 0:
            break
        context.remaining_steps -= 1
        emit(scope, draw_operation(draw, context, scope, depth=depth))
    if yield_:
        terminator = draw_yield(draw, context, scope, output_types)
    elif len(successors) == 2:
        terminator = draw_cond_branch(draw, context, scope, *successors)
    elif successors:
        terminator = draw_branch(draw, context, scope, successors[0])
    else:
        terminator = draw_return(draw, context, scope, output_types)
    emit(scope, terminator)
    return block


def draw_if_region(
    draw: DrawFn,
    context: Context,
    visible: Sequence[ir.Value],
    output_types: Sequence[ir.Type],
    *,
    depth: int = 1,
) -> ir.Region:
    """Draw an If arm, sharing allocation but copying the visible outer scope."""
    return ir.Region(
        [
            draw_block(
                draw,
                context,
                visible=visible,
                output_types=output_types,
                yield_=True,
                depth=depth,
            )
        ]
    )


def _draw_successors(draw: DrawFn, count: int) -> list[list[int]]:
    """Draw a reachable graph with at most two edges per block and none to entry.

    First create a random spanning tree, then add optional edges, including
    self-loops, backedges, and duplicate conditional destinations.
    """
    successors: list[list[int]] = [[] for _ in range(count)]
    for target in range(1, count):
        parents = [source for source in range(target) if len(successors[source]) < 2]
        successors[draw(st.sampled_from(parents))].append(target)
    if count > 1:
        for targets in successors:
            extra = draw(st.integers(0, 2 - len(targets)))
            targets.extend(draw(st.integers(1, count - 1)) for _ in range(extra))
    return successors


def _dominators(successors: Sequence[Sequence[int]]) -> list[set[int]]:
    """Find dominance by removing each block and checking entry reachability.

    This deliberately differs from the verifier's predecessor fixed point.
    The small generated CFGs make the simpler independent algorithm practical.
    """
    result: list[set[int]] = [set() for _ in successors]
    for removed in range(len(successors)):
        reachable: set[int] = set()
        pending = [0] if removed != 0 else []
        while pending:
            source = pending.pop()
            if source == removed or source in reachable:
                continue
            reachable.add(source)
            pending.extend(successors[source])
        for target in range(len(successors)):
            if target not in reachable:
                result[target].add(removed)
    return result


def draw_region(
    draw: DrawFn, context: Context, signature: ir.FunctionType
) -> ir.Region:
    """Draw a reachable function CFG, filling dominators before their uses.

    Preallocate all blocks and arguments so branches can reference forward
    blocks and backedges. Non-entry layout is permuted after filling bodies.
    The graph may contain multiple returns, cycles, or no returning path.
    """
    count = draw(st.integers(1, context.bounds.max_blocks))
    blocks = [
        ir.Block(
            tuple(draw_value(draw, context.supply, type_) for type_ in signature.inputs)
        )
    ]
    for _ in range(count - 1):
        arity = draw(st.integers(0, context.bounds.max_arity))
        blocks.append(
            ir.Block(tuple(draw_value(draw, context.supply) for _ in range(arity)))
        )
    successors = _draw_successors(draw, count)
    dominators = _dominators(successors)
    definitions: dict[int, tuple[ir.Value, ...]] = {}
    for index in sorted(
        range(count), key=lambda index: (len(dominators[index]), index)
    ):
        visible = tuple(
            value
            for parent in sorted(dominators[index] - {index})
            for value in definitions[parent]
        )
        block = draw_block(
            draw,
            context,
            block=blocks[index],
            visible=visible,
            output_types=signature.outputs,
            successors=[blocks[target] for target in successors[index]],
        )
        definitions[index] = (
            *block.arguments,
            *(value for op in block.operations for value in ir.get_results(op)),
        )
    return ir.Region([blocks[0], *draw(st.permutations(blocks[1:]))])


def _draw_interface_names(draw: DrawFn, arity: int) -> tuple[str | None, ...] | None:
    """Draw absent interface metadata or exactly one optional nonempty name per slot."""
    if draw(st.booleans()):
        return None
    names = st.one_of(st.none(), st.text(min_size=1, max_size=12))
    return tuple(draw(names) for _ in range(arity))


def draw_function(
    draw: DrawFn,
    *,
    name: str | None = None,
    signature: ir.FunctionType | None = None,
    functions: Mapping[str, ir.FunctionType] | None = None,
    globals: Mapping[str, ir.Type] | None = None,
    bounds: Bounds = DEFAULT_BOUNDS,
    external: bool | None = None,
) -> ir.Function:
    """Draw a declaration or definition against optional module symbol tables.

    With no symbols supplied, the function is self-contained. Supplied symbols
    must exist in the module where the caller places this function. A fresh
    context owns the function's IDs and generation budget.
    """
    name = draw(st.text(min_size=1, max_size=12)) if name is None else name
    signature = (
        draw_function_type(draw, max_arity=bounds.max_arity)
        if signature is None
        else signature
    )
    external = draw(st.booleans()) if external is None else external
    context = Context(
        bounds,
        {} if functions is None else functions,
        {} if globals is None else globals,
    )
    return ir.Function(
        name,
        signature,
        None if external else draw_region(draw, context, signature),
        _draw_interface_names(draw, len(signature.inputs)),
        _draw_interface_names(draw, len(signature.outputs)),
        draw_attributes(draw),
    )


def draw_module(
    draw: DrawFn,
    *,
    max_functions: int = 3,
    max_globals: int = 3,
    bounds: Bounds = DEFAULT_BOUNDS,
) -> ir.Module:
    """Draw an optionally empty module covering all IR nodes and type variants.

    Declare globals and every function signature before drawing any body, so
    calls can refer forward, backward, or recursively. Symbol names are unique
    across both namespaces. No builder, inference, or verifier is called.
    """
    if min(max_functions, max_globals) < 0:
        raise ValueError("module symbol bounds must be nonnegative")
    global_count = draw(st.integers(0, max_globals))
    function_count = draw(st.integers(0, max_functions))
    globals_ = [
        draw_global(draw, name=f"g{index}_{draw(st.text(max_size=8))}")
        for index in range(global_count)
    ]
    signatures = {
        f"f{index}_{draw(st.text(max_size=8))}": draw_function_type(
            draw, max_arity=bounds.max_arity
        )
        for index in range(function_count)
    }
    global_types = {global_.name: global_.type for global_ in globals_}
    functions_ = [
        draw_function(
            draw,
            name=name,
            signature=signature,
            functions=signatures,
            globals=global_types,
            bounds=bounds,
        )
        for name, signature in signatures.items()
    ]
    return ir.Module(functions_, globals_, draw_attributes(draw))


modules = st.composite(draw_module)
"""Strategy wrapper for draw_module, suitable for @given(modules())."""

functions = st.composite(draw_function)
"""Strategy wrapper for draw_function, suitable for @given(functions())."""
