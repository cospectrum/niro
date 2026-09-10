"""Simplification of tensor transpose operations."""

import dataclasses

from niro import ir, rewrite, verify
from niro.ir import VerifiedModule

__all__ = ["simplify_transposes"]


def simplify_transposes(module: VerifiedModule) -> VerifiedModule:
    """Remove identity transposes and compose transpose chains throughout a module.

    For successive permutations `p` and `q`, the composed permutation is `p[q[i]]`
    for each output axis `i`. If it is the identity, redirect uses to the original
    tensor. Otherwise, replace the consumer with one transpose of that tensor,
    preserving the consumer's result ID. Remove the producer when its result
    has no other uses.

    Follow SSA operands, including captures in nested regions; transposes need
    not be adjacent. Simplify until no further changes apply. Tensor rank must
    be known, but individual dimensions may be dynamic. Unknown-rank transposes
    are left unchanged.

    Args:
        module: Verified input module. Its functions and metadata are not mutated.

    Returns:
        A verified module with simplified transposes, or the input object if
        nothing changes. Unchanged IR and metadata may be shared.

    Examples:
        Two axis swaps cancel, leaving the function's argument as its result:

        Before:

        ```text
        module {
          func @round_trip(%0: tensor<2x3xf32>) -> tensor<2x3xf32> {
            %1 = transpose %0, [1, 0] : tensor<3x2xf32>
            %2 = transpose %1, [1, 0] : tensor<2x3xf32>
            return %2
          }
        }
        ```

        After:

        ```text
        module {
          func @round_trip(%0: tensor<2x3xf32>) -> tensor<2x3xf32> {
            return %0
          }
        }
        ```

        ```python
        from niro import ir, optimizations, verify

        tensor = ir.TensorType(ir.ScalarType.F32, (2, 3))
        swapped = ir.TensorType(ir.ScalarType.F32, (3, 2))
        x = ir.Value(ir.ValueId(0), tensor)
        y = ir.Value(ir.ValueId(1), swapped)
        result = ir.Value(ir.ValueId(2), tensor)
        block = ir.Block((x,), [
            ir.Transpose(y, x, (1, 0)),
            ir.Transpose(result, y, (1, 0)),
            ir.Return((result,)),
        ])
        function = ir.Function(
            "round_trip", ir.FunctionType((tensor,), (tensor,)), ir.Region([block])
        )
        module = verify.module(ir.Module(functions=[function]))
        optimized = optimizations.simplify_transposes(module)
        body = optimized.functions[0].first_block
        assert body is not None
        assert body.operations == [ir.Return((x,))]
        assert len(block.operations) == 3
        ```
    """
    functions = [_simplify_function(function) for function in module.functions]
    if all(a is b for a, b in zip(functions, module.functions, strict=True)):
        return module
    return verify.module(dataclasses.replace(module, functions=functions))


def _simplify_function(function: ir.Function) -> ir.Function:
    """Return a function with nested transposes simplified to a fixed point."""
    if function.body is None:
        return function
    while True:
        assert function.body is not None
        for op in ir.iter_ops(function.body):
            if not isinstance(op, ir.Transpose):
                continue
            updated = _simplify_transpose(function, op)
            if updated is function:
                continue
            function = updated
            # Edits invalidate references; resume traversal on the new function.
            break
        else:
            return function


def _simplify_transpose(function: ir.Function, op: ir.Transpose) -> ir.Function:
    """Return a function with an identity transpose removed or a chain composed.

    Preserve the input when no rewrite applies, including unknown-rank operands.
    Remove the producer only when every use belongs to the rewritten consumer.
    """
    tensor = op.operand.type
    assert isinstance(tensor, ir.TensorType)
    if tensor.rank is None:
        return function
    identity = tuple(range(tensor.rank))
    if op.permutation == identity:
        return rewrite.replace_op(
            function, op, (), replacements={op.result.id: op.operand}
        )

    definition = ir.get_definition(function, op.operand.id)
    if not isinstance(definition, ir.OpResult):
        return function
    producer = definition.owner
    if not isinstance(producer, ir.Transpose):
        return function

    permutation = tuple(producer.permutation[axis] for axis in op.permutation)
    removed: list[ir.Op] = [op]
    if all(use.owner is op for use in ir.iter_uses(function, producer.result.id)):
        removed.append(producer)
    if permutation == identity:
        return rewrite.replace_ops(
            function, removed, (), replacements={op.result.id: producer.operand}
        )
    replacement = ir.Transpose(op.result, producer.operand, permutation)
    return rewrite.replace_ops(
        function, removed, (replacement,), at=rewrite.before(function, op)
    )
