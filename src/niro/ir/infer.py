"""Infer operation result types from operand types and operation parameters.

Use `ir.infer` to derive result types before constructing operations. Builders
and operation validation use the same inference functions.

```python
from niro import ir

lhs = ir.TensorType(ir.ScalarType.F32, (2, 3))
rhs = ir.TensorType(ir.ScalarType.F32, (3, 4))
result_type = ir.infer.matmul_result_type(lhs, rhs)
transposed_type = ir.infer.transpose_result_type(result_type, (1, 0))
```
"""

from niro import ir
from niro.ir.types import ScalarType, TensorType, Type


def transpose_result_type(
    operand_type: Type, permutation: tuple[int, ...]
) -> TensorType:
    """Infer the tensor type after permuting axes, preserving unknown rank."""
    if not isinstance(operand_type, ir.TensorType):
        raise TypeError("transpose operand must be a tensor")
    if operand_type.shape is None:
        return operand_type
    if sorted(permutation) != list(range(len(operand_type.shape))):
        raise ValueError("transpose permutation must contain every dimension once")
    return ir.TensorType(
        operand_type.element_type,
        tuple(operand_type.shape[index] for index in permutation),
    )


def matmul_result_type(lhs: Type, rhs: Type) -> TensorType:
    """Infer the result type of rank-two matrix multiplication.

    Operand element types and known contracting dimensions must match.
    Unknown dimensions are preserved in the result shape.
    """
    if not isinstance(lhs, ir.TensorType) or not isinstance(rhs, ir.TensorType):
        raise TypeError("matmul operands must be tensors")
    if lhs.shape is None or rhs.shape is None:
        raise TypeError("matmul operands must be ranked tensors")
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise TypeError("matmul operands must be rank-two tensors")
    if lhs.element_type is not rhs.element_type:
        raise TypeError("matmul operand element types must match")
    lhs_inner, rhs_inner = lhs.shape[1], rhs.shape[0]
    if lhs_inner is not None and rhs_inner is not None and lhs_inner != rhs_inner:
        raise ValueError("matmul contracting dimensions must match")
    return ir.TensorType(lhs.element_type, (lhs.shape[0], rhs.shape[1]))


def tensor_extract_result_type(operand: Type, indices: tuple[Type, ...]) -> ScalarType:
    """Infer an element read, requiring a ranked tensor and one integer per axis."""
    if not isinstance(operand, ir.TensorType) or operand.shape is None:
        raise TypeError("tensor extract operand must be a ranked tensor")
    if len(indices) != len(operand.shape):
        raise ValueError("tensor extract requires one index per axis")
    if any(index not in (ir.ScalarType.I32, ir.ScalarType.I64) for index in indices):
        raise TypeError("tensor extract indices must be integer scalars")
    return operand.element_type
