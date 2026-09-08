"""Infer operation result types from operand types and operation parameters."""

from niro.ir.types import TensorType, Type


def transpose_result_type(
    operand_type: Type, permutation: tuple[int, ...]
) -> TensorType:
    """Infer the tensor type after permuting axes, preserving unknown rank."""
    if not isinstance(operand_type, TensorType):
        raise TypeError("transpose operand must be a tensor")
    if operand_type.shape is None:
        return operand_type
    if sorted(permutation) != list(range(len(operand_type.shape))):
        raise ValueError("transpose permutation must contain every dimension once")
    return TensorType(
        operand_type.element_type,
        tuple(operand_type.shape[index] for index in permutation),
    )


def matmul_result_type(lhs: Type, rhs: Type) -> TensorType:
    """Infer the result type of rank-two matrix multiplication.

    Operand element types and known contracting dimensions must match.
    Unknown dimensions are preserved in the result shape.
    """
    if not isinstance(lhs, TensorType) or not isinstance(rhs, TensorType):
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
    return TensorType(lhs.element_type, (lhs.shape[0], rhs.shape[1]))
