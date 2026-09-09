"""Generic operation accessors, re-exported in [`niro.ir`][]."""

from __future__ import annotations

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
from niro.ir.program import Region
from niro.ir.values import Value

__all__ = ["get_operands", "get_regions", "get_results"]


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
