"""Optimization passes over Niro IR modules.

Passes share the [`ModulePass`][niro.optimizations.ModulePass] interface.
Function-local algorithms can be composed inside a module pass; transformations
across functions can coordinate edits over the whole module.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from niro.ir import VerifiedModule
from niro.optimizations.inlining import inline_functions
from niro.optimizations.transpose import simplify_transposes

__all__ = ["ModulePass", "inline_functions", "simplify_transposes"]

type ModulePass = Callable[[VerifiedModule], VerifiedModule]
"""A callable that takes and returns a [`VerifiedModule`][niro.ir.VerifiedModule].

A pass must preserve program semantics and leave its input untouched. Construct
edits as new IR and verify the completed output with [`niro.verify.module`][].
An unchanged module may be returned directly. Treat shared IR and metadata as
immutable.

This alias describes the pass contract; it does not perform verification.
"""

if TYPE_CHECKING:
    import typing

    def _type_check_pass[**P](
        fn: Callable[typing.Concatenate[VerifiedModule, P], VerifiedModule],
    ) -> None:
        pass

    _type_check_pass(inline_functions)
    _type_check_pass(simplify_transposes)
