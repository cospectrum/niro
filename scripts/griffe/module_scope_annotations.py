"""Griffe extension resolving annotation names in module scope.

Griffe resolves the first name of a dotted annotation in the enclosing scope, so
a class member shadows a module-level import of the same name: in `niro.builder`
the `ModuleBuilder.ir` attribute shadows `import niro.ir as ir`, and `ir.Type`
resolves to `niro.builder.ModuleBuilder.ir.Type` instead of `niro.ir.Type`,
which silently drops the signature cross-reference.

Python evaluates annotations in module scope, never in class scope, so
re-parenting imported names to their module is both correct and enough to
restore the links.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import griffe

if TYPE_CHECKING:
    from collections.abc import Iterator


class ModuleScopeAnnotations(griffe.Extension):
    """Re-parent annotation names shadowed by class or function members."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        self._visit_module(pkg)

    def _visit_module(self, module: griffe.Module) -> None:
        imports = {name for name, member in module.members.items() if member.is_alias}
        for annotation in _annotations(module) if imports else ():
            for name in annotation.iterate(flat=True):
                if (
                    isinstance(name, griffe.ExprName)
                    and name.name in imports
                    and isinstance(name.parent, griffe.Object)
                    and name.parent is not module
                ):
                    name.parent = module
                    name.member = None
        for member in module.members.values():
            if isinstance(member, griffe.Module):
                self._visit_module(member)


def _annotations(obj: griffe.Module | griffe.Class) -> Iterator[griffe.Expr]:
    """Yield every annotation expression below `obj`, submodules excluded."""
    for member in obj.members.values():
        if isinstance(member, griffe.Class):
            yield from (base for base in member.bases if isinstance(base, griffe.Expr))
            yield from _annotations(member)
        elif isinstance(member, griffe.Function):
            for parameter in member.parameters:
                if isinstance(parameter.annotation, griffe.Expr):
                    yield parameter.annotation
            if isinstance(member.returns, griffe.Expr):
                yield member.returns
        elif isinstance(member, griffe.Attribute):
            if isinstance(member.annotation, griffe.Expr):
                yield member.annotation
            if isinstance(member.value, griffe.Expr):
                yield member.value
        elif isinstance(member, griffe.TypeAlias):
            if isinstance(member.value, griffe.Expr):
                yield member.value
