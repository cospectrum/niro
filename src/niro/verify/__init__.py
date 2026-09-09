"""Verification of completed Niro IR modules.

[`verify.module`][niro.verify.module] checks a module's structure, references, and
operations, returning the same module marked as
[`VerifiedModule`][niro.ir.VerifiedModule].

Verify it again after making changes.
"""

from niro.verify.program import module

__all__ = ["module"]
