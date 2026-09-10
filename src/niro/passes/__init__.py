"""Interface for passes over verified Niro IR modules.

Optimization, scheduling, and other IR transformations share the
[`Pass`][niro.passes.Pass] interface. Add implementations in modules named by
their transformation, such as `scheduling.py` or `inlining.py`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from niro.ir import VerifiedModule
from niro.passes.noop import noop

__all__ = ["Pass", "noop"]

type Pass = Callable[[VerifiedModule], VerifiedModule]
"""A transformation of verified IR that leaves its input untouched.

Construct edits as new IR and verify the completed output with
[`niro.verify.module`][]. Return the original module when nothing changes.
Treat shared IR and metadata as immutable and preserve program semantics.
This alias describes the contract; it does not run verification automatically.

Examples:
    Run a pass that leaves the module unchanged:

    ```python
    from niro import ir, passes, verify

    module = verify.module(ir.Module())
    assert passes.noop(module) is module
    ```
"""

if TYPE_CHECKING:
    _noop: Pass = noop
