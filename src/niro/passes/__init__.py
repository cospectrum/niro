"""Interface for passes over indexed, verified Niro IR modules.

Optimization, scheduling, and other IR transformations share the
[`Pass`][niro.passes.Pass] interface. Add implementations in modules named by
their transformation, such as `scheduling.py` or `inlining.py`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from niro.index import ModuleIndex
from niro.passes.noop import noop

__all__ = ["Pass", "noop"]

type Pass = Callable[[ModuleIndex], ModuleIndex]
"""A transformation of an indexed, verified module that leaves its input untouched.

Read the verified module through `index.module` and use the index's tables for
queries. Construct edits as new IR, verify the completed output with
[`niro.verify.module`][], then update its index with [`niro.index.reindex_module`][].
Return the original index when nothing changes. Treat shared IR and metadata as
immutable so the input and its index remain valid.

Passes must preserve program semantics. This alias describes the contract;
it does not run verification or refresh indexes automatically.

Examples:
    Run a pass that leaves the module unchanged:

    ```python
    from niro import index, ir, passes, verify

    indexed = index.index_module(verify.module(ir.Module()))
    assert passes.noop(indexed) is indexed
    ```
"""

if TYPE_CHECKING:
    _noop: Pass = noop
