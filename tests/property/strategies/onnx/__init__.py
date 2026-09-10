"""Composable Hypothesis strategies for valid ONNX graphs.

Use ``models()`` for the default operator set. Pass additional ``Operator``
objects to ``models(operators=...)`` to extend generation without changing the
graph assembler. See README.md for the rule contract and examples.
"""

from ._core import ELEMENT_TYPES as ELEMENT_TYPES
from ._core import OPSET_VERSION as OPSET_VERSION
from ._core import Context as Context
from ._core import Limits as Limits
from ._core import NodeSpec as NodeSpec
from ._core import Operator as Operator
from ._core import Value as Value
from ._core import tensor_shape as tensor_shape
from ._core import tensor_shapes as tensor_shapes
from ._graph import models as models
from ._operators import OPERATORS as OPERATORS
from ._operators import broadcast_binary as broadcast_binary
from ._operators import unary as unary
