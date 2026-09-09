"""Public Python API for Niro.

Conversion names are relative to Niro IR: `from_onnx` converts an ONNX model
to Niro IR, and `to_mlir` converts Niro IR to an MLIR module.

```python
import niro
import onnx

module = niro.from_onnx(onnx.load("model.onnx"))
mlir_module = niro.to_mlir(module)
text = niro.format_mlir(mlir_module)
```
"""

from niro import ir
from niro.mlir import format_mlir, to_mlir, write_mlir
from niro.onnx import from_onnx

__all__ = ["format_mlir", "from_onnx", "ir", "to_mlir", "write_mlir"]
