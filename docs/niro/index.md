Conversion names are relative to Niro IR: `from_onnx` converts an ONNX model
to Niro IR, and `to_mlir` converts Niro IR to an MLIR module. Use `format_mlir`
or `write_mlir` to produce textual MLIR from that module.

```python
import niro
import onnx

module = niro.from_onnx(onnx.load("model.onnx"))
mlir_module = niro.to_mlir(module)
text = niro.format_mlir(mlir_module)
```

::: niro
    options:
      heading_level: 1
      members:
        - from_onnx
        - to_mlir
        - format_mlir
        - write_mlir
      show_root_full_path: true
      show_root_heading: true
