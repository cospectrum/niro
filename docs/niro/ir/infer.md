Use `ir.infer` to derive result types before constructing operations. Builders
and operation validation use the same inference functions.

```python
from niro import ir

lhs = ir.TensorType(ir.ScalarType.F32, (2, 3))
rhs = ir.TensorType(ir.ScalarType.F32, (3, 4))
result_type = ir.infer.matmul_result_type(lhs, rhs)
transposed_type = ir.infer.transpose_result_type(result_type, (1, 0))
```

::: niro.ir.infer
    options:
      heading_level: 1
      filters: public
      show_root_full_path: true
      show_root_heading: true
