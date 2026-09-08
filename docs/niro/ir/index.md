Construct IR programs with `ir.ModuleBuilder`, `ir.FunctionBuilder`, and
`ir.BlockBuilder`. See the [builder reference](builder.md) for their methods.

```python
from niro import ir

module = ir.ModuleBuilder()
function = module.function(name="main", type=ir.FunctionType((), ()))
block = function.region().first_block()
block.return_()
```

Use [`ir.infer`](infer.md) to infer operation result types from operand types
and operation parameters.

::: niro.ir
    options:
      filters:
        - "!^_"
        - "!Builder$"
      heading_level: 1
      show_bases: false      
      show_if_no_docstring: true
      show_root_full_path: true
      show_root_heading: true
