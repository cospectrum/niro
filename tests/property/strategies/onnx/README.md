# ONNX model strategies

`models()` builds bounded ONNX graphs with Hypothesis. The default registry,
`OPERATORS`, contains Add, Mul, MatMul, Transpose, Identity, Relu, and If. Its frozen
fields provide type-checked access, such as `OPERATORS.matmul`. Iteration and
string lookup derive from those fields, so adding a default rule requires one
field declaration. String selection in `models()` remains supported. Pass an
`Operator` directly to add a generation rule without changing the registry:

```python
import onnx
from hypothesis import given

from tests.property.strategies import onnx as onnx_st


@given(
    onnx_st.models(
        operators=(
            "Identity",
            onnx_st.unary("Neg", element_types=(onnx.TensorProto.FLOAT,)),
        )
    )
)
def test_model(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
```

`unary()` covers operations that preserve a tensor's shape and element type.
`broadcast_binary()` covers two-input operations with multidirectional
broadcasting; its optional `output_element_type` supports comparisons such as
`Less`. These factories use standard-domain opsets 14 and newer and require no
attributes or input-value restrictions. Select an operator and input types for
which the construction rule is valid in the active opset.

`OPERATORS.if_` draws a scalar Boolean condition and two generated branches
with matching output types and shapes. Branches reuse the selected operator
rules, capture visible outer values, and may contain nested If nodes. Conditions
reuse available scalars or consume an initializer slot when BOOL is enabled.
`models(max_depth=2, max_branch_nodes=6)` sets the default nesting and branch
size limits; zero depth disables If. Branch outputs use Identity to bind a local
result, even when returning a captured value.

## Writing an operator rule

An `Operator(op_type, strategy, domain="")` associates an ONNX name with a
function that accepts a `Context` and returns a Hypothesis `SearchStrategy` of
`NodeSpec`, or `None` when no valid case is available. Inspect compatibility
before returning the strategy; make random choices inside Hypothesis onnx_st.

For example, this rule casts concrete INT32 tensors to INT64 at opset 14:

```python
import onnx
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from tests.property.strategies import onnx as onnx_st


def cast_to_int64(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if context.opsets.get("") != 14:
        return None
    candidates = tuple(
        value
        for value in context.values
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type == onnx.TensorProto.INT32
        and value.type.tensor_type.HasField("shape")
        and all(
            dimension.HasField("dim_value")
            for dimension in value.type.tensor_type.shape.dim
        )
    )
    if not candidates:
        return None

    @st.composite
    def cases(draw: DrawFn) -> onnx_st.NodeSpec:
        value = draw(st.sampled_from(candidates))
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        return onnx_st.NodeSpec(
            inputs=(value.name,),
            outputs=(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, shape),
            ),
            attributes=(onnx.helper.make_attribute("to", onnx.TensorProto.INT64),),
        )

    return cases()


cast = onnx_st.Operator("Cast", cast_to_int64)
models = onnx_st.models(
    operators=("Identity", cast),
    element_types=(onnx.TensorProto.INT32,),
    opsets={"": 14},
)
```

The assembler chooses among applicable rules, allocates node and value names,
adds auxiliary initializers and type information, and selects graph outputs.
Rules must supply compatible operands, accurate output types, valid attributes,
and any restrictions on tensor contents. For example, a Reshape rule must build
a shape tensor whose values preserve the input's element count.
Treat context values and protobuf objects passed to `NodeSpec` as read-only.

The extension types support more than fixed-arity tensor operations:

- `Context.values` is a tuple of `Value(name, type)`. The raw ONNX `TypeProto`
  represents tensor, sequence, optional, and other ONNX value types.
- `NodeSpec.inputs` contains existing value names, unnamed `TensorProto`
  auxiliary initializers, or `None` for omitted optional inputs.
- `NodeSpec.outputs` contains output `TypeProto` objects, or `None` for omitted
  optional outputs. Its length determines the node's output arity.
- `NodeSpec.attributes` contains `AttributeProto` objects, including tensor and
  graph attributes for Constant and control-flow rules. Use
  `Context.name_prefix` when naming values inside nested graphs.
- `Context.initializer_slots` reports the remaining initializer budget.
  `Context.limits` gives rank and dimension bounds and random-source element types;
  `Context.opsets` gives imported domain versions.
- `Context.max_depth` gives the remaining nesting depth. When supplied by the
  assembler, `Context.subgraphs(name, output_types)` draws a bounded child graph
  with those output types, each of which must have an available matching value.
  Subgraph rules must honor the depth limit and use unique names.

## Bounds and validity

`max_initializers` includes both randomly seeded and rule-supplied initializers.
Use `max_seed_initializers=0` to reserve that budget for rules. Input-free rules,
such as Constant, can generate graphs with `max_inputs=0` and no seeded values.
`min_nodes` requires at least that many nodes and raises if applicable rules
cannot reach the count; generation may otherwise stop when no rule applies.
`min_nodes` and `max_nodes` count top-level nodes. `max_branch_nodes` counts all
nodes in each generated If branch, including nested branches and output Identity
nodes. Thus built-in models contain at most
`max_nodes * (1 + 2 * max_branch_nodes)` nodes. Initializer budgets apply separately
to each graph; branches inherit their enclosing context's available slots.

Bounds cover random sources, concrete typed shapes, and auxiliary initializers.
Rules own the validity and bounds of symbolic dimensions, nested graphs, and
attribute tensors, as well as value-dependent constraints. Keep generated
models small enough for the property being tested.

`opsets` updates the default `{"": 14}` imports; `ir_version` is inferred from
known opsets unless supplied. Custom-domain rules require an explicit entry in
`opsets` and may require an explicit `ir_version`. The caller must provide the
corresponding schema/runtime when validating or executing them.
An ONNX schema alone cannot determine valid tensor shapes or contents for every
operator. Add and validate semantic rules for the operators a test needs; this
API does not automatically generate every ONNX operator.
