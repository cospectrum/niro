# Niro IR

Niro IR is a strongly typed, SSA-based intermediate representation. It accepts
input from frontends such as ONNX and is designed to lower to MLIR and other
backends.

This document defines the IR itself: its types, values, structure, operations,
and validity rules.

The current reference implementation is in `src/niro/ir.py`.

## Types

### `ScalarType`

`ScalarType` describes a single non-tensor value. The initial scalar types are
`BOOL`, `I32`, `I64`, `F32`, and `F64`.

### `TensorType`

`TensorType` describes a tensor by its element type and shape:

```python
TensorType(element_type=ScalarType.F32, shape=(3, 4))
```

Each shape dimension is either a non-negative integer or dynamic. A dynamic
dimension has an unknown size.

| Shape | Meaning | MLIR-like notation |
| --- | --- | --- |
| `None` | Unknown rank | `tensor<*xf32>` |
| `()` | Rank-zero tensor | `tensor<f32>` |
| `(3,)` | Vector with three elements | `tensor<3xf32>` |
| `(3, 4)` | 3-by-4 matrix | `tensor<3x4xf32>` |
| `(None, 4)` | Matrix with a dynamic first dimension | `tensor<?x4xf32>` |

A scalar and a rank-zero tensor are different types. `ScalarType.F32` is a
single `f32`, while `TensorType(ScalarType.F32, ())` is a tensor containing one
`f32`.

### `Type`

`Type` is the union of all types currently supported by Niro:

```python
type Type = ScalarType | TensorType
```

## SSA values

An operation is an instruction that consumes and produces values. In static
single-assignment (SSA) form, every value is defined exactly once and may be
used many times.

### `ValueId`

`ValueId` identifies a value. IDs are unique within an entire function,
including its nested regions. Different functions may reuse the same IDs.

Function inputs, block inputs, and operation results share the same
function-level ID namespace.

### `FuncId` and `OpId`

`FuncId` identifies a function and is unique within its module. `OpId`
identifies an operation and is unique within its entire function, including
nested regions. Different modules may reuse function IDs, and different
functions may reuse operation IDs.

Function names remain the public symbols used for calls and linking; a
function ID is its internal identity. Function, operation, and value IDs are
non-negative.

### `Value`

A `Value` is an immutable reference containing an ID and type:

```python
value = Value(ValueId(0), ScalarType.F32)
```

The ID and type cannot change. A value is defined when it appears as a block
argument or as an operation result. Later operations refer to that value as an
operand. The value's definition is identified by its position in the IR.

## Program structure

Niro uses the following hierarchy:

```text
Module
└── Function
    └── Region
        └── Block
            └── Operation
```

### `Block`

A block contains input values followed by an ordered sequence of operations:

```python
x = Value(ValueId(0), ScalarType.F32)
block = Block(arguments=(x,), operations=[])
```

Block arguments are values available from the start of the block. The entry
block of a function uses them for the function inputs. All blocks in a function
share its value-ID namespace.

### `Region`

A region owns blocks:

```python
entry = Block()
body = Region(blocks=[entry])
```

A function body is a region. Some operations, such as `If`, also own nested
regions. A nested region may use values visible at the containing operation.
Values created inside it can leave only through results of that operation.

Niro currently requires a single block in regions used for structured control
flow. General multi-block control flow may be supported later.

### `FunctionType`

`FunctionType` contains the input and output types of a function:

```python
signature = FunctionType(
    inputs=(ScalarType.F32, ScalarType.F32),
    outputs=(ScalarType.F32,),
)
```

A function may have any number of inputs and outputs, including zero.

### `Function`

A function has a name, signature, and optional body:

```python
external = Function(
    id=FuncId(0),
    name="print_f32",
    type=FunctionType(inputs=(ScalarType.F32,), outputs=()),
    body=None,
)
```

A function without a body is declared here but implemented elsewhere. For a
defined function, the entry block arguments must match the signature's input
types.

### `Module`

A module owns the functions in a program and provides the symbol scope used to
resolve function names:

```python
module = Module(functions=[external])
```

Function names must be unique within a module.

## Stored data

### `Literal`

A literal is data written directly into the IR instead of being calculated at
runtime. Examples include:

```python
True
42
2.0
b"raw data"
```

Niro literals may be booleans, integers, floating-point numbers, or bytes. A
literal does not specify an IR type by itself. The operation that introduces
it also provides the typed result.

Tensor constants are stored as contiguous, row-major bytes in little-endian
element order. Their element type and shape come from the result's
`TensorType`. They are not expanded into aggregate literals, since doing so
would add substantial object and pointer overhead for model weights.

### `Attribute`

An attribute is compile-time configuration or metadata, not a runtime value.
Generic attributes support an empty value, booleans, integers, floating-point
numbers, strings, bytes, and aggregate values.

Known operations use named, typed fields. For example,
`Transpose.permutation` is clearer than a string-keyed attribute. Generic
attributes are useful for `UnknownOp`, whose fields are not known in advance:

```python
attributes = {"alpha": 0.01}
```

Function and module attributes are reserved for metadata such as entry points,
linkage, visibility, target options, and debug information.

## Operations

Niro has a fixed set of known operation kinds. Adding an operation also requires
supporting its validity rules and lowering behavior.

### `Const`

`Const` introduces literal data as a typed SSA value:

```python
result = Value(ValueId(0), ScalarType.F32)
constant = Const(id=OpId(0), result=result, value=2.0)
```

Here, `2.0` is the literal stored in the IR. `result` is the `F32` value used by
later operations. A constant's literal must match its result type. A tensor
constant has a static shape and exactly the number of packed bytes implied by
its shape and element type.

### Arithmetic operations

`Add` and `Mul` compute element-wise or scalar arithmetic. `MatMul` performs
matrix multiplication. Each consumes two operands and produces one result:

```python
result = Value(ValueId(2), ScalarType.F32)
add = Add(id=OpId(0), result=result, lhs=left, rhs=right)
```

Arithmetic operands and results have compatible types. `MatMul` operates on
compatible rank-two tensors; its result shape follows from the operand shapes.

### `Transpose`

`Transpose` reorders tensor dimensions:

```python
transpose = Transpose(
    id=OpId(0),
    result=result,
    operand=input_value,
    permutation=(1, 0),
)
```

The permutation lists the input dimension used for each result dimension.

### `Call`

`Call` invokes a function by name:

```python
function_call = Call(
    id=OpId(0),
    callee="add",
    arguments=(lhs, rhs),
    results=(result,),
)
```

A call may have any number of arguments and results, including zero. The callee
may be a definition or external declaration in the same module.

### `Return`

`Return` terminates a function and returns zero or more values:

```python
return_op = Return(id=OpId(0), operands=(result,))
```

The returned values must match the function's output types.

### `Yield`

`Yield` terminates a nested region and passes values to the operation that owns
the region. Unlike `Return`, it does not return from the function:

```python
yield_op = Yield(id=OpId(0), operands=(result,))
```

### `If`

`If` selects one of two single-block regions using a boolean condition. Both
regions end with `Yield`, and those yielded values become the `If` results:

```python
if_op = If(
    id=OpId(0),
    results=(result,),
    condition=condition,
    then_region=Region(
        blocks=[Block(operations=[Yield(id=OpId(1), operands=(then_value,))])]
    ),
    else_region=Region(
        blocks=[Block(operations=[Yield(id=OpId(2), operands=(else_value,))])]
    ),
)
```

Both regions must yield the same number and types of values as `If.results`.
They may use values visible before the `If`, but their local values cannot be
used outside directly.

### `UnknownOp`

`UnknownOp` preserves an operation whose semantics are not yet modeled by Niro:

```python
unknown = UnknownOp(
    id=OpId(0),
    name="onnx.LeakyRelu",
    operands=(input_value,),
    results=(result,),
    attributes={"alpha": 0.01},
)
```

An unknown operation still has a complete SSA interface and follows the normal
scope and uniqueness rules. A backend may preserve it as a custom operation or
reject it with a clear diagnostic.

## Complete example

This example defines a function that adds two `F32` values.

```python
lhs = Value(ValueId(0), ScalarType.F32)
rhs = Value(ValueId(1), ScalarType.F32)
result = Value(ValueId(2), ScalarType.F32)

entry = Block(
    arguments=(lhs, rhs),
    operations=[
        Add(id=OpId(0), result=result, lhs=lhs, rhs=rhs),
        Return(id=OpId(1), operands=(result,)),
    ],
)

add = Function(
    id=FuncId(0),
    name="add",
    type=FunctionType(
        inputs=(ScalarType.F32, ScalarType.F32),
        outputs=(ScalarType.F32,),
    ),
    body=Region(blocks=[entry]),
)

module = Module(functions=[add])
```

## Well-formed IR

A Niro module has unique function names and function IDs. Within each function,
operation IDs and value IDs are unique, operands refer to visible definitions,
and uses obey SSA dominance.

Tensor dimensions are non-negative. Function inputs and returns match their
signatures. Operation operands and results have compatible types and shapes,
and calls name functions with matching signatures. Every region ends with the
terminator required by its containing operation, and values defined inside a
region are not visible outside it.
