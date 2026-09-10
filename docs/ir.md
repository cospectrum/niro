# Niro IR

Niro IR is a strongly typed, SSA-based intermediate representation. It accepts
input from frontends such as ONNX and is designed to lower to MLIR and other
backends.

This document defines the IR itself: its types, values, structure, operations,
and validity rules.

The current reference implementation is in `src/niro/ir/`.

## Contents

- [Types](#types)
- [SSA values](#ssa-values)
- [Program](#program)
- [Literals and attributes](#literals-and-attributes)
- [Operations](#operations)
- [Complete example](#complete-example)
- [Well-formed IR](#well-formed-ir)

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

### `Value`

A `Value` is an immutable reference containing an ID and type:

```python
value = Value(ValueId(0), ScalarType.F32)
```

The ID and type cannot change. A value is defined when it appears as a block
argument or as an operation result. Later operations refer to that value as an
operand. The value's definition is identified by its position in the IR.

## Program

Niro uses the following hierarchy. `Name*` means zero or more elements, and
`Name?` means zero or one:

```text
Module
├── Global*
└── Function*
    └── Region?
        └── Block*
            └── Op*
                └── Region*
```

Regions nested in operations repeat the same region/block/operation structure.
Validity rules below constrain the number of elements in each context.

### `Module`

A module owns globals and functions:

```python
module = Module()
```

Functions and globals share one symbol namespace. Their names must be unique
within the module.

#### `SymbolName`

`SymbolName` names a module-level declaration. References to functions and
globals are resolved by name. Operations are unnamed; their SSA results are
identified by `ValueId`.

### `Global`

A global is an immutable initialized module value:

```python
weight = Global(name="weight", type=weight_type, initializer=weight_data)
```

The initializer must match the global's declared type, even if the global is
unused. Tensor initializers require a static shape and exactly the number of
packed bytes implied by the shape and element type, following the same literal
rules as `Const`.

### `Function`

A function has a name, signature, and optional body:

#### Function type

`FunctionType` contains the input and output types of a function:

```python
signature = FunctionType(
    inputs=(ScalarType.F32, ScalarType.F32),
    outputs=(ScalarType.F32,),
)
```

A function may have any number of inputs and outputs, including zero.

#### Declaration or definition

```python
external = Function(
    name="print_f32",
    type=FunctionType(inputs=(ScalarType.F32,), outputs=()),
    body=None,
)
```

A function without a body is declared here but implemented elsewhere. A
defined function has an entry block whose arguments represent the inputs
described by its `FunctionType`.

`input_names` and `output_names` optionally describe the function's public
interface. A present sequence has the same arity as its side of the function
type. Each position contains either a non-empty name or `None` for an unnamed
item. These names are metadata and do not replace numeric SSA value IDs.

### `Region`

A region owns blocks:

```python
entry = Block()
body = Region(blocks=[entry])
```

A function body is a region. Some operations, such as `If`, also own nested
regions. A nested region may use values visible at the containing operation.
Values created inside it can leave only through results of that operation.

A region has one or more blocks, except opaque operations may own empty regions. Its first block is the entry; every other
block must be reachable from it through branches. Branches stay in their
immediately containing region and cannot target its entry block. Blocks and
regions each have a unique owner. Function bodies allow loops, including loops
with no exit. Block order after entry does not determine execution order. Each `If`
region is restricted to exactly one argument-free block ending in `Yield`.

### `Block`

A block contains input values followed by an ordered sequence of operations:

```python
x = Value(ValueId(0), ScalarType.F32)
block = Block(arguments=(x,), operations=[])
```

Block arguments are values available from the start of the block. The entry
block of a defined function uses them as its function inputs. Their types must
match `FunctionType.inputs` in the same order:

```python
tuple(
    argument.type for argument in function.first_block.arguments
) == function.type.inputs
```

An external function has no body and therefore no entry block or block
arguments. All blocks in a function share its value-ID namespace.

Other blocks receive their arguments from incoming branches. Each block ends
with `Branch`, `CondBranch`, an `UnknownOp` with successors, or the exit terminator required by its region:
`Return` for function bodies and `Yield` for nested regions. A terminator must
be the last operation in its block.

### `Op`

`Op` is the closed union of operation kinds that may appear in a block:

```python
type Op = (
    Const
    | GetGlobal
    | Transpose
    | TensorExtract
    | Add
    | Mul
    | MatMul
    | Call
    | Branch
    | CondBranch
    | Return
    | Yield
    | If
    | UnknownOp
)
```

The concrete operations are described in the Operations section below. Keeping
the union explicit makes generic operation handling exhaustive when new kinds
are added.

## Literals and attributes

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

Niro has a fixed set of known operation kinds. Backends must implement their
semantics or reject unsupported operations explicitly.

### `Const`

`Const` introduces literal data as a typed SSA value:

```python
result = Value(ValueId(0), ScalarType.F32)
constant = Const(result=result, literal=2.0)
```

Here, `2.0` is the literal stored in the IR. `result` is the `F32` value used by
later operations. A constant's literal must match its result type. A tensor
constant has a static shape and exactly the number of packed bytes implied by
its shape and element type.

### `GetGlobal`

`GetGlobal` references an immutable module global as an SSA value:

```python
Global(name="weight", type=weight_type, initializer=weight_data)

result = Value(ValueId(0), weight_type)
GetGlobal(name="weight", result=result)
```

The referenced symbol must name a global and its type must match the result.

### Arithmetic operations

`Add` and `Mul` compute element-wise or scalar arithmetic. `MatMul` performs
matrix multiplication. Each consumes two operands and produces one result:

```python
result = Value(ValueId(2), ScalarType.F32)
add = Add(result=result, lhs=left, rhs=right)
```

Arithmetic operands and results have compatible types. `MatMul` operates on
compatible rank-two tensors; its result shape follows from the operand shapes.

### `Transpose`

`Transpose` reorders tensor dimensions:

```python
transpose = Transpose(
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
    callee="add",
    arguments=(lhs, rhs),
    results=(result,),
)
```

A call may have any number of arguments and results, including zero. The callee
may be a definition or external declaration in the same module.

### `Branch` and `CondBranch`

`Branch` transfers control to a destination block, passing values to its block
arguments. `CondBranch` selects its true or false destination using a scalar
`BOOL` condition and passes the corresponding argument tuple. Both terminate
their block and produce no results.

```python
merged = Value(ValueId(3), ScalarType.I32)
join = Block(arguments=(merged,), operations=[Return((merged,))])
left = Block(operations=[Branch(target=join, arguments=(lhs,))])
right = Block(operations=[Branch(target=join, arguments=(rhs,))])
entry = Block(
    arguments=(condition, lhs, rhs),
    operations=[CondBranch(condition, true_target=left, false_target=right)],
)
body = Region(blocks=[entry, left, right, join])
```

Each edge's arguments must match its destination's block arguments in number,
order, and type. Both conditional edges may target the same block with different
arguments. Block arguments express merged and loop-carried values without a
separate phi operation.

### `Return`

`Return` terminates a function and returns zero or more values:

```python
return_op = Return(operands=(result,))
```

The returned values must match the function's output types.

### `Yield`

`Yield` terminates a nested region and passes values to the operation that owns
the region. Unlike `Return`, it does not return from the function:

```python
yield_op = Yield(operands=(result,))
```

### `TensorExtract`

`TensorExtract` reads one element of an immutable ranked tensor and returns its
scalar element type. It leaves the tensor unchanged. Supply one `I32` or `I64`
scalar index per axis; indices must be in bounds at execution. A rank-zero tensor
needs no indices:

```python
extract = TensorExtract(result=scalar, operand=tensor, indices=())
```

For example, extracting from `TensorType(BOOL, ())` produces `BOOL`, suitable
as an `If` condition.

### `If`

`If` selects one of two single-block regions using a scalar boolean condition
and produces zero or more results. Both regions are required. Each contains one
argument-free block ending in `Yield`; its yielded values become the `If` results:

```python
if_op = If(
    results=(result,),
    condition=condition,
    then_region=Region(blocks=[Block(operations=[Yield(operands=(then_value,))])]),
    else_region=Region(blocks=[Block(operations=[Yield(operands=(else_value,))])]),
)
```

Both regions must yield the same number and types of values as `If.results`.
An empty result tuple still requires both regions and their `Yield` terminators.
`Return`, `Branch`, and `CondBranch` cannot appear directly in either arm.
Other operations, including calls and nested `If` operations, follow their
normal validity rules. The regions may use values visible before the `If`, but
their local values cannot be used outside directly.

### `UnknownOp`

`UnknownOp` preserves an operation whose semantics are not yet modeled by Niro:

```python
unknown = UnknownOp(
    name="onnx.LeakyRelu",
    operands=(input_value,),
    results=(result,),
    attributes={"alpha": 0.01},
)
```

Optional `regions` and `successors` tuples preserve nested structure and CFG
edges. Regions have unique ownership, may capture enclosing values, and may be
empty. Nonempty regions follow SSA dominance rules and exit through `Yield` or
branches. Yield signatures and successor operand conventions remain opaque.
An unknown operation with successors terminates its block; targets must belong
to the enclosing region and cannot be its entry block.

An unknown operation still has a complete SSA interface and follows normal scope
and uniqueness rules. A backend may preserve it as a custom operation or reject
it with a clear diagnostic; the current MLIR backend rejects unknown operations.

## Complete example

This example defines a function that adds two `F32` values.

```python
lhs = Value(ValueId(0), ScalarType.F32)
rhs = Value(ValueId(1), ScalarType.F32)
result = Value(ValueId(2), ScalarType.F32)

entry = Block(
    arguments=(lhs, rhs),
    operations=[
        Add(result=result, lhs=lhs, rhs=rhs),
        Return(operands=(result,)),
    ],
)

add = Function(
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

A Niro module has unique symbol names across globals and functions. Within each
function, value IDs are unique, operands refer to visible definitions, and uses
obey SSA dominance.

A block dominates another block when every path from region entry to the latter
passes through the former. A value may be used in blocks dominated by its
defining block. Within its defining block, an operation result is available
only after that operation; block arguments are available from the start.
Nested regions may capture only values available before their owning operation.
Values defined on just one arm of a conditional cannot be used directly at a
join; pass them through the join's block arguments instead.

Tensor dimensions are non-negative. Function inputs and returns match their
signatures. Operation operands and results have compatible types and shapes,
and calls name functions with matching signatures. Every block has a valid
terminator, each region exit matches its containing operation, and values
defined inside a region are not visible outside it.
