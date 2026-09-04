# Niro

Niro compiles computation graphs from ONNX and future input formats through a
unified, strongly typed SSA IR. Its initial output target is MLIR.

## Install

```sh
uv tool install git+https://github.com/cospectrum/niro.git
```

## Use

Inspect an ONNX model's entry-point signature:

```sh
niro inspect signature model.onnx
```

Emit textual MLIR:

```sh
niro emit mlir model.onnx
```

Start with the [IR reference](ir.md), browse the [Python API](niro/index.md),
or read the [contribution guide](https://github.com/cospectrum/niro/blob/main/CONTRIBUTING.md).
