# niro

[![GitHub](https://img.shields.io/badge/GitHub-cospectrum%2Fniro-4C1D95?logo=github)](https://github.com/cospectrum/niro)
[![CI](https://github.com/cospectrum/niro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cospectrum/niro/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/cospectrum/niro/badge.svg?branch=main)](https://coveralls.io/github/cospectrum/niro?branch=main)

> **Work in progress:** Niro is at an early stage and is not yet ready for
> production use. Pull requests may be rejected while the design is still
> taking shape.

The goal of Niro is to compile computation graphs from [ONNX], TensorFlow,
PyTorch, and other frontends through one unified SSA IR.
The initial target is [MLIR], primarily using [xDSL]. Future backends may
include WebAssembly, [GIMPLE], and others.

[Documentation](https://cospectrum.github.io/niro/)

## Installation

### CLI

```sh
uv tool install git+https://github.com/cospectrum/niro.git
```

## Usage

### CLI

Emit textual MLIR to standard output:

```sh
niro emit mlir model.onnx
```

Use `-o` to write it to a file:

```sh
niro emit mlir model.onnx -o model.mlir
```

Niro infers the input format from recognized file extensions. Specify it when
reading from standard input:

```sh
niro emit mlir --input-format onnx < model.onnx | mlir-opt
```

You can also inspect a model's entry-point signature:

```sh
niro inspect signature model.onnx
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

[GIMPLE]: https://gcc.gnu.org/onlinedocs/gccint/GIMPLE.html
[MLIR]: https://mlir.llvm.org/
[ONNX]: https://onnx.ai/onnx/
[xDSL]: https://docs.xdsl.dev/
