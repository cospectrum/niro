# niro

[![GitHub](https://img.shields.io/badge/GitHub-cospectrum%2Fniro-0969DA?logo=github)](https://github.com/cospectrum/niro)
[![CI](https://github.com/cospectrum/niro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cospectrum/niro/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/cospectrum/niro/badge.svg?branch=main)](https://coveralls.io/github/cospectrum/niro?branch=main)

> **Work in progress:** Niro is at an early stage and is not yet ready for
> production use. Pull requests may be rejected while the design is still
> taking shape.

The goal of Niro is to compile computation graphs from [ONNX], TensorFlow,
PyTorch, and other frontends through one unified SSA IR.
The initial target is [MLIR], primarily using [xDSL]. Future backends may
include WebAssembly, [GIMPLE], and others.

The Niro IR is described in [docs/ir.md](docs/ir.md).

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

Inspect a model's entry-point signature:

```sh
niro inspect signature model.onnx
```

## Development

Run the local CI checks with:

```sh
nix run .#ci
```

### Style guide

We want the project to remain small, direct, and easy to understand without
compromising correctness or output quality. When contributing:

- Prefer simple designs that model the required semantics precisely over
  speculative abstractions.
- Use type hints throughout Python code.
- Keep the core IR independent of any single frontend or backend.
- Establish invariants at construction time, derive redundant information, so
  downstream code can trust constructed values.
- Keep code compact and minimal without sacrificing clarity or required
  behavior.
- Add tests for meaningful behavior and invariants.
- Keep documentation concise, and try to introduce concepts before relying on them.

[GIMPLE]: https://gcc.gnu.org/onlinedocs/gccint/GIMPLE.html
[MLIR]: https://mlir.llvm.org/
[ONNX]: https://onnx.ai/onnx/
[xDSL]: https://docs.xdsl.dev/
