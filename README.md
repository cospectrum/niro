# niro

[![GitHub](https://img.shields.io/badge/GitHub-cospectrum%2Fniro-4C1D95?logo=github)](https://github.com/cospectrum/niro)
[![CI](https://github.com/cospectrum/niro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cospectrum/niro/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/cospectrum/niro/badge.svg?branch=main)](https://coveralls.io/github/cospectrum/niro?branch=main)

Niro compiles computation graphs from [ONNX] (TensorFlow, PyTorch, [JAX][Jaxpr],
and others planned) through one unified, strongly typed SSA IR. [MLIR] is the
primary target; [GIMPLE], [QBE], and other backends may follow.

**[Read the documentation](https://cospectrum.github.io/niro/)** ·
[IR specification](https://cospectrum.github.io/niro/ir/) ·
[Python API](https://cospectrum.github.io/niro/niro/)

> [!WARNING]
> Niro is at an early stage and is not yet ready for production use.
> Pull requests may be rejected while the design is still taking shape.

## Install

```sh
uv tool install git+https://github.com/cospectrum/niro.git
```

## Usage

Compile a model to textual MLIR:

```sh
niro emit mlir model.onnx -o model.mlir
```

Input and output default to the standard streams, so Niro composes with other
tools. Pass `--input-format` when there is no file extension to infer it from:

```sh
niro emit mlir --input-format onnx < model.onnx | mlir-opt
```

Inspect a model's entry-point signature:

```sh
niro inspect signature model.onnx
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

[GIMPLE]: https://gcc.gnu.org/onlinedocs/gccint/GIMPLE.html
[Jaxpr]: https://docs.jax.dev/en/latest/jaxpr.html
[MLIR]: https://mlir.llvm.org/
[ONNX]: https://onnx.ai/onnx/
[QBE]: https://c9x.me/compile/doc/il.html
