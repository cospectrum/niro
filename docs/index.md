# Niro

Niro compiles computation graphs from [ONNX] (TensorFlow, PyTorch, [JAX][Jaxpr],
and others planned) through one unified, strongly typed SSA IR. [MLIR] is the
primary target; [GIMPLE], [QBE], and other backends may follow.

!!! warning "Work in progress"

    Niro is at an early stage and is not yet ready for production use.

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

## Next steps

- [IR specification](ir.md) — the types, values, and operations Niro compiles through.
- [Python API](niro/index.md) — build, inspect, import, and export Niro programs programmatically.
- [Contributing](https://github.com/cospectrum/niro/blob/main/CONTRIBUTING.md) — set up the
  development environment and run the checks.

[GIMPLE]: https://gcc.gnu.org/onlinedocs/gccint/GIMPLE.html
[Jaxpr]: https://docs.jax.dev/en/latest/jaxpr.html
[MLIR]: https://mlir.llvm.org/
[ONNX]: https://onnx.ai/onnx/
[QBE]: https://c9x.me/compile/doc/il.html
