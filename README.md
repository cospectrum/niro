# niro

[![GitHub](https://img.shields.io/badge/GitHub-cospectrum%2Fniro-0969DA?logo=github)](https://github.com/cospectrum/niro)
[![CI](https://github.com/cospectrum/niro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cospectrum/niro/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/cospectrum/niro/badge.svg?branch=main)](https://coveralls.io/github/cospectrum/niro?branch=main)

> **Work in progress:** Niro is at an early stage and is not yet ready for
> production use. Pull requests may be rejected while the design is still
> taking shape.

The goal of Niro is to compile computation graphs from ONNX, TensorFlow,
PyTorch, and other frontends through one unified SSA IR. The initial target is
MLIR, primarily using xDSL. Future backends may include WebAssembly, GIMPLE, and
others.

## Development

The custom SSA intermediate representation is documented in the
[Niro IR specification](docs/ir.md).

We want the project to remain small, direct, and easy to understand without
compromising correctness or output quality. When contributing:

- Prefer simple designs that model the required semantics precisely over
  speculative abstractions.
- Use type hints throughout Python code.
- Keep the core IR independent of any single frontend or backend.
- Add tests for meaningful behavior and invariants.
- Keep documentation concise, and introduce concepts before relying on them.
- Write code and prose with care for the people who will read them next.

Run the local CI checks with:

```sh
nix run .#ci
```
