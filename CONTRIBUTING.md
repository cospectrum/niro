# Contributing to Niro

Niro is an early-stage project. Its design is still evolving, so discuss large
changes before investing substantial work in them. See [todo.md](todo.md) for
the current roadmap and potential work.

## Development setup

Install the locked development environment with [uv]:

```sh
uv sync --locked
```

Alternatively, enter the Nix development shell, which provides uv:

```sh
nix develop
```

## Testing

Run the unit and integration tests during development:

```sh
uv run pytest tests
```

Run the end-to-end tests:

```sh
uv run pytest e2e
```

Run the complete local CI workflow before submitting a change:

```sh
nix run .#ci
```

Unit tests mirror the source tree under `tests/`. End-to-end tests live under
`e2e/`, grouped by the interface or workflow they exercise.

## Documentation

Preview the documentation with Zensical while editing it:

```sh
uv run zensical serve
```

Build it with Zensical:

```sh
uv run zensical build --clean
```

Write Python docstrings in [Google style][google-docstrings]. Use cross-references
for Python objects and modules so generated API references are clickable.

## Example models

Model generators are grouped by format under `scripts/`. They write a
serialized model to stdout so it can be saved to a file or piped directly into
Niro.

Inspect the signature of the example ONNX linear model:

```sh
uv run scripts/onnx/generate_linear.py \
  | uv run niro inspect signature --input-format onnx
```

Emit MLIR from it:

```sh
uv run scripts/onnx/generate_linear.py \
  | uv run niro emit mlir --input-format onnx
```

Or save it for repeated use:

```sh
uv run scripts/onnx/generate_linear.py > linear.onnx
```

## Style guide

We want the project to remain small, direct, and easy to understand without
compromising correctness or output quality. When contributing:

- Prefer compact, straightforward design that models the required semantics
  precisely, and keep the core IR independent of any single frontend or backend.
- Use type hints throughout Python code and derive redundant information rather
  than storing it.
- Keep IR dataclasses free of validation. Implement IR validity rules in
  `src/niro/ir/verify.py`; builders call its local `check_*` functions before
  insertion, and `niro.ir.verify` checks complete modules. Use assertions for
  internal invariants, and test meaningful behavior and validity rules.
- Give each check or verifier one IR type as its subject. Check field-specific
  rules with their owning IR type; avoid generic validators controlled by a
  diagnostic label.
- Keep documentation concise and introduce concepts before relying on them.

[uv]: https://docs.astral.sh/uv/
[google-docstrings]: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
