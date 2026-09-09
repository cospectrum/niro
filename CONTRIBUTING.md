# Contributing to Niro

Niro is an early-stage project. Its design is still evolving, so discuss large
changes before investing substantial work in them. See [todo.md](todo.md) for
the current roadmap and potential work.

## Development setup

Install the locked development environment with [uv]:

```sh
uv sync --locked
```


## Testing

Run the unit and integration tests during development:

```sh
uv run pytest tests/unit
```

Run the end-to-end tests:

```sh
uv run pytest tests/e2e
```

You can run the complete local CI workflow before submitting a change:

```sh
nix run .#ci
```

Unit tests mirror the source tree under `tests/unit/`. End-to-end tests live under
`tests/e2e/`, grouped by the interface or workflow they exercise.

## Documentation

Keep [docs/ir.md](docs/ir.md) language agnostic: it defines the IR's concepts,
structure, semantics, and validity rules. Do not add implementation details such
as Python dataclasses, inheritance, runtime validation mechanisms, or accessor
APIs. Document the Python API in docstrings and `docs/niro/ir/` instead.
Implementation-only refactors should not change the IR specification unless
they change its semantics. In structure diagrams, use `Name*` for zero or more
elements and `Name?` for zero or one.

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

Generate API reference pages from public members using `filters: public` or
filters that exclude private and internal names. Try not to enumerate `members`
explicitly in documentation directives; new public API members should appear
automatically.

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
- Establish invariants at construction time, use assertions to check internal
  invariants, and test meaningful behavior and invariants.
- Keep documentation concise and introduce concepts before relying on them.

[uv]: https://docs.astral.sh/uv/
[google-docstrings]: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
