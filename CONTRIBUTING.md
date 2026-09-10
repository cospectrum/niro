# Contributing to Niro

Niro's design is evolving. Discuss large changes before investing substantial
work; see [todo.md](todo.md) for the roadmap.

## Development

| Task | Command |
| --- | --- |
| Install locked dependencies with [uv] | `uv sync --locked` |
| Unit and integration tests | `uv run pytest tests/unit` |
| Property tests | `uv run pytest tests/property` |
| End-to-end tests | `uv run pytest tests/e2e` |
| Full local CI | `nix run .#ci` |
| Preview docs while editing | `uv run zensical serve` |
| Build docs | `uv run zensical build --clean` |

Test meaningful behavior and invariants. Unit tests mirror `src/` under
`tests/unit/`; group end-to-end tests under `tests/e2e/` by interface or workflow.

Property tests use Hypothesis under `tests/property/`, mirroring the source
modules where useful. Like unit tests, name files after the source module,
such as `optimizations/test_inlining.py`. Generate bounded,
valid programs and check semantic preservation as well as IR validity and input
immutability. Check idempotence when it is part of the pass behavior. CI runs
unit tests, property tests, then end-to-end tests. The shared Hypothesis profile
runs 200 examples per test. Use `--hypothesis-show-statistics` to inspect runtime
and generated-case events when tuning generators or the example budget. Prefer
per-test `@hypothesis.settings(max_examples=...)` overrides when a test needs a
different budget; consider total CI runtime as the suite grows.

When an API requires `VerifiedModule`, obtain it through `verify.module(module)`
(or the builder's `module.verify()`), including in tests. Calling
`ir.VerifiedModule(module)` directly only applies a type marker; it does not
validate the IR. Verify completed transformation results unless the API already
does so before returning.

## Code

- Keep designs small, clear, and correct. Avoid premature optimization;
  optimize measured bottlenecks later.
- Model Niro IR semantics precisely and independently of frontends and backends.
  Keep it high-level and functionalized: tensor updates produce new SSA values.
  Defer memory writes, bufferization, and in-place operations to later lowering
  passes, such as MLIR passes.
- Prefer functions, immutable dataclasses, and transformations returning new IR.
  Use behavior-owning classes only for shared mutable state (builders, value
  allocators) or resource lifecycles; avoid inheritance and classes that merely
  group functions.
- Name optimization modules by subject or transformation (`inlining.py`,
  `transpose.py`) and pass functions by action (`inline_functions`,
  `simplify_transposes`).
- Prefer guard clauses and early returns over nesting.
- Every function and type defined under `src/` must have a docstring, including
  private helpers, methods, classes, and type aliases. For functions, explain
  what they do and return, and any non-obvious assumptions or side effects.
  For types, explain what they represent and their invariants; place alias
  docstrings immediately after the declaration. A concise sentence is enough
  for simple definitions.
  Docstrings must describe the actual implementation and be updated in the same
  change whenever the behavior or contract changes.
- Type-hint all Python code. Trust annotations; reserve runtime type checks for
  external inputs and narrowing unions. Derive redundant information instead
  of storing it.
- Use `Mapping[K, V]` for read-only inputs; use `dict[K, V]` when mutation or a
  concrete dictionary is required.
- Establish invariants during construction. Use `assert` for internal
  preconditions, postconditions, and invariants whose violation indicates a bug;
  fail fast rather than recover. Use explicit exceptions for invalid external
  inputs and expected runtime failures. Keep assertions free of side effects.
- In implementation code and private annotations, prefer short module namespaces
  for Niro symbols (`ir.Op`, `verify.module`, `rewrite.erase_ops`) over importing
  individual names. Keep enough qualification to avoid ambiguity, e.g. `niro.onnx`
  to distinguish it from the `onnx` dependency.
- In public annotations, reference Niro types by directly imported names (`Op`)
  or full paths (`niro.ir.ops.Op`), not short aliases (`ir.Op`), so documentation
  links resolve. This linking requirement does not apply to dependency types.
- For standard-library and third-party imports, prefer module-qualified names
  in executable code (`collections.Counter`) and directly imported types in
  annotations (`Iterator`, `Mapping`).

## Documentation

Be concise, introduce concepts before using them, and update affected docs and
examples when APIs or behavior change. Keep shared content consistent.

- `README.md` and `docs/index.md`: minimal overview, installation, basic usage,
  and links. Keep shared content synchronized, allowing site-specific formatting
  and links. Update only when existing content needs changing; do not add
  announcements or descriptions of new modules, APIs, or features.
- [docs/ir.md](docs/ir.md): language-agnostic IR concepts, structure, semantics,
  and validity. No implementation details: dataclasses, inheritance, runtime
  validation mechanisms, or accessor APIs.
- Python docstrings: API explanations and examples in
  [Google style][google-docstrings], with cross-references to objects and modules.
- `docs/niro/`: API reference directives and rendering options only; no handwritten
  prose or examples. Use `filters: public` or exclude private/internal names.
  Avoid explicit `members` lists so new public members appear automatically.

Document new features on their relevant pages. Preview and build docs using the
commands above.

## Example models

Group generators by format under `scripts/`. Write serialized models to stdout
so they can be saved or piped into Niro:

```sh
uv run scripts/onnx/generate_linear.py | uv run niro inspect signature --input-format onnx
uv run scripts/onnx/generate_linear.py | uv run niro emit mlir --input-format onnx
uv run scripts/onnx/generate_linear.py > linear.onnx
```

[uv]: https://docs.astral.sh/uv/
[google-docstrings]: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
