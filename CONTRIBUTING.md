# Contributing to Niro

Niro's design is evolving. Discuss large changes before investing substantial
work; see [todo.md](todo.md) for the roadmap.

## Development

| Task | Command |
| --- | --- |
| Install locked dependencies with [uv] | `uv sync --locked` |
| Unit and integration tests | `uv run pytest tests/unit` |
| End-to-end tests | `uv run pytest tests/e2e` |
| Full local CI | `nix run .#ci` |
| Preview docs while editing | `uv run zensical serve` |
| Build docs | `uv run zensical build --clean` |

Test meaningful behavior and invariants. Unit tests mirror `src/` under
`tests/unit/`; group end-to-end tests under `tests/e2e/` by interface or workflow.

## Code

- Keep designs small and direct without sacrificing correctness or output
  quality. Model semantics precisely; keep core IR independent of frontends
  and backends.
- Keep Niro IR high-level and functionalized: tensor updates produce new SSA
  values. Defer memory writes, bufferization, and in-place operations to later
  lowering passes, such as MLIR passes.
- Prefer functions, immutable dataclasses, and transformations returning new IR.
  Use behavior-owning classes only for shared mutable state (builders, value
  allocators) or resource lifecycles; avoid inheritance and classes that merely
  group functions.
- Prefer guard clauses and early returns over nesting.
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
