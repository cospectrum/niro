# Contributing to Niro

Niro's design is evolving: discuss large changes before substantial work.
See [todo.md](todo.md) for the roadmap.

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

Test meaningful behavior and invariants. Unit tests mirror `src/`; Hypothesis
property tests mirror source modules where useful. Name both after source modules.
Group end-to-end tests by interface or workflow. CI runs unit, property (Ubuntu
only), then end-to-end tests.

Obtain required `VerifiedModule` values through `verify.module(module)` or the
builder's `module.verify()`, including in tests: `ir.VerifiedModule(module)` only
adds a type marker, without validation. Verify completed transformations unless
the API verifies results before returning.

## Code

- Keep designs small, clear, and correct; optimize only measured bottlenecks.
- Model IR semantics precisely, independently of frontends/backends. Keep IR
  high-level and functionalized: tensor updates produce new SSA values. Defer
  memory writes, bufferization, and in-place operations to later lowering (e.g. MLIR).
- Prefer functions, immutable dataclasses, and transformations returning new IR.
  Reserve behavior-owning classes for shared mutable state (builders, value
  allocators) or resource lifecycles; avoid inheritance and function-grouping classes.
- Compose focused functions in callers instead of selecting behaviors with flags.
- Name pass modules by subject or transformation and pass functions by action.
- Prefer guard clauses and early returns over nesting.
- Where possible, put public functions/methods and tests before private/test
  helpers, keeping helpers near the bottom of their module or class.
- Docstring every function and type under `src/`, including private helpers,
  methods, classes, and aliases. Explain functions' behavior, returns, and
  non-obvious assumptions/side effects; explain types' meaning and invariants.
  Put alias docstrings immediately after declarations. One sentence suffices for
  simple definitions. Match the implementation; update docstrings in the same
  change as behavior/contracts.
- Type-hint all Python code; trust annotations. Limit runtime type checks to
  external inputs and union narrowing. Derive redundant information; don't store it.
- Use `Mapping[K, V]` for read-only inputs, `dict[K, V]` for mutation or concrete dictionaries.
- Establish invariants at construction. Use side-effect-free `assert` for internal
  preconditions, postconditions, and invariants whose violation indicates a bug; fail fast.
  Use explicit exceptions for invalid external inputs and expected runtime failures.
- Prefer short Niro module namespaces in implementation/private annotations
  (`ir.Op`, `verify.module`, `rewrite.erase_ops`); qualify enough to avoid ambiguity
  (e.g. `niro.onnx` versus the `onnx` dependency).
- Public Niro type annotations must use direct imports (`Op`) or full paths
  (`niro.ir.ops.Op`), not short aliases (`ir.Op`), for documentation links.
  Dependency types are exempt.
- For standard-library/third-party imports, prefer module-qualified executable
  names (`collections.Counter`) and directly imported annotation types (`Iterator`, `Mapping`).

## Documentation

Be concise; introduce concepts before use. Update affected docs/examples with
API or behavior changes, document new features on relevant pages, and keep shared
content consistent. Preview and build with the commands above.

- Keep `CONTRIBUTING.md` as short as possible while preserving essential guidance.
- `README.md` and `docs/index.md`: synchronized minimal overview, installation,
  basic usage, and links; allow site-specific formatting/links. Only revise existing
  content as needed; no announcements or descriptions of new modules, APIs, or features.
- [docs/ir.md](docs/ir.md): language-agnostic IR concepts, structure, semantics,
  and validity. No implementation details: dataclasses, inheritance, runtime
  validation mechanisms, or accessor APIs.
- Python docstrings: [Google-style][google-docstrings] API explanations/examples
  with object/module cross-references.
- `docs/niro/`: API reference directives and rendering options only; no handwritten
  prose or examples. Use `filters: public` or exclude private/internal names.
  Omit explicit `members` lists so new public members appear automatically.

## Example models

Group generators by format under `scripts/`; emit serialized models to stdout
for saving or piping into Niro:

```sh
uv run scripts/onnx/generate_linear.py | uv run niro inspect signature --input-format onnx
uv run scripts/onnx/generate_linear.py | uv run niro emit mlir --input-format onnx
uv run scripts/onnx/generate_linear.py > linear.onnx
```

[uv]: https://docs.astral.sh/uv/
[google-docstrings]: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
