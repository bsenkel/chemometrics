# Repository Guidelines

## Project Structure & Module Organization

`src/lib.rs` exposes the library API. Public filters live in `src/smooth.rs`, per-spectrum normalization in `src/normalize.rs`, baseline correction in `src/baseline.rs`, principal component analysis behind the optional `pca` feature in `src/pca.rs`, shared errors in `src/error.rs`, and private Householder-QR, coefficient, sample-scaling, input-validation and thin-SVD helpers in `src/numeric.rs`. Integration tests live in `tests/smoothing.rs` for the filters, `tests/derivatives.rs` for Savitzky–Golay derivatives, `tests/normalization.rs` for SNV, `tests/baseline.rs` for detrending and `tests/pca.rs` for principal components; SciPy, NumPy and scikit-learn reference data and its generators live in `tests/fixtures/`. `examples/preprocessing.rs` demonstrates both filters, derivatives, SNV and detrending, and `examples/pca.rs` the PCA workflow. The dependency policy for the optional feature is in `deny.toml`. GitHub Actions configuration is in `.github/workflows/ci.yml`.

## Build, Test, and Development Commands

- `cargo build --locked`: compile the library.
- `cargo run --locked --example preprocessing`: run the preprocessing example.
- `cargo run --locked --features pca --example pca`: run the principal component example.
- `cargo test --locked --all-targets`: run unit and integration tests and compile examples.
- `cargo test --locked --all-features --all-targets`: the same including the optional `pca` feature.
- `cargo test --locked --doc`: test documentation examples, including the README.
- `cargo +1.85 test --locked --all-features --all-targets`: check the minimum supported Rust version when that toolchain is installed.
- `cargo fmt --all -- --check`: verify formatting; use `cargo fmt --all` to apply it.
- `cargo clippy --locked --all-features --all-targets -- -D warnings`: reject lint warnings.
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --all-features`: validate API documentation.
- `cargo deny --all-features check`: audit the optional dependency tree.
- `cargo package --locked`: verify packaging from a clean working tree.

## Rust Coding Guidelines

- Prioritize code correctness and clarity. Speed and efficiency are secondary priorities unless otherwise specified.
- Do not write organizational comments or comments that summarize the code. Comments should only be written in order to explain "why" the code is written in some way in the case there is a reason that is tricky or non-obvious.
- Prefer implementing functionality in existing files unless it is a new logical component. Avoid creating many small files.
- Avoid using functions that panic like `unwrap()` in library code, instead use mechanisms like `?` to propagate errors. Tests may `unwrap()` freely.

## Coding Style & Architecture

Use Rust 2024, maintain Rust 1.85 compatibility, and follow rustfmt's four-space indentation. Use `snake_case` for functions/modules and `PascalCase` for types. Document public APIs and errors in English. Unsafe code is forbidden.

Keep the default library dependency-free and independent of file formats. Preserve slice-based `f64` inputs, reusable immutable filters, and allocation-free `apply_into`. Keep numerical helpers private. Matrix dependencies stay optional and internal, reached only through `src/numeric.rs`, and never appear in the public API; avoid speculative traits and placeholder features.

## Testing Guidelines

Use Rust's built-in test framework with descriptive names such as `polynomial_preservation_including_edges`. Cover mathematical invariants, edge windows, invalid inputs, buffer preservation, reuse, and numerical failures. There is no percentage coverage threshold.

Compare reference values using the existing absolute-plus-relative tolerance. SciPy fixtures use version 1.18.1, with `mode="interp"` for Savitzky–Golay, `zscore(x, ddof=1)` for SNV and NumPy `polyfit` residuals for detrending; regenerate with `uv run tests/fixtures/generate.py`, `uv run tests/fixtures/generate_derivatives.py`, `uv run tests/fixtures/generate_normalization.py`, `uv run tests/fixtures/generate_baseline.py` and `uv run tests/fixtures/generate_pca.py`, which install the pinned dependencies declared inline in each script (PEP 723). The PCA fixture uses NumPy and is cross-checked against scikit-learn inside the generator. After changing a generator, rerun it and confirm the fixture diff is empty unless new reference data is intended. Review fixture changes independently of implementation changes. Normal Rust tests require no Python. CI covers Linux, macOS, Windows, and the MSRV.

## Conventions

- Follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
- Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. The history uses `feat:`, `fix:`, `docs:`, `test:` and `build:` prefixes; only the initial commit predates the convention.
- Follow the [Semantic Versioning](https://semver.org/) specification.
- Keep commits focused.
- Update `CHANGELOG.md` after every meaningful change (new features, bug fixes, breaking changes, deprecations, removals).
- The `CHANGELOG.md` is user-facing only. Refactoring, test infrastructure and documentation wording are deliberately left to the commit history.

## Commit & Pull Request Guidelines

Use a concise imperative subject after the type prefix, for example `test: add baseline correction tests`.

PRs should explain behavior changes, numerical choices, validation performed, and any API compatibility impact. Link relevant issues and update documentation for changed behavior. Publishing to crates.io remains a separate manual action.

## Security

- Never commit credentials, generated build products, or user data.
- Never expose personally identifiable machine or user information.
- Never override the configured Git author or committer identity.
