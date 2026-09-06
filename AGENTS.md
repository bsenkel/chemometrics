# Repository Guidelines

## Project Structure & Module Organization

`src/lib.rs` exposes the library API. Public filters live in `src/smooth.rs`, shared errors in `src/error.rs`, and private Householder-QR and coefficient calculations in `src/polynomial.rs`. Integration tests live in `tests/smoothing.rs`; SciPy reference data and its generator live in `tests/fixtures/`. `examples/smoothing.rs` demonstrates both filters. GitHub Actions configuration is in `.github/workflows/ci.yml`.

## Build, Test, and Development Commands

- `cargo build --locked`: compile the library.
- `cargo run --locked --example smoothing`: run the filter example.
- `cargo test --locked --all-targets`: run unit and integration tests and compile examples.
- `cargo test --locked --doc`: test documentation examples, including the README.
- `cargo +1.85 test --locked --all-targets`: check the minimum supported Rust version when that toolchain is installed.
- `cargo fmt --all -- --check`: verify formatting; use `cargo fmt --all` to apply it.
- `cargo clippy --locked --all-targets -- -D warnings`: reject lint warnings.
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps`: validate API documentation.
- `cargo package --locked`: verify packaging from a clean working tree.

## Coding Style & Architecture

Use Rust 2024, maintain Rust 1.85 compatibility, and follow rustfmt's four-space indentation. Use `snake_case` for functions/modules and `PascalCase` for types. Document public APIs and errors in English. Unsafe code is forbidden.

Keep the default library dependency-free and independent of file formats. Preserve slice-based `f64` inputs, reusable immutable filters, and allocation-free `apply_into`. Keep numerical helpers private. Future matrix dependencies should be optional; avoid speculative traits and placeholder features.

## Testing Guidelines

Use Rust's built-in test framework with descriptive names such as `polynomial_preservation_including_edges`. Cover mathematical invariants, edge windows, invalid inputs, buffer preservation, reuse, and numerical failures. There is no percentage coverage threshold.

Compare reference values using the existing absolute-plus-relative tolerance. SciPy fixtures use version 1.17.1 and `mode="interp"`; regenerate with `python tests/fixtures/generate.py` in a compatible environment. Review fixture changes independently of implementation changes. Normal Rust tests require no Python. CI covers Linux, macOS, Windows, and the MSRV.

## Commit & Pull Request Guidelines

No commits exist yet, so no historical convention is established. Use concise imperative subjects, for example `Add baseline correction tests`. Keep changes focused.

PRs should explain behavior changes, numerical choices, validation performed, and any API compatibility impact. Link relevant issues and update documentation for changed behavior. Publishing to crates.io remains a separate manual action.
