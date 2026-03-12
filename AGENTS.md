# RatioPath Agent Instructions

## Documentation Standards

- Add documentation for any new public functions, classes, or modules you create.
- Keep documentation examples aligned with the actual import paths exposed by the package.
- For API documentation, ensure docstrings are clear enough for mkdocstrings output.
- Use Google-style docstrings with clear `Args:`, `Returns:`, `Yields:`, `Raises:`, and `Examples:` sections when relevant.

## Testing And Validation

Before finishing, make sure these commands succeed without errors:

- `uvx ruff check`
- `uvx ruff format`
- `uvx mypy .`

Before opening a PR, also run:

- `uvx pytest`

If you changed documentation, docstrings that affect rendered docs, or MkDocs configuration, also run:

- `uvx mkdocs build --strict`

## Conventional Commits

Use the Conventional Commits specification for commit messages and PR titles.

Format: `<type>(<scope>): <subject>`

- **type**: `feat`, `fix`, `chore`, `refactor`, `test`, or `docs`
- **scope**: The affected module or area (e.g., `tiling`, `parsers`, `ray`, `augmentations`)
- **subject**: Clear, imperative mood description (e.g., "add support for multi-level tiling", "fix edge case in grid_tiles")

Examples:
- `feat(tiling): add stride validation for grid_tiles`
- `fix(parsers): handle missing geometry in GeoJSON relations`
- `docs(reference): update tilers API documentation`
- `test(model-selection): add stratified split edge case tests`
- `chore: update ruff configuration`

Apply this convention consistently to both commit messages and PR titles.
