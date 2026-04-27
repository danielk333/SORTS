# Development

Link to the [repository](https://github.com/danielk333/sorts).

We use [`mkdocs`](https://www.mkdocs.org/) and [`mkdocstrings`](https://mkdocstrings.github.io/) for documentations.

## Development setup

We provide both the `requirements.txt` and `pylock.toml`. \
(The `requirements.txt` is generated on a linux x86_64 platform)

Run these to install the packages:
```bash
# using pip
pip install -r requirements.txt

# using uv
uv pip install -r pylock.toml
```

The lock files are generated using `uv` at the moment:
```bash
uv export --resolution lowest-direct -o requirements.txt
uv export --resolution lowest-direct -o pylock.toml
```

## Tests

- Unit tests are under the `tests/unit` folder
- Integration tests are under the `tests/integration` folder

## Documentation compilation

Use the command `mkdocs serve` in the root folder.

For details please refer to [MkDocs documentation](https://www.mkdocs.org/).

## Style guide

- We generally follow [PEP 8](https://peps.python.org/pep-0008/) and use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings.

- This library is data-centric. We prefer standalone functions (rather than methods) to implement behavior.

- Prefer explicit function parameters instead of passing parameter containers.
  If a grouped representation is needed, use lightweight containers such as `NamedTuple`
  and unpack them at the call site (e.g., `*params`).

- We group functions and classes by functionality into modules.
  Prefer importing modules rather than individual functions or classes
  to maintain consistency and minimize circular import issues.

## Additional notes about the `docs` directory

### Placement of `examples` directory

The `examples` folder are located inside the `docs` directory because
[`mkdocs` does not work well for files that sit outside of the `docs_dir`](https://github.com/mkdocs/mkdocs/discussions/2911).

### The `docs_ignore` directory

It exist because it seems some mkdocs plugin cannot
exclude directories nicely.

e.g.
```yml
- mkdocs-jupyter:
    # globbing only works on single level
    # so `"_ignore/**/*"` actually only exactly exclude `"_ignore/<folder>/<file>"`
    ignore: ["_ignore/**/*"]
```
