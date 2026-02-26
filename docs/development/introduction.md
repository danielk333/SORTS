# Development

Link to the [repository](https://github.com/danielk333/sorts).

We use [`mkdocs`](https://www.mkdocs.org/) and [`mkdocstrings`](https://mkdocstrings.github.io/) for documentations.

## Development setup

use this to install the packages
```bash
uv sync --all-extras

# the sorts bundling info does not work well with `uv` at the moment
# so a separate install using `pip` sorts itself is needed.
uv pip install -e .
```

## Documentation compilation

Use this command in the root folder. For details please refer to [MkDocs documentation](https://www.mkdocs.org/).
` mkdocs serve`

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
