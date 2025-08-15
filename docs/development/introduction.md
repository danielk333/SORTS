# Development

Repository ([link](https://github.com/danielk333/sorts))

we use `mkdocs` ([link](https://www.mkdocs.org/)) and `mkdocstrings` ([link](https://mkdocstrings.github.io/)) for documentations.

## Development setup

use this to install the packages
```bash
uv sync --all-extras`

# the sorts bundling info does not work well with `uv` at the moment
# so a separate install using `pip` sorts itself is needed.
pip install -e .
```
