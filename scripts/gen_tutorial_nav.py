"""
This script uses a combination of `mkdocs` plugins to generate api docs.

- `mkdocs-gen-files`: for generating temporary file during documentation compilation
- `mkdocs-jupyter`: for rendering a `.py` or `.ipynb` file as documentation page
- `mkdocs-literate-nav`: for specifying navigations in markdown instead of yaml

Essentially, this script:

- gather python files,
- write a navigation entry, which `mkdocs` can use for docs generation

(Based on https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages)
"""

from pathlib import Path
import mkdocs_gen_files
from mkdocs_gen_files.nav import Nav

project_root_dpath = Path(__file__).parent.parent
docs_example_dpath = project_root_dpath / "docs" / "tutorial"


resultant_fpaths: list[Path] = [
    *docs_example_dpath.rglob("*.md"),
    *docs_example_dpath.rglob("*.py"),
    *docs_example_dpath.rglob("*.ipynb"),
]

# for each path in the gathered file paths,
# we write an navigation entry, which `mkdocs` can use for docs generation
nav = Nav()
for fpath in sorted(resultant_fpaths):
    r_fpath = fpath.relative_to(docs_example_dpath).with_suffix("")
    nav[*[p.replace("_", " ").capitalize() for p in r_fpath.parts]] = r_fpath.as_posix()

# write the navigation file
with mkdocs_gen_files.open(Path(docs_example_dpath.parts[-1]) / "nav.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
