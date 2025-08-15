# Based on https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages
#
# we walk through the example dir then generate a `nav.md` accordingly.
# `mkdocs-jupyter` plugin will generate the pages based on files which the `nav.md` points to.

from pathlib import Path
import mkdocs_gen_files
from mkdocs_gen_files.nav import Nav

project_root_dpath = Path(__file__).parent.parent
docs_example_dpath = project_root_dpath / "docs" / "examples"

nav = Nav()

resultant_fpaths: list[Path] = [
    *docs_example_dpath.rglob("*.py"),
    *docs_example_dpath.rglob("*.ipynb"),
]

for fpath in sorted(resultant_fpaths):
    r_fpath = fpath.relative_to(docs_example_dpath).with_suffix("")
    nav[*[p.replace("_", " ").capitalize() for p in r_fpath.parts]] = r_fpath.as_posix()


with mkdocs_gen_files.open(Path(docs_example_dpath.parts[-1]) / "nav.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
