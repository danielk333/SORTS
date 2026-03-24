"""
This script uses a combination of `mkdocs` plugins to generate api docs.

- `mkdocstrings`: for auto documentation content generation from a python file/module
- `mkdocs-gen-files`: for generating temporary file during documentation compilation
- `mkdocs-literate-nav`: for specifying navigations in markdown instead of yaml

Essentially, this script:

- gather python files,
- then generate a corresponding temporary markdown file for each,
- which contains `mkdocstrings` autodoc specifier `::: package.subpackage.module`

(Based on https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages)
"""

from pathlib import Path
import mkdocs_gen_files
from mkdocs_gen_files.nav import Nav

project_root_dpath = Path(__file__).parent.parent
src_dpath = project_root_dpath / "src"  # path for source code directory
root_mod_dpath = src_dpath / "sorts"  # path for top level `sorts` module
api_docs_dpath = project_root_dpath / "docs" / "api_reference"  # path for api docs directory


ignore_dpaths: list[Path] = [
    root_mod_dpath / "simulation_v1.py",
]

# we gather the paths for all python files, recursively,
# and exclude certain modules from docs generation
resultant_fpaths: list[Path] = []
for mod_path in root_mod_dpath.rglob("*.py"):
    if not any([mod_path.is_relative_to(ignore_dpath) for ignore_dpath in ignore_dpaths]):
        resultant_fpaths.append(mod_path)

# for each path in the gathered file paths,
# we generate a file that contains `mkdocstrings` autodoc specifier `::: package.subpackage.module`,
# with some skipping and adjustments as needed.
nav = Nav()
for fpath in sorted(resultant_fpaths):
    # `mod_path`: The module path will look like `project/lorem`.
    #     It will be used to build the `mkdocstrings` autodoc identifier.
    # `doc_path`: This is the partial path of the Markdown page for the module.
    # `full_doc_path`: This is the full path of the Markdown page within the docs.
    #     Here we put all reference pages into a reference folder.

    mod_path = fpath.relative_to(src_dpath).with_suffix("")     # e.g. <root_mod>/subpackage/.../module; fmt: skip
    doc_path = fpath.relative_to(src_dpath).with_suffix(".md")  # e.g. <root_mod>/subpackage/.../module.md; fmt: skip
    full_doc_path = Path(api_docs_dpath.parts[-1]) / doc_path     # e.g. <output_dir>/<root_mod>/subpackage/.../module.md; fmt: skip

    mod_path_parts = tuple(mod_path.parts)
    if mod_path_parts[-1] == "__init__":
        # for module files that named as `__init__` (disregarding file extension),
        # it will not have a `__init__` entry in the navigation,
        # instead, the module's heading in the navigation will be used.
        # it is achieved by setting the generated doc file name to "index.md"
        mod_path_parts = mod_path_parts[:-1]

        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    elif mod_path_parts[-1] == "__main__":
        # do not generate api docs for module files that named as `__main__` (disregarding file extension)
        continue

    elif mod_path_parts[-1].endswith("_test"):
        # do not generate api docs for module files with names that endswith "_test" (disregarding file extension)
        continue

    else:
        pass

    # progressively build the navigation object, which will be written to a navigation file later
    nav[mod_path_parts] = doc_path.as_posix()

    # add the file to mkdocs pages. the file is temporary and not actually written to the repos docs folder
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {".".join(mod_path_parts)}")

# write the navigation file
with mkdocs_gen_files.open(Path(api_docs_dpath.parts[-1]) / "nav.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
