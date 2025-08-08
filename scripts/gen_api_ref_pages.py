# Based on https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages

from pathlib import Path
import mkdocs_gen_files
from mkdocs_gen_files.nav import Nav

project_root = Path(__file__).parent.parent

docs_dname = "docs_mkdocs"
src_dname = "src"

docs_dpath = project_root / docs_dname
src_dpath = project_root / src_dname
root_mod_dpath = project_root / src_dname / "sorts"
output_dpath = docs_dpath / "api_reference"

output_dpath.mkdir(exist_ok=True)

nav = Nav()

# we only gen api docs for some specific modules for now
target_mod_paths: list[Path] = [
    root_mod_dpath / "controller_v2",
    root_mod_dpath / "schedule_v2.py",
    root_mod_dpath / "scheduler_v2",
    root_mod_dpath / "simulation_v2",
]

resultant_fpaths: list[Path] = []
for mod_path in target_mod_paths:
    if mod_path.is_file():
        resultant_fpaths.append(mod_path)
    elif mod_path.is_dir():
        resultant_fpaths.extend(mod_path.rglob("*.py"))
    else:
        print(f'path: "{mod_path}" is either file or directory, skipping')

for fpath in sorted(resultant_fpaths):
    mod_path = fpath.relative_to(src_dpath).with_suffix("")
    doc_path = fpath.relative_to(root_mod_dpath).with_suffix(".md")
    full_doc_path = output_dpath / doc_path

    mod_id_parts = tuple(mod_path.parts)
    if mod_id_parts[-1] == "__init__":
        mod_id_parts = mod_id_parts[:-1]

        # TODO: this seems to generate empty sub-section
        # doc_path = doc_path.with_name("index.md")
        # full_doc_path = full_doc_path.with_name("index.md")

    elif mod_id_parts[-1] == "__main__":
        continue
    elif mod_id_parts[-1].endswith("_test"):
        continue
    else:
        nav[mod_id_parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            identifier = ".".join(mod_id_parts)
            fd.write(f"::: {identifier}")

        mkdocs_gen_files.set_edit_path(full_doc_path, fpath.relative_to(project_root))


with mkdocs_gen_files.open(output_dpath / "nav.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
