"""Build local multiversion documentation from release tags.

The upstream ``sphinx-multiversion`` command loads each tag's historical Sphinx
configuration from an exported tree. Merlin's historical configs expect a live
Git checkout, so this script exports each exact release tag itself, then builds
those sources with the current Sphinx configuration and explicit multiversion
metadata. By default, only the latest patch tag for each major/minor series is
built and exposed under its minor-series name, for example ``0.1``, ``0.2``,
and ``0.3``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

from packaging.version import Version


RELEASE_TAG_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S %z"
SOURCE_SUFFIXES = (".rst", ".ipynb")


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    """Run a command and return its standard output.

        Parameters
        ----------
        command : list[str]
            Command and arguments to execute.
        cwd : pathlib.Path
            Directory where the command is executed.
        env : dict[str, str] | None
            Environment variables for the command. If omitted, the current
            process environment is inherited by subprocess.

        Returns
        -------
        str
            Standard output produced by the command.
    """
    completed_process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed_process.stdout


def list_release_tags(repo_path: Path) -> list[str]:
    """List exact semantic release tags available in the repository.

        Parameters
        ----------
        repo_path : pathlib.Path
            Path to the Merlin repository.

        Returns
        -------
        list[str]
            Tags matching the exact ``x.x.x`` release format, sorted by
            semantic version.
    """
    tag_output = run_command(["git", "tag", "--list"], cwd=repo_path)
    release_tags = [
        tag.strip()
        for tag in tag_output.splitlines()
        if RELEASE_TAG_PATTERN.fullmatch(tag.strip())
    ]
    return sorted(release_tags, key=Version)


def keep_latest_patch_per_minor(tags: list[str]) -> list[str]:
    """Keep only the latest patch release for each major/minor series.

        Parameters
        ----------
        tags : list[str]
            Exact semantic release tags.

        Returns
        -------
        list[str]
            Latest patch tag for each ``major.minor`` series, sorted by
            semantic version.
    """
    latest_tags_by_minor: dict[tuple[int, int], Version] = {}
    for tag in tags:
        version = Version(tag)
        minor_series = (version.major, version.minor)
        if minor_series not in latest_tags_by_minor:
            latest_tags_by_minor[minor_series] = version
            continue
        if version > latest_tags_by_minor[minor_series]:
            latest_tags_by_minor[minor_series] = version

    latest_tags = [str(version) for version in latest_tags_by_minor.values()]
    return sorted(latest_tags, key=Version)


def get_minor_series_name(tag: str) -> str:
    """Return the public version name for an exact release tag.

        Parameters
        ----------
        tag : str
            Exact semantic release tag.

        Returns
        -------
        str
            Major/minor version name, for example ``0.3`` for tag ``0.3.2``.
    """
    version = Version(tag)
    return f"{version.major}.{version.minor}"


def get_tag_commit_date(repo_path: Path, tag: str) -> str:
    """Return the commit date for a tag in Sphinx metadata format.

        Parameters
        ----------
        repo_path : pathlib.Path
            Path to the Merlin repository.
        tag : str
            Exact release tag to inspect.

        Returns
        -------
        str
            Commit date formatted for ``sphinx_multiversion`` metadata.
    """
    timestamp = run_command(
        ["git", "log", "-1", "--format=%cI", tag],
        cwd=repo_path,
    ).strip()
    return datetime.fromisoformat(timestamp).strftime(DATE_FORMAT)


def export_tag(repo_path: Path, tag: str, destination: Path) -> None:
    """Export a release tag into a temporary source tree.

        Parameters
        ----------
        repo_path : pathlib.Path
            Path to the Merlin repository.
        tag : str
            Exact release tag to export.
        destination : pathlib.Path
            Empty destination directory where the archive is extracted.

        Returns
        -------
        None
            The tag content is written to ``destination``.
    """
    destination.mkdir(parents=True)
    archive_process = subprocess.Popen(
        ["git", "archive", "--format=tar", tag],
        cwd=repo_path,
        stdout=subprocess.PIPE,
    )
    if archive_process.stdout is None:
        raise RuntimeError(f"Could not read git archive for {tag}.")

    try:
        with tarfile.open(fileobj=archive_process.stdout, mode="r|") as archive:
            if sys.version_info >= (3, 12):
                archive.extractall(destination, filter="data")
            else:
                archive.extractall(destination)
    finally:
        archive_process.stdout.close()

    return_code = archive_process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ["git", "archive", tag])


def discover_docnames(source_path: Path) -> list[str]:
    """Discover source document names for version-aware links.

        Parameters
        ----------
        source_path : pathlib.Path
            Sphinx source directory for one exported tag.

        Returns
        -------
        list[str]
            Document names without source suffixes, relative to
            ``source_path``.
    """
    docnames: set[str] = set()
    for suffix in SOURCE_SUFFIXES:
        for source_file in source_path.rglob(f"*{suffix}"):
            relative_path = source_file.relative_to(source_path)
            docnames.add(str(relative_path.with_suffix("")))
    return sorted(docnames)


def build_metadata(
    repo_path: Path,
    checkouts: dict[str, tuple[str, Path]],
    output_path: Path,
) -> dict:
    """Build the metadata consumed by ``sphinx_multiversion`` templates.

        Parameters
        ----------
        repo_path : pathlib.Path
            Path to the Merlin repository.
        checkouts : dict[str, tuple[str, pathlib.Path]]
            Mapping from public version name to exact tag and exported source
            tree.
        output_path : pathlib.Path
            Root directory where versioned HTML output is written.

        Returns
        -------
        dict
            Version metadata keyed by public version name.
    """
    metadata = {}
    for version_name, (tag, checkout_path) in checkouts.items():
        source_path = checkout_path / "docs" / "source"
        metadata[version_name] = {
            "name": version_name,
            "version": version_name,
            "release": version_name,
            "rst_prolog": "",
            "is_released": True,
            "source": "tags",
            "source_tag": tag,
            "creatordate": get_tag_commit_date(repo_path, tag),
            "basedir": str(checkout_path),
            "sourcedir": str(source_path),
            "outputdir": str(output_path / version_name),
            "confdir": str(repo_path / "docs" / "source"),
            "docnames": discover_docnames(source_path),
        }
    return metadata


def write_latest_redirect(output_path: Path, latest_version_name: str) -> None:
    """Write a root ``index.html`` redirect to the latest built version.

        Parameters
        ----------
        output_path : pathlib.Path
            Root directory containing all versioned HTML builds.
        latest_version_name : str
            Public version name that should be opened by default.

        Returns
        -------
        None
            The redirect page is written to ``output_path``.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    redirect_path = output_path / "index.html"
    redirect_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<meta charset="utf-8">',
                '<meta http-equiv="refresh" '
                f'content="0; url={latest_version_name}/index.html">',
                f'<link rel="canonical" href="{latest_version_name}/index.html">',
                f'<a href="{latest_version_name}/index.html">'
                f"Open Merlin {latest_version_name} documentation</a>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_version(
    *,
    repo_path: Path,
    checkout_path: Path,
    output_path: Path,
    version_name: str,
    tag: str,
    latest_version_name: str,
    metadata_path: Path,
    sphinx_build: str,
    sphinx_options: list[str],
) -> None:
    """Build one exported tag with the current Sphinx configuration.

        Parameters
        ----------
        repo_path : pathlib.Path
            Path to the Merlin repository that owns the current docs config.
        checkout_path : pathlib.Path
            Exported source tree for the tag being built.
        output_path : pathlib.Path
            Root directory for all versioned HTML output.
        version_name : str
            Public version name used for generated paths and selector labels.
        tag : str
            Exact release tag currently being built.
        latest_version_name : str
            Latest public version name among the selected versions.
        metadata_path : pathlib.Path
            JSON metadata file consumed by ``sphinx_multiversion``.
        sphinx_build : str
            Sphinx executable path or command name.
        sphinx_options : list[str]
            Additional options forwarded to ``sphinx-build``.

        Returns
        -------
        None
            HTML is written under ``output_path / version_name``.
    """
    source_path = checkout_path / "docs" / "source"
    version_output_path = output_path / version_name
    # Keep nbsphinx auxiliary images relative to the exported source tree.
    doctree_path = checkout_path / "docs" / "build" / "doctrees" / version_name
    version_output_path.mkdir(parents=True, exist_ok=True)
    doctree_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MERLIN_DOCS_REPO_PATH"] = str(checkout_path)
    env["MERLIN_DOCS_SOURCE_PATH"] = str(source_path)
    env["MERLIN_DOCS_VERSION"] = version_name
    # Keep generated caches inside the exported tree so local builds do not
    # depend on user-level writable cache directories.
    env.setdefault("PCVL_PERSISTENT_PATH", str(checkout_path / "docs" / ".pcvl_home"))
    env.setdefault("MPLCONFIGDIR", str(checkout_path / "docs" / "tmp_mpl"))
    env.setdefault("XDG_CACHE_HOME", str(checkout_path / "docs" / ".cache"))

    command = [
        sphinx_build,
        "-b",
        "html",
        "-d",
        str(doctree_path),
        "-D",
        f"smv_metadata_path={metadata_path}",
        "-D",
        f"smv_current_version={version_name}",
        "-D",
        f"smv_latest_version={latest_version_name}",
        *sphinx_options,
        "-c",
        str(repo_path / "docs" / "source"),
        str(source_path),
        str(version_output_path),
    ]
    subprocess.run(command, cwd=checkout_path / "docs", env=env, check=True)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the local multiversion builder.

        Returns
        -------
        argparse.Namespace
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build local Merlin documentation for exact x.x.x release tags.",
    )
    parser.add_argument(
        "--output",
        default="build/html",
        help="Output directory for versioned HTML builds.",
    )
    parser.add_argument(
        "--sphinx-build",
        default="sphinx-build",
        help="Sphinx executable to use.",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        help=(
            "Specific exact x.x.x tags to build. Defaults to the latest patch "
            "tag for each major/minor series. Built versions are named by "
            "major/minor unless --all-versions is used."
        ),
    )
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="Build every exact x.x.x release tag instead of one patch per series.",
    )
    parser.add_argument(
        "sphinx_options",
        nargs=argparse.REMAINDER,
        help="Extra options passed to sphinx-build after a '--' separator.",
    )
    arguments = parser.parse_args()
    if arguments.versions is not None and arguments.all_versions:
        parser.error("--versions and --all-versions cannot be used together.")
    return arguments


def main() -> int:
    """Build the selected release-tag documentation set.

        Returns
        -------
        int
            Process exit code. Zero indicates success.
    """
    arguments = parse_arguments()
    docs_path = Path(__file__).resolve().parent
    repo_path = docs_path.parent
    output_path = (docs_path / arguments.output).resolve()
    sphinx_build = arguments.sphinx_build
    # The build runs from exported tag directories. Relative executable paths
    # must therefore be resolved before changing working directory.
    if os.sep in sphinx_build or (os.altsep and os.altsep in sphinx_build):
        sphinx_build = str((Path.cwd() / sphinx_build).resolve())

    release_tags = list_release_tags(repo_path)
    selected_tags = arguments.versions
    if selected_tags is None:
        selected_tags = (
            release_tags
            if arguments.all_versions
            else keep_latest_patch_per_minor(release_tags)
        )
    invalid_tags = sorted(set(selected_tags) - set(release_tags))
    if invalid_tags:
        raise ValueError(
            "Only exact x.x.x release tags can be built. Invalid tags: "
            + ", ".join(invalid_tags)
        )
    if not selected_tags:
        raise RuntimeError("No exact x.x.x release tags found.")

    selected_tags = sorted(selected_tags, key=Version)
    use_minor_version_names = not arguments.all_versions
    selected_versions = {}
    for tag in selected_tags:
        version_name = get_minor_series_name(tag) if use_minor_version_names else tag
        if version_name in selected_versions:
            raise SystemExit(
                "Multiple selected tags resolve to "
                f"{version_name}: {selected_versions[version_name]}, {tag}. "
                "Use --all-versions or select one tag per minor series."
            )
        selected_versions[version_name] = tag
    latest_version_name = (
        get_minor_series_name(selected_tags[-1])
        if use_minor_version_names
        else selected_tags[-1]
    )
    # Rebuild the output directory from scratch so removed versions or pages do
    # not remain as stale local HTML.
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    sphinx_options = arguments.sphinx_options
    # argparse.REMAINDER preserves the separator used before raw Sphinx options.
    if sphinx_options[:1] == ["--"]:
        sphinx_options = sphinx_options[1:]

    with tempfile.TemporaryDirectory(prefix="merlin-docs-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        checkouts = {}
        for version_name, tag in selected_versions.items():
            checkout_path = temporary_path / version_name
            export_tag(repo_path, tag, checkout_path)
            checkouts[version_name] = (tag, checkout_path)

        metadata = build_metadata(repo_path, checkouts, output_path)
        metadata_path = temporary_path / "versions.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        for version_name, (tag, checkout_path) in checkouts.items():
            print(f"Building Merlin documentation for {version_name} from {tag}")
            build_version(
                repo_path=repo_path,
                checkout_path=checkout_path,
                output_path=output_path,
                version_name=version_name,
                tag=tag,
                latest_version_name=latest_version_name,
                metadata_path=metadata_path,
                sphinx_build=sphinx_build,
                sphinx_options=sphinx_options,
            )

    write_latest_redirect(output_path, latest_version_name)
    print(f"Built versions: {', '.join(selected_versions)}")
    print(f"Open {output_path / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
