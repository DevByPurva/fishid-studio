"""Guards the repository's citation metadata.

GitHub's "Cite this repository" button renders ``CITATION.cff``, and the DOIs in
``README.md`` / ``README.en.md`` point at the same Zenodo record. Those three
files are edited independently, so a release that renames an author, bumps the
version, or resolves a new DOI can silently leave one of them stale. These
tests fail loudly when that happens.

No YAML parser is used on purpose: the CI pytest step installs only the packages
listed in ``.github/workflows/ci.yml`` (pytest, httpx, fastapi, numpy, pandas,
opencv, pillow, tqdm) — PyYAML is not among them. The checks are exact enough to
catch drift without pulling in a dependency.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CITATION_FILE = REPO_ROOT / "CITATION.cff"
READMES = ("README.md", "README.en.md")

# The Zenodo DOI cited in CITATION.cff and both READMEs.
ZENODO_DOI = "10.5281/zenodo.24451918"


def _read(path):
    assert path.is_file(), f"{path.name} is missing at the repository root"
    return path.read_text(encoding="utf-8")


def test_citation_file_declares_required_fields():
    text = _read(CITATION_FILE)
    for field in ("cff-version", "title", "doi", "license"):
        assert re.search(rf"^{field}\s*:", text, re.MULTILINE), (
            f"CITATION.cff is missing the required '{field}' field"
        )


def test_citation_declares_expected_doi_and_license():
    text = _read(CITATION_FILE)
    assert re.search(rf"^doi\s*:\s*['\"]?{re.escape(ZENODO_DOI)}['\"]?\s*$", text, re.MULTILINE), (
        f"CITATION.cff must cite DOI {ZENODO_DOI}"
    )
    assert re.search(r"^license\s*:\s*MIT\s*$", text, re.MULTILINE), (
        "CITATION.cff must declare license: MIT"
    )


def test_citation_lists_both_authors():
    text = _read(CITATION_FILE)
    assert "da Silva de Souza" in text, "CITATION.cff is missing author José Vinícius da Silva de Souza"
    assert "Sant'Ana" in text, "CITATION.cff is missing author Rodrigo Sant'Ana"
    assert "rsantana@univali.br" in text, "CITATION.cff is missing co-author institutional email"


def test_citation_version_is_semver():
    text = _read(CITATION_FILE)
    match = re.search(r"^version\s*:\s*['\"]?([^'\"\s]+)['\"]?\s*$", text, re.MULTILINE)
    assert match, "CITATION.cff is missing a version field"
    assert re.fullmatch(r"\d+\.\d+\.\d+", match.group(1)), (
        f"CITATION.cff version {match.group(1)!r} is not a semantic version"
    )


@pytest.mark.parametrize("readme", READMES)
def test_readmes_cite_the_same_doi(readme):
    text = _read(REPO_ROOT / readme)
    assert ZENODO_DOI in text, f"{readme} does not contain the Zenodo DOI {ZENODO_DOI}"


def test_all_citation_sources_agree_on_the_doi():
    """CITATION.cff and both READMEs must not drift apart."""
    citation_doi = re.search(r"^doi\s*:\s*['\"]?([^'\"\s]+)['\"]?\s*$", _read(CITATION_FILE), re.MULTILINE)
    assert citation_doi, "CITATION.cff has no parseable doi field"

    for readme in READMES:
        assert citation_doi.group(1) in _read(REPO_ROOT / readme), (
            f"{readme} cites a different DOI than CITATION.cff ({citation_doi.group(1)})"
        )
