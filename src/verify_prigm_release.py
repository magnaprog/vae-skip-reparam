"""CPU-only release gate for an already compiled PriGM main.pdf.

Requires Poppler's pdfinfo, pdftotext and pdffonts, plus the final main.log and
LaTeX -recorder main.fls. This checks release consistency, not the core experiment
or scientific claims. Schemas 2 and 3 also use the combined builder's pure public
validator. No compiler, model loading, training or network operation is invoked.
An optional explicitly pinned presentation manifest binds replacement PDFs while
retaining all historical scientific source and asset checks.
"""
from __future__ import annotations

import argparse
import getpass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
STYLE_SHA256 = "67a960f35da6fecb5a0ce0202fa400459f4c8bb6c294f1a3fdd82fbbd39244c2"
# Import the builder only inside the schema-2/3 verification branch.
OUTPUTS = (
    "generated/results_rows.tex", "generated/all_runs_rows.tex",
    "generated/data_rows.tex", "generated/baseline_rows.tex",
    "generated/summary_values.tex", "figures/reconstruction_generation.png",
    "figures/full_record_context.png", "figures/prior_samples.png",
)
COMBINED_OUTPUTS = tuple(name for name in OUTPUTS if name != "figures/prior_samples.png") + (
    "generated/refinement_rows.tex", "generated/component_rows.tex",
    "generated/dart_diagnostic_rows.tex", "generated/dart_energy_rows.tex", "generated/combined_binding.tex",
    "figures/dart_distribution_diagnostics.png", "figures/dart_support_spectra.png",
)
DART_COMBINED_OUTPUTS = tuple(name for name in COMBINED_OUTPUTS if name not in (
    "generated/dart_diagnostic_rows.tex", "generated/dart_energy_rows.tex",
    "figures/dart_distribution_diagnostics.png", "figures/dart_support_spectra.png")) + (
    "generated/dart_record_rows.tex", "generated/dart_extension_rows.tex",
    "generated/dart_variance_rows.tex", "generated/dart_reference_rows.tex",
    "generated/dart_selected_rows.tex", "generated/dart_block_rows.tex",
    "figures/dart_ecdf_historical.png", "figures/dart_ecdf_fresh.png",
    "figures/dart_spectra_historical.png", "figures/dart_spectra_fresh.png",
    "figures/dart_variance_refit.png",
)
SOURCES = ("main.tex", "references.bib", "main.bbl", "neurips_2025.sty")
EVIDENCE = "paper_evidence_manifest.json"
MANIFEST = "submission_manifest.json"
PRESENTATION = "presentation_manifest.json"
PRESENTATION_REPLACEMENTS = {
    name: "figures/presentation/" + Path(name).stem + ".pdf"
    for name in DART_COMBINED_OUTPUTS if name.endswith(".png")
}
PRESENTATION_INPUTS = tuple("analysis/prigm/figures/" + name for name in (
    "plot_data.npz", "plot_data_manifest.json", "prepare_figure_data.py",
    "render_figures.py", "equivalence.json", "compare_plot_arrays.py",
))
# Installed TeX dependencies are not paper assets. Include the local TinyTeX
# distribution and generated font cache, but reject arbitrary external inputs.
TEX_ROOTS = tuple(Path(p) for p in (
    "/usr/share/texlive", "/usr/share/texmf", "/usr/local/texlive",
    "/usr/local/share/texmf", "/var/lib/texmf", "/etc/texmf", "/usr/share/fonts",
)) + tuple(Path.home() / ".TinyTeX" / name for name in ("texmf-dist", "texmf-var", "texmf-config", "texmf.cnf"))
AUXILIARIES = {"main.aux", "main.out", "main.fls", "main.log"}
EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CORRESPONDING_EMAIL = "kevinlee69720@g.ucla.edu"
CAMERA_READY_AUTHOR = "Kevin Lee"
WORKSHOP_TITLE = "2nd Workshop on Principles of Generative Modeling (PriGM)"
PRIVATE = re.compile(
    r"(?<!\w)/(?:home|Users|tmp|private|root|mnt|media|var|opt|srv|run|usr|etc|dev)(?:/|\b)"
    r"|(?<!\w)[A-Za-z]:[\\/]|\\\\[A-Za-z0-9_.-]+\\|file://"
    r"|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b"
    r"|\b(?:GPU|MIG)-[0-9a-f-]{8,}"
    r"|\b(?:host(?:name)?|user(?:name|[_ -]name)?|(?:gpu|device|machine)[_ -]?uuid|uid)\s*[:=]",
    re.IGNORECASE,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             "Expected a lowercase SHA256 digest")
    return value


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _safe_path(root, name):
    _require(isinstance(name, str) and name and not re.search(r"[\\:\x00-\x1f]", name)
             and all(p not in ("", ".", "..") for p in name.split("/")),
             "Unsafe relative asset path")
    path = root
    for part in name.split("/"):
        path = path / part
        _require(not path.is_symlink(), "Symlink asset path is not allowed")
    return path


def _json(path, *, expected_sha256=None):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    data = path.read_bytes()
    if expected_sha256 is not None:
        _require(sha256(data).hexdigest() == _digest(expected_sha256),
                 "Pinned JSON SHA256 differs: " + path.name)
    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs)


def _private_text(text):
    _require(not PRIVATE.search(text), "Machine identity or local path in public content")
    for identity in (getpass.getuser(), socket.gethostname()):
        # 'root' is also ordinary scientific/provenance vocabulary.
        if len(identity) >= 3 and identity.lower() != "root":
            _require(not re.search(r"(?<![\w])" + re.escape(identity) + r"(?![\w])", text, re.I),
                     "Host username or hostname in public content")


def _public_json(value):
    if isinstance(value, dict):
        for key, child in value.items():
            _require(not re.search(
                r"(?:^|_)(?:host|hostname|username|uid|uuid|runtime|timestamp|created_at|path)(?:$|_)",
                key, re.I), "Private field in public evidence manifest")
            _public_json(key)
            _public_json(child)
    elif isinstance(value, list):
        for child in value:
            _public_json(child)
    elif isinstance(value, str):
        _private_text(value)
        _require(not value.startswith(("/", "~")) and not EMAIL.search(value),
                 "Private value in public evidence manifest")
    elif isinstance(value, float):
        _require(math.isfinite(value), "Nonfinite JSON value")


def _uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def _presentation_manifest(directory, expected_sha256, schema, combined_pin):
    path = _safe_path(directory, PRESENTATION)
    if expected_sha256 is None and not path.exists():
        return None
    _require(expected_sha256 is not None,
             "Presentation requires --expected-presentation-sha256")
    _digest(expected_sha256)
    _require(schema == 3, "Presentation requires DART evidence schema 3")
    _require(path.is_file(), "Missing release file: " + PRESENTATION)
    # Authenticate these exact bytes before decoding or interpreting any paths.
    value = _json(path, expected_sha256=expected_sha256)
    _require(isinstance(value, dict) and type(value.get("schema_version")) is int
             and value["schema_version"] == 1, "Unsupported presentation schema")
    _digest(value.get("scientific_manifest_sha256"))
    _require(value.get("combined_input_sha256") == combined_pin,
             "Presentation combined input SHA256 differs")
    replacements = value.get("replacements")
    _require(isinstance(replacements, dict) and set(replacements) == set(PRESENTATION_REPLACEMENTS),
             "Presentation must replace exactly the seven DART figures")
    for name, target in PRESENTATION_REPLACEMENTS.items():
        item = replacements[name]
        _require(isinstance(item, dict) and set(item) == {"path", "sha256"}
                 and item["path"] == target, "Unexpected presentation replacement: " + name)
        _digest(item["sha256"])
    inputs = value.get("inputs_sha256")
    _require(isinstance(inputs, dict) and set(inputs) == set(PRESENTATION_INPUTS),
             "Presentation input inventory differs")
    for digest in inputs.values():
        _digest(digest)
    # The seven fixed relative replacement paths are the only permitted 'path'
    # fields. Optional style/font metadata still follows public-content checks.
    _public_json({key: item for key, item in value.items() if key != "replacements"})
    return value


def _source_checks(files, outputs=OUTPUTS, combined_pin=None, camera_ready=False):
    tex = _uncomment(files["main.tex"].read_text(encoding="utf-8"))
    compact = re.sub(r"\s+", "", tex)
    _require(r"\documentclass{article}" in compact, "Expected the article document class")
    modes = re.findall(r"\\usepackage\s*\[([^]]*)\]\s*\{neurips_2025\}", tex)
    _require(len(modes) == 1, "Expected one neurips_2025 package option list")
    options = [part.strip() for part in modes[0].split(",")]
    _require(len(options) == len(set(options)), "Duplicate neurips_2025 option")
    if camera_ready:
        # Retain bare [final] for historical accepted packages.
        _require(set(options) in ({"final"}, {"dblblindworkshop", "final"}),
                 "Camera-ready requires [dblblindworkshop,final] or legacy [final]")
        _require(re.search(r"\\workshoptitle\s*\{" + re.escape(WORKSHOP_TITLE) + r"\}", tex),
                 "Expected PriGM workshoptitle")
    else:
        _require(options == ["dblblindworkshop"],
                 "Require dblblindworkshop, not final/preprint")
    _require(not re.search(r"\\PassOptionsToPackage\s*\{[^}]*\}\s*\{neurips_2025\}"
                           r"|\\@(?:anonymousfalse|neuripsfinaltrue|preprinttrue)", tex),
             "Submission mode override is forbidden")
    for macro, value in (("@neuripsyear", "2026"), ("@neuripsordinal", "40th")):
        _require(re.findall(r"\\renewcommand\s*\{\\" + macro + r"\}\s*\{([^}]*)\}", tex) == [value],
                 "Expected intentional 2026/40th metadata overrides")
    _require(not re.search(
        r"\\(?:newgeometry|geometry|restoregeometry|linespread|fontsize|enlargethispage)\b"
        r"|\\(?:textwidth|textheight|oddsidemargin|evensidemargin|topmargin|baselineskip)\s*="
        r"|\\(?:setlength|addtolength|renewcommand)\s*\{?\\(?:textwidth|textheight|oddsidemargin|"
        r"evensidemargin|topmargin|baselineskip|baselinestretch|normalsize|small|section|@maketitle)\b"
        r"|\\usepackage(?:\[[^]]*\])?\{[^}]*(?:geometry|fullpage|titlesec|setspace)[^}]*\}", tex),
        "Main source modifies official formatting")
    subject = r"pdfsubject={PriGMcoreSHA256:\PrigmCoreSHA"
    if combined_pin is not None:
        subject += r";combinedSHA256:\PrigmCombinedSHA"
        binding = files["generated/combined_binding.tex"].read_text(encoding="utf-8")
        _require(re.findall(r"\\newcommand\{\\PrigmCombinedSHA\}\{([0-9a-f]{64})\}", binding)
                 == [_digest(combined_pin)], "Generated PrigmCombinedSHA differs from pinned combined SHA256")
    if camera_ready:
        _require(re.search(r"\\author\s*\{", tex) and CAMERA_READY_AUTHOR in tex
                 and not re.search(r"\\author\s*\{\s*Anonymous Authors\s*\}", tex),
                 "Camera-ready source author must include Kevin Lee")
        _require("pdfauthor={KevinLee}" in compact and subject + "}" in compact,
                 "Camera-ready must bind PDF Subject and set pdfauthor={Kevin Lee}")
    else:
        authors = re.findall(r"\\author\s*\{([^{}]*)\}", tex)
        _require(authors == ["Anonymous Authors"], "Source author must be Anonymous Authors")
        _require("pdfauthor={}" in compact and subject + "}" in compact,
                 "Source must bind PDF Subject to its evidence digests and leave Author blank")
    _require(len(re.findall(r"\\bibliography\s*\{", tex)) == 1 and re.search(
        r"\\clearpage\s*(?:\\bibliographystyle\s*\{(?:plainnat|unsrtnat)\}\s*)?"
        r"\\bibliography\s*\{references\}", tex),
        "Require clearpage immediately before the references bibliography")
    used = set()
    for name in ("main.tex", "main.bbl", *(n for n in outputs if n.endswith(".tex"))):
        text = _uncomment(files[name].read_text(encoding="utf-8"))
        commands = list(re.finditer(r"\\(input|include|includegraphics)\b\*?", text))
        for command in commands:
            argument = re.match(r"\s*(?:\[[^]]*\]\s*)?\{([^{}]+)\}", text[command.end():])
            _require(argument is not None, "Require literal braced local input paths")
            target = argument[1]
            suffix = (Path(target).suffix or ".png") if command[1] == "includegraphics" else ".tex"
            _require(suffix in ((".png", ".pdf") if command[1] == "includegraphics" else (".tex",)),
                     "Unbound local figure or TeX input")
            if not Path(target).suffix:
                target += suffix
            _safe_path(files["main.tex"].parent, target)
            _require(target in outputs and target.endswith(suffix), "Unbound local figure or TeX input")
            used.add(target)
    _require(used == set(outputs), "All bound generated assets must be included")
    summary = files["generated/summary_values.tex"].read_text(encoding="utf-8")
    pins = re.findall(r"\\newcommand\{\\PrigmCoreSHA\}\{([0-9a-f]{64})\}", summary)
    return tex, pins


def _recorder(files, pdf_mtime, outputs=OUTPUTS):
    root = files["main.tex"].parent
    lines = files["main.fls"].read_text(encoding="utf-8").splitlines()
    working = [line[4:] for line in lines if line.startswith("PWD ")]
    _require(len(working) == 1 and Path(working[0]).is_absolute(), "Missing or ambiguous recorder PWD")
    cwd = Path(working[0])
    required = {files[n] for n in (*SOURCES, *outputs) if n != "references.bib"}
    seen, pdf_output = set(), False
    for line in lines:
        kind, _, raw = line.partition(" ")
        if kind not in ("INPUT", "OUTPUT"):
            continue
        path = Path(os.path.abspath(cwd / raw.strip('"')))
        if kind == "OUTPUT":
            pdf_output |= path == files["main.pdf"]
            continue
        if path.is_relative_to(root):
            name = path.relative_to(root).as_posix()
            _safe_path(root, name)
            if name in AUXILIARIES:
                continue  # TeX can rewrite these after the PDF; never freshness inputs.
            _require(name in SOURCES or name in outputs, "Unbound local recorder input")
        else:
            path = path.resolve()
            _require(any(path.is_relative_to(p.resolve()) for p in TEX_ROOTS),
                     "Unbound external recorder input")
        _require(path.is_file(), "Missing recorded dependency")
        _require(path.stat().st_mtime_ns <= pdf_mtime, "Stale PDF: recorded dependency is newer")
        seen.add(path)
    _require(required <= seen and pdf_output, "Recorder missing required inputs or main.pdf output")


def _command(args):
    try:
        result = subprocess.run(args, check=True, capture_output=True, encoding="utf-8",
                                errors="strict", timeout=60, env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise ValueError(f"{args[0]} failed; cannot verify PDF") from error
    _require(not result.stderr.strip(), f"{args[0]} reported warnings/errors")
    return result.stdout


def _pdf_checks(files, pin, tex, combined_pin=None, camera_ready=False):
    info_text = _command(["pdfinfo", str(files["main.pdf"])])
    info = {}
    for line in info_text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            _require(key not in info, "Duplicate PDF metadata field")
            info[key] = value.strip()
    if camera_ready:
        _require(CAMERA_READY_AUTHOR in info.get("Author", ""),
                 "PDF Author metadata must name Kevin Lee")
    else:
        _require(not info.get("Author", ""), "PDF Author metadata must be blank")
    expected_subject = f"PriGM core SHA256: {pin}"
    if combined_pin is not None:
        expected_subject += f"; combined SHA256: {_digest(combined_pin)}"
    _require(info.get("Subject") == expected_subject, "PDF Subject differs from pinned evidence SHA256")
    _require(re.fullmatch(r"[1-9][0-9]*", info.get("Pages", "")), "Invalid PDF page count")
    _private_text(info_text)
    _require(not EMAIL.search(info_text), "Email in PDF metadata")
    count = int(info["Pages"])
    text = _command(["pdftotext", "-layout", "-enc", "UTF-8", str(files["main.pdf"]), "-"])
    _require(not any(unicodedata.category(c) in ("Co", "Cs", "Cc") and c not in "\n\r\t\f"
                     or c in "\ufffd\u25a1" for c in text)
             and not re.search(r"\(cid:\d+\)|\?\?", text), "Broken glyph encoding or unresolved PDF reference")
    _private_text(text)
    pages = text.split("\f")
    if not pages[-1].strip():
        pages.pop()
    _require(len(pages) == count and all(page.strip() for page in pages), "PDF text/page count mismatch")
    # Remove submission line numbers, but not substantive text above References.
    clean = []
    for page in pages:
        lines = [re.sub(r"\s{2,}\d+\s*$", "", line) for line in page.splitlines()]
        clean.append([re.sub(r"^\s*\d+\s+", "", line).strip() for line in lines if line.strip()])
    headings = [(i, j) for i, lines in enumerate(clean) for j, line in enumerate(lines) if line == "References"]
    _require(headings, "Missing References heading")
    main_pages, position = headings[0]
    _require(1 <= main_pages <= 4, "Main paper exceeds four pages or has no main pages")
    _require(all(line.isdigit() for line in clean[main_pages][:position]), "References must start a new page")
    first = "\n".join(clean[0])
    if camera_ready:
        _require(CAMERA_READY_AUTHOR in first and "Abstract" in first
                 and first.index(CAMERA_READY_AUTHOR) < first.index("Abstract"),
                 "Missing Kevin Lee author line before first-page Abstract")
    else:
        _require(re.search(r"(?m)^Anonymous Author(?:\(s\)|s)\s*$", first)
                 and "Abstract" in first
                 and first.index("Anonymous Author") < first.index("Abstract"),
                 "Missing anonymous author line before first-page Abstract")
    # Bibliographic email addresses are allowed only when present in BOTH bound
    # bibliography files and in the References region, never in body/appendices.
    # Camera-ready also allows the declared corresponding-author email.
    bib_emails = set(EMAIL.findall(files["references.bib"].read_text(encoding="utf-8")))
    bib_emails &= set(EMAIL.findall(files["main.bbl"].read_text(encoding="utf-8")))
    allowed_body_emails = {CORRESPONDING_EMAIL} if camera_ready else set()
    before, after = re.split(r"\\bibliography\s*\{references\}", tex, maxsplit=1)
    sections = lambda source: re.findall(r"\\section\*?\s*\{([^{}]+)\}", source)
    normalized_pages = [" ".join(" ".join(lines).split()) for lines in clean]
    flat = "\n".join(normalized_pages)
    ref_start = flat.index("References", sum(len(page) + 1 for page in normalized_pages[:main_pages]))
    appendix = sections(after)
    ref_end = flat.find(" ".join(appendix[0].split()), ref_start) if appendix else len(flat)
    _require(ref_end >= ref_start, "Missing first appendix section")
    for email in EMAIL.finditer(flat):
        if email.group() in allowed_body_emails:
            continue
        _require(ref_start <= email.start() < ref_end and email.group() in bib_emails,
                 "Non-bibliographic email in PDF")
    for source, region in ((before, " ".join(normalized_pages[:main_pages])),
                           (after, " ".join(normalized_pages[main_pages:]))):
        for section in sections(source):
            _require(" ".join(section.split()) in region, "Missing PDF text section: " + section)
    _font_checks(files["main.pdf"])
    return count, main_pages


def _font_checks(path, required_family=None):
    fonts = _command(["pdffonts", str(path)])
    rows = fonts.splitlines()
    separator = next((i for i, row in enumerate(rows) if re.fullmatch(r"[-\s]+", row) and "---" in row), None)
    _require(separator is not None and rows[separator + 1:], "No PDF fonts found")
    names = []
    for row in rows[separator + 1:]:
        fields = re.fullmatch(r".+\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*", row)
        _require(fields is not None and fields[1] == "yes", "Missing font embedding or invalid pdffonts row")
        names.append(re.sub(r"^[A-Z]{6}\+", "", row.split()[0]))
    if required_family is not None:
        _require(any(name == required_family or name.startswith(required_family + "-") for name in names),
                 "Missing embedded " + required_family + " in " + path.name)
    # 'uni=no' can be legitimate for TeX math fonts; extracted glyphs and the
    # compiler's missing-character diagnostics are checked instead.


def verify_release(paper_dir, expected_core_sha256, expected_combined_sha256=None, camera_ready=None,
                   *, expected_presentation_sha256=None):
    """Return only sanitized release facts; verification alone writes nothing.

    camera_ready: True/False forces a mode; None detects the final package option.
    expected_presentation_sha256 authenticates the optional presentation sidecar.
    """
    pin = _digest(expected_core_sha256)
    directory = Path(paper_dir).absolute()
    _require(not any(p.is_symlink() for p in (directory, *directory.parents)), "Symlink paper directory")
    directory = directory.resolve(strict=True)
    evidence_path = _safe_path(directory, EVIDENCE)
    _require(evidence_path.is_file(), "Missing release file: " + EVIDENCE)
    initial_evidence = _json(evidence_path)
    _require(isinstance(initial_evidence, dict), "Expected evidence manifest object")
    schema = initial_evidence.get("schema_version")
    _require(type(schema) is int and schema in (1, 2, 3), "Unsupported paper evidence schema")
    combined = schema in (2, 3)
    combined_pin = None
    if combined:
        _require(initial_evidence.get("purpose") == "combined"
                 and initial_evidence.get("profile") == ("dart_extension" if schema == 3 else "corrected_audit"),
                 "Unknown combined profile or purpose")
        _require(expected_combined_sha256 is not None, "Combined evidence requires --expected-combined-sha256")
        combined_pin = _digest(expected_combined_sha256)
        _require(initial_evidence.get("combined_input_sha256") == combined_pin,
                 "Evidence differs from pinned combined SHA256")
        from build_prigm_v2_paper import (AUDIT_OUTPUTS, DART_EXTENSION_OUTPUTS,
                                          FROZEN_NUMERICAL_SOURCE_SHA256, NUMERICAL_REVIEWED_SOURCES,
                                          validate_combined_manifest)
        expected_outputs = DART_COMBINED_OUTPUTS if schema == 3 else COMBINED_OUTPUTS
        builder_outputs = DART_EXTENSION_OUTPUTS if schema == 3 else AUDIT_OUTPUTS
        _require(set(builder_outputs) == set(expected_outputs), "Combined output contract differs")
        validate_combined_manifest(initial_evidence, pin)
    else:
        _require(initial_evidence.get("purpose") == "core", "Expected core evidence schema version 1")
        _require(expected_combined_sha256 is None, "Core-only evidence cannot use a combined SHA256 pin")
        expected_outputs = (*OUTPUTS, "generated/refinement_rows.tex") if "refinement" in initial_evidence else OUTPUTS
    presentation = _presentation_manifest(directory, expected_presentation_sha256, schema, combined_pin)
    compile_outputs = expected_outputs
    if presentation is not None:
        compile_outputs = tuple(PRESENTATION_REPLACEMENTS.get(name, name) for name in expected_outputs)
    names = (*SOURCES, *expected_outputs, *compile_outputs, EVIDENCE, "main.pdf", "main.log", "main.fls")
    files = {name: _safe_path(directory, name) for name in names}
    if presentation is not None:
        files[PRESENTATION] = _safe_path(directory, PRESENTATION)
        files.update({"presentation_input:" + name: _safe_path(ROOT, name) for name in PRESENTATION_INPUTS})
    files["builder"] = _safe_path(ROOT, "src/build_prigm_v2_paper.py" if combined else "src/build_prigm_paper.py")
    if combined:
        files.update({"derivation:" + name: _safe_path(ROOT, name)
                      for name in initial_evidence["input_index"]["source_sha256"]})
        if schema == 2:
            files.update({"numerical_source:" + name: _safe_path(ROOT, reviewed)
                          for name, reviewed in NUMERICAL_REVIEWED_SOURCES.items()})
    for name, path in files.items():
        _require(path.is_file(), "Missing release file: " + name)
    snapshots = {name: (_sha(path), path.stat().st_mtime_ns) for name, path in files.items()}
    evidence = _json(files[EVIDENCE])
    _require(isinstance(evidence, dict), "Expected evidence manifest object")
    _public_json(evidence)
    _require(evidence == initial_evidence, "Evidence changed during verification")
    if presentation is not None:
        _require(snapshots[PRESENTATION][0] == expected_presentation_sha256,
                 "Presentation manifest changed during verification")
        _require(presentation["scientific_manifest_sha256"] == snapshots[EVIDENCE][0],
                 "Presentation scientific manifest SHA256 differs")
        for item in presentation["replacements"].values():
            _require(item["sha256"] == snapshots[item["path"]][0],
                     "Presentation figure SHA256 differs: " + item["path"])
        for name, digest in presentation["inputs_sha256"].items():
            _require(digest == snapshots["presentation_input:" + name][0],
                     "Presentation input SHA256 differs: " + name)
    core_evidence = evidence["core"] if combined else evidence
    _require(core_evidence.get("run_manifest_sha256") == pin, "Evidence differs from pinned core SHA256")
    _require(evidence.get("builder_sha256") == snapshots["builder"][0], "Current builder SHA256 differs")
    if combined:
        for name, digest in evidence["input_index"]["source_sha256"].items():
            _require(digest == snapshots["derivation:" + name][0], "Current derivation source SHA256 differs: " + name)
        numerical_sources = evidence["input_index"]["numerical_validation"]["source_sha256"]
        if schema == 3:
            _require(numerical_sources == FROZEN_NUMERICAL_SOURCE_SHA256,
                     "Preserved numerical producer SHA256 differs")
        else:
            for name, digest in numerical_sources.items():
                _require(digest == snapshots["numerical_source:" + name][0],
                         "Reviewed numerical source SHA256 differs: " + name)
    if "refinement" in evidence:
        refinement = evidence["refinement"]
        _require(isinstance(refinement, dict), "Invalid refinement evidence")
        _digest(refinement.get("manifest_sha256"))
        _digest(refinement.get("archive_manifest_sha256"))
        _require(isinstance(refinement.get("results"), dict), "Missing refinement results")
    outputs = evidence.get("output_sha256")
    _require(isinstance(outputs, dict), "Missing output_sha256 map")
    for name in outputs:
        _safe_path(directory, name)
    _require(set(outputs) == set(expected_outputs), "Evidence must bind exactly the expected generated assets")
    for name, digest in outputs.items():
        _require(_digest(digest) == snapshots[name][0], "Generated asset SHA256 differs: " + name)
    _require(snapshots["neurips_2025.sty"][0] == STYLE_SHA256, "Official neurips_2025.sty SHA256 differs")
    raw_tex = files["main.tex"].read_text(encoding="utf-8")
    detected_final = any(
        "final" in (part.strip() for part in mode.split(","))
        for mode in re.findall(r"\\usepackage\s*\[([^]]*)\]\s*\{neurips_2025\}",
                               _uncomment(raw_tex)))
    if camera_ready is None:
        camera_ready = detected_final
    elif camera_ready:
        _require(detected_final, "Camera-ready flag requires the final option in main.tex")
    tex, pins = _source_checks(files, compile_outputs, combined_pin, camera_ready=camera_ready)
    _require(pins == [pin], "Generated PrigmCoreSHA differs from pinned core SHA256")
    pdf_mtime = snapshots["main.pdf"][1]
    freshness_inputs = (*SOURCES, *compile_outputs)
    if presentation is None:
        freshness_inputs += (EVIDENCE,)
    for name in freshness_inputs:
        _require(snapshots[name][1] <= pdf_mtime, "Stale PDF: dependency is newer: " + name)
    _recorder(files, pdf_mtime, compile_outputs)
    log = files["main.log"].read_text(encoding="utf-8", errors="replace")
    _require(not re.search(
        r"^!|\b(?:LaTeX|Package [\w.-]+|Class [\w.-]+) Error:|undefined|Emergency stop|Fatal error"
        r"|Overfull\s+\\[hv]box|Missing character:|Label\(s\) may have changed|Rerun to get"
        r"|Please\s+(?:\(re\))?run\s+(?:Biber|BibTeX)|:[0-9]+:.*(?:error|Runaway argument)"
        r"|Runaway argument", log, re.I | re.M), "LaTeX log has errors, unresolved references, glyphs or overfull boxes")
    pages, main_pages = _pdf_checks(files, pin, tex, combined_pin, camera_ready=camera_ready)
    if presentation is not None:
        for name in PRESENTATION_REPLACEMENTS.values():
            _font_checks(files[name], required_family="DejaVuSerif")
    completion = re.findall(r"Output written on [^\n]*main\.pdf\"?\s*\((\d+) pages?[,)]", log)
    _require(completion == [str(pages)], "LaTeX log lacks matching successful PDF completion")
    for name, path in files.items():
        _require((_sha(path), path.stat().st_mtime_ns) == snapshots[name], "Release input changed during verification")
    if camera_ready:
        verification = {"mode": "final", "year": 2026, "ordinal": "40th",
                        "official_style_unchanged": True, "author_metadata_blank": False,
                        "first_page_anonymous": False, "first_page_author": True,
                        "subject_core_bound": True,
                        "references_start_new_page": True, "public_identity_checks": True,
                        "text_sections_present": True, "glyph_checks": True, "fonts_embedded": True,
                        "latex_log_clean": True, "recorder_inputs_bound": True,
                        "dependencies_not_newer_than_pdf": True}
    else:
        verification = {"mode": "dblblindworkshop", "year": 2026, "ordinal": "40th",
                        "official_style_unchanged": True, "author_metadata_blank": True,
                        "first_page_anonymous": True, "subject_core_bound": True,
                        "references_start_new_page": True, "public_identity_checks": True,
                        "text_sections_present": True, "glyph_checks": True, "fonts_embedded": True,
                        "latex_log_clean": True, "recorder_inputs_bound": True,
                        "dependencies_not_newer_than_pdf": True}
    result = {
        "schema_version": schema, "run_manifest_sha256": pin,
        "pdf": {"file": "main.pdf", "sha256": snapshots["main.pdf"][0],
                "pages": pages, "main_pages": main_pages},
        "source_sha256": {name: snapshots[name][0] for name in SOURCES},
        "evidence_sha256": snapshots[EVIDENCE][0], "builder_sha256": snapshots["builder"][0],
        "generated_sha256": dict(outputs),
        "verification": verification,
    }
    if combined:
        result.update(profile=evidence["profile"], combined_input_sha256=combined_pin,
                      derivation_source_sha256=dict(evidence["input_index"]["source_sha256"]),
                      numerical_source_sha256=dict(evidence["input_index"]["numerical_validation"]["source_sha256"]))
        result["verification"]["subject_combined_bound"] = True
    if presentation is not None:
        result["presentation"] = {
            "manifest_sha256": snapshots[PRESENTATION][0],
            "output_sha256": {item["path"]: item["sha256"] for item in presentation["replacements"].values()},
            "inputs_sha256": dict(presentation["inputs_sha256"]),
        }
        result["verification"]["presentation_bound"] = True
        result["verification"]["presentation_fonts_embedded"] = True
    return result


def _write_manifest(directory, manifest):
    target = _safe_path(directory, MANIFEST)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", prefix=".submission-manifest-",
                                         dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--expected-combined-sha256")
    parser.add_argument("--expected-presentation-sha256",
                        help="Authenticate presentation_manifest.json when using the seven replacement PDFs")
    parser.add_argument("--camera-ready", action="store_true",
                        help="Require camera-ready packaging with Kevin Lee authorship "
                             "(also auto-detected from the final package option in main.tex)")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = verify_release(args.paper_dir, args.expected_core_sha256, args.expected_combined_sha256,
                                  camera_ready=True if args.camera_ready else None,
                                  expected_presentation_sha256=args.expected_presentation_sha256)
        if args.write_manifest:
            _write_manifest(args.paper_dir.resolve(), manifest)
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(1, f"Release verification failed: {error}\n")
    print(json.dumps({"verified": True, **manifest["pdf"], "manifest_written": args.write_manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
