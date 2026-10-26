"""Release gate tests use synthetic files and mocked Poppler output, not a real PDF."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import verify_prigm_release as release

PIN = "a" * 64
OLD, PDF_TIME, NEW = 1_000_000_000, 2_000_000_000, 3_000_000_000
FINAL_OPTIONS = ["final", "dblblindworkshop,final", "final,dblblindworkshop",
                 " dblblindworkshop ,\n final ", "\n final,\t dblblindworkshop "]


def put(path, text, mtime=OLD):
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(mtime, mtime))


def edit(path, old, new, mtime=OLD):
    text = path.read_text(encoding="utf-8")
    assert old in text
    put(path, text.replace(old, new), mtime)


def seal(paper):
    manifest = json.loads((paper / release.EVIDENCE).read_text())
    manifest["output_sha256"] = {name: release._sha(paper / name) for name in release.OUTPUTS}
    put(paper / release.EVIDENCE, json.dumps(manifest))


@pytest.fixture
def paper(tmp_path, monkeypatch):
    """Real file hashes/mtimes/recorder, fake PDF and command output only."""
    directory = tmp_path / "paper with spaces"
    directory.mkdir()
    (directory / "generated").mkdir()
    (directory / "figures").mkdir()
    (tmp_path / "src").mkdir()
    put(tmp_path / "src/build_prigm_paper.py", "# synthetic CPU builder identity\n")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release.getpass, "getuser", lambda: "private-user")
    monkeypatch.setattr(release.socket, "gethostname", lambda: "private-host")
    tex = r"""\documentclass{article}
\usepackage[dblblindworkshop]{neurips_2025}
\makeatletter
\renewcommand{\@neuripsyear}{2026}
\renewcommand{\@neuripsordinal}{40th}
\makeatother
\input{generated/summary_values.tex}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdfauthor={},pdfsubject={PriGM core SHA256: \PrigmCoreSHA}}
\author{Anonymous Authors}
\begin{document}
\maketitle
\begin{abstract}Fixture abstract.\end{abstract}
\section{Introduction}
\section{Results}
\section{Conclusion}
\input{generated/results_rows.tex}
\includegraphics[width=\linewidth]{figures/reconstruction_generation.png}
\clearpage
\bibliographystyle{plainnat}
\bibliography{references}
\clearpage
\appendix
\section{Density and scoring details}
\input{generated/data_rows.tex}
\input{generated/baseline_rows.tex}
\input{generated/all_runs_rows.tex}
\includegraphics{figures/full_record_context.png}
\includegraphics{figures/prior_samples.png}
\end{document}
"""
    put(directory / "main.tex", tex)
    put(directory / "references.bib", '@article{Lee2024, author={Lee, Kevin}, title={Public predecessor}}')
    put(directory / "main.bbl", r"\begin{thebibliography}{1} Kevin Lee. Public predecessor. \end{thebibliography}")
    style = ROOT / "paper/prigm/neurips_2025.sty"
    (directory / "neurips_2025.sty").write_bytes(style.read_bytes())
    os.utime(directory / "neurips_2025.sty", ns=(OLD, OLD))
    for name in release.OUTPUTS:
        put(directory / name, "synthetic generated asset\n")
    put(directory / "generated/summary_values.tex", r"\newcommand{\PrigmCoreSHA}{" + PIN + "}\n")
    put(directory / release.EVIDENCE, json.dumps({
        "schema_version": 1, "purpose": "core", "run_manifest_sha256": PIN,
        "builder_sha256": release._sha(tmp_path / "src/build_prigm_paper.py"),
        "output_sha256": {},
    }))
    seal(directory)
    put(directory / "main.pdf", "%PDF-fixture, not a compiled result", PDF_TIME)
    # TeX writes these late. They must not be treated as stale PDF dependencies.
    put(directory / "main.log", "This is pdfTeX\nOutput written on main.pdf (6 pages, 12345 bytes).\n", NEW)
    put(directory / "main.aux", "auxiliary cycle", NEW)
    put(directory / "main.out", "hyperref cycle", NEW)
    required = [n for n in (*release.SOURCES, *release.OUTPUTS) if n != "references.bib"]
    put(directory / "main.fls", "PWD " + str(directory) + "\n" +
        "".join("INPUT ./" + name + "\n" for name in (*required, "main.aux", "main.out")) +
        "OUTPUT main.pdf\n", NEW)
    outputs = {
        "pdfinfo": f"Title: Fixture\nAuthor:          \nSubject: PriGM core SHA256: {PIN}\nPages: 6\n",
        "pdftotext": "\f".join([
            "Fixture title\nAnonymous Author(s)\nAffiliation\nAddress\nemail\nAbstract\nFixture abstract.\n1 Introduction\n",
            "2 Results\nText.\n", "More results.\n", "3 Conclusion\nText.\n",
            "References\nKevin Lee. Public predecessor.\n5\n",
            "A Density and scoring details\nAdditional results.\n6\n",
        ]) + "\f",
        "pdffonts": "name                                 type              encoding         emb sub uni object ID\n"
                    "------------------------------------ ----------------- ---------------- --- --- --- ---------\n"
                    "ABCDEF+NimbusRoman-Regular            Type 1            Custom           yes yes yes     12  0\n"
                    "GHIJKL+CMMI10                        Type 1            Builtin          yes yes no      15  0\n",
    }
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        assert isinstance(args, list) and not kwargs.get("shell")
        assert kwargs["check"] and kwargs["timeout"] == 60
        assert kwargs["env"]["LC_ALL"] == "C"
        assert str(directory / "main.pdf") in args
        return subprocess.CompletedProcess(args, 0, outputs[args[0]], "")

    monkeypatch.setattr(release.subprocess, "run", run)
    return directory, outputs, calls


def verify(paper):
    return release.verify_release(paper[0], PIN)


def change_evidence(paper, update):
    path = paper[0] / release.EVIDENCE
    manifest = json.loads(path.read_text())
    update(manifest)
    put(path, json.dumps(manifest))


@pytest.mark.parametrize("url", ["https://arxiv.org/abs/2412.16375",
                                "https://example.org/dev/shm/results",
                                "https://example.org/private/archive"])
def test_https_citation_is_not_a_local_path(url):
    release._private_text(url)
    with pytest.raises(ValueError, match="Machine identity or local path"):
        release._private_text("C:/Users/private-user/results")


def test_verify_only_checks_exact_artifacts_without_writes(paper):
    directory, _, calls = paper
    before = {p.relative_to(directory): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in directory.rglob("*") if p.is_file()}
    result = verify(paper)
    after = {p.relative_to(directory): (p.read_bytes(), p.stat().st_mtime_ns)
             for p in directory.rglob("*") if p.is_file()}
    assert before == after
    assert result["pdf"] == {"file": "main.pdf", "sha256": release._sha(directory / "main.pdf"),
                             "pages": 6, "main_pages": 4}
    assert result["evidence_sha256"] == release._sha(directory / release.EVIDENCE)
    assert set(result["source_sha256"]) == set(release.SOURCES)
    assert set(result["generated_sha256"]) == set(release.OUTPUTS)
    assert [args[0] for args, _ in calls] == ["pdfinfo", "pdftotext", "pdffonts"]
    text = json.dumps(result)
    assert all(term not in text for term in (str(directory), "/home/", "private-user", "private-host", "timestamp"))
    assert not (directory / release.MANIFEST).exists()


def test_refinement_table_requires_its_hash_and_recorder_input(paper):
    directory, _, _ = paper
    name = "generated/refinement_rows.tex"
    put(directory / name, "synthetic refinement rows\n")
    edit(directory / "main.tex", r"\end{document}", "\\input{" + name + "}\n\\end{document}")
    put(directory / "main.fls", (directory / "main.fls").read_text() + "INPUT ./" + name + "\n", NEW)
    def add_refinement(value):
        value["refinement"] = {"manifest_sha256": "b" * 64,
                               "archive_manifest_sha256": "c" * 64, "results": {}}
        value["output_sha256"][name] = release._sha(directory / name)
    change_evidence(paper, add_refinement)
    result = verify(paper)
    assert name in result["generated_sha256"]
    put(directory / name, "changed refinement values\n")
    with pytest.raises(ValueError, match="Generated asset SHA256 differs"):
        verify(paper)


def test_cli_requires_pin_and_writes_manifest_only_on_request(paper, capsys):
    directory, _, _ = paper
    arguments = ["--paper-dir", str(directory)]
    with pytest.raises(SystemExit) as error:
        release.main(arguments)
    assert error.value.code == 2
    arguments += ["--expected-core-sha256", PIN]
    assert release.main(arguments) == 0
    assert not (directory / release.MANIFEST).exists()
    assert release.main([*arguments, "--write-manifest"]) == 0
    saved = json.loads((directory / release.MANIFEST).read_text())
    assert saved == verify(paper)
    assert not list(directory.glob(".submission-manifest-*"))
    assert '"manifest_written": true' in capsys.readouterr().out


@pytest.mark.parametrize("name", release.OUTPUTS)
def test_generated_asset_tampering_rejected(paper, name):
    put(paper[0] / name, "tampered")
    with pytest.raises(ValueError, match="Generated asset SHA256"):
        verify(paper)


@pytest.mark.parametrize("field", ["run_manifest_sha256", "builder_sha256"])
def test_pinned_core_and_current_builder_hash_required(paper, field):
    change_evidence(paper, lambda value: value.update({field: "0" * 64}))
    with pytest.raises(ValueError, match="pinned core|builder SHA256"):
        verify(paper)


def test_changed_current_builder_rejected(paper):
    put(release.ROOT / "src/build_prigm_paper.py", "changed builder")
    with pytest.raises(ValueError, match="builder SHA256"):
        verify(paper)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_exact_eight_outputs_required(paper, mutation):
    def update(value):
        if mutation == "missing":
            value["output_sha256"].pop(release.OUTPUTS[0])
        else:
            value["output_sha256"]["figures/unbound.png"] = "0" * 64
    change_evidence(paper, update)
    with pytest.raises(ValueError, match="exactly the expected"):
        verify(paper)


@pytest.mark.parametrize("name", ["../outside", "/outside", "figures/../outside", "./file", "figures//file",
                                   "figures\\file", "C:/outside", "figures/evil\x00.png"])
def test_unsafe_asset_paths_rejected(paper, name):
    change_evidence(paper, lambda value: value["output_sha256"].update({name: "0" * 64}))
    with pytest.raises(ValueError, match="Unsafe relative|Private value|Machine identity"):
        verify(paper)


@pytest.mark.parametrize("parent", [False, True])
def test_symlink_assets_and_parents_rejected(paper, parent):
    directory = paper[0]
    name = "figures" if parent else release.OUTPUTS[0]
    path = directory / name
    moved = directory / "moved"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=parent)
    with pytest.raises(ValueError, match="Symlink"):
        verify(paper)


@pytest.mark.parametrize("name", (*release.SOURCES, *release.OUTPUTS, release.EVIDENCE))
def test_every_bound_dependency_must_predate_pdf(paper, name):
    os.utime(paper[0] / name, ns=(NEW, NEW))
    with pytest.raises(ValueError, match="Stale PDF"):
        verify(paper)


def test_style_hash_is_official_and_not_patchable(paper):
    assert release._sha(ROOT / "paper/prigm/neurips_2025.sty") == release.STYLE_SHA256
    put(paper[0] / "neurips_2025.sty", "% patched style")
    with pytest.raises(ValueError, match="Official neurips"):
        verify(paper)


@pytest.mark.parametrize("old,new,message", [
    # Final options auto-enter camera-ready; this fixture lacks workshoptitle.
    ("dblblindworkshop", "final", "workshoptitle"),
    ("dblblindworkshop", "preprint", "dblblindworkshop"),
    ("dblblindworkshop", "dblblindworkshop,final", "workshoptitle"),
    (r"\clearpage\n", "", "clearpage"),
    ("Anonymous Authors", "Kevin Lee", "Source author"),
    ("pdfsubject={PriGM core SHA256: \\PrigmCoreSHA}", "pdfsubject={}", "PDF Subject"),
    (r"\begin{document}", r"\geometry{margin=0.5in}\begin{document}", "formatting"),
    (r"\begin{document}", r"\PassOptionsToPackage{final}{neurips_2025}\begin{document}", "mode override"),
])
def test_source_submission_contract(paper, old, new, message):
    edit(paper[0] / "main.tex", old.replace(r"\n", "\n"), new)
    with pytest.raises(ValueError, match=message):
        verify(paper)


@pytest.mark.parametrize("options", FINAL_OPTIONS)
@pytest.mark.parametrize("camera_ready", [True, None])
def test_named_author_passes_camera_ready_and_fails_anonymous_mode(paper, options, camera_ready):
    directory, outputs, _ = paper
    edit(directory / "main.tex", r"\usepackage[dblblindworkshop]{neurips_2025}",
         "\\usepackage[" + options + "]{neurips_2025}\n"
         "\\workshoptitle{2nd Workshop on Principles of Generative Modeling (PriGM)}")
    edit(directory / "main.tex",
         r"\author{Anonymous Authors}",
         "\\author{%\n Kevin Lee\\orcidlink{0009-0004-0388-9260}$^{1,2}$ \\\\\n"
         " [4pt] $^1$University of California at Los Angeles \\\\\n"
         " $^2$NASA Jet Propulsion Laboratory \\\\\n}")
    edit(directory / "main.tex", "pdfauthor={}", "pdfauthor={Kevin Lee}")
    seal(directory)
    outputs["pdfinfo"] = outputs["pdfinfo"].replace(
        "Author:          ", "Author:          Kevin Lee")
    outputs["pdftotext"] = outputs["pdftotext"].replace(
        "Anonymous Author(s)", "Kevin Lee")
    # Corresponding email on page 1 is allowed in camera-ready mode.
    outputs["pdftotext"] = outputs["pdftotext"].replace(
        "Kevin Lee\nAffiliation",
        "Kevin Lee\nAffiliation\nkevinlee69720@g.ucla.edu")
    result = release.verify_release(directory, PIN, camera_ready=camera_ready)
    assert result["verification"]["mode"] == "final"
    assert result["verification"]["author_metadata_blank"] is False
    assert result["verification"]["first_page_anonymous"] is False
    assert result["verification"]["first_page_author"] is True
    with pytest.raises(ValueError, match="Require dblblindworkshop"):
        release.verify_release(directory, PIN, camera_ready=False)
    # Switching to anonymous mode leaves a named TeX author, which must fail.
    edit(directory / "main.tex", "[" + options + "]", "[dblblindworkshop]")
    edit(directory / "main.tex", "pdfauthor={Kevin Lee}", "pdfauthor={}")
    with pytest.raises(ValueError, match="Anonymous Authors"):
        verify(paper)


@pytest.mark.parametrize("options", FINAL_OPTIONS)
@pytest.mark.parametrize("camera_ready", [True, None])
def test_camera_ready_mode_rejects_anonymous_source_author(paper, options, camera_ready):
    directory, _, calls = paper
    # Retain the anonymous TeX author while enabling final mode.
    edit(directory / "main.tex", r"\usepackage[dblblindworkshop]{neurips_2025}",
         "\\usepackage[" + options + "]{neurips_2025}\n"
         "\\workshoptitle{2nd Workshop on Principles of Generative Modeling (PriGM)}")
    edit(directory / "main.tex", "pdfauthor={}", "pdfauthor={Kevin Lee}")
    with pytest.raises(ValueError, match="Camera-ready source author must include Kevin Lee"):
        release.verify_release(directory, PIN, camera_ready=camera_ready)
    assert calls == []


@pytest.mark.parametrize("options", [
    "final,final", "dblblindworkshop,dblblindworkshop",
    "dblblindworkshop,final,final", "dblblindworkshop,dblblindworkshop,final",
    "preprint,final", "dblblindworkshop,preprint,final", "dblblindworkshop,preprint",
    "main,final", "sglblindworkshop,final", "dblblindworkshop,position,final",
    "dblblindworkshop,main", "final,", "dblblindworkshop,,final",
])
@pytest.mark.parametrize("camera_ready", [True, None, False])
def test_unsupported_or_repeated_style_options_rejected(paper, options, camera_ready):
    directory, _, calls = paper
    edit(directory / "main.tex", "[dblblindworkshop]", "[" + options + "]")
    with pytest.raises(ValueError, match="Duplicate neurips_2025 option|Camera-ready|Require dblblindworkshop"):
        release.verify_release(directory, PIN, camera_ready=camera_ready)
    assert calls == []


def test_commented_clearpage_does_not_count(paper):
    edit(paper[0] / "main.tex", "\\clearpage\n\\bibliographystyle", "% \\clearpage\n\\bibliographystyle")
    with pytest.raises(ValueError, match="clearpage"):
        verify(paper)


@pytest.mark.parametrize("style", ["plainnat", "unsrtnat"])
def test_supported_bibliography_styles_require_page_break(paper, style):
    edit(paper[0] / "main.tex", "{plainnat}", "{" + style + "}")
    assert verify(paper)["pdf"]["main_pages"] == 4
    edit(paper[0] / "main.tex", "\\clearpage\n\\bibliographystyle", "\\bibliographystyle")
    with pytest.raises(ValueError, match="clearpage"):
        verify(paper)


def test_other_bibliography_styles_remain_rejected(paper):
    edit(paper[0] / "main.tex", "{plainnat}", "{abbrvnat}")
    with pytest.raises(ValueError, match="clearpage"):
        verify(paper)


def test_subject_and_generated_macro_must_match_pin(paper):
    directory, outputs, _ = paper
    outputs["pdfinfo"] = outputs["pdfinfo"].replace(PIN, "b" * 64)
    with pytest.raises(ValueError, match="PDF Subject"):
        verify(paper)
    outputs["pdfinfo"] = outputs["pdfinfo"].replace("b" * 64, PIN)
    edit(directory / "generated/summary_values.tex", PIN, "b" * 64)
    seal(directory)
    with pytest.raises(ValueError, match="Generated PrigmCoreSHA"):
        verify(paper)


def test_more_than_four_main_pages_rejected(paper):
    outputs = paper[1]
    pages = outputs["pdftotext"].split("\f")
    pages.insert(4, "Extra main text.\n")
    outputs["pdftotext"] = "\f".join(pages)
    outputs["pdfinfo"] = outputs["pdfinfo"].replace("Pages: 6", "Pages: 7")
    with pytest.raises(ValueError, match="exceeds four"):
        verify(paper)


def test_references_must_be_first_substantive_text_on_page(paper):
    outputs = paper[1]
    outputs["pdftotext"] = outputs["pdftotext"].replace("References\n", "Leftover main text.\nReferences\n")
    with pytest.raises(ValueError, match="References must start"):
        verify(paper)


def test_output_contract_matches_builder_without_importing_experiment():
    tree = ast.parse((ROOT / "src/build_prigm_paper.py").read_text())
    outputs = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "OUTPUTS" for target in node.targets))
    assert release.OUTPUTS == ast.literal_eval(outputs)


def test_wrapped_section_headings_with_line_numbers_are_readable(paper):
    edit(paper[0] / "main.tex", r"\section{Results}", r"\section{Measured experimental results}")
    outputs = paper[1]
    outputs["pdftotext"] = outputs["pdftotext"].replace(
        "2 Results", "2 Measured experimental    23\nresults    24")
    assert verify(paper)["verification"]["text_sections_present"]


def test_line_numbers_and_valid_public_self_citation_allowed(paper):
    outputs = paper[1]
    outputs["pdftotext"] = outputs["pdftotext"].replace("References\n", "123 References\n")
    outputs["pdftotext"] = outputs["pdftotext"].replace("Kevin Lee.", "124 Kevin Lee.")
    assert verify(paper)["verification"]["first_page_anonymous"]


def test_author_metadata_is_not_excused_by_legitimate_citation(paper):
    outputs = paper[1]
    assert "Kevin Lee" in outputs["pdftotext"]
    outputs["pdfinfo"] = outputs["pdfinfo"].replace("Author:          ", "Author: Kevin Lee")
    with pytest.raises(ValueError, match="Author metadata"):
        verify(paper)


def test_empty_author_can_be_omitted_by_pdfinfo(paper):
    paper[1]["pdfinfo"] = paper[1]["pdfinfo"].replace("Author:          \n", "")
    assert verify(paper)["verification"]["author_metadata_blank"]


@pytest.mark.parametrize("leak", ["/home/someone/private", "C:\\Users\\someone", "private-user", "private-host",
                                   "GPU-12345678-1234-1234-1234-123456789abc", "uid=1000", "hostname=other-machine",
                                   "private@example.org"])
def test_private_identity_in_pdf_rejected(paper, leak):
    paper[1]["pdftotext"] = paper[1]["pdftotext"].replace("More results.", leak)
    with pytest.raises(ValueError, match="Machine identity|Host username|Non-bibliographic email"):
        verify(paper)


@pytest.mark.parametrize("tool", ["pdftotext", "pdfinfo"])
@pytest.mark.parametrize("path", ["/dev/shm/review-archive/diagnostics.npz", "/dev/fd/63",
                                  "/run/shm/review-archive/diagnostics.npz",
                                  "/private/var/folders/review/diagnostics.npz"])
def test_private_filesystem_paths_in_pdf_rejected(paper, tool, path):
    if tool == "pdftotext":
        paper[1][tool] = paper[1][tool].replace("More results.", "Scratch artifact: " + path)
    else:
        paper[1][tool] += "Producer: Scratch artifact " + path + "\n"
    with pytest.raises(ValueError, match="Machine identity or local path"):
        verify(paper)


@pytest.mark.parametrize("field,value", [("hostname", "other-machine"), ("uid", 1000),
                                         ("gpu_uuid", "redacted"), ("extra", "/home/someone/data"),
                                         ("extra", "GPU-12345678-1234-1234-1234-123456789abc")])
def test_evidence_must_already_be_public(paper, field, value):
    change_evidence(paper, lambda manifest: manifest.update({field: value}))
    with pytest.raises(ValueError, match="Private field|Machine identity"):
        verify(paper)


def test_bibliographic_email_allowed_only_in_bound_references(paper):
    directory, outputs, _ = paper
    email = "public@reference.org"
    for name in ("references.bib", "main.bbl"):
        put(directory / name, (directory / name).read_text() + " " + email)
    outputs["pdftotext"] = outputs["pdftotext"].replace("Public predecessor.", "Public predecessor. " + email)
    assert verify(paper)["pdf"]["main_pages"] == 4
    outputs["pdftotext"] = outputs["pdftotext"].replace("Additional results.", email)
    with pytest.raises(ValueError, match="Non-bibliographic email"):
        verify(paper)


@pytest.mark.parametrize("mutation", ["unembedded", "empty", "malformed"])
def test_font_embedding_required(paper, mutation):
    outputs = paper[1]
    if mutation == "unembedded":
        outputs["pdffonts"] = outputs["pdffonts"].replace("yes yes yes", "no  yes yes")
    elif mutation == "empty":
        outputs["pdffonts"] = "\n".join(outputs["pdffonts"].splitlines()[:2])
    else:
        outputs["pdffonts"] += "unparseable font\n"
    with pytest.raises(ValueError, match="font embedding|No PDF fonts"):
        verify(paper)


@pytest.mark.parametrize("glyph", ["\ufffd", "\ue000", "\x00", "\u25a1", "(cid:123)", "??"])
def test_broken_text_encoding_rejected(paper, glyph):
    paper[1]["pdftotext"] += glyph
    with pytest.raises(ValueError, match="glyph encoding"):
        verify(paper)


@pytest.mark.parametrize("text", ["References", "Anonymous Author(s)", "Results", "Density and scoring details"])
def test_missing_text_sections_and_anonymous_line_rejected(paper, text):
    paper[1]["pdftotext"] = paper[1]["pdftotext"].replace(text, "missing")
    with pytest.raises(ValueError, match="Missing|anonymous"):
        verify(paper)


@pytest.mark.parametrize("diagnostic", [
    "! Undefined control sequence.", "LaTeX Error: Missing document.",
    "./main.tex:12: Undefined control sequence.", "Package natbib Warning: Citation `x' undefined.",
    "LaTeX Warning: Reference `x' on page 2 undefined.", "LaTeX Warning: There were undefined references.",
    "Overfull \\hbox (1.0pt too wide)", "Overfull \\vbox (1.0pt too high)",
    "Missing character: There is no X in font nullfont!", "LaTeX Warning: Label(s) may have changed.",
    "Package rerunfilecheck Warning: Rerun to get outlines right", "Runaway argument?",
])
def test_latex_log_diagnostics_rejected(paper, diagnostic):
    path = paper[0] / "main.log"
    put(path, path.read_text() + diagnostic + "\n", NEW)
    with pytest.raises(ValueError, match="LaTeX log has errors"):
        verify(paper)


def test_latin1_latex_log_is_scanned_without_decode_failure(paper):
    path = paper[0] / "main.log"
    path.write_bytes(path.read_bytes() + b"M. C. Eble and F. I. Gonz\xe1lez.\n")
    os.utime(path, ns=(NEW, NEW))
    assert verify(paper)["verification"]["latex_log_clean"]


def test_log_must_record_success_for_same_pdf_page_count(paper):
    edit(paper[0] / "main.log", "6 pages", "5 pages", NEW)
    with pytest.raises(ValueError, match="matching successful"):
        verify(paper)


@pytest.mark.parametrize("name", [n for n in (*release.SOURCES, *release.OUTPUTS) if n != "references.bib"])
def test_each_actual_input_must_appear_in_recorder(paper, name):
    edit(paper[0] / "main.fls", "INPUT ./" + name + "\n", "", NEW)
    with pytest.raises(ValueError, match="Recorder missing"):
        verify(paper)


@pytest.mark.parametrize("name", ["extra.tex", "figures/extra.png", "shadow.sty", "extra.aux"])
def test_unbound_local_recorder_input_rejected(paper, name):
    path = paper[0] / "main.fls"
    put(paper[0] / name, "unbound input")
    put(path, path.read_text() + "INPUT " + name + "\n", NEW)
    with pytest.raises(ValueError, match="Unbound local recorder"):
        verify(paper)


def test_unbound_external_recorder_input_rejected(paper):
    directory = paper[0]
    external = directory.parent / "unbound.tex"
    put(external, "external input")
    path = directory / "main.fls"
    put(path, path.read_text() + "INPUT " + str(external) + "\n", NEW)
    with pytest.raises(ValueError, match="Unbound external"):
        verify(paper)


@pytest.mark.parametrize("command", [r"\input{extra.tex}", r"\includegraphics{figures/extra.png}", r"\input extra.tex"])
def test_unbound_source_input_rejected_even_if_recorder_omits_it(paper, command):
    edit(paper[0] / "main.tex", r"\end{document}", command + r"\end{document}")
    with pytest.raises(ValueError, match="Unbound local|literal braced"):
        verify(paper)


def test_all_five_fragments_must_be_included_not_just_exist(paper):
    edit(paper[0] / "main.tex", r"\input{generated/all_runs_rows.tex}", "")
    with pytest.raises(ValueError, match="All bound generated"):
        verify(paper)


def test_installed_tex_dependencies_are_checked_for_freshness(paper, monkeypatch):
    directory = paper[0]
    system = directory.parent / "texmf"
    system.mkdir()
    dependency = system / "article.cls"
    put(dependency, "installed document class")
    monkeypatch.setattr(release, "TEX_ROOTS", (system,))
    path = directory / "main.fls"
    put(path, path.read_text() + "INPUT " + str(dependency) + "\n", NEW)
    assert verify(paper)["verification"]["recorder_inputs_bound"]
    os.utime(dependency, ns=(NEW, NEW))
    with pytest.raises(ValueError, match="Stale PDF"):
        verify(paper)


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("pdfinfo", 60),
                                     subprocess.CalledProcessError(1, "pdfinfo")])
def test_tool_failure_never_passes_or_writes(paper, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(release.subprocess, "run", fail)
    with pytest.raises(ValueError, match="pdfinfo failed"):
        verify(paper)
    assert not (paper[0] / release.MANIFEST).exists()


def test_tool_warnings_fail_closed(paper, monkeypatch):
    monkeypatch.setattr(release.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, paper[1][args[0]], "Syntax Warning: bad font"))
    with pytest.raises(ValueError, match="reported warnings"):
        verify(paper)


def test_inputs_changed_during_pdf_tools_are_rejected(paper, monkeypatch):
    run = release.subprocess.run
    def mutate(args, **kwargs):
        result = run(args, **kwargs)
        if args[0] == "pdffonts":
            put(paper[0] / "main.tex", "changed during verification")
        return result
    monkeypatch.setattr(release.subprocess, "run", mutate)
    with pytest.raises(ValueError, match="changed during verification"):
        verify(paper)


def test_atomic_write_failure_preserves_previous_manifest(paper, monkeypatch):
    target = paper[0] / release.MANIFEST
    put(target, "previous release")
    def fail(*args):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(release.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        release._write_manifest(paper[0], verify(paper))
    assert target.read_text() == "previous release"
    assert not list(paper[0].glob(".submission-manifest-*"))


def test_failed_verification_preserves_previous_manifest(paper, capsys):
    directory = paper[0]
    put(directory / release.MANIFEST, "previous release")
    (directory / "main.pdf").unlink()
    with pytest.raises(SystemExit) as error:
        release.main(["--paper-dir", str(directory), "--expected-core-sha256", PIN, "--write-manifest"])
    assert error.value.code == 1
    assert not capsys.readouterr().out
    assert (directory / release.MANIFEST).read_text() == "previous release"


def test_duplicate_and_nonfinite_manifest_values_rejected(paper):
    path = paper[0] / release.EVIDENCE
    original = path.read_text()
    put(path, original[:-1] + ', "purpose": "core"}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        verify(paper)
    put(path, original[:-1] + ', "extra": NaN}')
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        verify(paper)
