"""Presentation boundary tests with real hashes/FLS and synthetic scientific assets.

The existing combined-manifest tests cover scientific schema validation. Here
that validator and Poppler are isolated so failures identify the release boundary.
No raw experiment, real PDF, font installation or GPU is needed.
"""
import json
import os
from pathlib import Path
import re

import pytest

from test_prigm_release import NEW, PIN, ROOT, paper, put
import build_prigm_v2_paper as builder
import verify_prigm_release as release


COMBINED_PIN = "b" * 64


def write_json(path, value):
    put(path, json.dumps(value, sort_keys=True))


def recorder(directory, outputs):
    names = [name for name in (*release.SOURCES, *outputs) if name != "references.bib"]
    put(directory / "main.fls", "PWD " + str(directory) + "\n"
        + "".join("INPUT ./" + name + "\n" for name in names) + "OUTPUT main.pdf\n", NEW)


@pytest.fixture
def presented(paper, tmp_path, monkeypatch):
    directory, commands, poppler_calls = paper
    source_hashes = {}
    for name in builder.DART_EXTENSION_SOURCE_IDS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        source_hashes[name] = release._sha(target)
    for name in release.DART_COMBINED_OUTPUTS:
        put(directory / name, "synthetic historical output\n")
    put(directory / "generated/summary_values.tex", r"\newcommand{\PrigmCoreSHA}{" + PIN + "}\n")
    put(directory / "generated/combined_binding.tex", r"\newcommand{\PrigmCombinedSHA}{" + COMBINED_PIN + "}\n")
    scientific = {
        "schema_version": 3, "purpose": "combined", "profile": "dart_extension",
        "combined_input_sha256": COMBINED_PIN,
        "builder_sha256": source_hashes["src/build_prigm_v2_paper.py"],
        "core": {"run_manifest_sha256": PIN},
        "input_index": {"source_sha256": source_hashes,
                        "numerical_validation": {"source_sha256": dict(builder.FROZEN_NUMERICAL_SOURCE_SHA256)}},
        "output_sha256": {name: release._sha(directory / name) for name in release.DART_COMBINED_OUTPUTS},
    }
    write_json(directory / release.EVIDENCE, scientific)
    validation_calls = []

    def validate(value, expected_core):
        assert expected_core == PIN
        assert value["schema_version"] == 3 and value["profile"] == "dart_extension"
        validation_calls.append(expected_core)
        return value

    monkeypatch.setattr(builder, "validate_combined_manifest", validate)
    replacements = {}
    for old, name in release.PRESENTATION_REPLACEMENTS.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        put(path, "%PDF-synthetic presentation " + name + "\n")
        replacements[old] = {"path": name, "sha256": release._sha(path)}
    inputs = {}
    for name in release.PRESENTATION_INPUTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        put(target, "synthetic authenticated input " + name + "\n")
        inputs[name] = release._sha(target)
    sidecar = {
        "schema_version": 1, "scientific_manifest_sha256": release._sha(directory / release.EVIDENCE),
        "combined_input_sha256": COMBINED_PIN, "replacements": replacements, "inputs_sha256": inputs,
        "style": {"family": "DejaVu Serif", "mathtext": "stix", "size_pt": 8},
    }
    write_json(directory / release.PRESENTATION, sidecar)
    tex = (directory / "main.tex").read_text()
    tex = tex.replace(r"pdfsubject={PriGM core SHA256: \PrigmCoreSHA}",
                      r"pdfsubject={PriGM core SHA256: \PrigmCoreSHA; combined SHA256: \PrigmCombinedSHA}")
    tex = tex.replace(r"\includegraphics{figures/prior_samples.png}", "")
    for old, new in release.PRESENTATION_REPLACEMENTS.items():
        tex = tex.replace("{" + old + "}", "{" + new + "}")
    extra = [name for name in release.DART_COMBINED_OUTPUTS if name not in release.OUTPUTS]
    additions = "\n".join((r"\input{" + name + "}" if name.endswith(".tex") else
                            r"\includegraphics{" + release.PRESENTATION_REPLACEMENTS[name] + "}") for name in extra)
    put(directory / "main.tex", tex.replace(r"\end{document}", additions + "\n" + r"\end{document}"))
    compile_outputs = tuple(release.PRESENTATION_REPLACEMENTS.get(name, name)
                            for name in release.DART_COMBINED_OUTPUTS)
    recorder(directory, compile_outputs)
    commands["pdfinfo"] = (f"Title: Fixture\nAuthor: \nSubject: PriGM core SHA256: {PIN}; "
                           f"combined SHA256: {COMBINED_PIN}\nPages: 6\n")
    figure_fonts = {name: commands["pdffonts"].replace("NimbusRoman-Regular", "DejaVuSerif")
                    for name in release.PRESENTATION_REPLACEMENTS.values()}
    command = release._command

    def run(args):
        if args[0] == "pdffonts" and args[1] != str(directory / "main.pdf"):
            return figure_fonts[Path(args[1]).relative_to(directory).as_posix()]
        return command(args)

    monkeypatch.setattr(release, "_command", run)
    return {"directory": directory, "repo": tmp_path, "sidecar": sidecar, "scientific": scientific,
            "pin": release._sha(directory / release.PRESENTATION), "compile_outputs": compile_outputs,
            "poppler_calls": poppler_calls, "validation_calls": validation_calls, "figure_fonts": figure_fonts}


def verify(state, **kwargs):
    options = {"expected_presentation_sha256": state["pin"], **kwargs}
    return release.verify_release(state["directory"], PIN, COMBINED_PIN, **options)


def reseal(state):
    path = state["directory"] / release.PRESENTATION
    write_json(path, state["sidecar"])
    return release._sha(path)


def test_presentation_binds_new_files_and_preserves_historical_bindings(presented):
    directory = presented["directory"]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in presented["repo"].rglob("*") if p.is_file()}
    result = verify(presented)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
    assert presented["validation_calls"] == [PIN]
    assert len(result["generated_sha256"]) == 21
    assert result["generated_sha256"] == presented["scientific"]["output_sha256"]
    assert result["derivation_source_sha256"] == presented["scientific"]["input_index"]["source_sha256"]
    assert result["presentation"] == {
        "manifest_sha256": presented["pin"],
        "output_sha256": {item["path"]: item["sha256"] for item in presented["sidecar"]["replacements"].values()},
        "inputs_sha256": presented["sidecar"]["inputs_sha256"],
    }
    assert result["verification"]["presentation_bound"]
    assert result["verification"]["presentation_fonts_embedded"]
    assert len(result["presentation"]["output_sha256"]) == 7
    assert len([name for name in presented["compile_outputs"] if name.endswith(".tex")]) == 14
    assert not (directory / release.MANIFEST).exists()


def test_historical_schema_three_route_stays_available(presented):
    directory = presented["directory"]
    (directory / release.PRESENTATION).unlink()
    tex = (directory / "main.tex").read_text()
    for old, new in release.PRESENTATION_REPLACEMENTS.items():
        tex = tex.replace("{" + new + "}", "{" + old + "}")
    put(directory / "main.tex", tex)
    recorder(directory, release.DART_COMBINED_OUTPUTS)
    result = verify(presented, expected_presentation_sha256=None)
    assert "presentation" not in result
    assert "presentation_bound" not in result["verification"]
    assert result["generated_sha256"] == presented["scientific"]["output_sha256"]


@pytest.mark.parametrize("pin", [None, "0" * 64, "ABC", True])
def test_explicit_presentation_pin_is_required_and_exact(presented, pin):
    with pytest.raises(ValueError):
        verify(presented, expected_presentation_sha256=pin)
    assert presented["poppler_calls"] == []


def test_wrong_pin_precedes_json_decoding(presented):
    put(presented["directory"] / release.PRESENTATION, "not JSON")
    with pytest.raises(ValueError, match="Pinned JSON SHA256 differs"):
        verify(presented)


@pytest.mark.parametrize("pin", [None, "original"])
def test_replacement_pdfs_require_the_sidecar(presented, pin):
    (presented["directory"] / release.PRESENTATION).unlink()
    with pytest.raises(ValueError, match="Missing release file|Unbound local figure"):
        verify(presented, expected_presentation_sha256=presented["pin"] if pin else None)


@pytest.mark.parametrize("mutation", ["schema", "boolean_schema", "science", "combined", "missing_map",
                                     "extra_map", "renamed_pdf", "duplicate_pdf", "extra_field",
                                     "bad_hash", "missing_input", "extra_input", "private_metadata"])
def test_resealed_invalid_contract_is_rejected(presented, mutation):
    value = presented["sidecar"]
    old, target = next(iter(value["replacements"].items()))
    if mutation == "schema":
        value["schema_version"] = 2
    elif mutation == "boolean_schema":
        value["schema_version"] = True
    elif mutation == "science":
        value["scientific_manifest_sha256"] = "0" * 64
    elif mutation == "combined":
        value["combined_input_sha256"] = "0" * 64
    elif mutation == "missing_map":
        del value["replacements"][old]
    elif mutation == "extra_map":
        value["replacements"]["figures/extra.png"] = dict(target)
    elif mutation == "renamed_pdf":
        target["path"] = "figures/presentation/other.pdf"
    elif mutation == "duplicate_pdf":
        other = next(item for name, item in value["replacements"].items() if name != old)
        target["path"] = other["path"]
    elif mutation == "extra_field":
        target["unchecked"] = True
    elif mutation == "bad_hash":
        target["sha256"] = "not a digest"
    elif mutation == "missing_input":
        del value["inputs_sha256"][release.PRESENTATION_INPUTS[0]]
    elif mutation == "extra_input":
        value["inputs_sha256"]["analysis/prigm/figures/extra.npz"] = "0" * 64
    else:
        value["style"]["font_file"] = "/home/private-user/font.ttf"
    with pytest.raises(ValueError):
        verify(presented, expected_presentation_sha256=reseal(presented))
    assert presented["poppler_calls"] == []


@pytest.mark.parametrize("name", release.PRESENTATION_INPUTS)
def test_every_presentation_input_is_authenticated(presented, name):
    put(presented["repo"] / name, "tampered plotting input")
    with pytest.raises(ValueError, match="Presentation input SHA256 differs"):
        verify(presented)


@pytest.mark.parametrize("name", release.PRESENTATION_INPUTS)
def test_every_presentation_input_is_required(presented, name):
    (presented["repo"] / name).unlink()
    with pytest.raises(ValueError, match="Missing release file: presentation_input:"):
        verify(presented)


@pytest.mark.parametrize("kind", ["cache", "mapping"])
def test_resealing_a_changed_input_cannot_replace_the_reviewed_pin(presented, kind):
    if kind == "cache":
        name = release.PRESENTATION_INPUTS[0]
        path = presented["repo"] / name
        put(path, "tampered cache")
        presented["sidecar"]["inputs_sha256"][name] = release._sha(path)
    else:
        target = next(iter(presented["sidecar"]["replacements"].values()))
        target["path"] = "figures/presentation/arbitrary.pdf"
    assert reseal(presented) != presented["pin"]
    with pytest.raises(ValueError, match="Pinned JSON SHA256 differs"):
        verify(presented)


@pytest.mark.parametrize("name", tuple(release.PRESENTATION_REPLACEMENTS.values()))
@pytest.mark.parametrize("mutation", ["tampered", "missing", "stale"])
def test_each_compiled_pdf_is_bound_and_fresh(presented, name, mutation):
    path = presented["directory"] / name
    if mutation == "tampered":
        put(path, "%PDF-tampered")
    elif mutation == "missing":
        path.unlink()
    else:
        os.utime(path, ns=(NEW, NEW))
    with pytest.raises(ValueError, match="Presentation figure SHA256 differs|Missing release file|Stale PDF"):
        verify(presented)


def test_only_compile_inputs_affect_pdf_freshness(presented):
    directory = presented["directory"]
    for name in (*release.PRESENTATION_REPLACEMENTS, release.EVIDENCE, release.PRESENTATION):
        os.utime(directory / name, ns=(NEW, NEW))
    for name in release.PRESENTATION_INPUTS:
        os.utime(presented["repo"] / name, ns=(NEW, NEW))
    verify(presented)
    path = directory / "generated/dart_block_rows.tex"
    os.utime(path, ns=(NEW, NEW))
    with pytest.raises(ValueError, match="Stale PDF"):
        verify(presented)


@pytest.mark.parametrize("name", tuple(release.PRESENTATION_REPLACEMENTS) + ("generated/dart_block_rows.tex",))
def test_historical_assets_remain_authenticated(presented, name):
    put(presented["directory"] / name, "changed historical asset")
    with pytest.raises(ValueError, match="Generated asset SHA256 differs"):
        verify(presented)


@pytest.mark.parametrize("name", builder.DART_EXTENSION_SOURCE_IDS)
def test_all_historical_sources_remain_authenticated(presented, name):
    path = presented["repo"] / name
    path.write_bytes(path.read_bytes() + b"\n# changed historical source\n")
    with pytest.raises(ValueError, match="Current (builder|derivation source) SHA256 differs"):
        verify(presented)


@pytest.mark.parametrize("mutation", ["missing_pdf", "historical_png", "extra_pdf"])
def test_recorder_uses_only_the_declared_compilation_set(presented, mutation):
    path = presented["directory"] / "main.fls"
    text = path.read_text()
    if mutation == "missing_pdf":
        text = text.replace("INPUT ./" + next(iter(release.PRESENTATION_REPLACEMENTS.values())) + "\n", "")
    else:
        name = next(iter(release.PRESENTATION_REPLACEMENTS)) if mutation == "historical_png" else "figures/extra.pdf"
        if mutation == "extra_pdf":
            put(presented["directory"] / name, "extra asset")
        text += "INPUT ./" + name + "\n"
    put(path, text, NEW)
    with pytest.raises(ValueError, match="Recorder missing required inputs|Unbound local recorder input"):
        verify(presented)


@pytest.mark.parametrize("mutation", ["old_png", "undeclared_pdf", "unused_pdf"])
def test_tex_must_use_exactly_the_seven_replacements(presented, mutation):
    path = presented["directory"] / "main.tex"
    old, new = next(iter(release.PRESENTATION_REPLACEMENTS.items()))
    text = path.read_text()
    if mutation == "unused_pdf":
        text = re.sub(r"\\includegraphics(?:\[[^]]*\])?\{" + re.escape(new) + r"\}", "", text)
    else:
        text = text.replace(new, old if mutation == "old_png" else "figures/presentation/unbound.pdf")
    put(path, text)
    with pytest.raises(ValueError, match="Unbound local figure|All bound generated assets must be included"):
        verify(presented)


@pytest.mark.parametrize("kind", ["manifest", "cache", "pdf"])
def test_new_files_cannot_be_symlinks(presented, kind):
    path = (presented["repo"] / release.PRESENTATION_INPUTS[0] if kind == "cache" else
            presented["directory"] / (release.PRESENTATION if kind == "manifest" else
                                      next(iter(release.PRESENTATION_REPLACEMENTS.values()))))
    original = path.with_name(path.name + ".retained")
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ValueError, match="Symlink asset path"):
        verify(presented)


@pytest.mark.parametrize("kind", ["manifest", "cache", "pdf"])
def test_new_files_are_rechecked_after_pdf_validation(presented, monkeypatch, kind):
    path = (presented["repo"] / release.PRESENTATION_INPUTS[0] if kind == "cache" else
            presented["directory"] / (release.PRESENTATION if kind == "manifest" else
                                      next(iter(release.PRESENTATION_REPLACEMENTS.values()))))
    command = release._command

    def mutate(args):
        result = command(args)
        if args[0] == "pdfinfo":
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(release, "_command", mutate)
    with pytest.raises(ValueError, match="Release input changed during verification"):
        verify(presented)


def test_cli_pin_and_submission_manifest(presented, capsys):
    directory = presented["directory"]
    args = ["--paper-dir", str(directory), "--expected-core-sha256", PIN,
            "--expected-combined-sha256", COMBINED_PIN, "--write-manifest"]
    with pytest.raises(SystemExit) as error:
        release.main(args)
    assert error.value.code == 1
    assert not (directory / release.MANIFEST).exists()
    assert release.main([*args, "--expected-presentation-sha256", presented["pin"]]) == 0
    saved = json.loads((directory / release.MANIFEST).read_text())
    assert saved == verify(presented)
    assert '"manifest_written": true' in capsys.readouterr().out


@pytest.mark.parametrize("name", tuple(release.PRESENTATION_REPLACEMENTS.values()))
@pytest.mark.parametrize("mutation", ["wrong_family", "unembedded"])
def test_every_presentation_pdf_embeds_dejavu_serif(presented, name, mutation):
    text = presented["figure_fonts"][name]
    presented["figure_fonts"][name] = (text.replace("DejaVuSerif", "OtherSerif") if mutation == "wrong_family"
                                      else text.replace("yes yes yes", "no  yes yes"))
    with pytest.raises(ValueError, match="Missing embedded DejaVuSerif|Missing font embedding"):
        verify(presented)
