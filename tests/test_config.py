"""The two guards every stage relies on, and the type rule."""

import pytest

import config


def test_require_names_the_missing_path_and_the_variable_to_set(tmp_path):
    absent = tmp_path / "nope.tsv"
    with pytest.raises(SystemExit) as exc:
        config.require(absent)
    message = str(exc.value)
    assert str(absent) in message
    assert "ROTATE_EMB" in message, "the message has to say which variable to set"


def test_require_accepts_existing_paths(tmp_path):
    present = tmp_path / "here.tsv"
    present.write_text("x\n")
    config.require(present)  # must not raise


def test_refuse_overwrite_blocks_an_existing_output(tmp_path):
    existing = tmp_path / "evaluation_summary.tsv"
    existing.write_text("model\troc\n")
    with pytest.raises(SystemExit) as exc:
        config.refuse_overwrite(existing)
    assert "refusing to overwrite" in str(exc.value)
    assert existing.read_text() == "model\troc\n", "the guard must not touch the file"


def test_refuse_overwrite_passes_when_nothing_is_there(tmp_path):
    config.refuse_overwrite(tmp_path / "absent.tsv")  # must not raise


def test_treats_object_types_are_the_biolink_disease_slots():
    """detect.py's type_rule_violation column is only defensible if this is the
    Biolink pair a treats edge may point at."""
    assert config.VALID_TREATS_OBJECT_TYPES == (
        "biolink:Disease",
        "biolink:PhenotypicFeature",
    )
