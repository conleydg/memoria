"""Tests for ground truth collection (ADR-0006) - entirely against fake
AppleScript output and a fake search function. Never touches Photos or
any real image content: the actual AppleScript call (run_photos_search)
is a real I/O boundary that's injected, not exercised, in every test
below. See docs/how-ai-was-used.md for why this split matters - the
*tool* is fully testable now; the real ground truth data it produces
against a real library is a separate, later, human-triggered step.
"""

import json

from memoria.ground_truth import (
    Match,
    _escape_applescript_string,
    collect_ground_truth,
    load_ground_truth,
    parse_search_output,
    save_ground_truth,
)


class TestEscapeApplescriptString:
    def test_plain_string_is_unchanged(self):
        assert _escape_applescript_string("dog") == "dog"

    def test_quotes_are_escaped(self):
        assert _escape_applescript_string('say "hi"') == 'say \\"hi\\"'

    def test_backslashes_are_escaped_before_quotes(self):
        # Wrong order would double-escape the backslash just inserted
        # for the quote - this pins the correct order.
        assert _escape_applescript_string('a\\"b') == 'a\\\\\\"b'


class TestParseSearchOutput:
    def test_parses_multiple_lines(self):
        raw = "uuid-1\tIMG_0001.HEIC\nuuid-2\tIMG_0002.HEIC\n"
        assert parse_search_output(raw) == [
            Match(uuid="uuid-1", filename="IMG_0001.HEIC"),
            Match(uuid="uuid-2", filename="IMG_0002.HEIC"),
        ]

    def test_empty_output_is_a_real_zero_result_answer(self):
        # ADR-0026: "no matches" is meaningful ground truth, not an
        # error or something to skip.
        assert parse_search_output("") == []

    def test_blank_lines_are_skipped(self):
        raw = "uuid-1\tIMG_0001.HEIC\n\n\nuuid-2\tIMG_0002.HEIC\n"
        assert len(parse_search_output(raw)) == 2

    def test_malformed_line_is_skipped_not_fatal(self):
        raw = "uuid-1\tIMG_0001.HEIC\nthis line has no tab\nuuid-2\tIMG_0002.HEIC\n"
        result = parse_search_output(raw)
        assert result == [
            Match(uuid="uuid-1", filename="IMG_0001.HEIC"),
            Match(uuid="uuid-2", filename="IMG_0002.HEIC"),
        ]

    def test_filename_containing_a_tab_is_kept_whole(self):
        # split(sep, 1) - only the first tab is a delimiter.
        raw = "uuid-1\tweird\tfilename.jpg\n"
        assert parse_search_output(raw) == [Match(uuid="uuid-1", filename="weird\tfilename.jpg")]


class TestCollectGroundTruth:
    def test_runs_every_query_through_the_injected_search_fn(self):
        fake_responses = {
            "dog": "uuid-1\tIMG_0001.HEIC\n",
            "beach": "uuid-2\tIMG_0002.HEIC\nuuid-3\tIMG_0003.HEIC\n",
        }
        result = collect_ground_truth(
            ["dog", "beach"], search_fn=lambda q: fake_responses[q]
        )
        assert result["dog"] == [Match(uuid="uuid-1", filename="IMG_0001.HEIC")]
        assert len(result["beach"]) == 2

    def test_zero_result_query_is_recorded_as_empty_list_not_omitted(self):
        result = collect_ground_truth(["giraffe"], search_fn=lambda q: "")
        assert "giraffe" in result
        assert result["giraffe"] == []

    def test_never_calls_the_real_applescript_function(self):
        # If this test ever calls the real run_photos_search, it'll
        # hang or fail outside a machine with Photos.app - proof the
        # injected search_fn is actually being used, not the default.
        calls = []
        collect_ground_truth(["a", "b"], search_fn=lambda q: calls.append(q) or "")
        assert calls == ["a", "b"]


class TestSaveAndLoadGroundTruth:
    def test_round_trip_preserves_data(self, tmp_path):
        original = {
            "dog": [Match(uuid="uuid-1", filename="IMG_0001.HEIC")],
            "giraffe": [],
        }
        path = str(tmp_path / "ground_truth.json")
        save_ground_truth(original, path)
        loaded = load_ground_truth(path)
        assert loaded == original

    def test_saved_file_is_plain_readable_json(self, tmp_path):
        path = str(tmp_path / "ground_truth.json")
        save_ground_truth({"dog": [Match(uuid="uuid-1", filename="a.jpg")]}, path)
        with open(path) as f:
            raw = json.load(f)
        assert raw == {"dog": [{"uuid": "uuid-1", "filename": "a.jpg"}]}
