import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from argos.domain.utils import (  # noqa: E402
    count_jsonl_entries,
    ensure_dir,
    flatten_dict,
    read_last_jsonl_line,
    slugify_machine_id,
    utc_timestamp_str,
)
from argos.common.utils import redact_keys, retry_operation, setup_logging  # noqa: E402


class TestSlugifyMachineId(unittest.TestCase):
    """Tests for slugify_machine_id."""

    def test_slugify_special_chars(self):
        self.assertEqual(slugify_machine_id("My Machine!"), "my-machine")

    def test_slugify_underscores_preserved(self):
        self.assertEqual(slugify_machine_id("EDGE_01"), "edge_01")

    def test_slugify_none_uses_default(self):
        result = slugify_machine_id(None)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_slugify_empty_string(self):
        result = slugify_machine_id("")
        self.assertIsInstance(result, str)


class TestFlattenDict(unittest.TestCase):
    """Tests for flatten_dict."""

    def test_single_nested(self):
        self.assertEqual(flatten_dict({"a": {"b": 1}}), {"a.b": 1})

    def test_mixed_nesting(self):
        self.assertEqual(flatten_dict({"a": 1, "b": {"c": 2}}), {"a": 1, "b.c": 2})

    def test_deep_nesting(self):
        self.assertEqual(flatten_dict({"a": {"b": {"c": 3}}}), {"a.b.c": 3})

    def test_empty_dict(self):
        self.assertEqual(flatten_dict({}), {})


class TestUtcTimestampStr(unittest.TestCase):
    """Tests for utc_timestamp_str."""

    def test_returns_string(self):
        result = utc_timestamp_str()
        self.assertIsInstance(result, str)

    def test_custom_format(self):
        result = utc_timestamp_str("%Y-%m-%d")
        self.assertRegex(result, r"^\d{4}-\d{2}-\d{2}$")


class TestEnsureDir(unittest.TestCase):
    """Tests for ensure_dir."""

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "subdir" / "nested"
            result = ensure_dir(new_path)
            self.assertTrue(new_path.exists())
            self.assertTrue(new_path.is_dir())
            self.assertEqual(result, new_path)

    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            result = ensure_dir(path)
            self.assertEqual(result, path)


class TestCountJsonlEntries(unittest.TestCase):
    """Tests for count_jsonl_entries."""

    def test_counts_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n')
            f.write('{"b": 2}\n')
            f.write('{"c": 3}\n')
            path = Path(f.name)
        try:
            self.assertEqual(count_jsonl_entries(path), 3)
        finally:
            path.unlink()

    def test_ignores_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n')
            f.write("\n")
            f.write('{"b": 2}\n')
            path = Path(f.name)
        try:
            self.assertEqual(count_jsonl_entries(path), 2)
        finally:
            path.unlink()

    def test_nonexistent_file_returns_zero(self):
        path = Path("/nonexistent/path/file.jsonl")
        self.assertEqual(count_jsonl_entries(path), 0)


class TestReadLastJsonlLine(unittest.TestCase):
    """Tests for read_last_jsonl_line."""

    def test_reads_last_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n')
            f.write('{"b": 2}\n')
            f.write('{"c": 3}\n')
            path = Path(f.name)
        try:
            self.assertEqual(read_last_jsonl_line(path), '{"c": 3}')
        finally:
            path.unlink()

    def test_ignores_trailing_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n')
            f.write("\n")
            f.write("\n")
            path = Path(f.name)
        try:
            self.assertEqual(read_last_jsonl_line(path), '{"a": 1}')
        finally:
            path.unlink()

    def test_empty_file_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        try:
            self.assertIsNone(read_last_jsonl_line(path))
        finally:
            path.unlink()

    def test_nonexistent_file_returns_none(self):
        path = Path("/nonexistent/path/file.jsonl")
        self.assertIsNone(read_last_jsonl_line(path))


class TestSetupLogging(unittest.TestCase):
    """Tests for setup_logging."""

    def test_setup_logging_no_error(self):
        # Should not raise
        setup_logging()
        setup_logging(level=20)  # INFO


class TestRedactKeys(unittest.TestCase):
    """Tests for redact_keys."""

    def test_redacts_specified_keys(self):
        payload = {"username": "user", "password": "secret", "data": 123}
        result = redact_keys(payload, {"password"})
        self.assertEqual(result["username"], "user")
        self.assertEqual(result["password"], "***")
        self.assertEqual(result["data"], 123)

    def test_no_redaction_when_empty_set(self):
        payload = {"a": 1, "b": 2}
        result = redact_keys(payload, set())
        self.assertEqual(result, payload)

    def test_multiple_keys_redacted(self):
        payload = {"a": 1, "b": 2, "c": 3}
        result = redact_keys(payload, {"a", "c"})
        self.assertEqual(result["a"], "***")
        self.assertEqual(result["b"], 2)
        self.assertEqual(result["c"], "***")


class TestRetryOperation(unittest.TestCase):
    """Tests for retry_operation."""

    def test_success_on_first_try(self):
        call_count = [0]

        def success_func():
            call_count[0] += 1
            return "success"

        wrapped = retry_operation(success_func, attempts=3, wait_seconds=0.01)
        result = wrapped()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 1)

    def test_retries_on_failure(self):
        call_count = [0]

        def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("fail")
            return "success"

        wrapped = retry_operation(fail_then_succeed, attempts=3, wait_seconds=0.01)
        result = wrapped()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)

    def test_raises_after_all_retries(self):
        def always_fail():
            raise ValueError("always fails")

        wrapped = retry_operation(always_fail, attempts=2, wait_seconds=0.01)
        with self.assertRaises(ValueError):
            wrapped()

    def test_specific_exception_types(self):
        call_count = [0]

        def fail_with_type():
            call_count[0] += 1
            raise TypeError("type error")

        wrapped = retry_operation(
            fail_with_type,
            attempts=2,
            wait_seconds=0.01,
            exceptions=(TypeError,),
        )
        with self.assertRaises(TypeError):
            wrapped()
        self.assertEqual(call_count[0], 2)


if __name__ == "__main__":
    unittest.main()
