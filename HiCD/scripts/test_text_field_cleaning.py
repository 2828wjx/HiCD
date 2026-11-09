import unittest

from text_field_cleaning import (
    clean_code_text,
    clean_description_text,
    clean_record_text,
)


class TextFieldCleaningTests(unittest.TestCase):
    def test_code_cleaning_preserves_indentation_and_comments(self):
        raw = "\r\n  int x = 1;   \r\n\r\n\r\n  // keep me  \r\n"
        self.assertEqual(
            clean_code_text(raw),
            "int x = 1;\n\n  // keep me",
        )

    def test_description_whitespace_is_collapsed(self):
        self.assertEqual(
            clean_description_text("  alpha\r\n beta\t gamma  "),
            "alpha beta gamma",
        )

    def test_cleaning_is_idempotent(self):
        raw_code = "x();  \n\n\n y();\n"
        raw_description = "alpha   beta\n gamma"
        code = clean_code_text(raw_code)
        description = clean_description_text(raw_description)
        self.assertEqual(clean_code_text(code), code)
        self.assertEqual(
            clean_description_text(description), description
        )

    def test_record_field_priority_matches_dataset_loader(self):
        cleaned = clean_record_text({
            "code": " code();  ",
            "func": "ignored();",
            "cve_description": "  one   two ",
            "commit_message": "ignored",
        })
        self.assertEqual(cleaned["code"], "code();")
        self.assertEqual(cleaned["description"], "one two")


if __name__ == "__main__":
    unittest.main()
