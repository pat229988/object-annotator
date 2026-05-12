import os
import unittest

from object_annotator.libs import resources  # noqa: F401
from object_annotator.libs.stringBundle import StringBundle


class TestStringBundle(unittest.TestCase):
    def test_loadDefaultBundle_withoutError(self):
        str_bundle = StringBundle.get_bundle("en")
        self.assertEqual(
            str_bundle.get_string("openDir"),
            "Open Dir",
            "Fail to load the default bundle",
        )

    def test_fallback_withoutError(self):
        str_bundle = StringBundle.get_bundle("zh-TW")
        self.assertEqual(
            str_bundle.get_string("openDir"),
            "\u958b\u555f\u76ee\u9304",
            "Fail to load the zh-TW bundle",
        )

    def test_setInvaleLocaleToEnv_printErrorMsg(self):
        prev_lc = os.environ.get("LC_ALL")
        prev_lang = os.environ.get("LANG")
        try:
            os.environ["LC_ALL"] = "UTF-8"
            os.environ["LANG"] = "UTF-8"
            str_bundle = StringBundle.get_bundle()
            self.assertEqual(
                str_bundle.get_string("openDir"),
                "Open Dir",
                "Fail to load the default bundle",
            )
        finally:
            if prev_lc is None:
                os.environ.pop("LC_ALL", None)
            else:
                os.environ["LC_ALL"] = prev_lc
            if prev_lang is None:
                os.environ.pop("LANG", None)
            else:
                os.environ["LANG"] = prev_lang


if __name__ == "__main__":
    unittest.main()
