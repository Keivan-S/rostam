"""The packaged version and the runtime constant must agree.

`rostam.__version__` sat at 0.1.0 across the 0.1.1 release: anything reporting a
client version — a bug report, a user agent, a diagnostic — named a release that
was not the one running. Nothing caught it, because nothing compared the two.

This does. It reads pyproject.toml with a regex rather than a TOML parser
because the package supports Python 3.9 and `tomllib` arrived in 3.11; the field
it needs is one line and unambiguous.
"""

import os
import re
import unittest

import rostam

_PYPROJECT = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")


class VersionTest(unittest.TestCase):
    def test_runtime_version_matches_the_package_metadata(self):
        with open(_PYPROJECT, encoding="utf-8") as fh:
            text = fh.read()
        # The [project] table's own version, not a dependency pin.
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        self.assertIsNotNone(m, "no version field found in pyproject.toml")
        self.assertEqual(
            m.group(1),
            rostam.__version__,
            "pyproject.toml and rostam.__version__ disagree; bump both",
        )


if __name__ == "__main__":
    unittest.main()
