"""Integration tests against the installed native Pd-L2Ork executable."""
from pathlib import Path
import tempfile
import unittest
from custom_extractor import Composition, add_track, export_tweeter
from custom_extractor.native_check import check_native, find_pd
from test_custom_extractor import fixture


@unittest.skipUnless(find_pd(), "Pd-L2Ork executable unavailable; playback is not simulated")
class NativeCompatibilityTests(unittest.TestCase):
    def test_native_synthetic_all_fields_and_escaped_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all-fields.txt"
            path.write_bytes(fixture().encode("utf-8"))
            result = check_native(path)
            self.assertEqual(result["status"], "passed")
            self.assertGreater(result["live_atom_types_verified"], 0)

    def test_native_generated_track_and_numeric_keyboard_symbol(self):
        c = Composition()
        add_track(c, 14, {"carrier": "square", "modulator": "sine", "inst-name": "Added Track", "loop-length": 2000, "note-dur": 300, "octave": 0}, [{"step": 0, "pitch": 66}, {"step": 1, "pitch": 60.5}, {"step": 2, "pitch": 130.25}, {"step": 3, "pitch": -12}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.txt"
            export_tweeter(c, path)
            result = check_native(path)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["pd_writer_numeric_symbol_losses"], 1)
