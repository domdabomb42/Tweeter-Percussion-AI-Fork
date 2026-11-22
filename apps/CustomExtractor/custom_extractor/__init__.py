"""CustomExtractor: native Tweeter snapshots <-> editable JSON; no MIDI or ML."""
from .models import Composition
from .extractor import extract_tweeter, extract_text
from .exporter import export_tweeter, render_tweeter
from .json_adapter import save_json, load_json, dumps, loads
from .validator import validate, ValidationError
from .parser import FormatError
from .edits import set_parameter, set_parameters, add_note, remove_note, set_loop_length, add_track

extract = extract_tweeter
export = export_tweeter

__all__ = ["Composition", "extract_tweeter", "extract_text", "export_tweeter", "render_tweeter",
           "save_json", "load_json", "dumps", "loads", "validate", "ValidationError", "FormatError",
           "set_parameter", "set_parameters", "add_note", "remove_note", "set_loop_length", "add_track", "extract", "export"]
