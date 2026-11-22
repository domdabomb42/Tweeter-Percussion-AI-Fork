# Exact Tweeter session time shifts

This directory creates new native Tweeter `.txt` sessions by changing where
each existing 64-slot tracker pattern begins. It adds no notes and changes no
instrument settings. The inputs are the 51 original files in `../sessions/`;
they are never written to. The generated files are **not yet used by any
training script**.

## Run

From this directory, with Python 3.10 or newer:

```powershell
python augment.py --dry-run
python augment.py
python -m unittest discover -s tests -v
```

The dry run reads all inputs and reports mathematically valid offsets without
writing a manifest or generated files. The normal run writes
`generated/<source-id>/<source-id>_shift_<00..63>.txt` and `manifest.json`.
Offset zero is a byte-for-byte copy of the source. Each run is deterministic;
rerunning replaces only its own generated destination files and manifest.
The manifest gives every file's original name, collision-safe source ID,
reference loop, offset, exact time shift, each track's loop and integer step
shift, source/output hashes, and validation result. It also lists every
skipped offset and the track-specific reason.

## Native format and algorithm

Tweeter stores records in a Pd `coll` file. A record's coll key is an ordering
key, **not** musical time. Each instrument has 64 `tracker-data` positions:
pitch zero is a rest, and a nonzero value is an active note. Matching
`tracker-char` records hold the displayed keyboard cell. Each instrument has
its own `loop-length` in milliseconds; step `p` occurs at `p * loop_ms / 64`
within that track's cycle. See `../core/L2OrkTweet.pd` and
`../../CustomExtractor/README.md` for the verified native interpretation.

For each source, `L_ref` is the longest saved valid track loop. At reference
offset `k`, the exact common time shift is `T = k * L_ref / 64`. A track with
loop `L_i` requires `s_i = 64*T/L_i = k*L_ref/L_i` tracker steps. The script
generates the offset only if **every track's** `s_i` is an integer and every
track has a known loop length. It then changes each native `tracker-data` and
`tracker-char` position `p` to `(p - s_i) mod 64`, including explicit rests.
It changes only the slot atom, preserving all record keys, other values and ordering.
The arithmetic uses exact rational numbers made from the native decimal
values; fractional shifts are never rounded. Loop lengths and note durations
are not stretched.

CustomExtractor's high-level native exporter regenerates display cells from
pitch edits. To retain every original display cell and explicit rest exactly,
this script uses CustomExtractor's native parser, changes only slot atoms,
exports with its atomic writer, then validates and reimports with
CustomExtractor.

The saved `tracker-time` triple contains the next playback step, remaining
delay and running flag. It stays fixed: rotating the **content** relative to
this fixed transport phase is what changes the starting musical point.
Rotating both content and cursor would cancel the intended time shift. Other
playback/editor flags such as `insert`, `delete` and `hover` are not tracker
slot addresses and remain unchanged. Legacy slot-64 records are outside the
active 0..63 tracker and remain untouched. CustomExtractor retains them as
native history but does not treat them as notes.

## Validation and limits

CustomExtractor parses and validates each source and each generated file. The
script checks every track ID/order, all non-note settings and global settings,
note counts and pitches, the exact expected position mapping, equal actual
time shifts modulo each track's loop, inverse rotation, every native
note/rest/display record, unchanged unrelated records, export bytes and
reimported structure. Source hashes are checked again after generation.
Failures are recorded; failed destination files are removed. Unit tests cover
equal and different loop lengths, 2x and 4x loop speeds, fractional shifts,
wraparound, empty tracks, repeated rhythms, missing loops and deterministic
reruns.

The current 51-file corpus has 3,264 possible source/offset combinations.
The dry run found **1,836 exact candidates** and **1,428 skipped offsets**.
The largest source limitation is fractional step shifts caused by differing
loop ratios. One source has tracks with no saved loop length, so only its
offset-zero copy is safe. The final manifest records the actual number of
generated and validated files.

This is an exact rotation of the saved note grid, not a snapshot of sounding
audio. Reverb/echo tails and the live audio engine's current phase are not
stored as restartable buffers in these files. Native loading or later
transport synchronization can also change the live playback cursor. The
files preserve the saved synthesis/effect settings, but the script does not
claim byte-identical rendered audio after a time shift.

**Future training must split by original session before adding these files.**
Every augmented child must stay in the same train, validation or test
partition as its parent; otherwise the model can see the same composition on
both sides of an evaluation split. Existing splits and training scripts are
unchanged by this task.
