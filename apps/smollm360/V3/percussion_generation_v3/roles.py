"""Audited roles from native inst-name metadata, never MIDI pitch/channel rules.

The lists are exact names seen in the bundled Tweeter sessions. A new or
unclear name is unknown until reviewed; it is never silently made a negative.
"""

PERCUSSION = frozenset({
    " Crotales and Hi-Hat", "Crotales and Hi-Hat", "80s Toms - Caleb",
    "Acoustikick - Caleb", "Acoustitom - Caleb & Ico", "Better Hi-Hat (m)",
    "clap", "clap - Caleb", "Claptastic", "electrikick",
    "electrikick - Caleb", "electrikick 2 - Caleb", "Explosive Kick",
    "Ghost Snare - Caleb", "Ghost-Toms", "Hi-Hat-ish", "Kick'n'Tom",
    "Kinda Hi-Hat (m)", "Laser Snare - Caleb", "New Snare",
    "Percs Texture", "Pinksnare", "shaker - Caleb", "Sine Kick",
    "Slow-Ghost-Toms - Caleb", "Snare ARG", "snare body 3.5 - Caleb",
    "snare noise 3 - Caleb", "Snare, where art thou?",
    "Snarelicious by Ico", "Totally Hi-Hat", "Z Bass Drum",
    "Z Drums by Caden Vandervort", "Zclap",
})

# Exact metadata naming a pitched voice or non-drum effect. Ambiguous names
# are intentionally absent from both lists.
NON_PERCUSSION = frozenset({
    "(h)arpy by Caleb Bittenbender", "8-bit Wind", "Arg Bass",
    "Bagpipe-Like Lead", "Bass", "Bassic - Caleb", "Bassthing",
    "Bell - tanner", "boinky bassline", "Boquita BAss",
    "chord voice 1", "chord voice 2", "Clarinet-ish echo pad",
    "Cosmic Whistle", "Deep Bass", "Deep Bass Higher Double",
    "Deep dark synthy bass", "Deep Drone", "delayed melody sine",
    "Digi-rimba", "Digixylo - Caleb", "Distant Voice",
    "Distorted Sub by Jared Zimmermann", "DoubleBassLicious",
    "Drone Bass Extended", "Echo Digi-rimba", "echoey sine bass",
    "High Bass Arpeggios", "House Bass - Andrew Wickman",
    "Jelly Sub", "Lead", "Lead Arpeggios", "Long Lead",
    "Long Overlapping Lead", "Marimba_Dialing", "multilayer echo pad",
    "new drone", "pad high melody", "padcito", "Pulsing_Drone",
    "Pure Sine Bass", "Resonant Bass by Jacob Smith", "ROll bass - tanner",
    "sadlead", "Saw ish - Alex", "Saw ish - Alex Turse",
    "Simple Bass", "Slow Attack - Andrew Wickman",
    "Space Lead", "Stasis Lead 48k", "Static_Bass", "Stratosphere Pad",
    "Sweeping pad", "Sweet sine pad", "Synth Bass", "trumpet interjection",
    "underwater melody", "vinyl whistle - Caleb", "whistle",
    "Whistle by (TeeJay95)", "Wooden Flute",
    "Zither Forever", "Zithery Reverberant Attack Pad",
})

# These names do not prove a drum role or a melodic role by themselves.
# "808" might be a bass, "Steel drum" is pitched, and sound-effect names
# can be played rhythmically; none is assigned an inferred role.


def role(track):
    name = track["instrument"]["name"]
    if not track["notes"]:
        return "silent"
    if name in PERCUSSION:
        return "percussion"
    if name in NON_PERCUSSION:
        return "non_percussion"
    return "ambiguous"
