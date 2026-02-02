# evaluation/config.py

"""
Define evaluation suites by mapping a suite name to a list of game_ids.
"""

EVALUATION_GAMES = {
    # A small suite for quick debugging and testing
    "debug_suite": [
        "vc33-9851e02b",
    ],
    # Single game suite for testing non-as66 games
    "_suite": [
        "ls20-cb3b57cc",
    ],
    # Full 6-game evaluation suite
    "standard_suite": [
        "as66-821a4dcad9c2",
        "ls20-fa137e247ce6",
        "ft09-b8377d4b7815",
        "vc33-9851e02b",
        "lp85-d265526edbaa",
        "sp80-0605ab9e5b2a",
    ],
}
