import os

# The reach envelope takes ~10 s to build; unit tests opt in explicitly.
os.environ.setdefault("DESKBUDDY_REACH_MAP", "0")
