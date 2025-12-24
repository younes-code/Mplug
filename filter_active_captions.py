import json
import re
from collections import defaultdict

# =========================
# CONFIG
# =========================
CAPTIONS_PATH = "captions.txt"
JSON_FILENAME = "global_temporal_selection.json"
OUTPUT_PATH = "captions_active_frames.txt"

# =========================
# HELPERS
# =========================
def timestamp_to_seconds(ts: str) -> float:
    ts = ts.replace(".jpg", "")
    parts = ts.split(":")

    if len(parts) == 3:  # MM:SS:ms
        m, s, ms = parts
        return int(m) * 60 + int(s) + int(ms) / 1000.0
    elif len(parts) == 4:  # HH:MM:SS:ms
        h, m, s, ms = parts
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    else:
        raise ValueError(f"Unsupported timestamp format: {ts}")


def is_inside_any_window(t: float, windows) -> bool:
    return any(start <= t <= end for start, end in windows)


def base_video_name(name: str) -> str:
    """
    Fighting041_01_x264 -> Fighting041_x264
    """
    return re.sub(r"_\d+_x264$", "_x264", name)

# =========================
# LOAD ACTIVE WINDOWS
# =========================
with open(JSON_FILENAME, "r") as f:
    data = json.load(f)

# Exact name → windows
windows_exact = {}

# Base name → list of windows from all splits
windows_by_base = defaultdict(list)

for entry in data:
    name = entry["video_name"]
    windows = entry.get("active_windows", [])

    windows_exact[name] = windows

    base = base_video_name(name)
    windows_by_base[base].extend(windows)

print(f"[INFO] Loaded windows for {len(windows_exact)} DVS videos")
print(f"[INFO] Found {len(windows_by_base)} base videos (with splits)")

# =========================
# PROCESS CAPTIONS
# =========================
total = 0
kept = 0
missing_video = 0
used_split_match = 0

with open(CAPTIONS_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in fin:
        if "##" not in line:
            continue

        total += 1

        left, caption = line.split("##", 1)
        left = left.strip()

        try:
            video_frame, timestamp = left.split()
            video_name = video_frame.split("/")[0]
            t_sec = timestamp_to_seconds(timestamp)
        except Exception as e:
            print(f"[WARN] Parse error: {line.strip()} ({e})")
            continue

        # 1️⃣ Exact match
        windows = windows_exact.get(video_name)

        # 2️⃣ Try split-based match
        if windows is None:
            base = video_name
            windows = windows_by_base.get(base)
            if windows:
                used_split_match += 1

        if not windows:
            missing_video += 1
            continue

        if is_inside_any_window(t_sec, windows):
            fout.write(line)
            kept += 1

# =========================
# REPORT
# =========================
print("===================================")
print("[DONE]")
print(f"Total captions           : {total}")
print(f"Active captions          : {kept}")
print(f"Recovered via splits     : {used_split_match}")
print(f"Missing video names      : {missing_video}")
print(f"Saved to                 : {OUTPUT_PATH}")
print("===================================")
