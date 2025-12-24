# =========================
# VIDEO-LEVEL ANALYSIS (FINAL ANSWER)
# =========================

caption_videos = set(all_by_video.keys())
active_videos = set(active_by_video.keys())

# 1️⃣ Videos present in captions but absent from JSON
missing_videos = sorted(caption_videos - json_videos)

# 2️⃣ JSON videos that have NO active frames
json_no_active = sorted(json_videos - active_videos)

# 3️⃣ JSON videos WITH active frames
json_with_active = sorted(json_videos & active_videos)

# 4️⃣ Coverage
coverage = 100.0 * len(json_with_active) / len(json_videos)

print("\n=========== VIDEO-LEVEL SUMMARY ===========")
print(f"Videos in JSON                    : {len(json_videos)}")
print(f"Videos in captions                : {len(caption_videos)}")
print(f"Missing videos entirely           : {len(missing_videos)}")
print(f"JSON videos with no active frames : {len(json_no_active)}")
print(f"JSON videos with active frames    : {len(json_with_active)}")
print(f"Coverage of anomaly videos        : {coverage:.2f}%")
print("===========================================\n")

# =========================
# SAVE VIDEO LISTS
# =========================
with open("missing_videos.txt", "w") as f:
    for v in missing_videos:
        f.write(v + "\n")

with open("json_videos_no_active_frames.txt", "w") as f:
    for v in json_no_active:
        f.write(v + "\n")

with open("json_videos_with_active_frames.txt", "w") as f:
    for v in json_with_active:
        f.write(v + "\n")

print("[INFO] Video lists saved:")
print(" - missing_videos.txt")
print(" - json_videos_no_active_frames.txt")
print(" - json_videos_with_active_frames.txt")
