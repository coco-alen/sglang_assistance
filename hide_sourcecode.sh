# pyarmor info               # shows valid license (trial or full)

# # 1) Paths
# SRC_DIR=/sgl-workspace/sglang
# SRC_FILE=$SRC_DIR/python/sglang/srt/multimodal/processors/internvl.py
# OUT=$SRC_DIR/python/sglang/srt/multimodal/processors/internvl_obf

# # 2) Generate obfuscated file (only this file)
# rm "$OUT"
# pyarmor gen -O "$OUT" "$SRC_FILE"

# # 3) Inspect output
# find "$OUT" -maxdepth 2 -type f

# # 4) Install PyArmor runtime so imports resolve
# SITE=/usr/local/lib/python3.12/dist-packages
# RUNTIME_DIR=$(find "$OUT" -maxdepth 1 -type d -name 'pyarmor_runtime_*' -print -quit)
# cp -a "$RUNTIME_DIR" "$SITE"/

# # 5) Backup and replace the original file
# cp -a "$SRC_FILE" "${SRC_FILE}.bak"
# cp -a "$OUT/internvl.py" "$SRC_FILE"
# chmod 0644 "$SRC_FILE"


pyarmor info               # shows valid license (trial or full)

# 1) Paths
SRC_DIR=/sgl-workspace/sglang
SRC_FILE=$SRC_DIR/python/sglang/srt/models/internvl_flash.py
OUT=$SRC_DIR/python/sglang/srt/models/internvl_flash_obf

# 2) Generate obfuscated file (only this file)
rm "$OUT"
pyarmor gen -O "$OUT" "$SRC_FILE"

# 3) Inspect output
find "$OUT" -maxdepth 2 -type f

# 4) Install PyArmor runtime so imports resolve
SITE=/usr/local/lib/python3.12/dist-packages
RUNTIME_DIR=$(find "$OUT" -maxdepth 1 -type d -name 'pyarmor_runtime_*' -print -quit)
cp -a "$RUNTIME_DIR" "$SITE"/

# 5) Backup and replace the original file
cp -a "$SRC_FILE" "${SRC_FILE}.bak"
cp -a "$OUT/internvl_flash.py" "$SRC_FILE"
chmod 0644 "$SRC_FILE"



# recover
# [ -f "${SRC_FILE}.bak" ] && mv -f "${SRC_FILE}.bak" "$SRC_FILE" && rm -rf "$SRC_DIR/__pycache__"
