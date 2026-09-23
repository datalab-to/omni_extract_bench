"""Assemble the captured frames into one looping GIF.

Frames are shot at 2x and downscaled here, which supersamples the text. Every frame is
then mapped onto ONE shared palette: a GIF stores a local palette per frame otherwise, and
a shared one also lets Pillow write only the rectangle that changed between frames.
"""
import json, sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).parent
SRC = HERE / "frames"
OUT = HERE / "scoring.gif"
W, H = 1100, 560
COLORS = int(sys.argv[1]) if len(sys.argv) > 1 else 224

paths = sorted(SRC.glob("f*.png"))
delays = json.loads((SRC / "delays.json").read_text())
assert len(paths) == len(delays), f"{len(paths)} frames vs {len(delays)} delays"

frames = [Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS) for p in paths]

# NEAREST keeps the swatches to real pixel colours, not blends from a smooth resize.
tile = (W // 4, H // 4)
master = Image.new("RGB", (tile[0], tile[1] * len(frames)))
for i, f in enumerate(frames):
    master.paste(f.resize(tile, Image.NEAREST), (0, i * tile[1]))
palette = master.quantize(colors=COLORS, method=Image.MEDIANCUT)

quantised = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

quantised[0].save(
    OUT, save_all=True, append_images=quantised[1:],
    duration=delays, loop=0, disposal=1, optimize=True,
)

total = sum(delays) / 1000
print(f"{OUT.name}: {OUT.stat().st_size/1e6:.2f} MB, {len(frames)} frames, "
      f"{total:.1f}s, {W}x{H}, {COLORS} colours")
