"""
Generates the comprehensive Medi-Script-AI block diagram: a single,
information-dense figure showing the internal composition of every
stage (not just the coarse 3-box flow), intended as the paper's primary
architecture figure. Designed to span both columns (figure*).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42

# ---------------------------------------------------------------- palette
NAVY      = "#1f3a5f"
NAVY_FILL = "#e8eef5"
BLUE      = "#3f6fa8"
BLUE_FILL = "#eaf1f8"
TEAL      = "#2f7a7d"
TEAL_FILL = "#e7f3f3"
GREEN     = "#2f7d5a"
GREEN_FILL= "#e6f2ec"
INK       = "#16191d"
MUTED     = "#5b6472"
SHADOW    = "#c7ccd3"
WHITE     = "#ffffff"
CHIP_FILL = "#f4f5f7"
CHIP_EDGE = "#c6cbd2"

fig, ax = plt.subplots(figsize=(9.8, 4.45))
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)
ax.axis("off")

# ---------------------------------------------------------------- helpers
def shadow(x0, y0, w, h, r=1.1):
    ax.add_patch(FancyBboxPatch((x0 + 0.45, y0 - 0.45), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=0, facecolor=SHADOW, alpha=0.55, zorder=1))

def panel(x0, y0, w, h, title, title_color, body_fill, r=1.1, title_h=6.2):
    shadow(x0, y0, w, h, r)
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=1.4, edgecolor=title_color, facecolor=body_fill, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, y0 + h - title_h), w, title_h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=0, facecolor=title_color, zorder=3))
    ax.add_patch(Rectangle((x0, y0 + h - title_h), w, title_h * 0.5,
                 facecolor=title_color, edgecolor="none", zorder=3))
    ax.text(x0 + w / 2, y0 + h - title_h / 2, title, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color="white", zorder=4, linespacing=1.6)

def subbox(x0, y0, w, h, text, edge, fill=WHITE, fontsize=7.3, r=0.55):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=5))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=6, linespacing=1.35)

def chip(xc, yc, text, w, fontsize=6.3):
    h = 2.5
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=1.25",
                 linewidth=0.9, edgecolor=CHIP_EDGE, facecolor=CHIP_FILL, zorder=6))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fontsize,
            color=MUTED, zorder=7, style="italic")

def varrow(x, y0, y1, color=MUTED, lw=1.3):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=9, linewidth=lw, color=color, zorder=5))

def harrow(x0, x1, y, color=INK, lw=1.5, label=None, label_dy=2.0, fontsize=6.3):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=11, linewidth=lw, color=color, zorder=9))
    if label:
        ax.text((x0 + x1) / 2, y + label_dy, label, ha="center", va="bottom",
                fontsize=fontsize, color=MUTED, style="italic", zorder=10,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

def diag(x0, y0, x1, y1, color=MUTED, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-",
                 mutation_scale=8, linewidth=lw, color=color, zorder=4))

# ============================================================== layout
IN_X, IN_W = 1, 8
S1_X, S1_W = 13, 19
S2_X, S2_W = 36, 31
S3_X, S3_W = 71, 18
OUT_X, OUT_W = 93, 7

# ============================================================== INPUT
iy0, ih = 12, 18
shadow(IN_X, iy0, IN_W, ih, r=1.0)
ax.add_patch(FancyBboxPatch((IN_X, iy0), IN_W, ih, boxstyle="round,pad=0,rounding_size=1.0",
             linewidth=1.4, edgecolor=INK, facecolor="white", zorder=2))
gx, gy = IN_X + IN_W / 2, iy0 + ih - 6.3
ax.add_patch(FancyBboxPatch((gx - 3, gy - 2.1), 6, 4.2, boxstyle="round,pad=0,rounding_size=0.35",
             linewidth=1.0, edgecolor=MUTED, facecolor=CHIP_FILL, zorder=3))
ax.add_patch(Circle((gx - 1.5, gy + 0.7), 0.55, facecolor="white", edgecolor=MUTED, linewidth=0.8, zorder=4))
tri = mpatches.Polygon([[gx - 2.2, gy - 1.3], [gx - 0.2, gy + 0.9], [gx + 1.0, gy - 0.1], [gx + 2.3, gy - 1.3]],
                        closed=True, facecolor=MUTED, edgecolor="none", zorder=4)
ax.add_patch(tri)
ax.text(IN_X + IN_W / 2, iy0 + 4.6, "Input", ha="center", va="center", fontsize=8.8, fontweight="bold", color=INK)
ax.text(IN_X + IN_W / 2, iy0 + 1.9, "handwritten\nprescription\nphoto", ha="center", va="center",
        fontsize=6.0, color=MUTED, linespacing=1.3)

harrow(IN_X + IN_W, S1_X, 21)

# ============================================================== STAGE 1
s1y, s1h = 2, 38
panel(S1_X, s1y, S1_W, s1h, "STAGE 1\nFIELD LOCALIZATION", NAVY, NAVY_FILL)

bx0, bw = S1_X + 1.4, S1_W - 2.8
subbox(bx0, 27.0, bw, 6.0, "Backbone\nCSPDarknet-style feature\nextraction", NAVY, fontsize=6.5)
varrow(S1_X + S1_W / 2, 27.0, 24.9)
subbox(bx0, 18.4, bw, 6.0, "Neck\nPAN-FPN multi-scale\nfeature fusion", NAVY, fontsize=6.5)
varrow(S1_X + S1_W / 2, 18.4, 16.3)
subbox(bx0, 9.8, bw, 6.0, "Detection Head\n\u2192 6 field classes\n(box + class + conf.)", NAVY, fontsize=6.5)

chip(S1_X + S1_W / 2, 6.0, "YOLOv8-n \u00b7 3.0M params \u00b7 8.1 GFLOPs", S1_W - 1.2, fontsize=5.7)
chip(S1_X + S1_W / 2, 3.0, "2.5 ms / crop (Tesla T4)", S1_W - 1.2, fontsize=5.7)

harrow(S1_X + S1_W, S2_X, 21, label="per-class crops")

# ============================================================== STAGE 2
s2y, s2h = 2, 38
panel(S2_X, s2y, S2_W, s2h, "STAGE 2\nHANDWRITING RECOGNITION", BLUE, BLUE_FILL)

ax.text(S2_X + 1.3, 31.0, "TrOCR", fontsize=7.4, fontweight="bold", color=NAVY, ha="left", va="center")
subbox(S2_X + 1.3, 25.2, 10.2, 5.2, "ViT patch\nencoder", BLUE, fontsize=6.4)
harrow(S2_X + 11.7, S2_X + 13.3, 27.8, color=MUTED, lw=1.0)
subbox(S2_X + 13.5, 25.2, 15.8, 5.2, "Autoregressive decoder\nbeam=4, lexicon-constrained", BLUE, fontsize=6.0)

ax.text(S2_X + 1.3, 15.0, "EasyOCR", fontsize=7.4, fontweight="bold", color=TEAL, ha="left", va="center")
subbox(S2_X + 1.3, 9.6, 28.0, 5.2, "Independent CNN + LSTM + CTC\nreader (cross-check pass)", TEAL,
       fill=TEAL_FILL, fontsize=6.2)

mx, my = S2_X + S2_W - 3.4, 19.2
diag(S2_X + 21.0, 25.2, mx - 1.2, my + 1.5)
diag(S2_X + 21.0, 12.2, mx - 1.2, my - 1.5)
ax.add_patch(Circle((mx, my), 1.3, facecolor="white", edgecolor=INK, linewidth=1.2, zorder=6))
ax.text(mx, my, "\u2248", ha="center", va="center", fontsize=8.5, color=INK, zorder=7)
ax.text(mx, my + 3.0, "RapidFuzz\nagreement", ha="center", va="center", fontsize=5.6,
        color=MUTED, style="italic", linespacing=1.15)

chip(S2_X + S2_W / 2, 3.0, "4 ranked candidates + confidence label (High / Low)", S2_W - 1.2, fontsize=5.8)

harrow(S2_X + S2_W, S3_X, 21, label="text + confidence")

# ============================================================== STAGE 3
s3y, s3h = 2, 38
panel(S3_X, s3y, S3_W, s3h, "STAGE 3\nPOST-PROCESSING", GREEN, GREEN_FILL)

cx0, cw = S3_X + 1.3, S3_W - 2.6
subbox(cx0, 28.6, cw, 5.4, "Trie lexicon\n1,142 drugs + 35 symptoms", GREEN, fontsize=6.2)
varrow(S3_X + S3_W / 2, 28.6, 26.6)
subbox(cx0, 21.2, cw, 5.4, "RapidFuzz correction\n\u2265 80% similarity", GREEN, fontsize=6.2)
varrow(S3_X + S3_W / 2, 21.2, 19.2)
subbox(cx0, 13.8, cw, 5.4, "Bigram / unigram LM\nrescores 4 beams", GREEN, fontsize=6.2)
varrow(S3_X + S3_W / 2, 13.8, 11.8)
subbox(cx0, 6.0, cw, 5.4, "Confidence gate\nbelow threshold \u2192 field empty", GREEN, fontsize=6.0)

harrow(S3_X + S3_W, OUT_X, 21)

# ============================================================== OUTPUT
oy0, oh = 12, 18
shadow(OUT_X, oy0, OUT_W, oh, r=1.0)
ax.add_patch(FancyBboxPatch((OUT_X, oy0), OUT_W, oh, boxstyle="round,pad=0,rounding_size=1.0",
             linewidth=1.4, edgecolor=INK, facecolor="white", zorder=2))
ax.text(OUT_X + OUT_W / 2, oy0 + oh - 6.3, "{ }", ha="center", va="center", fontsize=16,
        fontweight="bold", color=GREEN, zorder=4)
ax.text(OUT_X + OUT_W / 2, oy0 + 4.6, "Output", ha="center", va="center", fontsize=8.8,
        fontweight="bold", color=INK)
ax.text(OUT_X + OUT_W / 2, oy0 + 1.9, "structured\nJSON record", ha="center", va="center",
        fontsize=6.4, color=MUTED, linespacing=1.3)

plt.tight_layout(pad=0.2)
plt.savefig("model_block_diagram.pdf", bbox_inches="tight", pad_inches=0.06)
print("saved model_block_diagram.pdf")
