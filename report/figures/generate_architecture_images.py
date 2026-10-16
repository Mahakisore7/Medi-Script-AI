"""
Generates crisp, high-resolution PNG images for all 5 architecture diagrams
in Medi-Script-AI_Architecture_Diagram.md so they render in ANY markdown viewer,
Overleaf, or browser without requiring a Mermaid extension.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42

NAVY = "#1f3a5f"
BLUE = "#2563eb"
LIGHT_BLUE = "#dbeafe"
TEAL = "#0d9488"
LIGHT_TEAL = "#ccfbf1"
AMBER = "#d97706"
LIGHT_AMBER = "#fef3c7"
PURPLE = "#7c3aed"
LIGHT_PURPLE = "#ede9fe"
GREEN = "#16a34a"
LIGHT_GREEN = "#dcfce7"
GRAY = "#4b5563"
LIGHT_GRAY = "#f3f4f6"
BORDER = "#9ca3af"


def draw_box(
    ax, x, y, w, h, title, subtitle="", bg=LIGHT_BLUE, border=BLUE, radius=0.03
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=bg,
        edgecolor=border,
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=border,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=8,
            color=GRAY,
            zorder=3,
        )
    else:
        ax.text(
            x + w / 2,
            y + h * 0.50,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=border,
            zorder=3,
        )


def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5),
        zorder=4,
    )
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mid_x,
            mid_y + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=GRAY,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8
            ),
            zorder=5,
        )


# ==================== 1. MASTER PIPELINE DIAGRAM ====================
def make_master_diagram(out_path):
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "Medi-Script-AI: Complete End-to-End Master Pipeline",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        0.5,
        0.92,
        "Input Prescription (H=1024, W=768, 3) -> Two-Lane Processing -> Verified Clinical JSON",
        ha="center",
        fontsize=9.5,
        color=GRAY,
    )

    # Input Photo
    draw_box(
        ax,
        0.02,
        0.52,
        0.13,
        0.18,
        "INPUT PHOTO",
        "(1024, 768, 3)\nuint8 BGR",
        bg="#f8fafc",
        border=NAVY,
    )

    # Stage 1 YOLO
    draw_box(
        ax,
        0.20,
        0.65,
        0.17,
        0.18,
        "STAGE 1: YOLOv8-n",
        "Letterbox -> (1,3,640,480)\n5 Boxes: B (5, 4)",
        bg=LIGHT_BLUE,
        border=BLUE,
    )
    draw_arrow(ax, 0.15, 0.65, 0.20, 0.70, "Unchanged P")

    # Lane A: Medicine Line Processing
    draw_box(
        ax,
        0.42,
        0.65,
        0.15,
        0.18,
        "STAGE 2a: ENHANCE",
        "x2 -> CLAHE -> Threshold\nCrop 1: (128, 640, 3)",
        bg=LIGHT_TEAL,
        border=TEAL,
    )
    draw_arrow(ax, 0.37, 0.74, 0.42, 0.74, "Crops (h,w,3)")

    draw_box(
        ax,
        0.62,
        0.73,
        0.15,
        0.14,
        "TrOCR-Large (Tier 2)",
        "ViT (24L) + RoBERTa (12L)\nresize (1,3,384,384)",
        bg=LIGHT_PURPLE,
        border=PURPLE,
    )
    draw_box(
        ax,
        0.62,
        0.55,
        0.15,
        0.14,
        "Gemini Vision (Tier 1)",
        "High-speed zero-shot\nFallback to TrOCR",
        bg=LIGHT_AMBER,
        border=AMBER,
    )
    draw_arrow(ax, 0.57, 0.74, 0.62, 0.80, "Enhanced Crop")
    draw_arrow(ax, 0.57, 0.74, 0.62, 0.62, "Enhanced Crop")

    # Stage 3 Verify
    draw_box(
        ax,
        0.81,
        0.65,
        0.16,
        0.18,
        "STAGE 3: VERIFY",
        "TF-IDF n-grams x Lexicon\n384,604 Drugs (cos >= 0.60)",
        bg=LIGHT_GREEN,
        border=GREEN,
    )
    draw_arrow(ax, 0.77, 0.80, 0.81, 0.76, "Raw String")
    draw_arrow(ax, 0.77, 0.62, 0.81, 0.72, "Raw String")

    # Lane B: Global Page Processing
    draw_box(
        ax,
        0.20,
        0.20,
        0.17,
        0.18,
        "STAGE 4a: EasyOCR",
        "Full Page Detection\nN text fragments",
        bg=LIGHT_GRAY,
        border=GRAY,
    )
    draw_arrow(ax, 0.15, 0.55, 0.20, 0.30, "Unchanged P")

    draw_box(
        ax,
        0.42,
        0.20,
        0.17,
        0.18,
        "STAGE 4b: SPATIAL RULES",
        "IoA <= 0.10 Filter vs YOLO\nDoctor / Hospital / Date",
        bg=LIGHT_GRAY,
        border=GRAY,
    )
    draw_arrow(ax, 0.37, 0.29, 0.42, 0.29, "Fragments (N)")
    draw_arrow(ax, 0.28, 0.65, 0.45, 0.38, "YOLO Boxes (IoA filter)")

    draw_box(
        ax,
        0.64,
        0.20,
        0.16,
        0.18,
        "STAGE 4c: ANCHORS",
        "Horizontal search:\nName, Age, Sex, Date",
        bg=LIGHT_GRAY,
        border=GRAY,
    )
    draw_arrow(ax, 0.59, 0.29, 0.64, 0.29, "Classified")

    # Stage 5 LLM & Final Output
    draw_box(
        ax,
        0.84,
        0.20,
        0.14,
        0.28,
        "STAGE 5: LLM",
        "Dosage Advice translation\n(1-0-1 -> morning/night)\nFills residual 'Other'",
        bg=LIGHT_AMBER,
        border=AMBER,
    )
    draw_arrow(ax, 0.89, 0.65, 0.89, 0.48, "Verified Drugs")
    draw_arrow(ax, 0.80, 0.29, 0.84, 0.29, "Patient Data")

    # Final Record Output
    draw_box(
        ax,
        0.45,
        0.02,
        0.28,
        0.11,
        "FINAL STRUCTURED JSON",
        "7 Keys: Medical Text, Patient, Date, Doctor...",
        bg="#1e293b",
        border="#0f172a",
    )
    draw_arrow(ax, 0.84, 0.25, 0.73, 0.08, "Structured Output")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ==================== 2. CROP ENHANCEMENT DIAGRAM ====================
def make_enhance_diagram(out_path):
    fig, ax = plt.subplots(figsize=(11, 2.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.88,
        "Stage 2a: Crop Enhancement Pipeline (Dimensions at Each Step)",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )

    steps = [
        ("Raw Crop", "(64, 320, 3)\nuint8 BGR"),
        ("x2 Scale", "(128, 640, 3)\nLanczos-4"),
        ("Grayscale", "(128, 640)\nuint8 gray"),
        ("CLAHE", "(128, 640)\n4x4 tiles, clip=2.5"),
        ("Threshold", "(128, 640)\nOtsu / Adaptive"),
        ("Final Crop", "(128, 640, 3)\n3-channel binary"),
    ]

    x = 0.02
    w = 0.13
    for i, (title, sub) in enumerate(steps):
        bg = LIGHT_TEAL if i == len(steps) - 1 else LIGHT_BLUE
        border = TEAL if i == len(steps) - 1 else BLUE
        draw_box(ax, x, 0.22, w, 0.48, title, sub, bg=bg, border=border)
        if i < len(steps) - 1:
            draw_arrow(ax, x + w, 0.46, x + w + 0.035, 0.46)
        x += w + 0.035

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ==================== 3. TrOCR TENSOR FLOW DIAGRAM ====================
def make_trocr_diagram(out_path):
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.90,
        "Stage 2b: TrOCR-Large Tensor Shapes & Internal Attention Flow",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )

    nodes = [
        ("Enhanced Crop", "(128, 640, 3)\nuint8"),
        ("pixel_values", "(1, 3, 384, 384)\nNorm float32"),
        ("576 Patches", "(1, 576, 768)\n16x16 grid"),
        ("ViT Encoder", "(1, 577, 1024)\n24 Layers (Top-4 Unfrozen)"),
        ("RoBERTa Decoder", "(2, t, 1024)\n12 Layers (Cross-Attention)"),
        ("Output String", "Beam Search (k=2)\n<= 24 tokens"),
    ]

    x = 0.02
    w = 0.135
    for i, (title, sub) in enumerate(nodes):
        bg = LIGHT_PURPLE if "Encoder" in title or "Decoder" in title else LIGHT_BLUE
        border = PURPLE if "Encoder" in title or "Decoder" in title else BLUE
        draw_box(ax, x, 0.24, w, 0.50, title, sub, bg=bg, border=border)
        if i < len(nodes) - 1:
            draw_arrow(ax, x + w, 0.49, x + w + 0.032, 0.49)
        x += w + 0.032

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p1 = os.path.join(script_dir, "arch_master_pipeline.png")
    p2 = os.path.join(script_dir, "arch_crop_enhancement.png")
    p3 = os.path.join(script_dir, "arch_trocr_tensor_flow.png")
    make_master_diagram(p1)
    make_enhance_diagram(p2)
    make_trocr_diagram(p3)
    print("Generated all architecture diagram PNGs successfully!")
