import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUT_DIR = r"c:\WAR-FL\Output\Figures"
os.makedirs(OUT_DIR, exist_ok=True)
DOC_DIR = r"c:\WAR-FL\Documentation\Figures"
os.makedirs(DOC_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'

# ==============================================================================
# 1. FIGURE 1: REVISED FL-WEAR ARCHITECTURE
# ==============================================================================
def draw_fig1():
    fig, ax = plt.subplots(figsize=(12.0, 5.6), dpi=300)
    ax.set_xlim(0, 122)
    ax.set_ylim(0, 56)
    ax.axis('off')
    
    # Title
    ax.text(61, 52.5, "Revised FL-WeAR Architecture: Physical Data Ownership (1 Welder = 1 Federated Client)",
            fontsize=12, fontweight='bold', ha='center', va='center', color='#0F172A')

    # Styles
    box_blue   = dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc="#F0F9FF", ec="#0284C7", lw=1.8)
    box_teal   = dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc="#F0FDFA", ec="#0D9488", lw=1.8)
    box_amber  = dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc="#FEFCE8", ec="#CA8A04", lw=1.8)
    box_rose   = dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc="#FFF1F2", ec="#E11D48", lw=1.8)
    box_purple = dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc="#FAF5FF", ec="#9333EA", lw=1.8)
    box_indigo = dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc="#EEF2FF", ec="#4F46E5", lw=1.8)
    box_slate  = dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc="#F8FAFC", ec="#475569", lw=1.8)

    # Worker Clients (Left Column)
    ax.text(10, 42, "Worker Client 1\n9-Axis IMU (50 Hz)\nLocal Raw Data $D_1$",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#0369A1', bbox=box_blue)
    ax.text(10, 27, "Worker Client $k$\n9-Axis IMU (50 Hz)\nLocal Raw Data $D_k$",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#0369A1', bbox=box_blue)
    ax.text(10, 12, "Worker Client 5\n9-Axis IMU (50 Hz)\nLocal Raw Data $D_5$",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#0369A1', bbox=box_blue)

    # Local Processing Engine Box
    ax.text(33, 27, "Local Processing Engine\n• Butterworth Bandpass & Z-score\n• Sliding Window ($W=100, 2.0$ s)\n• CNN-LSTM Local Optimization\n• DP-SGD ($L_2$ Clip $C=1.0, \\sigma=1.45$)",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color='#0F766E', bbox=box_teal)

    # Authenticated Encrypted Transport
    ax.text(56, 27, "Secure Transport\n• Authenticated TLS 1.3\n• AES-256 GCM Payload\n• Ephemeral Key Rotation\n• Gradients Only (No Raw Data)",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color='#A16207', bbox=box_amber)

    # Aggregation Server
    ax.text(78, 27, "Aggregation Server\n• Client Selection & Sync\n• Sample-Weighted Aggregation:\n  $w^{t+1} = \\sum_{k=1}^K \\frac{n_k}{n} w_k^{t+1}$\n• FedAvg / FedProx / SCAFFOLD",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color='#BE123C', bbox=box_rose)

    # Global Model Box
    ax.text(99, 27, "Global Welder\nHAR Model\n$w^{t+1}$",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#3730A3', bbox=box_indigo)

    # Top Right: Privacy Accounting & Audit Log
    ax.text(108, 43, "Privacy Accounting & Audit Log\n• Per-Client RDP Moments Accountant\n• Cumulative Budget $\\varepsilon = 2.98 \\leq 3.0$\n• Empirical Yeom MIA Audit (51.27%)",
            ha='center', va='center', fontsize=7.6, fontweight='bold', color='#7E22CE', bbox=box_purple)

    # Bottom Right: Management Output
    ax.text(108, 11, "Management & Industrial Output\n• Ergonomics & Posture Summary\n• Welder Activity Statistics\n• Zero Raw Data Centralization",
            ha='center', va='center', fontsize=7.6, fontweight='bold', color='#334155', bbox=box_slate)

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color="#475569", lw=1.6, mutation_scale=14)
    ax.annotate("", xy=(22.5, 30), xytext=(19.0, 42), arrowprops=arrow_kw)
    ax.annotate("", xy=(22.5, 27), xytext=(19.0, 27), arrowprops=arrow_kw)
    ax.annotate("", xy=(22.5, 24), xytext=(19.0, 12), arrowprops=arrow_kw)

    ax.annotate("", xy=(46.5, 27), xytext=(43.5, 27), arrowprops=arrow_kw)
    ax.annotate("", xy=(68.5, 27), xytext=(65.5, 27), arrowprops=arrow_kw)
    ax.annotate("", xy=(91.5, 27), xytext=(88.5, 27), arrowprops=arrow_kw)

    ax.annotate("", xy=(101, 38), xytext=(99, 32), arrowprops=arrow_kw)
    ax.annotate("", xy=(101, 16), xytext=(99, 22), arrowprops=arrow_kw)

    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "fig_1_system_architecture.png")
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(DOC_DIR, "fig_1_system_architecture.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Fig 1 regenerated -> {p1}")

# ==============================================================================
# 2. FIGURE 2: LEAKAGE-FREE EVALUATION PIPELINE
# ==============================================================================
def draw_fig2():
    fig, ax = plt.subplots(figsize=(12.0, 4.8), dpi=300)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 48)
    ax.axis('off')

    # Title
    ax.text(60, 44.5, "Leakage-Free Evaluation Pipeline: Session-Level Partitioning Before Window Segmentation",
            fontsize=12, fontweight='bold', ha='center', va='center', color='#0F172A')

    # Boxes definition
    boxes = [
        ("Step 1: Continuous IMU\nRaw Data Acquisition\n\n• 5 Professional Welders\n• 4 Sessions per Welder\n• 9-Axis IMU @ 50 Hz\n• 136,000 Total Samples",
         12, 22, "#EFF6FF", "#3B82F6", "#1E40AF"),
        ("Step 2: Session-Level\nSubject Partitioning\n\n• Leave-One-Subject-Out\n• 4 Welders -> Train Set\n• 1 Welder -> Holdout Test\n• Zero Temporal Overlap!",
         36, 22, "#ECFDF5", "#10B981", "#065F46"),
        ("Step 3: Signal Preprocessing\n& Filtering\n\n• 4th-Order Butterworth\n  Bandpass (0.5 - 20 Hz)\n• Z-score Standardization\n  Fit on Train Set Only",
         60, 22, "#FEF3C7", "#F59E0B", "#92400E"),
        ("Step 4: Sliding-Window\nSegmentation\n\n• Window $W = 100$ (2.0 s)\n• Step = 50 samples (1.0 s)\n• 50% Overlap within Set\n• Total 2,719 3D Windows",
         84, 22, "#F3E8FF", "#A855F7", "#6B21A8"),
        ("Step 5: Federated Training\n& Zero-Leakage Eval\n\n• Local DP-SGD on $D_{train}$\n• FedAvg Model Sync\n• Test on Unseen Subject\n• Unbiased Metric Eval",
         108, 22, "#FFE4E6", "#F43F5E", "#9F1239")
    ]

    for title, cx, cy, fc, ec, tc in boxes:
        bbox_style = dict(boxstyle="round,pad=0.6,rounding_size=0.8", fc=fc, ec=ec, lw=1.8)
        ax.text(cx, cy, title, ha='center', va='center', fontsize=7.8, fontweight='bold', color=tc, bbox=bbox_style)

    # Arrows between consecutive steps
    arrow_kw = dict(arrowstyle="-|>", color="#334155", lw=2.0, mutation_scale=16)
    ax.annotate("", xy=(24.5, 22), xytext=(22.5, 22), arrowprops=arrow_kw)
    ax.annotate("", xy=(48.5, 22), xytext=(46.5, 22), arrowprops=arrow_kw)
    ax.annotate("", xy=(72.5, 22), xytext=(70.5, 22), arrowprops=arrow_kw)
    ax.annotate("", xy=(96.5, 22), xytext=(94.5, 22), arrowprops=arrow_kw)

    # Bottom Callout: Key Anti-Leakage Guarantee
    callout = dict(boxstyle="round,pad=0.4", fc="#FEF2F2", ec="#EF4444", lw=1.2)
    ax.text(60, 4.0, "★ Key Anti-Leakage Guarantee: Session partitioning occurs BEFORE sliding-window creation to prevent adjacent-window data leakage.",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#B91C1C', bbox=callout)

    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "fig_2_fl_architecture.png")
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(DOC_DIR, "fig_2_fl_architecture.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Fig 2 regenerated -> {p2}")

# ==============================================================================
# 3. FIGURE 3: HYBRID CNN-LSTM LOCAL CLASSIFIER ARCHITECTURE
# ==============================================================================
def draw_fig3():
    fig, ax = plt.subplots(figsize=(12.0, 5.2), dpi=300)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # Title
    ax.text(60, 46.5, "Hybrid CNN-LSTM Classifier Network Topology with Layer Dimensions & Hyperparameters",
            fontsize=12, fontweight='bold', ha='center', va='center', color='#0F172A')

    # Layer nodes - Row 1 (Feature Extraction)
    r1_nodes = [
        ("Input Segment\n$W \\times F = 100 \\times 9$\n(Acc, Gyro, Mag)", 11, 32, "#E2E8F0", "#475569", "#1E293B"),
        ("Conv1D - Block 1\n64 Filters, $k=3$, ReLU\nBatchNorm1D\nOutput: $100 \\times 64$", 31, 32, "#DBEAFE", "#2563EB", "#1E40AF"),
        ("MaxPool1D - 1\nKernel=2, Stride=2\n\nOutput: $50 \\times 64$", 51, 32, "#E0E7FF", "#4F46E5", "#3730A3"),
        ("Conv1D - Block 2\n128 Filters, $k=3$, ReLU\nBatchNorm1D\nOutput: $50 \\times 128$", 71, 32, "#DBEAFE", "#2563EB", "#1E40AF"),
        ("MaxPool1D - 2\nKernel=2, Stride=2\n\nOutput: $25 \\times 128$", 91, 32, "#E0E7FF", "#4F46E5", "#3730A3"),
        ("Dropout\nRate $p = 0.2$\n\nOutput: $25 \\times 128$", 109, 32, "#F1F5F9", "#64748B", "#334155")
    ]

    for title, cx, cy, fc, ec, tc in r1_nodes:
        bbox_style = dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc=fc, ec=ec, lw=1.6)
        ax.text(cx, cy, title, ha='center', va='center', fontsize=7.5, fontweight='bold', color=tc, bbox=bbox_style)

    # Layer nodes - Row 2 (Sequence Modeling & Classification Head)
    r2_nodes = [
        ("Bidirectional LSTM\nHidden Size = 128\nBatch First = True\nOutput: $25 \\times 256$", 109, 13, "#FEF3C7", "#D97706", "#92400E"),
        ("LSTM Last-Step\nFeature Vector\n\nOutput: $1 \\times 256$", 86, 13, "#FEF3C7", "#D97706", "#92400E"),
        ("Dropout & Dense 1\n128 Units, ReLU\nDropout $p = 0.3$\nOutput: $1 \\times 128$", 63, 13, "#D1FAE5", "#059669", "#065F46"),
        ("Dense 2 (Head)\n64 Units, ReLU\nDropout $p = 0.2$\nOutput: $1 \\times 64$", 40, 13, "#D1FAE5", "#059669", "#065F46"),
        ("Softmax Output\n6 Activity Classes\n$(C_1, \\dots, C_6)$\nProbability Vector", 17, 13, "#FFE4E6", "#E11D48", "#9F1239")
    ]

    for title, cx, cy, fc, ec, tc in r2_nodes:
        bbox_style = dict(boxstyle="round,pad=0.5,rounding_size=0.6", fc=fc, ec=ec, lw=1.6)
        ax.text(cx, cy, title, ha='center', va='center', fontsize=7.5, fontweight='bold', color=tc, bbox=bbox_style)

    # Arrows in Row 1 (Left to Right)
    arrow_kw = dict(arrowstyle="-|>", color="#334155", lw=1.8, mutation_scale=14)
    ax.annotate("", xy=(21.0, 32), xytext=(19.0, 32), arrowprops=arrow_kw)
    ax.annotate("", xy=(42.0, 32), xytext=(40.0, 32), arrowprops=arrow_kw)
    ax.annotate("", xy=(61.5, 32), xytext=(59.5, 32), arrowprops=arrow_kw)
    ax.annotate("", xy=(82.0, 32), xytext=(80.0, 32), arrowprops=arrow_kw)
    ax.annotate("", xy=(102.5, 32), xytext=(100.0, 32), arrowprops=arrow_kw)

    # Transition Arrow: Row 1 End -> Row 2 Start
    trans_kw = dict(arrowstyle="-|>", color="#2563EB", lw=2.0, mutation_scale=16)
    ax.annotate("", xy=(109, 21.5), xytext=(109, 24.5), arrowprops=trans_kw)

    # Arrows in Row 2 (Right to Left)
    ax.annotate("", xy=(97.0, 13), xytext=(99.5, 13), arrowprops=arrow_kw)
    ax.annotate("", xy=(74.5, 13), xytext=(77.0, 13), arrowprops=arrow_kw)
    ax.annotate("", xy=(51.5, 13), xytext=(54.0, 13), arrowprops=arrow_kw)
    ax.annotate("", xy=(28.5, 13), xytext=(31.0, 13), arrowprops=arrow_kw)

    plt.tight_layout()
    p3 = os.path.join(OUT_DIR, "fig_3_cnn_lstm_architecture.png")
    fig.savefig(p3, dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(DOC_DIR, "fig_3_cnn_lstm_architecture.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Fig 3 regenerated -> {p3}")

if __name__ == "__main__":
    draw_fig1()
    draw_fig2()
    draw_fig3()
    print("\n[SUCCESS] Figures 1, 2, and 3 perfected!")
