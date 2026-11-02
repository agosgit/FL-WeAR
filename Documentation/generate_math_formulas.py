import os
import matplotlib
import matplotlib.pyplot as plt

def render_math_formula_stack(lines_with_heights, filename, out_dir, fontsize=14, color="#1B365D", dpi=300):
    """
    Renders multiple LaTeX lines with explicit per-line vertical allocations
    to guarantee zero overlapping, crystal clear typography, and optimal padding.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    total_height = sum(h for _, h in lines_with_heights) + 0.3
    fig = plt.figure(figsize=(8.5, total_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    curr_y = total_height - 0.15
    for line, h in lines_with_heights:
        y_pos = (curr_y - h / 2.0) / total_height
        ax.text(0.01, y_pos, line, fontsize=fontsize, color=color,
                ha='left', va='center',
                family='serif', math_fontfamily='stix')
        curr_y -= h
        
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', transparent=True, pad_inches=0.08)
    plt.close(fig)
    print(f"[CLEAN RENDER] {filename} successfully generated.")
    return out_path

def generate_all_system_formulas():
    out_dir = r"c:\WAR-FL\Documentation\formulas"
    
    # 1. System 1: Butterworth Filter
    render_math_formula_stack([
        (r'$|H(f)|^2 = \frac{1}{1 + \left(\frac{f}{f_c}\right)^{2n}}$', 0.8)
    ], "eq_butterworth.png", out_dir, fontsize=15)
    
    # 1. System 1: Kalman Filter
    render_math_formula_stack([
        (r'$\mathbf{Prediction:} \quad \hat{x}_{k|k-1} = \hat{x}_{k-1|k-1}, \quad P_{k|k-1} = P_{k-1|k-1} + Q$', 0.6),
        (r'$\mathbf{Kalman\ Gain:} \quad K_k = \frac{P_{k|k-1}}{P_{k|k-1} + R}$', 0.8),
        (r'$\mathbf{State\ Update:} \quad \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \cdot (z_k - \hat{x}_{k|k-1})$', 0.6),
        (r'$\mathbf{Covariance\ Update:} \quad P_{k|k} = (1 - K_k) \cdot P_{k|k-1}$', 0.6)
    ], "eq_kalman.png", out_dir, fontsize=13)
    
    # 1. System 1: Z-Score Normalization
    render_math_formula_stack([
        (r'$\mu_j = \frac{1}{N_{train} \cdot W} \sum_{i=1}^{N_{train}} \sum_{t=1}^W x_{i,t,j}$', 0.85),
        (r'$\sigma_j = \left[ \frac{1}{N_{train} \cdot W} \sum_{i=1}^{N_{train}} \sum_{t=1}^W (x_{i,t,j} - \mu_j)^2 \right]^{1/2}$', 0.85),
        (r'$x_{i,t,j}^{norm} = \frac{x_{i,t,j} - \mu_j}{\sigma_j}$', 0.75)
    ], "eq_zscore.png", out_dir, fontsize=13)
    
    # 2. System 2: Class-Weighted Loss
    render_math_formula_stack([
        (r'$w_c = \frac{N_{local}}{C \cdot N_c}$', 0.75),
        (r'$\mathcal{L}_{local}(\theta) = -\frac{1}{B} \sum_{i=1}^B w_{y_i} \sum_{c=0}^{C-1} y_{i,c} \log \hat{y}_{i,c}(\theta)$', 0.85)
    ], "eq_class_weighted_loss.png", out_dir, fontsize=13.5)
    
    # 2. System 2: DP-SGD Optimizer
    render_math_formula_stack([
        (r'$\bar{g}_t = \frac{g_t}{\max\left(1, \frac{\|g_t\|_2}{C_{clip}}\right)}$', 0.85),
        (r'$g_t^{DP} = \bar{g}_t + \mathcal{N}\left(0, \sigma^2 C_{clip}^2 \mathbf{I}\right)$', 0.65),
        (r'$w_{t+1} = w_t - \eta \cdot g_t^{DP}$', 0.6)
    ], "eq_dp_sgd.png", out_dir, fontsize=13.5)
    
    # 3. System 3: FedAvg
    render_math_formula_stack([
        (r'$w^{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_k^{t+1}, \quad \text{where } n = \sum_{k=1}^K n_k$', 0.8)
    ], "eq_fedavg.png", out_dir, fontsize=14)
    
    # 3. System 3: FedProx
    render_math_formula_stack([
        (r'$\mathcal{L}_{FedProx}(w; w^t) = \mathcal{L}_{local}(w) + \frac{\mu}{2} \|w - w^t\|_2^2$', 0.75)
    ], "eq_fedprox.png", out_dir, fontsize=14)
    
    # 3. System 3: SCAFFOLD
    render_math_formula_stack([
        (r'$g_k^{corrected} = g_k - c_k + c$', 0.6),
        (r'$c_k^{t+1} = c_k^t - c^t + \frac{1}{K \eta_L} (w^t - w_k^{t+1})$', 0.8)
    ], "eq_scaffold.png", out_dir, fontsize=13.5)
    
    # 4. System 4: Reward Distribution Mechanism (Section III-G)
    render_math_formula_stack([
        (r'$S_k^{(t)} = \alpha \hat{q}_k^{(t)} + \beta \widehat{\Delta A_k^{(t)}} + \gamma \hat{r}_k^{(t)}$', 0.65),
        (r'$\hat{q}_k^{(t)} = \frac{n_k}{\sum_{j=1}^K n_j}, \qquad \widehat{\Delta A_k^{(t)}} = \frac{\max(0, \Delta A_k^{(t)})}{\sum_{j=1}^K \max(0, \Delta A_j^{(t)})}, \qquad \hat{r}_k^{(t)} = \frac{1/\tau_k}{\sum_{j=1}^K 1/\tau_j}$', 0.95),
        (r'$\text{where weights constraint satisfies: } \alpha + \beta + \gamma = 1.0$', 0.55)
    ], "eq_reward_mechanism.png", out_dir, fontsize=13)
    
    # 4. System 4: Latency Profiling
    render_math_formula_stack([
        (r'$T_{total} = T_{buffer} + T_{prep} + T_{infer} + T_{decision}$', 0.6),
        (r'$T_{total} = 1000.0\text{ ms} + 5.0\text{ ms} + 381.61\text{ ms} + 0.5\text{ ms} = 1387.11\text{ ms}$', 0.6)
    ], "eq_latency.png", out_dir, fontsize=13)

if __name__ == '__main__':
    generate_all_system_formulas()
