import os
import matplotlib
import matplotlib.pyplot as plt

def render_equation_stack(lines, filename, out_dir, fontsize=14, color="#1B365D", dpi=300):
    """
    Renders a stack of LaTeX math lines with dynamic height calculation
    and generous line spacing to completely eliminate any vertical overlaps.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    if isinstance(lines, str):
        lines = [lines]
        
    # Calculate required figure dimensions
    # For each line, determine if it has tall fractions or summations
    line_heights = []
    for line in lines:
        if r'\frac' in line or r'\sum' in line or r'\max' in line:
            line_heights.append(0.85) # tall math
        else:
            line_heights.append(0.55) # normal math
            
    total_height = sum(line_heights) + 0.4
    fig = plt.figure(figsize=(8, total_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Place lines from top to bottom
    curr_y = total_height - 0.2
    for i, line in enumerate(lines):
        h = line_heights[i]
        y_pos = (curr_y - h/2) / total_height
        ax.text(0.02, y_pos, line, fontsize=fontsize, color=color,
                ha='left', va='center',
                family='serif', math_fontfamily='stix')
        curr_y -= h
        
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close(fig)
    print(f"[FIXED RENDER] {filename} saved successfully.")

def test_all_renders():
    out_dir = r"c:\WAR-FL\Documentation\formulas"
    
    # 1. System 1
    render_equation_stack(
        r'$|H(f)|^2 = \frac{1}{1 + \left(\frac{f}{f_c}\right)^{2n}}$',
        "eq_butterworth.png", out_dir, fontsize=15
    )
    
    render_equation_stack([
        r'$\mathbf{Prediction:} \quad \hat{x}_{k|k-1} = \hat{x}_{k-1|k-1}, \quad P_{k|k-1} = P_{k-1|k-1} + Q$',
        r'$\mathbf{Update:} \quad K_k = \frac{P_{k|k-1}}{P_{k|k-1} + R}$',
        r'$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \cdot (z_k - \hat{x}_{k|k-1})$',
        r'$P_{k|k} = (1 - K_k) \cdot P_{k|k-1}$'
    ], "eq_kalman.png", out_dir, fontsize=13)
    
    render_equation_stack([
        r'$\mu_j = \frac{1}{N_{train} \cdot W} \sum_{i=1}^{N_{train}} \sum_{t=1}^W x_{i,t,j}$',
        r'$\sigma_j = \sqrt{\frac{1}{N_{train} \cdot W} \sum_{i=1}^{N_{train}} \sum_{t=1}^W (x_{i,t,j} - \mu_j)^2}$',
        r'$x_{i,t,j}^{norm} = \frac{x_{i,t,j} - \mu_j}{\sigma_j}$'
    ], "eq_zscore.png", out_dir, fontsize=13)
    
    # 2. System 2
    render_equation_stack([
        r'$w_c = \frac{N_{local}}{C \cdot N_c}$',
        r'$\mathcal{L}_{local}(\theta) = -\frac{1}{B} \sum_{i=1}^B w_{y_i} \sum_{c=0}^{C-1} y_{i,c} \log \hat{y}_{i,c}(\theta)$'
    ], "eq_class_weighted_loss.png", out_dir, fontsize=13.5)
    
    render_equation_stack([
        r'$\bar{g}_t = \frac{g_t}{\max\left(1, \frac{\|g_t\|_2}{C_{clip}}\right)}$',
        r'$g_t^{DP} = \bar{g}_t + \mathcal{N}\left(0, \sigma^2 C_{clip}^2 \mathbf{I}\right)$',
        r'$w_{t+1} = w_t - \eta \cdot g_t^{DP}$'
    ], "eq_dp_sgd.png", out_dir, fontsize=13.5)
    
    # 3. System 3
    render_equation_stack(
        r'$w^{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_k^{t+1}, \quad \text{where } n = \sum_{k=1}^K n_k$',
        "eq_fedavg.png", out_dir, fontsize=14
    )
    
    render_equation_stack(
        r'$\mathcal{L}_{FedProx}(w; w^t) = \mathcal{L}_{local}(w) + \frac{\mu}{2} \|w - w^t\|_2^2$',
        "eq_fedprox.png", out_dir, fontsize=14
    )
    
    render_equation_stack([
        r'$g_k^{corrected} = g_k - c_k + c$',
        r'$c_k^{t+1} = c_k^t - c^t + \frac{1}{K \eta_L} (w^t - w_k^{t+1})$'
    ], "eq_scaffold.png", out_dir, fontsize=13.5)
    
    # 4. System 4 - Fixed spacing for Reward Mechanism
    render_equation_stack([
        r'$S_k^{(t)} = \alpha \hat{q}_k^{(t)} + \beta \widehat{\Delta A_k^{(t)}} + \gamma \hat{r}_k^{(t)}$',
        r'$\hat{q}_k^{(t)} = \frac{n_k}{\sum_{j=1}^K n_j}, \qquad \widehat{\Delta A_k^{(t)}} = \frac{\max(0, \Delta A_k^{(t)})}{\sum_{j=1}^K \max(0, \Delta A_j^{(t)})}, \qquad \hat{r}_k^{(t)} = \frac{1/\tau_k}{\sum_{j=1}^K 1/\tau_j}$',
        r'$\text{where weights constraint: } \alpha + \beta + \gamma = 1.0$'
    ], "eq_reward_mechanism.png", out_dir, fontsize=13)
    
    render_equation_stack([
        r'$T_{total} = T_{buffer} + T_{prep} + T_{infer} + T_{decision}$',
        r'$T_{total} = 1000.0\text{ ms} + 5.0\text{ ms} + 381.61\text{ ms} + 0.5\text{ ms} = 1387.11\text{ ms}$'
    ], "eq_latency.png", out_dir, fontsize=13)

if __name__ == '__main__':
    test_all_renders()
