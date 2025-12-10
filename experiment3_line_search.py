import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from test_functions import get_test_function
from optimization_methods import armijo_line_search, strong_wolfe_line_search

def bfgs_method_with_initial_step(f, grad_f, x0, max_iter=1000, tol=1e-12,
                                   line_search='wolfe', initial_alpha=1.0):
    
    def armijo_custom(f, grad_f, x, p):
        alpha = initial_alpha
        rho = 0.5
        c1 = 1e-4
        fx = f(x)
        gx = grad_f(x)
        directional_derivative = np.dot(gx, p)
        f_evals = 0
        g_evals = 0
        for _ in range(50):
            f_evals += 1
            if f(x + alpha * p) <= fx + c1 * alpha * directional_derivative:
                return alpha, f_evals, g_evals
            alpha *= rho
        return alpha, f_evals, g_evals

    def wolfe_custom(f, grad_f, x, p):
        alpha = initial_alpha
        rho = 0.5
        c1 = 1e-4
        c2 = 0.4
        fx = f(x)
        gx = grad_f(x)
        directional_derivative = np.dot(gx, p)
        f_evals = 0
        g_evals = 0
        for _ in range(50):
            x_new = x + alpha * p
            fx_new = f(x_new)
            f_evals += 1
            if fx_new > fx + c1 * alpha * directional_derivative:
                alpha *= rho
                continue
            gx_new = grad_f(x_new)
            g_evals += 1
            new_directional_derivative = np.dot(gx_new, p)
            if new_directional_derivative >= c2 * directional_derivative:
                return alpha, f_evals, g_evals
            alpha *= 2.0
        return alpha, f_evals, g_evals

    def strong_wolfe_custom(f, grad_f, x, p):
        alpha = initial_alpha
        rho = 0.5
        c1 = 1e-4
        c2 = 0.9
        fx = f(x)
        gx = grad_f(x)
        directional_derivative = np.dot(gx, p)
        f_evals = 0
        g_evals = 0
        for _ in range(50):
            x_new = x + alpha * p
            fx_new = f(x_new)
            f_evals += 1
            if fx_new > fx + c1 * alpha * directional_derivative:
                alpha *= rho
                continue
            gx_new = grad_f(x_new)
            g_evals += 1
            new_directional_derivative = np.dot(gx_new, p)
            if abs(new_directional_derivative) <= c2 * abs(directional_derivative):
                return alpha, f_evals, g_evals
            if new_directional_derivative < 0:
                alpha *= 2.0
            else:
                alpha *= rho
        return alpha, f_evals, g_evals

    def no_line_search(f, grad_f, x, p):
        return 1.0, 0, 0
    x = x0.copy()
    n = len(x0)
    H = np.eye(n)

    x_history = [x.copy()]
    f_history = [f(x)]
    grad_norm_history = [np.linalg.norm(grad_f(x))]
    curvature_condition = []
    alpha_history = []
    f_evals_history = []
    g_evals_history = []

    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tol:
            break

        p = -H @ grad

        if line_search == 'none':
            alpha, f_evals, g_evals = no_line_search(f, grad_f, x, p)
        elif line_search == 'armijo':
            alpha, f_evals, g_evals = armijo_custom(f, grad_f, x, p)
        elif line_search == 'wolfe':
            alpha, f_evals, g_evals = wolfe_custom(f, grad_f, x, p)
        else:  # strong_wolfe
            alpha, f_evals, g_evals = strong_wolfe_custom(f, grad_f, x, p)

        alpha_history.append(alpha)
        f_evals_history.append(f_evals)
        g_evals_history.append(g_evals)

        x_new = x + alpha * p
        grad_new = grad_f(x_new)

        s = x_new - x
        y = grad_new - grad

        ys = np.dot(y, s)
        curvature_condition.append(ys)

        if ys > 1e-10:
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        x = x_new
        x_history.append(x.copy())
        f_history.append(f(x))
        grad_norm_history.append(grad_norm)

    return {
        'x': x,
        'f': f(x),
        'iterations': k + 1,
        'x_history': np.array(x_history),
        'f_history': np.array(f_history),
        'grad_norm_history': np.array(grad_norm_history),
        'curvature_condition': np.array(curvature_condition),
        'alpha_history': np.array(alpha_history),
        'f_evals_history': np.array(f_evals_history),
        'g_evals_history': np.array(g_evals_history),
        'total_f_evals': sum(f_evals_history) + k + 2,
        'total_g_evals': sum(g_evals_history) + k + 2,
        'converged': grad_norm < tol
    }

os.makedirs('./exp3', exist_ok=True)
func = get_test_function('rosenbrock', n=2)
x0 = np.array([-1.2, 1.0])

INITIAL_STEP_SIZE = 1.0

strategies = {
    'No line search (alpha=1)': 'none',
    'Armijo': 'armijo',
    'Regular Wolfe': 'wolfe',
    'Strong Wolfe': 'strong_wolfe'
}

results = {}
colors = {
    'No line search (alpha=1)': 'red',
    'Armijo': 'orange',
    'Regular Wolfe': 'blue',
    'Strong Wolfe': 'green'
}
markers = {
    'No line search (alpha=1)': 's',
    'Armijo': '^',
    'Regular Wolfe': 'D',
    'Strong Wolfe': 'o'
}


for name, strategy in strategies.items():
    result = bfgs_method_with_initial_step(func, func.gradient, x0.copy(),
                        max_iter=200, tol=1e-12, line_search=strategy,
                        initial_alpha=INITIAL_STEP_SIZE)
    results[name] = result
    curvature = result.get('curvature_condition', [])
    if len(curvature) > 0:
        curvature_array = np.array(curvature)
        violations = np.sum(curvature_array <= 0)
        positive_count = np.sum(curvature_array > 0)
        if violations > 0:
            print(f"    WARNING: Curvature condition violated {violations} times!")
        else:
            print(f"    SUCCESS: Curvature condition satisfied throughout")


fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))

x_range = np.linspace(-1.5, 1.5, 200)
y_range = np.linspace(-0.5, 1.5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

levels = [0.1, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 500, 800, 1000]
ax1.contour(X, Y, Z, levels=levels, colors='gray', linewidths=0.7, alpha=0.5)

plot_order = ['No line search (alpha=1)', 'Armijo', 'Regular Wolfe', 'Strong Wolfe']
for name in plot_order:
    if name in results and results[name] is not None:
        result = results[name]
        x_hist = result['x_history']
        ax1.plot(x_hist[:, 0], x_hist[:, 1], '-',
                color=colors[name], marker=markers[name], markersize=4,
                label=name, linewidth=1.5, alpha=0.8)

# Mark start and optimum
ax1.plot(x0[0], x0[1], 'ko', markersize=10, label='Start', zorder=10)
ax1.plot(func.optimal_x[0], func.optimal_x[1], 'r*', markersize=15, label='Optimum', zorder=10)

ax1.set_xlabel('$x_1$', fontsize=11)
ax1.set_ylabel('$x_2$', fontsize=11)
ax1.set_title('Optimization Paths on Rosenbrock Function', fontsize=12, fontweight='bold')
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.savefig('./exp3/fig1_optimization_paths.png', dpi=300, bbox_inches='tight')
plt.close()

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

for name in plot_order:
    if name in results and results[name] is not None:
        result = results[name]
        iters = np.arange(1, len(result['f_history']) + 1)  # Start from 1 for log scale
        f_vals = np.array(result['f_history']) - func.optimal_f + 1e-16
        ax2.loglog(iters, f_vals, '-', color=colors[name],
                  marker=markers[name], markersize=3, label=name, linewidth=1.5)

ax2.set_xlabel('Iteration $k$', fontsize=11)
ax2.set_ylabel('$f(x_k) - f^*$', fontsize=11)
ax2.set_title('Function Value Convergence (log-log scale)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('./exp3/fig2_function_convergence.png', dpi=300, bbox_inches='tight')
plt.close()

fig3, (ax3_top, ax3_bottom) = plt.subplots(2, 1, figsize=(10, 8),
                                           gridspec_kw={'height_ratios': [1, 2]})
all_curvature_values = []
for name in plot_order:
    if name in results and results[name] is not None:
        curv = results[name].get('curvature_condition', [])
        all_curvature_values.extend([c for c in curv if c > 10])

if len(all_curvature_values) > 0:
    y_top_max = max(all_curvature_values) * 1.1
    y_top_min = 100
else:
    y_top_max = 1000
    y_top_min = 100

y_bottom_max = 5
y_bottom_min = -1

for name in plot_order:
    if name in results and results[name] is not None:
        result = results[name]
        curvature = result.get('curvature_condition', [])
        if len(curvature) > 0:
            iters = np.arange(len(curvature))
            ax3_top.plot(iters, curvature, '-', color=colors[name],
                        marker=markers[name], markersize=3, label=name, linewidth=1.5)

ax3_top.set_ylim(int(y_top_min), int(y_top_max))
ax3_top.set_xlim(1, 100)
ax3_top.set_ylabel('$y_k^T s_k$', fontsize=11)
ax3_top.set_title('Curvature Condition (Must be > 0 for BFGS)', fontsize=12, fontweight='bold')
ax3_top.legend(loc='upper right', fontsize=9)
ax3_top.grid(True, alpha=0.3)
ax3_top.spines['bottom'].set_visible(False)
ax3_top.xaxis.set_visible(False)

linestyles = {'No line search (alpha=1)': '-', 'Armijo': '--', 'Regular Wolfe': '-.', 'Strong Wolfe': ':'}
plot_order_bottom = ['No line search (alpha=1)', 'Strong Wolfe', 'Regular Wolfe', 'Armijo']  # Armijo on top

for name in plot_order_bottom:
    if name in results and results[name] is not None:
        result = results[name]
        curvature = result.get('curvature_condition', [])
        if len(curvature) > 0:
            iters = np.arange(len(curvature))
            ax3_bottom.plot(iters, curvature, linestyles[name], color=colors[name],
                           marker=markers[name], markersize=4, label=name,
                           linewidth=2, markevery=3, alpha=0.9)

ax3_bottom.axhline(y=0, color='black', linestyle='--', linewidth=2,
                   label='$y^T s = 0$ (critical)', zorder=10)
ax3_bottom.fill_between([-5, 105], -1, 0, color='red', alpha=0.1,
                        label='Violation region')

ax3_bottom.set_ylim(y_bottom_min, y_bottom_max)
ax3_bottom.set_xlim(1, 100)
ax3_bottom.set_xlabel('Iteration $k$', fontsize=11)
ax3_bottom.set_ylabel('$y_k^T s_k$', fontsize=11)
ax3_bottom.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax3_bottom.grid(True, alpha=0.3)
ax3_bottom.spines['top'].set_visible(False)
ax3_bottom.set_title('Near-zero detail (note: different line styles)', fontsize=10, style='italic')


d = .015 
kwargs = dict(transform=ax3_top.transAxes, color='k', clip_on=False, linewidth=1)
ax3_top.plot((-d, +d), (-d, +d), **kwargs)
ax3_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax3_bottom.transAxes)
ax3_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax3_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax3_bottom.text(0.02, 0.95, 'Focus: Near-zero region\nViolations occur when y^T s < 0\nArmijo (dashed) & Wolfe (dotted) overlap',
                transform=ax3_bottom.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)  # Reduce space between subplots
plt.savefig('./exp3/fig3_curvature_condition.png', dpi=300, bbox_inches='tight')
plt.close()
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))

# Plot all methods
for name in plot_order:
    if name in results and results[name] is not None:
        result = results[name]
        iters = np.arange(len(result['grad_norm_history']))
        ax4.semilogy(iters, result['grad_norm_history'], '-',
                    color=colors[name], marker=markers[name], markersize=3,
                    label=name, linewidth=1.5)

ax4.set_xlabel('Iteration $k$', fontsize=11)
ax4.set_ylabel('$\\|\\nabla f(x_k)\\|$', fontsize=11)
ax4.set_title('Gradient Norm Convergence', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 120)

plt.tight_layout()
plt.savefig('./exp3/fig4_gradient_norm.png', dpi=300, bbox_inches='tight')
plt.close()

fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))

for name in ['Armijo', 'Regular Wolfe', 'Strong Wolfe']:
    if name in results and results[name] is not None:
        result = results[name]
        alpha_hist = result.get('alpha_history', [])
        if len(alpha_hist) > 0:
            iters = np.arange(len(alpha_hist))
            ax5.semilogy(iters, alpha_hist, '-', color=colors[name],
                        marker=markers[name], markersize=4, label=name,
                        linewidth=2, markevery=2)

ax5.set_xlabel('Iteration $k$', fontsize=11)
ax5.set_ylabel('Step Size $\\alpha_k$', fontsize=11)
ax5.set_title('Step Size Adaptation (How Line Search Adjusts $\\alpha$)', fontsize=12, fontweight='bold')
ax5.legend(loc='best', fontsize=10)
ax5.grid(True, alpha=0.3, which='both')
ax5.set_xlim(0, 100)

# Add text annotation
ax5.text(0.98, 0.97, 'Strong Wolfe adapts step size\nmore aggressively for faster convergence',
         transform=ax5.transAxes, fontsize=9, verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('./exp3/fig5_step_size_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

summary_rows = []
for name, result in results.items():
    if result is not None:
        curvature = result.get('curvature_condition', [])
        violations = np.sum(np.array(curvature) <= 0) if len(curvature) > 0 else 0
        violation_rate = violations / len(curvature) * 100 if len(curvature) > 0 else 0

        summary_rows.append({
            'Line_Search_Strategy': name,
            'Converged': result['converged'],
            'Iterations': result['iterations'],
            'Final_f(x)': result['f'],
            'Final_Error': np.linalg.norm(result['x'] - func.optimal_x),
            'Final_Gradient_Norm': result['grad_norm_history'][-1],
            'Curvature_Violations': violations,
            'Violation_Rate_%': violation_rate
        })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv('./exp3/summary_comparison.csv', index=False)


# 2. Detailed iteration data for each strategy
for name, result in results.items():
    if result is not None:
        # Clean name for filename
        filename = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').lower()

        x_hist = result['x_history']
        f_hist = result['f_history']
        grad_hist = result['grad_norm_history']
        curv = result.get('curvature_condition', [])

        # Pad curvature array if needed
        curv_padded = list(curv) + [np.nan] * (len(f_hist) - len(curv))

        detailed_data = {
            'Iteration': np.arange(len(f_hist)),
            'x1': x_hist[:, 0],
            'x2': x_hist[:, 1],
            'f(x)': f_hist,
            'Gradient_Norm': grad_hist,
            'y^T_s': curv_padded,
            'Error_to_Optimum': [np.linalg.norm(x - func.optimal_x) for x in x_hist]
        }

        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv(f'./exp3/{filename}_iterations.csv', index=False)

# 3. Curvature condition comparison
curv_comparison = {}
max_len = 0
for name, result in results.items():
    if result is not None:
        curv = result.get('curvature_condition', [])
        curv_comparison[name] = curv
        max_len = max(max_len, len(curv))

# Create dataframe with all curvature values
curv_data = {'Iteration': np.arange(max_len)}
for name, curv in curv_comparison.items():
    curv_padded = list(curv) + [np.nan] * (max_len - len(curv))
    curv_data[name] = curv_padded

df_curvature = pd.DataFrame(curv_data)
df_curvature.to_csv('./exp3/curvature_comparison.csv', index=False)

for name, result in results.items():
    if result is not None:
        curvature = result.get('curvature_condition', [])
        if len(curvature) > 0:
            violations = np.sum(np.array(curvature) <= 0)
            print(f"\n{name}:")
            if violations == 0:
                print(f"  [OK] PERFECT: All y^T s > 0 (curvature condition satisfied)")
                print(f"  [OK] H_k maintained positive definiteness throughout")
            else:
                print(f"  [WARNING] {violations} curvature violations detected")
                print(f"  [WARNING] H_k may have lost positive definiteness")

# Check if Wolfe satisfied theory
wolfe_result = results.get('Strong Wolfe')
if wolfe_result is not None:
    curv = wolfe_result.get('curvature_condition', [])
    violations = np.sum(np.array(curv) <= 0) if len(curv) > 0 else 0
    if violations == 0:
        print("[OK] THEORY CONFIRMED: Strong Wolfe conditions guarantee y^T s > 0")
    else:
        print("[?] UNEXPECTED: Strong Wolfe had violations (numerical issues?)")


# Use realistic range of initial step sizes
initial_alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
robustness_results = {
    'Armijo': {'alphas': [], 'iterations': [], 'converged': [], 'violations': []},
    'Regular Wolfe': {'alphas': [], 'iterations': [], 'converged': [], 'violations': []},
    'Strong Wolfe': {'alphas': [], 'iterations': [], 'converged': [], 'violations': []}
}

for alpha0 in initial_alphas:
    print(f"Testing alpha_0 = {alpha0}...")

    for method_name in ['Armijo', 'Regular Wolfe', 'Strong Wolfe']:
        if method_name == 'Armijo':
            strategy = 'armijo'
        elif method_name == 'Regular Wolfe':
            strategy = 'wolfe'
        else:  # Strong Wolfe
            strategy = 'strong_wolfe'
        
        result = bfgs_method_with_initial_step(func, func.gradient, x0.copy(),
                            max_iter=200, tol=1e-12, line_search=strategy,
                            initial_alpha=alpha0)
        
        curvature = result.get('curvature_condition', [])
        violations = np.sum(np.array(curvature) <= 0) if len(curvature) > 0 else 0
        
        robustness_results[method_name]['alphas'].append(alpha0)
        robustness_results[method_name]['iterations'].append(result['iterations'])
        robustness_results[method_name]['converged'].append(result['converged'])
        robustness_results[method_name]['violations'].append(violations)
        
        
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5))

# Define markers for each method: circle, triangle, square
robustness_markers = {
    'Armijo': 'o',           # circle
    'Regular Wolfe': '^',    # triangle
    'Strong Wolfe': 's'      # square
}

# Left plot: Iterations vs Initial Step Size
# Plot order: circle (bottom), square (middle), triangle (top)
for method_name in ['Armijo', 'Strong Wolfe', 'Regular Wolfe']:
    data = robustness_results[method_name]
    ax6a.semilogx(data['alphas'], data['iterations'], '-',
                  marker=robustness_markers[method_name],
                  color=colors[method_name], markersize=8,
                  label=method_name, linewidth=2)

ax6a.set_xlabel('Initial Step Size $\\alpha_0$', fontsize=11)
ax6a.set_ylabel('Iterations to Convergence', fontsize=11)
ax6a.set_title('Robustness: Sensitivity to Initial Step Size', fontsize=12)
ax6a.legend(loc='best', fontsize=10)
ax6a.grid(True, alpha=0.3)
zorder_map = {'Strong Wolfe': 1, 'Regular Wolfe': 2, 'Armijo': 3}
for method_name in ['Strong Wolfe', 'Regular Wolfe', 'Armijo']:
    data = robustness_results[method_name]
    ax6b.semilogx(data['alphas'], data['violations'], '-',
                  marker=robustness_markers[method_name],
                  color=colors[method_name], markersize=8,
                  label=method_name, linewidth=2, zorder=zorder_map[method_name])

ax6b.set_xlabel('Initial Step Size $\\alpha_0$', fontsize=11)
ax6b.set_ylabel('Curvature Violations', fontsize=11)
ax6b.set_title('Robustness: Curvature Condition Stability', fontsize=12)
ax6b.legend(loc='best', fontsize=10)
ax6b.grid(True, alpha=0.3)
ax6b.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='No violations (ideal)')
ax6b.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig('./exp3/fig6_robustness_test.png', dpi=300, bbox_inches='tight')
plt.close()

# Save robustness data
robustness_data_rows = []
for alpha0 in initial_alphas:
    row = {'Initial_Alpha': alpha0}
    for method_name in ['Armijo', 'Regular Wolfe', 'Strong Wolfe']:
        data = robustness_results[method_name]
        idx = data['alphas'].index(alpha0)
        row[f'{method_name}_Iterations'] = data['iterations'][idx]
        row[f'{method_name}_Converged'] = data['converged'][idx]
        row[f'{method_name}_Violations'] = data['violations'][idx]
    robustness_data_rows.append(row)

df_robustness = pd.DataFrame(robustness_data_rows)
df_robustness.to_csv('./exp3/robustness_comparison.csv', index=False)


cost_analysis = []
for name, result in results.items():
    if result is not None:
        iters = result['iterations']
        total_f = result.get('total_f_evals', iters)
        total_g = result.get('total_g_evals', iters)
        avg_f_per_iter = total_f / iters if iters > 0 else 0
        avg_g_per_iter = total_g / iters if iters > 0 else 0
        
        cost_analysis.append({
            'Method': name,
            'Iterations': iters,
            'Total_F_Evals': total_f,
            'Total_G_Evals': total_g,
            'Avg_F_per_Iter': avg_f_per_iter,
            'Avg_G_per_Iter': avg_g_per_iter,
            'Total_Cost': total_f + total_g 
        })

df_cost = pd.DataFrame(cost_analysis)
df_cost.to_csv('./exp3/computational_cost_analysis.csv', index=False)
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))

methods = [name for name in plot_order if name in results and results[name] is not None]
iterations = [results[name]['iterations'] for name in methods]
avg_f_evals = [df_cost[df_cost['Method']==name]['Avg_F_per_Iter'].values[0] for name in methods]
total_cost = [df_cost[df_cost['Method']==name]['Total_Cost'].values[0] for name in methods]
method_colors = [colors[name] for name in methods]


x_pos = np.arange(len(methods))
width = 0.35

ax7a_twin = ax7a.twinx()
bars1 = ax7a.bar(x_pos - width/2, iterations, width, label='Iterations',
                 color=method_colors, alpha=0.7)
bars2 = ax7a_twin.bar(x_pos + width/2, avg_f_evals, width, label='Avg F-evals/iter',
                      color='gray', alpha=0.5)

ax7a.set_xlabel('Method', fontsize=11)
ax7a.set_ylabel('Total Iterations', fontsize=11, color='black')
ax7a_twin.set_ylabel('Avg Function Evaluations per Iteration', fontsize=11, color='gray')
ax7a.set_title('Iteration Count vs Per-Iteration Cost', fontsize=12, fontweight='bold')
ax7a.set_xticks(x_pos)
ax7a.set_xticklabels([m.replace(' (alpha=1)', '\n(α=1)') for m in methods], fontsize=9)
ax7a.tick_params(axis='y', labelcolor='black')
ax7a_twin.tick_params(axis='y', labelcolor='gray')
ax7a.legend(loc='upper left', fontsize=9)
ax7a_twin.legend(loc='upper right', fontsize=9)
ax7a.grid(True, alpha=0.3, axis='y')


bars3 = ax7b.bar(methods, total_cost, color=method_colors, alpha=0.7)
ax7b.set_xlabel('Method', fontsize=11)
ax7b.set_ylabel('Total Cost (F-evals + G-evals)', fontsize=11)
ax7b.set_title('Total Computational Cost', fontsize=12, fontweight='bold')
ax7b.set_xticklabels([m.replace(' (alpha=1)', '\n(α=1)') for m in methods], fontsize=9)
ax7b.grid(True, alpha=0.3, axis='y')


for i, (bar, cost) in enumerate(zip(bars3, total_cost)):
    height = bar.get_height()
    ax7b.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(cost)}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('./exp3/fig7_computational_cost.png', dpi=300, bbox_inches='tight')
plt.close()