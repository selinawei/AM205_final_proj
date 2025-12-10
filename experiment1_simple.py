import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from optimization_methods import bfgs_method
from test_functions import get_test_function

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 5),
    'lines.linewidth': 1.5,
    'lines.markersize': 4
})

func = get_test_function('rosenbrock', n=2)
x0 = np.array([-1.2, 1.0])

print(f"Initial point: x0 = {x0}")
print(f"Target: x* = (1, 1), f* = 0")

from optimization_methods import steepest_descent, newton_method
import time

n_trials = 10
print(f"\nRunning each method {n_trials} times for accurate timing...")
print("Running Gradient Descent...")
times_gd = []
for trial in range(n_trials):
    start_time = time.perf_counter()
    result_gd = steepest_descent(func, func.gradient, x0.copy(), max_iter=10000, tol=1e-8)
    times_gd.append(time.perf_counter() - start_time)
time_gd = np.mean(times_gd)
print(f"  Iterations: {result_gd['iterations']}")
print(f"  Final f(x): {result_gd['f']:.2e}")
print(f"  ||x - x*||: {np.linalg.norm(result_gd['x'] - func.optimal_x):.2e}")
print(f"  Avg Time: {time_gd:.6f} seconds (±{np.std(times_gd):.6f})")

print("Running BFGS...")
times_bfgs = []
for trial in range(n_trials):
    start_time = time.perf_counter()
    result_bfgs = bfgs_method(func, func.gradient, x0.copy(), max_iter=1000, tol=1e-8)
    times_bfgs.append(time.perf_counter() - start_time)
time_bfgs = np.mean(times_bfgs)
print(f"  Iterations: {result_bfgs['iterations']}")
print(f"  Final f(x): {result_bfgs['f']:.2e}")
print(f"  ||x - x*||: {np.linalg.norm(result_bfgs['x'] - func.optimal_x):.2e}")
print(f"  Avg Time: {time_bfgs:.6f} seconds (±{np.std(times_bfgs):.6f})")

print("Running Newton's Method...")
times_newton = []
for trial in range(n_trials):
    start_time = time.perf_counter()
    result_newton = newton_method(func, func.gradient, func.hessian, x0.copy(), max_iter=1000, tol=1e-8)
    times_newton.append(time.perf_counter() - start_time)
time_newton = np.mean(times_newton)
print(f"  Iterations: {result_newton['iterations']}")
print(f"  Final f(x): {result_newton['f']:.2e}")
print(f"  ||x - x*||: {np.linalg.norm(result_newton['x'] - func.optimal_x):.2e}")
print(f"  Avg Time: {time_newton:.6f} seconds (±{np.std(times_newton):.6f})")

# Store results
results = {
    'Gradient Descent': result_gd,
    'BFGS': result_bfgs,
    'Newton': result_newton
}

# Store timing information
method_times = {
    'Gradient Descent': time_gd,
    'BFGS': time_bfgs,
    'Newton': time_newton
}

# Save data for all methods
for method_name, result in results.items():
    filename = f"./exp1/experiment1_{method_name.replace(' ', '_').lower()}_data.csv"
    df = pd.DataFrame({
        'k': range(len(result['x_history'])),
        'x1': result['x_history'][:, 0],
        'x2': result['x_history'][:, 1],
        'f(x)': result['f_history'],
        '||grad f||': result['grad_norm_history']
    })
    df.to_csv(filename, index=False)

print(f"\nData saved for all methods")

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.add_subplot(111)
x1 = np.linspace(-1.5, 1.5, 200)
x2 = np.linspace(-0.5, 1.5, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = (1 - X1)**2 + 100*(X2 - X1**2)**2

levels = [0.1, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 500, 800, 1000]
ax1.contour(X1, X2, Z, levels=levels, colors='gray', linewidths=0.7, alpha=0.5)
colors = {'Gradient Descent': 'green', 'BFGS': 'blue', 'Newton': 'red'}
markers = {'Gradient Descent': 's', 'BFGS': 'o', 'Newton': '^'}
for method_name, result in results.items():
    x_hist = result['x_history']
    color = colors[method_name]
    marker = markers[method_name]

    ax1.plot(x_hist[:, 0], x_hist[:, 1], '-', color=color,
             linewidth=1.2, alpha=0.6)
    step = max(1, len(x_hist) // 10) 
    ax1.plot(x_hist[::step, 0], x_hist[::step, 1], marker,
             color=color, markersize=4, label=method_name, alpha=0.8)

ax1.plot(x0[0], x0[1], 'ko', markersize=8, label='Start',
         markerfacecolor='white', markeredgewidth=1, zorder=10)
ax1.plot(1, 1, 'k*', markersize=8, label='Optimum', zorder=10)

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Optimization Trajectories Comparison')
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(True, alpha=0.3)


ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('./exp1/experiment1_trajectory.png', dpi=300, bbox_inches='tight')
print("Figure saved: ./exp1/experiment1_trajectory.png")
plt.close()


fig2 = plt.figure(figsize=(6, 5))
ax2 = fig2.add_subplot(111)

for method_name, result in results.items():
    iters = np.arange(len(result['f_history']))
    f_gap = result['f_history'] - func.optimal_f + 1e-16
    f_gap = np.maximum(f_gap, 1e-16)
    color = colors[method_name]
    marker = markers[method_name]
    ax2.semilogy(iters, f_gap, '-', color=color, marker=marker,
                 linewidth=1.5, markersize=4, label=method_name)

ax2.set_xlabel('Iteration $k$')
ax2.set_ylabel('$f(x_k) - f^*$ (log scale)')
ax2.set_title('Convergence Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 40)
ax2.set_ylim(1e-17, None)

plt.tight_layout()
plt.savefig('./exp1/experiment1_convergence.png', dpi=300, bbox_inches='tight')
plt.close()


convergence_data = {}

for method_name, result in results.items():
    x_hist = result['x_history']

    errors = np.array([np.linalg.norm(x - func.optimal_x) for x in x_hist])
    converged_mask = errors < 0.1 
    converged_indices = np.where(converged_mask)[0]

    if len(converged_indices) >= 5:
        idx_start = max(0, len(errors) - 7)
        errors_to_analyze = errors[idx_start:]
        ratios_linear = errors_to_analyze[1:] / (errors_to_analyze[:-1] + 1e-20)
        ratios_quadratic = errors_to_analyze[1:] / (errors_to_analyze[:-1]**2 + 1e-20)

        avg_linear_ratio = np.mean(ratios_linear)
        std_linear_ratio = np.std(ratios_linear)
        if avg_linear_ratio < 0.1 and np.mean(ratios_quadratic[-3:]) < 100:
            conv_type = "Superlinear/Quadratic"
        elif avg_linear_ratio < 0.5:
            conv_type = "Superlinear"
        elif avg_linear_ratio < 0.95:
            conv_type = "Linear"
        else:
            conv_type = "Slow/Sublinear"

        convergence_data[method_name] = {
            'errors': errors_to_analyze,
            'ratios_linear': ratios_linear,
            'ratios_quadratic': ratios_quadratic,
            'avg_ratio': avg_linear_ratio,
            'std_ratio': std_linear_ratio,
            'type': conv_type
        }
    else:
        convergence_data[method_name] = {
            'type': 'Not converged',
            'avg_ratio': np.nan
        }


for method_name, data in convergence_data.items():
    if data['type'] != 'Not converged':
        if len(data['errors']) >= 4:
            log_errors = np.log(data['errors'][:-1] + 1e-20)
            log_errors_next = np.log(data['errors'][1:] + 1e-20)
            if np.all(np.isfinite(log_errors)) and np.all(np.isfinite(log_errors_next)):
                p = np.polyfit(log_errors, log_errors_next, 1)
                order = p[0]
                print(f"  Estimated convergence order: {order:.2f}")
                if order > 1.8:
                    print(f"    => Quadratic convergence (order approx 2)")
                elif order > 1.2:
                    print(f"    => Superlinear convergence (1 < order < 2)")
                elif 0.9 < order <= 1.1:
                    print(f"    => Linear convergence (order approx 1)")
                else:
                    print(f"    => Sublinear convergence (order < 1)")

conv_rate_rows = []
for method_name, data in convergence_data.items():
    if data['type'] != 'Not converged':
        for i, (err, ratio) in enumerate(zip(data['errors'][:-1], data['ratios_linear'])):
            conv_rate_rows.append({
                'Method': method_name,
                'Iteration': i,
                'Error': err,
                'Next_Error': data['errors'][i+1],
                'Ratio_e(k+1)/e(k)': ratio,
                'Type': data['type']
            })

df_conv_rate = pd.DataFrame(conv_rate_rows)
df_conv_rate.to_csv('./exp1/convergence_rate_analysis.csv', index=False)

summary_rows = []
for method_name, data in convergence_data.items():
    result = results[method_name]
    summary_rows.append({
        'Method': method_name,
        'Total_Iterations': result['iterations'],
        'Total_Time_(s)': method_times[method_name],
        'Avg_Time_per_Iter_(ms)': (method_times[method_name] / result['iterations']) * 1000,
        'Final_Error': np.linalg.norm(result['x'] - func.optimal_x),
        'Final_f(x)': result['f'],
        'Convergence_Type': data['type'],
        'Avg_Ratio': data.get('avg_ratio', np.nan),
        'Std_Ratio': data.get('std_ratio', np.nan),
        'Convergence_Order': np.nan
    })

    if data['type'] != 'Not converged' and len(data['errors']) >= 4:
        log_errors = np.log(data['errors'][:-1] + 1e-20)
        log_errors_next = np.log(data['errors'][1:] + 1e-20)
        if np.all(np.isfinite(log_errors)) and np.all(np.isfinite(log_errors_next)):
            p = np.polyfit(log_errors, log_errors_next, 1)
            summary_rows[-1]['Convergence_Order'] = p[0]

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv('./exp1/convergence_summary.csv', index=False)

fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111)

# Log-log plot of ||e_{k+1}|| vs ||e_k|| with linear fit
for method_name, result in results.items():
    x_hist = result['x_history']
    errors = np.array([np.linalg.norm(x - func.optimal_x) for x in x_hist])
    
    # Remove zeros (machine precision artifacts) and very large errors
    valid_mask = (errors > 1e-16) & (errors < 1.0)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) >= 3:
        # Get errors in valid region
        errors_valid = errors[valid_indices]
        log_ek = np.log10(errors_valid[:-1])
        log_ek_plus_1 = np.log10(errors_valid[1:])
        
        color = colors[method_name]
        marker = markers[method_name]
        
        ax3.plot(log_ek, log_ek_plus_1, marker, color=color, 
                 markersize=8, label=method_name, alpha=0.7)
        
        if len(log_ek) >= 2:
            if method_name == 'Gradient Descent':
                fit_indices = slice(None) 
            elif method_name == 'Newton':
                fit_mask = (errors_valid[:-1] < 0.02) & (errors_valid[:-1] > 1e-16)
                if np.sum(fit_mask) >= 2:
                    fit_log_ek = log_ek[fit_mask]
                    fit_log_ek_plus_1 = log_ek_plus_1[fit_mask]
                else:
                    fit_log_ek = log_ek[-3:] if len(log_ek) >= 3 else log_ek
                    fit_log_ek_plus_1 = log_ek_plus_1[-3:] if len(log_ek_plus_1) >= 3 else log_ek_plus_1
            else:  # BFGS
                fit_mask = errors_valid[:-1] < 0.1
                if np.sum(fit_mask) >= 3:
                    fit_log_ek = log_ek[fit_mask]
                    fit_log_ek_plus_1 = log_ek_plus_1[fit_mask]
                else:
                    fit_log_ek = log_ek
                    fit_log_ek_plus_1 = log_ek_plus_1
                    
            if method_name == 'Gradient Descent':
                coeffs = np.polyfit(log_ek, log_ek_plus_1, 1)
            else:
                coeffs = np.polyfit(fit_log_ek, fit_log_ek_plus_1, 1)
                
            p_order = coeffs[0]  
            
            log_ek_fit = np.linspace(log_ek.min(), log_ek.max(), 100)
            log_ek_plus_1_fit = coeffs[0] * log_ek_fit + coeffs[1]
            ax3.plot(log_ek_fit, log_ek_plus_1_fit, '--', color=color, 
                     linewidth=2, alpha=0.8,
                     label=f'{method_name} fit (order={p_order:.2f})')

ax3.plot([-15, 0], [-15, 0], 'k:', linewidth=1.5, alpha=0.5, label='Linear (order=1)')
ax3.plot([-15, 0], [-30, 0], 'k--', linewidth=1.5, alpha=0.5, label='Quadratic (order=2)')

ax3.set_xlabel(r'$\log_{10} ||x_k - x^*||$', fontsize=14)
ax3.set_ylabel(r'$\log_{10} ||x_{k+1} - x^*||$', fontsize=14)
ax3.set_title('Convergence Order Analysis', fontsize=15)
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-10, 0)
ax3.set_ylim(-25, 5)

plt.tight_layout()
plt.savefig('./exp1/convergence_rate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

for method_name, result in results.items():
    final_error = np.linalg.norm(result['x'] - func.optimal_x)
    conv_rate = convergence_data[method_name]['avg_ratio'] if convergence_data[method_name]['type'] != 'Not converged' else np.nan
    if np.isnan(conv_rate):
        conv_rate_str = "N/A"
    else:
        conv_rate_str = f"{conv_rate:.4f}"
    print(f"{method_name:<20} {result['iterations']:>12d} {result['f']:>15.2e} {final_error:>15.2e} {conv_rate_str:>12}")


# Helper function to find iteration where threshold is reached
def find_convergence_iter(x_history, optimal_x, threshold=1e-9):
    errors = np.array([np.linalg.norm(x - optimal_x) for x in x_history])
    converged_indices = np.where(errors < threshold)[0]
    if len(converged_indices) > 0:
        return converged_indices[0]
    return None

# Create performance comparison table
performance_table = []

for method_name, result in results.items():
    # Find iteration to reach 1e-9 precision
    iter_to_precision = find_convergence_iter(result['x_history'], func.optimal_x, threshold=1e-9)
    
    # Calculate metrics
    final_error = np.linalg.norm(result['x'] - func.optimal_x)
    final_f = result['f']
    total_iters = result['iterations']
    total_time = method_times[method_name]
    
    if method_name == 'Gradient Descent':
        func_evals = total_iters
        grad_evals = total_iters
        hess_evals = 0
    elif method_name == 'BFGS':
        func_evals = total_iters * 3
        grad_evals = total_iters
        hess_evals = 0
    else:
        func_evals = total_iters * 3  
        grad_evals = total_iters
        hess_evals = total_iters
    
    performance_table.append({
        'Method': method_name,
        'Iterations_to_1e-9': iter_to_precision if iter_to_precision is not None else np.nan,
        'Time_to_1e-9_(s)': (total_time * iter_to_precision / total_iters) if iter_to_precision is not None else np.nan,
        'Total_Iterations': total_iters,
        'Total_Time_(s)': total_time,
        'Final_Error': final_error,
        'Final_f(x)': final_f,
        'Func_Evals': func_evals,
        'Grad_Evals': grad_evals,
        'Hess_Evals': hess_evals,
        'Avg_Time_per_Iter_(ms)': (total_time / total_iters) * 1000
    })

df_performance = pd.DataFrame(performance_table)

# Save to CSV
df_performance.to_csv('./exp1/performance_comparison.csv', index=False)
bfgs_data = results['BFGS']
df_bfgs = pd.DataFrame({
    'k': range(min(10, len(bfgs_data['x_history']))),
    'x1': bfgs_data['x_history'][:10, 0],
    'x2': bfgs_data['x_history'][:10, 1],
    'f(x)': bfgs_data['f_history'][:10],
    '||grad f||': bfgs_data['grad_norm_history'][:10]
})