import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Roofline model parameters
bandwidth = 6.72e11  # in bytes per second
peak_performance = 16.31e12  # in FLOPS

# Data for plotting
operational_intensity = np.linspace(0.001, 10**5, 400)
bandwidth_limited_performance = bandwidth * operational_intensity

# Read the CSV file
csv_file = 'pyCuda/kernel_execution_times.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)
print(data)


operations = [
    {"name": "Matrix multiplication", "flops": 16382, "mem": 513, "oi": 32},
    {"name": "Compute delta2", "flops": 1, "mem": 513, "oi": 1.949e-3},
    {"name": "Add bias", "flops": 1, "mem": 3, "oi": 0.33333},
    {"name": "Sigmoid", "flops": 2, "mem": 2, "oi": 1},
    {"name": "Softmax", "flops": 514, "mem": 258, "oi": 2},
    {"name": "Compute delta1", "flops": 5, "mem": 3, "oi": 1.6667},
    {"name": "Compute derivative W", "flops": 16384, "mem": 514, "oi": 31.8754},
    {"name": "Compute derivative b", "flops": 258, "mem": 259, "oi": 1}
]

# Update operations with means and standard deviations
for op in operations:
    kernel_name = op["name"]
    if kernel_name in data.columns:
        op["flop_s"] = data[kernel_name].mean()
        op["flop_s_std"] = data[kernel_name].std()

# Use a distinct color palette
colors = plt.cm.tab10(np.linspace(0, 1, len(operations)))

# Plot the Roofline model
plt.figure(figsize=(15, 8))
plt.plot(operational_intensity, bandwidth_limited_performance, label='Bandwidth limitation', color='blue')
plt.hlines(peak_performance, min(operational_intensity), max(operational_intensity), label='Peak performance', colors='red')

# Add operation points and annotations
for idx, op in enumerate(operations):
    if "flop_s" in op:  # Check if the FLOPS value is available
        plt.scatter(op["oi"], op["flop_s"], color=colors[idx], alpha=0.7)
        plt.errorbar(op["oi"], op["flop_s"], yerr=op["flop_s_std"], fmt='o', color=colors[idx], capsize=5, alpha=0.7)
        if idx == 0:
            plt.annotate(op["name"], (op["oi"], op["flop_s"]),
                         textcoords="offset points", xytext=(0, 60),  # Adjust the offset
                         ha='center', color=colors[idx],
                         arrowprops=dict(arrowstyle="->", linestyle="dotted", color=colors[idx]),
                         fontsize=12)
        else:
            plt.annotate(op["name"], (op["oi"], op["flop_s"]),
                         textcoords="offset points", xytext=(0, 10 + 20 * (idx % 2)),  # Adjust the offset
                         ha='center', color=colors[idx],
                         arrowprops=dict(arrowstyle="->", linestyle="dotted", color=colors[idx]),
                         fontsize=12)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Operational Intensity (FLOPs/byte)', fontsize=14)
plt.ylabel('Performance (FLOPS)', fontsize=14)
plt.title('Roofline model - Nvidia Quadro RTX 6000', fontsize=14)
plt.legend(loc='upper left', fontsize=14)

plt.grid(True)
plt.show()
