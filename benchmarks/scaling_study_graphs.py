"""
this documents process the datas in file ./docs/source/quantum_expert_area/input_shape_logs/scaling_study_benchmark.csv

and plot them into 6 different graphs, to see how the wcnn model scales in term of :
- RAM
- memory footprint
- number of parameters.

graphs are saved on : ./docs/source/_static/img/graph_scaling_study.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


csv_filename = "scaling_study_benchmark.csv"
df = pd.read_csv(csv_filename)

df = df[df["status"] == "SUCCESS"].copy()
df["forward_backward_time_sec"] = df["forward_backward_time_sec"].astype(float)
df["num_parameters"] = df["num_parameters"].astype(float)
df["peak_memory_mb"] = df["peak_memory_mb"].astype(float)


df_batch = df[df["image_height"] == 4].sort_values(by="batch_size")
df_height = df[df["batch_size"] == 1].sort_values(by="image_height")


fig, axs = plt.subplots(3, 2, figsize=(16, 15))
ax1, ax2 = axs[0, 0], axs[0, 1]  # Line 1 : Runtime
ax3, ax4 = axs[1, 0], axs[1, 1]  # Line 2 : Parameters
ax5, ax6 = axs[2, 0], axs[2, 1]  # Line 3 : RAM


x_batch = df_batch["batch_size"].values
y_time_batch = df_batch["forward_backward_time_sec"].values
if len(x_batch) > 1:
    coefs = np.polyfit(x_batch, y_time_batch, 1)
    ax1.plot(x_batch, y_time_batch, 'o', label="Datas")
    ax1.plot(x_batch, np.polyval(coefs, x_batch), '-', color='red', label=f"Tendance\ny={coefs[0]:.2e}*x + {coefs[1]:.2e}")
ax1.set_title("Runtime vs Batch Size (Image: 4x4)")
ax1.set_ylabel("Time (s)")
ax1.grid(True, linestyle='--', alpha=0.7)
if len(x_batch) > 1: ax1.legend()

x_height = df_height["image_height"].values
y_time_height = df_height["forward_backward_time_sec"].values
if len(x_height) > 2:
    coefs = np.polyfit(x_height, y_time_height, 2)
    ax2.plot(x_height, y_time_height, 'o', label="Datas")
    ax2.plot(x_height, np.polyval(coefs, x_height), '-', color='orange', label=f"Tendance\ny={coefs[0]:.2e}*x² + {coefs[1]:.2e}*x + {coefs[2]:.2e}")
ax2.set_title("Runtime vs Image Height (Batch: 1)")
ax2.grid(True, linestyle='--', alpha=0.7)
if len(x_height) > 2: ax2.legend()


y_params_batch = df_batch["num_parameters"].values
if len(x_batch) > 0:
    ax3.plot(x_batch, y_params_batch, 'o-', color='purple', label="Paramètres")
    ax3.set_ylim(0, max(y_params_batch) * 1.5) 
ax3.set_title("Model footprint vs Batch Size")
ax3.set_ylabel("Total number of parameters")
ax3.grid(True, linestyle='--', alpha=0.7)
if len(x_batch) > 0: ax3.legend()

y_params_height = df_height["num_parameters"].values
if len(x_height) > 0:
    ax4.plot(x_height, y_params_height, 'o', color='purple', label="Paramètres réels")
    if len(x_height) > 2:
        coefs = np.polyfit(x_height, y_params_height, 2)
        ax4.plot(x_height, np.polyval(coefs, x_height), '-', color='blue', label=f"Tendance\ny={coefs[0]:.2e}*x² + {coefs[1]:.2e}*x + {coefs[2]:.2e}")
    ax4.set_ylim(0, max(y_params_height) * 1.5)
ax4.set_title("Model footprint vs Image Height")
ax4.grid(True, linestyle='--', alpha=0.7)
if len(x_height) > 0: ax4.legend()


y_ram_batch = df_batch["peak_memory_mb"].values
if len(x_batch) > 1:
    coefs = np.polyfit(x_batch, y_ram_batch, 1)
    ax5.plot(x_batch, y_ram_batch, 'o', color='green', label="real datas")
    ax5.plot(x_batch, np.polyval(coefs, x_batch), '-', color='teal', label=f"Tendance\ny={coefs[0]:.2e}*x + {coefs[1]:.2e}")
ax5.set_title("RAM consumption (Python) vs Batch Size")
ax5.set_xlabel("Batch Size")
ax5.set_ylabel("memory (MB)")
ax5.grid(True, linestyle='--', alpha=0.7)
if len(x_batch) > 1: ax5.legend()

y_ram_height = df_height["peak_memory_mb"].values
if len(x_height) > 2:
    coefs = np.polyfit(x_height, y_ram_height, 2)
    ax6.plot(x_height, y_ram_height, 'o', color='green', label="Data")
    ax6.plot(x_height, np.polyval(coefs, x_height), '-', color='teal', label=f"Tendany\ny={coefs[0]:.2e}*x² + {coefs[1]:.2e}*x + {coefs[2]:.2e}")
ax6.set_title("RAM Consumption (Python) vs Image Height")
ax6.set_xlabel("Image Height")
ax6.grid(True, linestyle='--', alpha=0.7)
if len(x_height) > 2: ax6.legend()

# ==========================================
# Save the graphs
# ==========================================
plt.tight_layout()
output_path = "./docs/source/_static/img/graph_scaling_study.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph saved on : {output_path}")