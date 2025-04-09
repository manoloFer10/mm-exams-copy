import matplotlib.pyplot as plt
import numpy as np

# Example data
model_sizes = ["Qwen-VL-3B", "Qwen-VL-7B", "Qwen-VL-72B"]  # X-axis labels
x_axis = [3, 7, 72]  # Actual x-axis values
overall = [35.63, 39.60, 52.95]  # Overall performance
mm = [33.79, 36.88, 48.41]  # Multimodal performance
text = [38.53, 43.96, 60.01]  # Text-only performance

# Create the plot
plt.figure(figsize=(8, 5))

# Line plot with x_axis values
plt.plot(x_axis, mm, linestyle="dashed", marker="o", label="Multimodal")
plt.plot(x_axis, text, linestyle="dashed", marker="o", label="Text-only")
plt.plot(x_axis, overall, linestyle="dashed", marker="o", label="Overall")

# Add labels and title
plt.ylabel("Performance")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.xscale("log")
# Set x-axis ticks and labels
# plt.xticks(x_axis, model_sizes)  # Use x_axis for positions and model_sizes for labels
plt.xticks(x_axis, model_sizes, rotation=45, ha="right")  # Rotate labels by 45 degrees

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.savefig("scaling_results.svg", format="svg", bbox_inches="tight")
plt.show()
# ##############################################
# # Example data
# model_sizes = ["Aya-Vision-8B", "Aya-Vision-32B"]  # X-axis labels
# x_axis = [8, 32]  # Actual x-axis values
# overall = [35.10674, 39.662]  # Overall performance
# mm = [32.36469, 36.2844]  # Multimodal performance
# text = [39.3039, 44.9886]  # Text-only performance

# # Create the plot
# plt.figure(figsize=(8, 5))

# # Line plot with x_axis values
# plt.plot(x_axis, mm, linestyle="dashed", marker="o", label="Multimodal")
# plt.plot(x_axis, text, linestyle="dashed", marker="o", label="Text-only")
# plt.plot(x_axis, overall, linestyle="dashed", marker="o", label="Overall")

# # Add labels and title
# plt.xlabel("Log Model Size")
# plt.ylabel("Performance")
# plt.title("Scaling Laws of the Model")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)

# plt.xscale("log")
# # Set x-axis ticks and labels
# # plt.xticks(x_axis, model_sizes)  # Use x_axis for positions and model_sizes for labels
# plt.xticks(x_axis, model_sizes, rotation=45, ha="right")  # Rotate labels by 45 degrees

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Combined data
# qwen_sizes = ["Qwen-VL-3B", "Qwen-VL-7B", "Qwen-VL-72B"]
# qwen_x = [3, 7, 72]
# qwen_overall = [35.63, 39.60, 52.95]
# qwen_mm = [33.79, 36.88, 48.41]
# qwen_text = [38.53, 43.96, 60.01]

# aya_sizes = ["Aya-Vision-8B", "Aya-Vision-32B"]
# aya_x = [8, 32]
# aya_overall = [35.11, 39.66]
# aya_mm = [32.36, 36.28]
# aya_text = [39.30, 44.99]

# # Create the plot with larger size
# plt.figure(figsize=(12, 7))

# colors = {
#     "mm": "#1f77b4",  # Classic blue
#     "text": "#ff7f0e",  # Classic orange
#     "overall": "#2ca02c",  # Classic green
# }

# plt.plot(
#     qwen_x,
#     qwen_text,
#     linestyle="dashed",
#     marker="o",
#     color=colors["text"],
#     markersize=8,
#     linewidth=2,
#     label="Qwen Text-only",
# )
# plt.plot(
#     aya_x,
#     aya_text,
#     linestyle="solid",
#     marker="s",
#     color=colors["text"],
#     markersize=8,
#     linewidth=2,
#     label="Aya Text-only",
# )
# plt.plot(
#     qwen_x,
#     qwen_mm,
#     linestyle="dashed",
#     marker="o",
#     color=colors["mm"],
#     markersize=8,
#     linewidth=2,
#     label="Qwen Multimodal",
# )
# plt.plot(
#     aya_x,
#     aya_mm,
#     linestyle="solid",
#     marker="s",
#     color=colors["mm"],
#     markersize=8,
#     linewidth=2,
#     label="Aya Multimodal",
# )
# plt.plot(
#     qwen_x,
#     qwen_overall,
#     linestyle="dashed",
#     marker="o",
#     color=colors["overall"],
#     markersize=8,
#     linewidth=2,
#     label="Qwen Overall",
# )
# plt.plot(
#     aya_x,
#     aya_overall,
#     linestyle="solid",
#     marker="s",
#     color=colors["overall"],
#     markersize=8,
#     linewidth=2,
#     label="Aya Overall",
# )

# # Add labels and title with larger font
# plt.ylabel("Accuracy", fontsize=12)

# # Customize legend
# legend = plt.legend(
#     loc="upper left",
#     frameon=True,
#     framealpha=0.9,
#     fontsize=10,
# )

# legend.get_frame().set_edgecolor("#DDDDDD")

# # Customize grid and axes
# plt.grid(True, linestyle="--", alpha=0.4, color="gray")
# plt.gca().set_facecolor("#FAFAFA")  # Very light gray background
# plt.gca().spines["top"].set_visible(False)
# plt.gca().spines["right"].set_visible(False)

# # Log scale and ticks
# plt.xscale("log")
# all_x = qwen_x + aya_x
# all_labels = qwen_sizes + aya_sizes
# plt.xticks(all_x, all_labels, rotation=45, ha="right", fontsize=10)
# plt.yticks(fontsize=10)

# # Add slight padding
# plt.xlim(2, 80)  # Add some padding on x-axis
# plt.ylim(30, 65)  # Adjust y-axis range to fit all data

# # Adjust layout
# plt.tight_layout()

# # Show the plot
# plt.show()
