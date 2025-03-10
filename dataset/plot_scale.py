import matplotlib.pyplot as plt
import numpy as np

# Example data
model_sizes = ["Qwen-VL-3B", "Qwen-VL-7B", "Qwen-VL-72B"]  # X-axis labels
mm = [0.323, 0.391, 0.327]  # Multimodal performance
text = [0.353, 0.433, 0.87]  # Text-only performance
overall = [0.333, 0.41, 0.331]  # Overall performance

# Create the plot
x = np.arange(len(model_sizes))  # X-axis positions
width = 0.2  # Width of the bars (for bar plot)

# Line plot
plt.figure(figsize=(8, 5))
plt.plot(model_sizes, mm, linestyle="dashed", marker="o", label="Multimodal")
plt.plot(model_sizes, text, linestyle="dashed", marker="o", label="Text-only")
plt.plot(model_sizes, overall, linestyle="dashed", marker="o", label="Overall")

# Add labels and title
plt.xlabel("Model Size")
plt.ylabel("Performance")
plt.title("Scaling Laws of the Model")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Show the plot
plt.show()

# Alternatively, you can use a bar plot
plt.figure(figsize=(8, 5))
plt.bar(x - width, mm, width, label="Multimodal")
plt.bar(x, text, width, label="Text-only")
plt.bar(x + width, overall, width, label="Overall")

# Add labels and title
plt.xlabel("Model Size")
plt.ylabel("Performance")
plt.title("Scaling Laws of the Model")
plt.xticks(x, model_sizes)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Show the plot
plt.show()
