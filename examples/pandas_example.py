"""Example: Using NoviCode in pandas mode.

Start the agent:
    novicode --mode pandas

Then try these prompts:

1. "Load a CSV and show basic statistics"
2. "Create a bar chart from sample data"
3. "Show a correlation heatmap with seaborn"
"""

# This is what the agent might generate for prompt #2:

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Category": ["A", "B", "C", "D", "E"],
    "Value": [23, 45, 12, 67, 34],
}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 5))
plt.bar(df["Category"], df["Value"], color="steelblue")
plt.title("Sample Bar Chart")
plt.xlabel("Category")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig("bar_chart.png", dpi=150)
plt.show()
