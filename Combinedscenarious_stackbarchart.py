import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Excel file
file_path = 'D:/FireData_27May2025/Excel_SSP-prone_Forestdensity.xlsx'
xls = pd.ExcelFile(file_path)
df = xls.parse('Sheet1')

# Rename columns
df.columns = [
    "Division", "Susceptibility", 
    "Current_OF", "Current_MDF", "Current_VDF", 
    "SSP126_OF", "SSP126_MDF", "SSP126_VDF", 
    "SSP585_OF", "SSP585_MDF", "SSP585_VDF"
]

# Convert to numeric
df[df.columns[2:]] = df[df.columns[2:]].apply(pd.to_numeric, errors='coerce')

# Susceptibility order
sus_order = ["High", "Moderate", "Least"]
df["Susceptibility"] = pd.Categorical(df["Susceptibility"], categories=sus_order, ordered=True)

# Melt into long-form for plotting
scenarios = {
    "Current": ["Current_OF", "Current_MDF", "Current_VDF"],
    "SSP1-26": ["SSP126_OF", "SSP126_MDF", "SSP126_VDF"],
    "SSP5-85": ["SSP585_OF", "SSP585_MDF", "SSP585_VDF"]
}

plot_df = pd.DataFrame()

for scenario_name, cols in scenarios.items():
    temp = df[["Division", "Susceptibility"] + cols].copy()
    temp.columns = ["Division", "Susceptibility", "OF", "MDF", "VDF"]
    temp["Scenario"] = scenario_name
    plot_df = pd.concat([plot_df, temp], ignore_index=True)

# Create combined label
plot_df["Group"] = plot_df["Susceptibility"].astype(str) + "\n" + plot_df["Division"]
plot_df.sort_values(by=["Susceptibility", "Division", "Scenario"], inplace=True)

# Unique group positions
groups = plot_df["Group"].unique()
x = np.arange(len(groups))
bar_width = 0.2

# Scenario position offsets with spacing
bar_width = 0.18
group_spacing = 0.06  # small gap between grouped bars
offsets = {"Current": -bar_width - group_spacing, "SSP1-26": 0, "SSP5-85": bar_width + group_spacing}
hatches = {"Current": "", "SSP1-26": "//", "SSP5-85": "xx"}
colors = {"OF": "#efa00f", "MDF": "#64d05f", "VDF": "#36802d"}

fig, ax = plt.subplots(figsize=(18, 8))

for scenario in scenarios.keys():
    scenario_data = plot_df[plot_df["Scenario"] == scenario]
    scenario_data = scenario_data.set_index("Group").loc[groups]  # Ensure correct order

    xpos = x + offsets[scenario]
    of_vals = scenario_data["OF"].values
    mdf_vals = scenario_data["MDF"].values
    vdf_vals = scenario_data["VDF"].values

    # Stack bars with hatch
    ax.bar(xpos, of_vals, width=bar_width, label=f"{scenario} - OF", 
           color=colors["OF"], hatch=hatches[scenario], edgecolor='black')
    ax.bar(xpos, mdf_vals, width=bar_width, bottom=of_vals, label=f"{scenario} - MDF", 
           color=colors["MDF"], hatch=hatches[scenario], edgecolor='black')
    ax.bar(xpos, vdf_vals, width=bar_width, bottom=of_vals + mdf_vals, label=f"{scenario} - VDF", 
           color=colors["VDF"], hatch=hatches[scenario], edgecolor='black')

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(groups, rotation=45, ha='right')
ax.set_ylabel("Area")
ax.set_title("Forest Density by Scenario, Division, and Susceptibility (with textures)")
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Unique legend by combining forest type and hatch
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicates
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left', title="Scenario & Forest Type")

plt.tight_layout()
plt.show()

