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

# Scenario configs
scenarios = {
    "Current": ["Current_OF", "Current_MDF", "Current_VDF"],
    "SSP1-26": ["SSP126_OF", "SSP126_MDF", "SSP126_VDF"],
    "SSP5-85": ["SSP585_OF", "SSP585_MDF", "SSP585_VDF"]
}

# Susceptibility order for sorting
sus_order = ["High", "Moderate", "Least"]

# Loop over each scenario to create separate plots
for scenario_name, cols in scenarios.items():
    # Create long-form DataFrame
    scenario_df = df[["Division", "Susceptibility"] + cols].copy()
    scenario_df.columns = ["Division", "Susceptibility", "OF", "MDF", "VDF"]
    scenario_df["Susceptibility"] = pd.Categorical(scenario_df["Susceptibility"], categories=sus_order, ordered=True)
    scenario_df.sort_values(by=["Susceptibility", "Division"], inplace=True)
    scenario_df["Label"] = scenario_df["Susceptibility"].astype(str) + "\n" + scenario_df["Division"].astype(str)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scenario_df))
    bar_width = 0.6

    of_vals = scenario_df["OF"].values
    mdf_vals = scenario_df["MDF"].values
    vdf_vals = scenario_df["VDF"].values

    # Draw stacked bars
    ax.bar(x, of_vals, width=bar_width, label="OF", color="#efa00f")
    ax.bar(x, mdf_vals, width=bar_width, bottom=of_vals, label="MDF", color="#64d05f")
    ax.bar(x, vdf_vals, width=bar_width, bottom=of_vals + mdf_vals, label="VDF", color="#36802d")

    # Susceptibility separation lines
    prev_susc = None
    for i, susc in enumerate(scenario_df["Susceptibility"]):
        if susc != prev_susc and prev_susc is not None:
            ax.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=1)
        prev_susc = susc

    # Formatting
    
    scenario_df["Label"] = scenario_df["Susceptibility"].astype(str) + "\n" + scenario_df["Division"].astype(str)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_df["Label"], rotation=45, ha='right')
    ax.set_ylabel("Area")
    ax.set_title(f"Forest Density Distribution - {scenario_name}")
    ax.legend(title="Forest Density")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
