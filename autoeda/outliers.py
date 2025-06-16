# py module for outlier detection, capping and removal of uploaded csv
import pandas as pd
import numpy as np
from scipy.stats import zscore
import json
import os

# 1. Load and inspect the scaled dataset
df = pd.read_csv("/notebooks/output-files/autoEDA_scaled_output.csv")
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Containers for results
detection_methods = {}
outliers_detected = {}
rows_with_outliers = set()
outlier_flags = pd.DataFrame(index=df.index)

# 2. Detect Skewness and apply outlier detection
for col in numeric_cols:
    skew_val = df[col].skew()
    if abs(skew_val) < 1.0:
        # 3a. Z-score method
        method = "Z-score"
        z_scores = zscore(df[col])
        outliers = np.abs(z_scores) > 3
        lower_bound = df[col].mean() - 3 * df[col].std()
        upper_bound = df[col].mean() + 3 * df[col].std()
    else:
        # 3b. IQR method
        method = "IQR"
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

    detection_methods[col] = method
    outliers_detected[col] = int(outliers.sum())
    rows_with_outliers.update(df[outliers].index)

    # 4. Flag outliers
    outlier_flags[f"{col}_is_outlier"] = outliers.astype(int)

# Save flagged dataset
flagged_df = pd.concat([df, outlier_flags], axis=1)
flagged_df.to_csv("autoEDA_outliers_flagged.csv", index=False)

# 5. Outlier Capping (Winsorization)
capped_df = df.copy()
for col in numeric_cols:
    method = detection_methods[col]
    if method == "Z-score":
        lower_bound = df[col].mean() - 3 * df[col].std()
        upper_bound = df[col].mean() + 3 * df[col].std()
    else:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

    capped_df[col] = np.clip(df[col], lower_bound, upper_bound)

capped_df.to_csv("autoEDA_outliers_capped.csv", index=False)

# 6. Outlier Removal
rows_with_outliers = list(rows_with_outliers)
removed_df = df.drop(index=rows_with_outliers)
removed_df.to_csv("autoEDA_outliers_removed.csv", index=False)

# 7. Report Generation
report = {
    "detection_methods": detection_methods,
    "outliers_detected": outliers_detected,
    "rows_removed": len(rows_with_outliers),
    "output_files": {
        "flagged": "autoEDA_outliers_flagged.csv",
        "capped": "autoEDA_outliers_capped.csv",
        "removed": "autoEDA_outliers_removed.csv"
    }
}

with open("autoEDA_outlier_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("✅ Outlier detection, flagging, capping, and removal complete.")
print("📄 Report saved to: autoEDA_outlier_report.json")
