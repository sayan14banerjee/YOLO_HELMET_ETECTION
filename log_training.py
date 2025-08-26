import pandas as pd
from datetime import datetime
import os

# Paths
run_dir = "runs/detect/train"   # change if you trained multiple times (train2, train3...)
csv_path = os.path.join(run_dir, "results.csv")
weights_path = os.path.join(run_dir, "weights", "best.pt")
log_file = "docs/training_logs.md"

# Load training results
df = pd.read_csv(csv_path)

# Get last epoch row (best summary is usually last line)
last_row = df.iloc[-1]

precision = last_row["metrics/precision(B)"]
recall = last_row["metrics/recall(B)"]
map50 = last_row["metrics/mAP50(B)"]
map50_95 = last_row["metrics/mAP50-95(B)"]

# Append to markdown log
with open(log_file, "a") as f:
    f.write(f"\n## Run ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
    f.write(f"- **Backbone:** yolov8n.pt\n")
    f.write(f"- **Epochs:** {len(df)}\n")
    f.write(f"- **Best Model:** `{weights_path}`\n")
    f.write(f"- **Results:**\n")
    f.write(f"  - Precision: {precision:.4f}\n")
    f.write(f"  - Recall: {recall:.4f}\n")
    f.write(f"  - mAP@50: {map50:.4f}\n")
    f.write(f"  - mAP@50-95: {map50_95:.4f}\n")
    f.write(f"- Training Curves: ![Results]({run_dir}/results.png)\n")
