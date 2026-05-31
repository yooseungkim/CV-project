import sys
import json
import os

# Set up paths
sys.path.insert(0, os.path.abspath("."))

import app

# Populate necessary globals for app.build_app()
app.CONCEPT_CONFIG = {
  "has_bill_shape": {
    "type": "categorical",
    "classes": [
      "curved_(up_or_down)",
      "dagger",
      "hooked",
      "needle",
      "hooked_seabird"
    ]
  },
  "has_size": {
    "type": "categorical",
    "classes": [
      "large",
      "small"
    ]
  },
  "numerical_feature": {
    "type": "numerical",
    "min": 0.0,
    "max": 10.0
  }
}

concepts_flat = []
total_dims = 0
app.CONCEPT_GROUPS = []

for name, info in app.CONCEPT_CONFIG.items():
    ctype = info.get("type", "numerical")
    if ctype == "categorical":
        classes = info.get("classes", [])
        classes_str = [str(c) for c in classes]
        group = {
            "name": name,
            "type": "categorical",
            "classes": classes_str,
            "flat_indices": list(range(total_dims, total_dims + len(classes)))
        }
        for cls_val in classes:
            concepts_flat.append(f"{name}_{cls_val}")
        total_dims += len(classes)
    else:
        group = {
            "name": name,
            "type": "numerical",
            "min": float(info.get("min", 0.0)),
            "max": float(info.get("max", 1.0)),
            "flat_indices": [total_dims]
        }
        concepts_flat.append(name)
        total_dims += 1
    app.CONCEPT_GROUPS.append(group)

app.CONCEPT_NAMES = concepts_flat
app.NUM_CONCEPTS = total_dims

print("Building Gradio application interface...")
blocks_app = app.build_app()
print("🎉 Gradio interface built successfully without errors!")
