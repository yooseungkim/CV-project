import json
import os

def main():
    attr_path = "data/CUB_200_2011/attributes/attributes.txt"
    out_path = "data/CUB_200_2011/concept_config.json"
    
    if not os.path.exists(attr_path):
        print(f"Error: {attr_path} not found.")
        return
        
    attributes = []
    with open(attr_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # 1 has_bill_shape::curved_(up_or_down)
                attr_name = parts[1]
                attributes.append(attr_name)
                
    print(f"Total parsed attributes: {len(attributes)}")
    
    # Group by concept name (part before ::)
    grouped = {}
    ordered_concepts = []
    
    for attr in attributes:
        if "::" in attr:
            concept, category = attr.split("::", 1)
        else:
            concept = attr
            category = "present"
            
        if concept not in grouped:
            grouped[concept] = []
            ordered_concepts.append(concept)
        grouped[concept].append(category)
        
    # Build concept_config
    concept_config = {}
    for concept in ordered_concepts:
        classes = grouped[concept]
        # Check if there is only 1 class and it is 'present'
        if len(classes) == 1 and classes[0] == "present":
            concept_config[concept] = {
                "type": "numerical",
                "min": 0.0,
                "max": 1.0
            }
        else:
            concept_config[concept] = {
                "type": "categorical",
                "classes": classes
            }
            
    # Verify that the flattened concepts match the original attributes list
    flattened = []
    for concept, info in concept_config.items():
        if info["type"] == "categorical":
            for cls in info["classes"]:
                flattened.append(f"{concept}::{cls}")
        else:
            flattened.append(concept)
            
    assert len(flattened) == len(attributes), f"Mismatch in length: {len(flattened)} vs {len(attributes)}"
    for i, (f, a) in enumerate(zip(flattened, attributes)):
        assert f == a, f"Mismatch at index {i}: {f} vs {a}"
        
    print("Verification passed! Flattened concepts match the original order perfectly.")
    
    # Save the new concept config
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(concept_config, f, indent=2)
    print(f"Saved new concept configuration to: {out_path}")

if __name__ == "__main__":
    main()
