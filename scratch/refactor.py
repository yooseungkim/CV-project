import os

def main():
    main_py_path = "main.py"
    if not os.path.exists(main_py_path):
        print("Error: main.py not found in current directory.")
        return

    with open(main_py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Identify helpers block to delete
    helpers_start_anchor = "def str2bool(v):"
    helpers_end_anchor = "def parse_args():"
    
    start_helpers = content.find(helpers_start_anchor)
    end_helpers = content.find(helpers_end_anchor)

    if start_helpers == -1 or end_helpers == -1:
        print("Error: Could not find helper functions boundary in main.py.")
        return

    # 2. Identify training loops block to delete
    loops_start_anchor = "def train_phase1("
    loops_end_anchor = "def main():"

    start_loops = content.find(loops_start_anchor)
    end_loops = content.find(loops_end_anchor)

    if start_loops == -1 or end_loops == -1:
        print("Error: Could not find training loops boundary in main.py.")
        return

    # Verify order of slices
    assert start_helpers < end_helpers
    assert end_helpers < start_loops
    assert start_loops < end_loops

    print(f"Original content size: {len(content)} characters.")
    print(f"Helpers block size to delete: {end_helpers - start_helpers} characters.")
    print(f"Loops block size to delete: {end_loops - start_loops} characters.")

    # 3. Construct new content
    # New imports to add at the top (right after other src.utils imports)
    import_anchor = "from src.utils.visualization import generate_concept_heatmaps\n"
    import_idx = content.find(import_anchor)
    if import_idx == -1:
        print("Error: Could not find import anchor in main.py.")
        return
    
    insert_pos = import_idx + len(import_anchor)

    # Reconstruct in parts
    part1 = content[:insert_pos]
    
    # Add new modular imports
    part2 = (
        "\n# Modularized utility, loss, and training loop imports\n"
        "from src.utils.helpers import str2bool, str_or_float, str_or_bool, calculate_pos_weights, get_dataset_choices\n"
        "from src.utils.losses import SigmoidFocalLoss, GroupCrossEntropyLoss\n"
        "from src.utils.train_loops import train_phase1, train_phase2, train_phase3\n\n"
    )
    
    # From insert_pos to start of helpers (excluding the helpers themselves)
    part3 = content[insert_pos:start_helpers]
    
    # From parse_args to train_phase1 (excluding train_phase1 itself)
    part4 = content[end_helpers:start_loops]
    
    # From main() to end of file
    part5 = content[end_loops:]

    new_content = part1 + part2 + part3 + part4 + part5

    # Backup original main.py
    backup_path = "main.py.bak"
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Backup created at: {backup_path}")

    # Write refactored main.py
    with open(main_py_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Successfully refactored main.py! New size: {len(new_content)} characters.")

if __name__ == "__main__":
    main()
