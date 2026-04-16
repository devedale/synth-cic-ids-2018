import json

nb_path = r"CheckDataset.ipynb"

try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    markdowns = []
    # Clear outputs to avoid massive file sizes and extract markdown
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            if "execution_count" in cell:
                cell["execution_count"] = None
        elif cell.get("cell_type") == "markdown":
            markdowns.append({"index": i, "source": "".join(cell["source"])})

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    with open("markdown_texts.json", "w", encoding="utf-8") as f:
        json.dump(markdowns, f, indent=2)

    print("Success: Notebook stripped of outputs and markdowns extracted.")
except Exception as e:
    print("Error:", e)
