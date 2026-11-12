"""Read-only: dump code cells of notebooks to a text file in the scratchpad (argv[1]=nb, argv[2]=out)."""
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        f.write(f"\n##### CELL {i} [{c['cell_type']}]\n{src}\n")
