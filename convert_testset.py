from pathlib import Path
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from pprint import pformat

input_dir = Path("CPSea/Model_Generation_and_Evaluation/TestSet/LNR/LNR_receptors")

parser = PDBParser(QUIET=True)
sequences = {}

for pdb_f in sorted(input_dir.rglob("*.pdb")):
    if pdb_f.name.startswith("._"):
        continue

    structure = parser.get_structure(pdb_f.stem, pdb_f)
    chains = []
    for chain in structure[0]:
        sequence = "".join([seq1(res3.resname) for res3 in chain if is_aa(res3, standard=False)])
        if sequence:
            chains.append((chain.id, sequence))

    if len(chains) == 1:
        sequences[pdb_f.stem] = chains[0][1]
    else:
        for chain_id, sequence in chains:
            sequences[f"{pdb_f.stem}_{chain_id}"] = sequence

with open("src/LNR_Sequences.py", "w") as f:
    f.write("LNR_Sequences = ")
    f.write(pformat(sequences, sort_dicts=True))
    f.write("\n")

print("Total of", len(sequences), "sequences saved to src/LNR_Sequences.py")
