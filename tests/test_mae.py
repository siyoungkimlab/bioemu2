import shlex

import mdtraj
import numpy as np
import pytest
from mdtraj.core.topology import Double, Single

from bioemu2 import mae_text, save_frames, save_mae
from bioemu2.data_interface import Ligand, Protein, System
from bioemu2.structure import AllAtomStructure
from bioemu2.trajectory_io import coords_angstrom_to_trajectory


def _blocks(text: str) -> list[str]:
    """The f_m_ct blocks of an MAE text, header excluded."""
    parts = text.split("f_m_ct {")
    assert parts[0].strip().startswith("{")  # the version block
    return parts[1:]


def _atom_rows(block: str) -> list[list[str]]:
    body = block.split("m_atom[")[1].split("}")[0]
    rows = [line.strip() for line in body.splitlines()]
    data = rows[rows.index(":::") + 1 : len(rows) - 1 - rows[::-1].index(":::")]
    # quoted fields keep their quotes and may hold spaces (" C1 ", "LIG ")
    return [shlex.split(row, posix=False) for row in data if row and row != ":::"]


def _bond_rows(block: str) -> list[tuple[int, int, int]]:
    if "m_bond[" not in block:
        return []
    body = block.split("m_bond[")[1].split("}")[0]
    rows = [line.strip() for line in body.splitlines()]
    data = rows[rows.index(":::") + 1 : len(rows) - 1 - rows[::-1].index(":::")]
    out = []
    for row in data:
        if row and row != ":::":
            _, first, second, order = row.split()
            out.append((int(first), int(second), int(order)))
    return out


def test_mae_text_writes_one_block_per_frame_with_topology_orders_and_charges() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain(chain_id="L")
    residue = topology.add_residue("LIG", chain, resSeq=7)
    carbon = topology.add_atom("C1", mdtraj.element.carbon, residue)
    oxygen = topology.add_atom("O1", mdtraj.element.oxygen, residue, formal_charge=-1)
    nitrogen = topology.add_atom("NA", mdtraj.element.nitrogen, residue)
    topology.add_bond(carbon, oxygen, type=Double, order=2)
    topology.add_bond(carbon, nitrogen, type=Single, order=1)
    trajectory = mdtraj.Trajectory(
        xyz=np.array([[[0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [0.0, 0.14, 0.0]]] * 2), topology=topology
    )

    text = mae_text(trajectory, titles=["first", "second"])

    blocks = _blocks(text)
    assert len(blocks) == 2
    assert '"first"' in blocks[0] and '"second"' in blocks[1]
    rows = _atom_rows(blocks[0])
    assert [row[0] for row in rows] == ["1", "2", "3"]
    assert [row[1] for row in rows] == ["7", "7", "7"]  # residue number
    assert rows[0][2] == '"L"'  # chain name
    assert rows[0][3] == '"LIG "'  # residue name padded to four
    assert rows[0][4] == '" C1 "' and rows[2][4] == '" NA "'  # one-letter element indented
    assert [row[5] for row in rows] == ["6", "8", "7"]  # atomic numbers
    assert [row[6] for row in rows] == ["0", "-1", "0"]  # formal charges; unknown is 0
    assert [float(value) for value in rows[1][7:10]] == pytest.approx([1.2, 0.0, 0.0])  # A
    assert _bond_rows(blocks[0]) == [(1, 2, 2), (1, 3, 1)]
    assert _bond_rows(blocks[1]) == [(1, 2, 2), (1, 3, 1)]


def test_mae_text_takes_bond_orders_and_charges_from_structure_and_kekulizes() -> None:
    system = System(
        entities=[
            Protein(sequence="GA", asym_ids=["A"]),
            Ligand(smiles="c1ccccc1C(=O)[O-]", asym_ids=["L"]),  # benzoate
        ]
    )
    structure = AllAtomStructure.from_system(system)
    trajectory = coords_angstrom_to_trajectory(structure.coords, system.topology)

    text = mae_text(trajectory, structure=structure)

    (block,) = _blocks(text)
    rows = _atom_rows(block)
    assert len(rows) == structure.num_nodes
    charges = [int(row[6]) for row in rows]
    assert sum(charges) == -1 and charges.count(-1) == 1
    ligand_indices = {
        index + 1 for index, row in enumerate(rows) if row[2] == '"L"'
    }
    ligand_bonds = [b for b in _bond_rows(block) if b[0] in ligand_indices and b[1] in ligand_indices]
    orders = sorted(order for _, _, order in ligand_bonds)
    # benzoate heavy atoms: ring 3 single + 3 double (kekulized), C-C single, C=O double, C-O single
    assert orders == [1, 1, 1, 1, 1, 2, 2, 2, 2]
    # every topology bond is present, including the peptide bond between residues
    assert len(_bond_rows(block)) >= trajectory.topology.n_bonds


def test_save_mae_rejects_structure_with_other_atoms(tmp_path) -> None:
    system = System(entities=[Protein(sequence="GA", asym_ids=["A"])])
    structure = AllAtomStructure.from_system(system)
    trajectory = coords_angstrom_to_trajectory(structure.coords, system.topology)
    other = AllAtomStructure.from_system(System(entities=[Protein(sequence="GAG", asym_ids=["A"])]))
    with pytest.raises(ValueError, match="same atoms"):
        save_mae(trajectory, tmp_path / "x.mae", structure=other)


def test_mae_text_rejects_bad_inputs() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain(chain_id="A")
    residue = topology.add_residue("GLY", chain)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(np.zeros((1, 1, 3)), topology)
    with pytest.raises(ValueError, match="at least one frame"):
        mae_text(trajectory[0:0])
    with pytest.raises(ValueError, match="one entry per"):
        mae_text(trajectory, titles=["a", "b"])
    with pytest.raises(ValueError, match="shape"):
        mae_text(trajectory, bfactors=np.zeros((2, 1)))


def test_save_frames_writes_mae_on_request(tmp_path) -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain(chain_id="A")
    residue = topology.add_residue("GLY", chain, resSeq=1)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(np.zeros((3, 1, 3)), topology)

    save_frames(trajectory, tmp_path / "plain")
    save_frames(trajectory, tmp_path / "with_mae", mae=True)

    assert not (tmp_path / "plain" / "samples.mae").exists()
    # The topology is MAE by default: the first frame alone, and no CIF beside it.
    assert len(_blocks((tmp_path / "plain" / "topology.mae").read_text())) == 1
    assert not (tmp_path / "plain" / "topology.cif").exists()
    save_frames(trajectory, tmp_path / "with_cif", cif=True)
    assert (tmp_path / "with_cif" / "topology.mae").is_file()
    assert (tmp_path / "with_cif" / "topology.cif").is_file()
    text = (tmp_path / "with_mae" / "samples.mae").read_text()
    assert len(_blocks(text)) == 3
    assert text.startswith("{\n  s_m_m2io_version")


def test_save_mae_writes_bfactors(tmp_path) -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain(chain_id="A")
    residue = topology.add_residue("GLY", chain, resSeq=1)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    topology.add_atom("N", mdtraj.element.nitrogen, residue)
    trajectory = mdtraj.Trajectory(np.zeros((1, 2, 3)), topology)
    path = save_mae(trajectory, tmp_path / "b.mae", bfactors=np.array([[87.5, 12.25]]))
    rows = _atom_rows(_blocks(path.read_text())[0])
    assert [row[10] for row in rows] == ["87.50", "12.25"]
