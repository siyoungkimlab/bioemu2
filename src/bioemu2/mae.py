"""Write sampled structures as Maestro (MAE) files.

MAE carries what XTC and the topology CIF do not: every bond with its order and
every atom with its formal charge, so a sample can be prepared for simulation
without re-perceiving the ligand. Each frame becomes one ``f_m_ct`` block, so a
whole trajectory is one file that Maestro, PyMOL, and boonza read as a list of
structures.

Bond orders and charges come from the :class:`AllAtomStructure` the system was
built from when it is given, where ligand templates keep them; aromatic bonds are
kekulized through RDKit because MAE has no aromatic order. Without the structure,
the MDTraj topology's bond orders and formal charges are used, and unknown ones
are written as single bonds and zero charge.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mdtraj
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BondType as RDBondType

from bioemu2.data_interface.const import BondType
from bioemu2.structure.all_atom_structure import AllAtomStructure

logger = logging.getLogger(__name__)

__all__ = ["mae_text", "save_mae"]

_VERSION_BLOCK = "{\n  s_m_m2io_version\n  :::\n  2.0.0\n}\n"
_ATOM_COLUMNS = (
    "i_m_residue_number",
    "s_m_chain_name",
    "s_m_pdb_residue_name",
    "s_m_pdb_atom_name",
    "i_m_atomic_number",
    "i_m_formal_charge",
    "r_m_x_coord",
    "r_m_y_coord",
    "r_m_z_coord",
    "r_m_pdb_tfactor",
)
_BOND_COLUMNS = ("i_m_from", "i_m_to", "i_m_order")
_RDKIT_BOND = {
    BondType.SINGLE: RDBondType.SINGLE,
    BondType.DOUBLE: RDBondType.DOUBLE,
    BondType.TRIPLE: RDBondType.TRIPLE,
    BondType.AROMATIC: RDBondType.AROMATIC,
    BondType.OTHER_OR_UNK: RDBondType.SINGLE,
}
_ORDER = {RDBondType.SINGLE: 1, RDBondType.DOUBLE: 2, RDBondType.TRIPLE: 3}


def _quote(value: object) -> str:
    """Quote an MAE string value, escaping what the format reserves."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _pdb_atom_name(name: str, symbol: str) -> str:
    """Pad an atom name into its four-character PDB column.

    ``s_m_pdb_atom_name`` is the PDB field: a one-letter element is indented by
    one space, as a PDB file writes it, so readers that use the padding find the
    element in the right column.
    """
    name = name[:4]
    if len(name) < 4 and name[:1].isalpha() and len(symbol) < 2:
        name = " " + name
    return f"{name:<4}"


def _kekulized_orders(
    elements: list[int],
    charges: list[int],
    bond_index: torch.Tensor,
    bond_type: torch.Tensor,
) -> dict[tuple[int, int], int]:
    """Integer bond orders by atom pair, with aromatic bonds kekulized.

    The molecule is built from the heavy atoms and bonds as RDKit sees them with
    implicit hydrogens. If kekulization fails, which a damaged template can
    cause, the aromatic bonds are written as single bonds and a warning says so.
    """
    mol = Chem.RWMol()
    for element, charge in zip(elements, charges, strict=True):
        atom = Chem.Atom(int(element))
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)
    pairs = [tuple(sorted(pair)) for pair in bond_index.t().tolist()]
    types = [BondType(int(value)) for value in bond_type.tolist()]
    for (first, second), kind in zip(pairs, types, strict=True):
        if first == second or mol.GetBondBetweenAtoms(first, second) is not None:
            continue
        mol.AddBond(first, second, _RDKIT_BOND[kind])
        if kind is BondType.AROMATIC:
            mol.GetAtomWithIdx(first).SetIsAromatic(True)
            mol.GetAtomWithIdx(second).SetIsAromatic(True)
    orders: dict[tuple[int, int], int] = {}
    if any(kind is BondType.AROMATIC for kind in types):
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except (Chem.KekulizeException, Chem.AtomValenceException, ValueError) as error:
            logger.warning(
                "Aromatic bonds could not be kekulized and are written as single bonds: %s",
                error,
            )
    for bond in mol.GetBonds():
        pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        orders[tuple(sorted(pair))] = _ORDER.get(bond.GetBondType(), 1)
    return orders


def _bonds(
    topology: mdtraj.Topology, structure: AllAtomStructure | None
) -> list[tuple[int, int, int]]:
    """(from, to, order) for every bond of the topology, 0-based atom indices.

    Orders come from ``structure`` where it has the bond, then from the MDTraj
    bond, then default to single. Bonds in ``structure`` that the topology lacks
    are included too, so a ligand keeps every bond of its template.
    """
    orders: dict[tuple[int, int], int] = {}
    if structure is not None:
        if structure.num_nodes != topology.n_atoms:
            raise ValueError(
                f"structure has {structure.num_nodes} atoms but the topology has "
                f"{topology.n_atoms}; they must describe the same atoms in the same order."
            )
        orders = _kekulized_orders(
            structure.element.tolist(),
            structure.charge.tolist(),
            structure.bond_index,
            structure.bond_type,
        )
    for bond in topology.bonds:
        pair = tuple(sorted((bond[0].index, bond[1].index)))
        if pair in orders:
            continue
        order = bond.order
        orders[pair] = int(order) if order in (1, 2, 3) else 1
    return [(first, second, order) for (first, second), order in sorted(orders.items())]


def _charges(topology: mdtraj.Topology, structure: AllAtomStructure | None) -> list[int]:
    if structure is not None:
        return [int(round(float(charge))) for charge in structure.charge.tolist()]
    return [
        0 if atom.formal_charge is None else int(atom.formal_charge) for atom in topology.atoms
    ]


def mae_text(
    trajectory: mdtraj.Trajectory,
    *,
    structure: AllAtomStructure | None = None,
    titles: list[str] | None = None,
    bfactors: np.ndarray | None = None,
    properties: list[dict[str, float]] | None = None,
) -> str:
    """The MAE text for ``trajectory``: one ``f_m_ct`` block per frame.

    Args:
        trajectory: Frames to write, in nanometres as MDTraj keeps them.
        structure: The system's structure, for bond orders and formal charges.
            It must have the trajectory's atoms in the same order.
        titles: One ``s_m_title`` per frame; ``sample_<i>`` by default.
        bfactors: Optional ``(frames, atoms)`` values for ``r_m_pdb_tfactor``,
            such as a per-atom confidence; zero by default.
        properties: One dict of ct-level properties per frame, such as the
            steering energy of each potential. Keys must be MAE property names
            (``r_`` real, ``i_`` integer, ``s_`` string, ``b_`` boolean) and
            every frame must carry the same keys, because the block header
            names them once.
    """
    if trajectory.n_frames == 0:
        raise ValueError("trajectory must contain at least one frame.")
    if trajectory.n_atoms == 0:
        raise ValueError("trajectory must contain at least one atom.")
    if titles is not None and len(titles) != trajectory.n_frames:
        raise ValueError("titles must contain one entry per trajectory frame.")
    if bfactors is not None and np.shape(bfactors) != (trajectory.n_frames, trajectory.n_atoms):
        raise ValueError("bfactors must have shape (frames, atoms).")
    prop_names: tuple[str, ...] = ()
    if properties is not None:
        if len(properties) != trajectory.n_frames:
            raise ValueError("properties must contain one dict per trajectory frame.")
        prop_names = tuple(properties[0])
        for entry in properties:
            if tuple(entry) != prop_names:
                raise ValueError("every properties entry must carry the same keys.")
        for name in prop_names:
            if not name.startswith(("r_", "i_", "s_", "b_")):
                raise ValueError(
                    f"MAE property {name!r} needs an r_/i_/s_/b_ type prefix.")
    topology = trajectory.topology
    bonds = _bonds(topology, structure)
    charges = _charges(topology, structure)
    atoms = list(topology.atoms)
    xyz = trajectory.xyz * 10.0  # nm -> A
    chunks = [_VERSION_BLOCK]
    for frame in range(trajectory.n_frames):
        title = titles[frame] if titles is not None else f"sample_{frame}"
        lines = ["f_m_ct {", "  s_m_title"]
        lines.extend(f"  {name}" for name in prop_names)
        lines.append("  :::")
        lines.append(f"  {_quote(title)}")
        if prop_names:
            entry = properties[frame]
            for name in prop_names:
                value = entry[name]
                if name.startswith("r_"):
                    lines.append(f"  {float(value):.6f}")
                elif name.startswith("i_"):
                    lines.append(f"  {int(value)}")
                elif name.startswith("b_"):
                    lines.append(f"  {int(bool(value))}")
                else:
                    lines.append(f"  {_quote(value)}")
        lines.append(f"  m_atom[{len(atoms)}] {{")
        lines.extend(f"    {column}" for column in _ATOM_COLUMNS)
        lines.append("    :::")
        for index, atom in enumerate(atoms):
            residue = atom.residue
            chain_name = residue.chain.chain_id or ""
            symbol = atom.element.symbol if atom.element is not None else ""
            number = atom.element.atomic_number if atom.element is not None else 0
            x, y, z = xyz[frame, index]
            bfactor = float(bfactors[frame, index]) if bfactors is not None else 0.0
            lines.append(
                f"    {index + 1} {residue.resSeq} {_quote(chain_name)} "
                f"{_quote(f'{residue.name[:4]:<4}')} {_quote(_pdb_atom_name(atom.name, symbol))} "
                f"{number} {charges[index]} {x:.6f} {y:.6f} {z:.6f} {bfactor:.2f}"
            )
        lines.extend(["    :::", "  }"])
        if bonds:
            lines.append(f"  m_bond[{len(bonds)}] {{")
            lines.extend(f"    {column}" for column in _BOND_COLUMNS)
            lines.append("    :::")
            for index, (first, second, order) in enumerate(bonds, start=1):
                lines.append(f"    {index} {first + 1} {second + 1} {order}")
            lines.extend(["    :::", "  }"])
        lines.append("}")
        chunks.append("\n".join(lines) + "\n")
    return "\n".join(chunks)


def save_mae(
    trajectory: mdtraj.Trajectory,
    path: str | Path,
    *,
    structure: AllAtomStructure | None = None,
    titles: list[str] | None = None,
    bfactors: np.ndarray | None = None,
    properties: list[dict[str, float]] | None = None,
) -> Path:
    """Write ``trajectory`` to ``path`` as MAE; see :func:`mae_text`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mae_text(trajectory, structure=structure, titles=titles,
                             bfactors=bfactors, properties=properties))
    return path
