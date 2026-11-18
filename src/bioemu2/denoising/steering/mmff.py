"""MMFF94 angle and torsion parameters for SMILES ligands, for steering.

Why this exists at all. BioEmu2 already ships bond and angle restraints, but
neither reaches a small molecule:

  * ``BondLengthPotential`` and ``BondAnglePotential`` resolve their reference
    values from ``assets/stereo_chemistry.yaml``, which covers 34 residues --
    the 20 amino acids, MSE, the nucleotides, and the peptide/NA link types.
    A ligand angle matches nothing and falls through to
    ``ANGLE_FALLBACK = (60, 180, 3)`` degrees, which at the default
    ``sigma_tolerance=10`` is an allowed window of 30 to 180 degrees. Bonds fall
    through to 0.8 to 2.5 A. A flat six-ring at 120 degrees and 1.39 A sits
    inside both, so the restraint exerts no force on it.

  * The RDKit per-residue constraints (PoseBusters, chiral, stereo, planar) are
    built only for ligands given by CCD code; ``ligand_ccd_codes_from_system``
    skips SMILES and InChI ligands, and the potentials then evaluate empty
    tensors.

So a SMILES ligand reaches the denoiser with effectively no geometric restraint,
which is what lets the model return benzene's geometry when given cyclohexane.

The two terms here are the MMFF94 forms, scoped to ligand atoms:

  angle    E = k (1 - cos(theta - theta0)), theta0 from MMFF's own typing.
           Zero force at theta0, force bounded by k, and at theta0 = 180 it
           reduces to MMFF's linear-bend form, so nitriles and alkynes need no
           special case.

  torsion  E = 1/2 [V1 (1 + cos phi) + V2 (1 - cos 2 phi) + V3 (1 + cos 3 phi)]

Atom identity is resolved by name, not by index. ``_template_from_rdmol``
builds a ligand as ``RemoveAllHs(AddHs(mol))`` after assigning names with
``_assign_mol_atom_names``, so rebuilding the molecule the same way and reading
the names back off the batch gives a mapping that does not depend on either
side's atom ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from bioemu2.data_interface import System
from bioemu2.data_interface.system import Ligand
from bioemu2.denoising.steering.feats import (
    _atom_name_idx_map,
    _iter_ligand_residues,
    convert_densebatch_to_potential_features,
)
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.templates import _assign_mol_atom_names

logger = logging.getLogger(__name__)

# Reference bond stiffness, kcal/mol/A^2: MMFF's own value for an sp3 C-C.
# Bond force constants are reported as multiples of this.
K_REF = 306.4


@dataclass
class MMFFTerms:
    """Packed ligand MMFF terms, shaped like the other potentials' features.

    angle_index is [1, 3, P] holding (outer_i, outer_k, center) to match
    ``AnglePotential``'s row convention; torsion_index is [1, 4, Q] holding
    (i, j, k, l) in path order, which is what ``DihedralPotential`` expects.
    """

    bond_index: torch.Tensor
    bond_r0: torch.Tensor
    bond_kb: torch.Tensor
    angle_index: torch.Tensor
    angle_theta0: torch.Tensor
    torsion_index: torch.Tensor
    torsion_v1: torch.Tensor
    torsion_v2: torch.Tensor
    torsion_v3: torch.Tensor


def ligand_smiles_from_system(system: System | None) -> dict[int, str]:
    """Map ``{int_asym_id: smiles}`` for SMILES-specified ligand chains.

    Mirrors ``ligand_smiles_from_system``'s CCD counterpart: chains are numbered
    in entity order so the ids line up with ``batch.asym_id``.
    """
    if system is None:
        return {}
    smiles: dict[int, str] = {}
    int_asym_id = 0
    for entity in system.entities:
        for _ in entity.asym_ids:
            if isinstance(entity, Ligand) and getattr(entity, "smiles", None):
                smiles[int_asym_id] = entity.smiles
            int_asym_id += 1
    return smiles


def _named_mol(smiles: str):
    """Rebuild the ligand exactly as the template builder does, names included."""
    from rdkit.Chem import AddHs, MolFromSmiles

    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_w_hs = AddHs(mol)
    _assign_mol_atom_names(mol_w_hs)
    return mol_w_hs


def _bond_terms(mol_w_hs, properties) -> list[tuple[str, str, float, float]]:
    """Every heavy-atom bond, as (name_i, name_j, r0 in A, force constant).

    GetMMFFBondStretchParams returns (bondType, kb, r0); kb is in mdyn/A, which
    the 143.9325/2 factor converts to kcal/mol/A^2 so this term is in the same
    units as the torsion term's V1/V2/V3.
    """
    terms: list[tuple[str, str, float, float]] = []
    for bond in mol_w_hs.GetBonds():
        i_atom, j_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        if i_atom.GetAtomicNum() == 1 or j_atom.GetAtomicNum() == 1:
            continue
        params = properties.GetMMFFBondStretchParams(
            mol_w_hs, i_atom.GetIdx(), j_atom.GetIdx()
        )
        if params is None:
            continue
        kb, r0 = float(params[1]), float(params[2])
        # Normalized by K_REF, an sp3 C-C, rather than left in kcal/mol/A^2.
        # The angle term carries k = 1 per angle, so a bond term in true MMFF
        # units is ~300x stiffer on the same guidance_weight scale; at 0.10 it
        # diverged and blew every ligand out to 14-24 A bonds. Relative
        # stiffness survives: an amide C=O stays ~3x a C-C.
        terms.append((i_atom.GetProp("name"), j_atom.GetProp("name"),
                      r0, 0.5 * 143.9325 * kb / K_REF))
    return terms


def _angle_terms(mol_w_hs, properties) -> list[tuple[str, str, str, float]]:
    """Every heavy-atom bond angle, as (name_i, name_k, name_center, theta0 rad)."""
    import math

    terms: list[tuple[str, str, str, float]] = []
    for center in mol_w_hs.GetAtoms():
        if center.GetAtomicNum() == 1:
            continue
        neighbours = [n for n in center.GetNeighbors() if n.GetAtomicNum() != 1]
        for a in range(len(neighbours)):
            for b in range(a + 1, len(neighbours)):
                i, j, k = neighbours[a].GetIdx(), center.GetIdx(), neighbours[b].GetIdx()
                params = properties.GetMMFFAngleBendParams(mol_w_hs, i, j, k)
                if params is None:
                    continue
                # (angleType, ka, theta0 in degrees)
                theta0 = math.radians(float(params[2]))
                terms.append((
                    neighbours[a].GetProp("name"),
                    neighbours[b].GetProp("name"),
                    center.GetProp("name"),
                    theta0,
                ))
    return terms


def _torsion_terms(mol_w_hs, properties) -> list[tuple[str, str, str, str, float, float, float]]:
    """Heavy-atom torsions worth restraining, as (names..., V1, V2, V3).

    Scope, carried over from the Boltz measurements: every torsion about a ring
    bond, plus non-ring torsions whose V2 is positive. A positive V2 is a
    planarity constraint (an amide, a conjugated bond) rather than a
    conformational preference. In a ring the angle term opposes the torsion term
    and the pair reproduces the full force field; a freely rotatable torsion with
    V2 <= 0 has nothing to oppose it here and widens the distribution instead of
    sharpening it.
    """
    terms: list[tuple[str, str, str, str, float, float, float]] = []
    for bond in mol_w_hs.GetBonds():
        j_atom, k_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        if j_atom.GetAtomicNum() == 1 or k_atom.GetAtomicNum() == 1:
            continue
        j, k = j_atom.GetIdx(), k_atom.GetIdx()
        in_ring = bond.IsInRing()
        for i_atom in j_atom.GetNeighbors():
            if i_atom.GetIdx() == k or i_atom.GetAtomicNum() == 1:
                continue
            for l_atom in k_atom.GetNeighbors():
                if l_atom.GetIdx() == j or l_atom.GetAtomicNum() == 1:
                    continue
                if i_atom.GetIdx() == l_atom.GetIdx():
                    continue  # three-membered ring: i and l are the same atom
                params = properties.GetMMFFTorsionParams(
                    mol_w_hs, i_atom.GetIdx(), j, k, l_atom.GetIdx()
                )
                if params is None:
                    continue
                # (torsionType, V1, V2, V3) -- the leading element is the type
                # index, not V1. Reading params[0:3] puts benzene's
                # planarity-enforcing V2 into the V3 slot and inverts it.
                v1, v2, v3 = float(params[1]), float(params[2]), float(params[3])
                if v1 == 0.0 and v2 == 0.0 and v3 == 0.0:
                    continue
                if not in_ring and v2 <= 0.0:
                    continue
                terms.append((
                    i_atom.GetProp("name"), j_atom.GetProp("name"),
                    k_atom.GetProp("name"), l_atom.GetProp("name"), v1, v2, v3,
                ))
    return terms


def mmff_terms_from_system(system: System) -> MMFFTerms:
    """Collect MMFF angle and torsion terms for every SMILES ligand in the system.

    Returns empty (but correctly shaped) tensors when the system holds no SMILES
    ligand, so the potentials become no-ops rather than failing.
    """
    from rdkit import RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    structure = AllAtomStructure.from_system(system, mode="structure")
    batch = batch_collate([structure])
    feats = convert_densebatch_to_potential_features(batch)
    device = batch.coords.device
    atom_chain = batch.upsample_to_atoms(feats["asym_id"]).long()
    smiles_by_chain = ligand_smiles_from_system(system)

    bond_rows: list[list[int]] = [[], []]
    r0s: list[float] = []
    kbs: list[float] = []
    angle_rows: list[list[int]] = [[], [], []]
    theta0: list[float] = []
    torsion_rows: list[list[int]] = [[], [], [], []]
    v1s: list[float] = []
    v2s: list[float] = []
    v3s: list[float] = []

    for atom_lo, atom_hi in _iter_ligand_residues(batch, 0, atom_chain[0]):
        chain_int = int(atom_chain[0, atom_lo].item())
        smiles = smiles_by_chain.get(chain_int)
        if smiles is None:
            continue
        mol_w_hs = _named_mol(smiles)
        if mol_w_hs is None:
            logger.warning("MMFF steering: RDKit could not parse SMILES %r", smiles)
            continue
        properties = AllChem.MMFFGetMoleculeProperties(mol_w_hs)
        if properties is None:
            logger.warning("MMFF steering: no MMFF94 typing for SMILES %r", smiles)
            continue

        # The batch stores heavy atoms only, so map by name through the
        # hydrogen-free molecule the template builder actually used.
        from rdkit.Chem import RemoveAllHs

        mol_wo_hs = RemoveAllHs(mol_w_hs)
        idx_map = _atom_name_idx_map(batch, 0, atom_lo, atom_hi, mol_wo_hs)
        if not idx_map:
            logger.warning("MMFF steering: no atom-name match for SMILES %r", smiles)
            continue
        by_name = {
            mol_wo_hs.GetAtomWithIdx(ref_idx).GetProp("name"): global_idx
            for ref_idx, global_idx in idx_map.items()
        }

        for name_i, name_j, r0, kb in _bond_terms(mol_w_hs, properties):
            if name_i in by_name and name_j in by_name:
                bond_rows[0].append(by_name[name_i])
                bond_rows[1].append(by_name[name_j])
                r0s.append(r0)
                kbs.append(kb)

        for name_i, name_k, name_c, value in _angle_terms(mol_w_hs, properties):
            if name_i in by_name and name_k in by_name and name_c in by_name:
                angle_rows[0].append(by_name[name_i])
                angle_rows[1].append(by_name[name_k])
                angle_rows[2].append(by_name[name_c])
                theta0.append(value)

        for name_i, name_j, name_k, name_l, v1, v2, v3 in _torsion_terms(
            mol_w_hs, properties
        ):
            names = (name_i, name_j, name_k, name_l)
            if not all(n in by_name for n in names):
                continue
            for row, name in zip(torsion_rows, names):
                row.append(by_name[name])
            v1s.append(v1)
            v2s.append(v2)
            v3s.append(v3)

    def pack_index(rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.long, device=device).reshape(1, len(rows), -1)

    def pack_values(values: list[float]) -> torch.Tensor:
        return torch.tensor([values], dtype=torch.float32, device=device)

    return MMFFTerms(
        bond_index=pack_index(bond_rows),
        bond_r0=pack_values(r0s),
        bond_kb=pack_values(kbs),
        angle_index=pack_index(angle_rows),
        angle_theta0=pack_values(theta0),
        torsion_index=pack_index(torsion_rows),
        torsion_v1=pack_values(v1s),
        torsion_v2=pack_values(v2s),
        torsion_v3=pack_values(v3s),
    )
