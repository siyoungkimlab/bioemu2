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
import math
from dataclasses import dataclass

import torch

from bioemu2.data_interface import System
from bioemu2.data_interface.system import Ligand
from bioemu2.denoising.steering.feats import (
    MMFF_ELEMENTS,
    _atom_name_idx_map,
    _iter_ligand_residues,
    convert_densebatch_to_potential_features,
    unsupported_ligand_reason,
)
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.templates import _assign_mol_atom_names

logger = logging.getLogger(__name__)

# MMFF's own force constants are carried in full, in kcal/mol/A^2 and
# kcal/mol/rad^2. The published forms are
#   EB = 143.9325 * (kb/2) * dr^2 (+ cubic)   -> 143.9325/2 * kb  per A^2
#   EA = 0.043844 * (ka/2) * dth_deg^2 (+...) -> 143.9325/2 * ka per rad^2
# The angle term here is a cosine well, E = k (1 - cos dth), whose quadratic
# coefficient is k/2 rather than k, so matching MMFF's curvature needs
# k = 143.9325 * ka. The extra factor of 2 is the cosine form, not a units
# slip.
#
# These are large: an sp3 C-C bond is 306.4 and a carbonyl 932. The guidance
# loop takes unit descent steps, so guidance_weight has to be set against
# them -- that is a weighting decision, kept out of the parameters.


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
    angle_ka: torch.Tensor       # [1, P] 143.9325 * ka, kcal/mol/rad^2
    sb_index: torch.Tensor       # [1, 3, S] (outer_i, outer_k, center)
    sb_theta0: torch.Tensor      # [1, S] radians
    sb_r0_ij: torch.Tensor       # [1, S] angstrom, i-centre bond
    sb_r0_kj: torch.Tensor       # [1, S] angstrom, k-centre bond
    sb_kba_ijk: torch.Tensor     # [1, S] md/rad, weights dr_ij
    sb_kba_kji: torch.Tensor     # [1, S] md/rad, weights dr_kj
    torsion_index: torch.Tensor
    pair_index: torch.Tensor      # [1, 2, P]  1-4 pairs
    pair_rstar: torch.Tensor      # [1, P]  R*, angstrom
    pair_eps: torch.Tensor        # [1, P]  epsilon, kcal/mol
    pair_qq: torch.Tensor         # [1, P]  q_i q_j, 0.75 already applied
    nb_index: torch.Tensor        # [1, 2, N] pairs separated by >3 bonds
    nb_rstar: torch.Tensor        # [1, N]  R*, angstrom
    nb_eps: torch.Tensor          # [1, N]  epsilon, kcal/mol
    nb_qq: torch.Tensor           # [1, N]  q_i q_j, unscaled
    hydrogen_ref: torch.Tensor    # [1, 3, H] global indices of (parent, B, A)
    hydrogen_geom: torch.Tensor   # [1, H, 3] (r, theta, phi), radians
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
        # MMFF's own kb, in kcal/mol/A^2: 143.9325/2 * kb. An sp3 C-C is
        # 306.4 and a carbonyl 932.
        terms.append((i_atom.GetProp("name"), j_atom.GetProp("name"),
                      r0, 0.5 * 143.9325 * kb))
    return terms


def _angle_terms(mol_w_hs, properties) -> list[
        tuple[str, str, str, float, float]]:
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
                    143.9325 * float(params[1]),
                ))
    return terms


def _torsion_terms(mol_w_hs, properties) -> list[tuple[str, str, str, str, float, float, float]]:
    """Every MMFF torsion about a heavy-heavy bond, as (names..., V1, V2, V3).

    The central bond j-k must be heavy, but i and l may be hydrogens, so a
    ring C-C bond contributes all nine i-j-k-l paths rather than the single
    heavy one. That matters: an all-atom force field spreads a bond's
    torsional barrier over every path through it, and the heavy path alone is
    not a weaker version of the sum but a different function -- on cyclohexane
    all nine paths minimise at 60 and 180 degrees while the heavy path alone
    minimises at 180 and puts a maximum near the chair's 60. The hydrogens are
    reconstructed at evaluation time by HydrogenMixin.

    A torsion is dropped only when RDKit has no parameters for it or when
    V1 = V2 = V3 = 0. There is no scope filter: an earlier version kept only
    ring bonds plus non-ring torsions with V2 > 0, which was a workaround for
    the heavy-atom path having the wrong shape, and reconstructing hydrogens
    fixes that at the source.
    """
    terms: list[tuple[str, str, str, str, float, float, float]] = []
    for bond in mol_w_hs.GetBonds():
        j_atom, k_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        if j_atom.GetAtomicNum() == 1 or k_atom.GetAtomicNum() == 1:
            continue
        j, k = j_atom.GetIdx(), k_atom.GetIdx()
        in_ring = bond.IsInRing()
        for i_atom in j_atom.GetNeighbors():
            # Hydrogens are allowed at the OUTER positions: they carry most of
            # an sp3 bond's torsional barrier, and without them the heavy path
            # alone has the wrong shape. The central bond stays heavy.
            if i_atom.GetIdx() == k:
                continue
            for l_atom in k_atom.GetNeighbors():
                if l_atom.GetIdx() == j:
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
                # No scope filter: every torsion is kept. The old rule
                # (ring bonds, plus non-ring torsions with V2 > 0) was a
                # workaround for the heavy-atom path having the wrong shape,
                # which reconstructing hydrogens fixes at the source.
                terms.append((
                    i_atom.GetProp("name"), j_atom.GetProp("name"),
                    k_atom.GetProp("name"), l_atom.GetProp("name"), v1, v2, v3,
                ))
    return terms



def _stretchbend_terms(mol_w_hs, properties) -> list[
        tuple[str, str, str, float, float, float, float, float]]:
    """Stretch-bend cross terms, as (name_i, name_k, name_center,
    theta0 rad, r0_ij, r0_kj, kba_ijk, kba_kji).

    MMFF couples each angle to its two bonds:

        EBA = 2.51210 * (kba_ijk * dr_ij + kba_kji * dr_kj) * dth_deg

    The two kba are NOT interchangeable -- kba_ijk weights the i-centre bond
    and kba_kji the k-centre bond -- so the atom order that produced them must
    be the order the potential uses.

    Same scope as _angle_terms: heavy centre, heavy neighbours.
    """
    terms: list[tuple[str, str, str, float, float, float, float, float]] = []
    for center in mol_w_hs.GetAtoms():
        if center.GetAtomicNum() == 1:
            continue
        neighbours = [n for n in center.GetNeighbors() if n.GetAtomicNum() != 1]
        for a in range(len(neighbours)):
            for b in range(a + 1, len(neighbours)):
                i, j, k = (neighbours[a].GetIdx(), center.GetIdx(),
                           neighbours[b].GetIdx())
                sb = properties.GetMMFFStretchBendParams(mol_w_hs, i, j, k)
                ang = properties.GetMMFFAngleBendParams(mol_w_hs, i, j, k)
                b_ij = properties.GetMMFFBondStretchParams(mol_w_hs, i, j)
                b_kj = properties.GetMMFFBondStretchParams(mol_w_hs, k, j)
                if sb is None or ang is None or b_ij is None or b_kj is None:
                    continue
                kba_ijk, kba_kji = float(sb[1]), float(sb[2])
                if kba_ijk == 0.0 and kba_kji == 0.0:
                    continue
                terms.append((
                    neighbours[a].GetProp("name"),
                    neighbours[b].GetProp("name"),
                    center.GetProp("name"),
                    math.radians(float(ang[2])),
                    float(b_ij[2]),
                    float(b_kj[2]),
                    kba_ijk,
                    kba_kji,
                ))
    return terms


def _hydrogen_terms(mol_w_hs) -> list[tuple[str, str, str, str, float, float, float]]:
    """Each hydrogen as (name, parent, B, A, r, theta deg, phi deg).

    The molecule is embedded and minimised here purely to measure those
    internals; the values are local geometry and do not depend on the
    conformer beyond the rotamer of a freely rotating group, which for a
    torsion sum is immaterial.
    """
    import math

    import numpy as np
    from rdkit.Chem import AllChem

    work = type(mol_w_hs)(mol_w_hs)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(work, params) != 0:
        return []
    if AllChem.MMFFHasAllMoleculeParams(work):
        AllChem.MMFFOptimizeMolecule(work, maxIters=2000)
    else:
        AllChem.UFFOptimizeMolecule(work, maxIters=2000)
    pos = np.array(work.GetConformer().GetPositions())

    def angle(a, b, c):
        u, v = pos[a] - pos[b], pos[c] - pos[b]
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(u.dot(v))))))

    def dihedral(a, b, c, d):
        b0, b1, b2 = pos[a] - pos[b], pos[c] - pos[b], pos[d] - pos[c]
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - b0.dot(b1) * b1
        w = b2 - b2.dot(b1) * b1
        return math.degrees(math.atan2(np.cross(b1, v).dot(w), v.dot(w)))

    out = []
    for atom in work.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        h = atom.GetIdx()
        parent = atom.GetNeighbors()[0]
        if parent.GetAtomicNum() == 1:
            continue
        b_opts = [n.GetIdx() for n in parent.GetNeighbors() if n.GetAtomicNum() != 1]
        if not b_opts:
            continue
        b = b_opts[0]
        a_opts = [n.GetIdx() for n in work.GetAtomWithIdx(b).GetNeighbors()
                  if n.GetAtomicNum() != 1 and n.GetIdx() != parent.GetIdx()]
        a_opts += [x for x in b_opts if x != b]
        if not a_opts:
            continue
        a = a_opts[0]
        out.append((
            atom.GetProp("name"), parent.GetProp("name"),
            work.GetAtomWithIdx(b).GetProp("name"),
            work.GetAtomWithIdx(a).GetProp("name"),
            float(np.linalg.norm(pos[h] - pos[parent.GetIdx()])),
            angle(h, parent.GetIdx(), b),
            dihedral(h, parent.GetIdx(), b, a),
        ))
    return out


def _pair14_terms(mol_w_hs, properties):
    """1-4 pairs as (name_i, name_j, R_star, epsilon, q_i*q_j).

    A 1-4 pair is exactly three bonds apart, taken from the topological
    distance matrix so rings are handled correctly. MMFF applies no scaling to
    the 1-4 van der Waals and 0.75 to the electrostatics; the 0.75 is folded
    into the stored charge product here so the potential needs no constant.
    """
    from rdkit import Chem

    ELECTROSTATIC_14 = 0.75

    dmat = Chem.GetDistanceMatrix(mol_w_hs)
    out = []
    n = mol_w_hs.GetNumAtoms()
    for i in range(n):
        for j in range(i + 1, n):
            if dmat[i][j] != 3:
                continue
            params = properties.GetMMFFVdWParams(i, j)
            if params is None:
                continue
            # (R*_unscaled, eps_unscaled, R*, eps) -- the scaled pair is wanted
            r_star, epsilon = float(params[2]), float(params[3])
            q = (properties.GetMMFFPartialCharge(i)
                 * properties.GetMMFFPartialCharge(j) * ELECTROSTATIC_14)
            out.append((mol_w_hs.GetAtomWithIdx(i).GetProp("name"),
                        mol_w_hs.GetAtomWithIdx(j).GetProp("name"),
                        r_star, epsilon, float(q)))
    return out


def _nonbonded_terms(mol_w_hs, properties):
    """Pairs separated by more than three bonds: (name_i, name_j, sigma, eps, qq).

    1-2, 1-3 and 1-4 are excluded. The 1-4s are excluded because they are a
    separate term carrying MMFF's own 0.75 electrostatic scaling; counting
    them here as well would double them.

    R* is MMFF's pair minimum distance and is kept as such: the potential
    uses MMFF's own buffered 14-7, which is written in terms of R*, so there
    is nothing to convert.
    """
    from rdkit import Chem

    dmat = Chem.GetDistanceMatrix(mol_w_hs)
    out = []
    n = mol_w_hs.GetNumAtoms()
    for i in range(n):
        for j in range(i + 1, n):
            if dmat[i][j] <= 3:
                continue
            params = properties.GetMMFFVdWParams(i, j)
            if params is None:
                continue
            # (R*_unscaled, eps_unscaled, R*, eps) -- the scaled pair is wanted
            r_star, epsilon = float(params[2]), float(params[3])
            q = (properties.GetMMFFPartialCharge(i)
                 * properties.GetMMFFPartialCharge(j))
            out.append((mol_w_hs.GetAtomWithIdx(i).GetProp("name"),
                        mol_w_hs.GetAtomWithIdx(j).GetProp("name"),
                        r_star, epsilon, float(q)))
    return out


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
    pair_rows: list[list[int]] = [[], []]
    pair_rstar: list[float] = []
    nb_rows = [[], []]
    nb_rstar, nb_eps, nb_qq = [], [], []
    pair_eps: list[float] = []
    pair_qq: list[float] = []
    hydrogen_rows: list[list[int]] = [[], [], []]
    hydrogen_geom: list[list[float]] = []
    angle_rows: list[list[int]] = [[], [], []]
    theta0: list[float] = []
    angle_kas: list[float] = []
    sb_rows: list[list[int]] = [[], [], []]
    sb_theta0: list[float] = []
    sb_r0_ij: list[float] = []
    sb_r0_kj: list[float] = []
    sb_kba_ijk: list[float] = []
    sb_kba_kji: list[float] = []
    torsion_rows: list[list[int]] = [[], [], [], []]
    v1s: list[float] = []
    v2s: list[float] = []
    v3s: list[float] = []

    for atom_lo, atom_hi in _iter_ligand_residues(batch, 0, atom_chain[0]):
        chain_int = int(atom_chain[0, atom_lo].item())
        smiles = smiles_by_chain.get(chain_int)
        if smiles is None:
            continue
        reason = unsupported_ligand_reason(smiles, MMFF_ELEMENTS)
        if reason is not None:
            logger.info("MMFF steering: leaving %r unsteered (%s)", smiles, reason)
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

        # Hydrogens are not predicted by the model; they are reconstructed
        # from heavy atoms and appended after the real atoms, so a torsion can
        # name one by an index past the atom count.
        n_atoms = int(batch.atom_mask.shape[1])
        extended: dict[str, int] = {}
        for h_name, p_name, b_name, a_name, r, theta, phi in _hydrogen_terms(mol_w_hs):
            if not all(n in by_name for n in (p_name, b_name, a_name)):
                continue
            extended[h_name] = n_atoms + len(hydrogen_geom)
            for row, value in zip(hydrogen_rows,
                                  (by_name[p_name], by_name[b_name], by_name[a_name])):
                row.append(value)
            hydrogen_geom.append([r, math.radians(theta), math.radians(phi)])

        def resolve_ext(*names: str) -> tuple[int, ...] | None:
            out = []
            for n in names:
                if n in by_name:
                    out.append(by_name[n])
                elif n in extended:
                    out.append(extended[n])
                else:
                    return None
            return tuple(out)

        for name_i, name_j, rst, epsilon, qq in _nonbonded_terms(
                mol_w_hs, properties):
            got = resolve_ext(name_i, name_j)
            if got is None:
                continue
            nb_rows[0].append(got[0])
            nb_rows[1].append(got[1])
            nb_rstar.append(rst)
            nb_eps.append(epsilon)
            nb_qq.append(qq)

        for name_i, name_j, r_star, epsilon, qq in _pair14_terms(mol_w_hs, properties):
            got = resolve_ext(name_i, name_j)
            if got is None:
                continue
            pair_rows[0].append(got[0])
            pair_rows[1].append(got[1])
            pair_rstar.append(r_star)
            pair_eps.append(epsilon)
            pair_qq.append(qq)

        for name_i, name_k, name_c, value, ka in _angle_terms(
                mol_w_hs, properties):
            if name_i in by_name and name_k in by_name and name_c in by_name:
                angle_rows[0].append(by_name[name_i])
                angle_rows[1].append(by_name[name_k])
                angle_rows[2].append(by_name[name_c])
                theta0.append(value)
                angle_kas.append(ka)

        for (name_i, name_k, name_c, th0, r0ij, r0kj, k1, k2
             ) in _stretchbend_terms(mol_w_hs, properties):
            if name_i in by_name and name_k in by_name and name_c in by_name:
                sb_rows[0].append(by_name[name_i])
                sb_rows[1].append(by_name[name_k])
                sb_rows[2].append(by_name[name_c])
                sb_theta0.append(th0)
                sb_r0_ij.append(r0ij)
                sb_r0_kj.append(r0kj)
                sb_kba_ijk.append(k1)
                sb_kba_kji.append(k2)

        for name_i, name_j, name_k, name_l, v1, v2, v3 in _torsion_terms(
            mol_w_hs, properties
        ):
            got = resolve_ext(name_i, name_j, name_k, name_l)
            if got is None:
                continue
            for row, value in zip(torsion_rows, got):
                row.append(value)
            v1s.append(v1)
            v2s.append(v2)
            v3s.append(v3)

    def pack_index(rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.long, device=device).reshape(1, len(rows), -1)

    def pack_values(values: list[float]) -> torch.Tensor:
        return torch.tensor([values], dtype=torch.float32, device=device)

    def h_index(rows):
        return torch.tensor(rows, dtype=torch.long, device=device).reshape(1, 3, -1)

    return MMFFTerms(
        pair_index=torch.tensor(pair_rows, dtype=torch.long,
                                device=device).reshape(1, 2, -1),
        pair_rstar=pack_values(pair_rstar), pair_eps=pack_values(pair_eps),
        pair_qq=pack_values(pair_qq),
        nb_index=torch.tensor(nb_rows, dtype=torch.long,
                              device=device).reshape(1, 2, -1),
        nb_rstar=pack_values(nb_rstar), nb_eps=pack_values(nb_eps),
        nb_qq=pack_values(nb_qq),
        hydrogen_ref=h_index(hydrogen_rows),
        hydrogen_geom=(torch.tensor([hydrogen_geom], dtype=torch.float32, device=device)
                       if hydrogen_geom
                       else torch.zeros((1, 0, 3), dtype=torch.float32, device=device)),
        bond_index=pack_index(bond_rows),
        bond_r0=pack_values(r0s),
        bond_kb=pack_values(kbs),
        angle_index=pack_index(angle_rows),
        angle_theta0=pack_values(theta0),
        angle_ka=pack_values(angle_kas),
        sb_index=pack_index(sb_rows),
        sb_theta0=pack_values(sb_theta0),
        sb_r0_ij=pack_values(sb_r0_ij),
        sb_r0_kj=pack_values(sb_r0_kj),
        sb_kba_ijk=pack_values(sb_kba_ijk),
        sb_kba_kji=pack_values(sb_kba_kji),
        torsion_index=pack_index(torsion_rows),
        torsion_v1=pack_values(v1s),
        torsion_v2=pack_values(v2s),
        torsion_v3=pack_values(v3s),
    )
