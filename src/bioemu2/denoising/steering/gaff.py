"""GAFF2 parameters for SMILES ligands, via boonza, for steering.

Why GAFF2 alongside MMFF. The MMFF terms fix geometry and nothing else: across
11 saturated rings they bring ring pucker from 0.00-0.13 A to 0.16-0.24 A
against a true 0.15-0.29, and bond lengths onto MMFF's r0 to about 0.001 A.
They leave every directional preference wrong, because all three are bonded and
the preferences are not -- methylcyclohexane comes back 0 axial of 500 against a
true 5%, the proline Cg exo/endo preference is inverted, ribose pucker is
inverted along with its response to phosphate charge, and the fluorine gauche
effect comes out backwards.

What this adds is the nonbonded half: GAFF2 Lennard-Jones with AM1-BCC charges,
plus GAFF2's own bonded terms (a 7-term Fourier torsion rather than MMFF's 3).

Heavy atoms only, because that is what the models predict. That is not obviously
enough -- the 1,3-diaxial repulsion behind methylcyclohexane's A value is partly
a hydrogen contact -- so it was measured rather than assumed: for the two chairs,
the GAFF2 nonbonded energy favours equatorial by 0.215 kcal/mol over all atoms
and by 0.587 kcal/mol over heavy atoms alone. Dropping hydrogens removes
attractive terms along with repulsive ones and the preference survives.

Parameterization runs antechamber and takes about 12 s per ligand, so it is
cached by SMILES rather than repeated per diffusion sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from bioemu2.data_interface import System
from bioemu2.denoising.steering.feats import (
    GAFF_ELEMENTS,
    unsupported_ligand_reason,
    _atom_name_idx_map,
    _iter_ligand_residues,
    convert_densebatch_to_potential_features,
)
from bioemu2.denoising.steering.mmff import ligand_smiles_from_system
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate
from bioemu2.denoising.steering.gaff_params import map_to, parameters_for
from bioemu2.denoising.steering.gaff_params import DEFAULT_PROTEIN_FF

# Which cached parameterization to read. None selects pure
# GAFF2 -- no protein force field, no pseudopeptide matching,
# no CMAP -- which has its own cache key. Set it before
# sampling to compare the two.
PROTEIN_FF: str | None = DEFAULT_PROTEIN_FF
from bioemu2.structure.templates import _assign_mol_atom_names

logger = logging.getLogger(__name__)


@dataclass
class GAFFTerms:
    """Packed GAFF2 terms, indices already mapped onto the batch's atoms."""

    bond_index: torch.Tensor      # [1, 2, P]
    bond_r0: torch.Tensor         # [1, P]
    bond_fc: torch.Tensor
    angle_index: torch.Tensor     # [1, 3, P]  (i, k, centre)
    angle_theta0: torch.Tensor    # radians
    angle_fc: torch.Tensor
    torsion_index: torch.Tensor   # [1, 4, P]
    torsion_phi0: torch.Tensor    # radians
    torsion_fc: torch.Tensor      # [1, P, 7]
    hydrogen_ref: torch.Tensor    # [1, 3, H]  global indices of (parent, B, A)
    hydrogen_geom: torch.Tensor   # [1, H, 3]  (r, theta, phi), radians
    nb_index: torch.Tensor        # [1, 2, P]  all pairs MINUS exclusions
    nb_sigma: torch.Tensor        # [1, P]  Lorentz-Berthelot combined
    nb_eps: torch.Tensor          # [1, P]
    nb_qq: torch.Tensor           # [1, P]  q_i * q_j
    pair_index: torch.Tensor      # [1, 2, P]  1-4 pairs, hydrogens included
    pair_aij: torch.Tensor        # [1, P]  LJ A, scaling already applied
    pair_bij: torch.Tensor        # [1, P]  LJ B
    pair_qij: torch.Tensor        # [1, P]  q_i q_j, scaling already applied
    cmap_index: torch.Tensor      # [1, 8, P]  phi quadruple then psi quadruple
    cmap_energy: torch.Tensor     # [1, P, size, size], kcal/mol
    cmap_origin: torch.Tensor     # [1, P, 2]  grid origin in radians


def gaff_terms_from_system(system: System, ligand_ff: str | None = None) -> GAFFTerms:
    """Collect the force-field terms for every SMILES ligand in the system.

    ``ligand_ff`` is the ligand force field ("gaff2", "openff-2.2.1", ...);
    None is gaff_params.LIGAND_FF. OpenFF shares GAFF2's functional forms, so
    the same terms carry either.
    """
    structure = AllAtomStructure.from_system(system, mode="structure")
    batch = batch_collate([structure])
    feats = convert_densebatch_to_potential_features(batch)
    device = batch.coords.device
    atom_chain = batch.upsample_to_atoms(feats["asym_id"]).long()
    smiles_by_chain = ligand_smiles_from_system(system)

    bond_rows: list[list[int]] = [[], []]
    bond_r0: list[float] = []
    bond_fc: list[float] = []
    angle_rows: list[list[int]] = [[], [], []]
    angle_theta0: list[float] = []
    angle_fc: list[float] = []
    torsion_rows: list[list[int]] = [[], [], [], []]
    torsion_phi0: list[float] = []
    torsion_fc: list[list[float]] = []
    hydrogen_rows: list[list[int]] = [[], [], []]
    hydrogen_geom: list[list[float]] = []
    nb_rows: list[list[int]] = [[], []]
    nb_sigma: list[float] = []
    nb_eps: list[float] = []
    nb_qq: list[float] = []
    pair_rows: list[list[int]] = [[], []]
    pair_aij: list[float] = []
    pair_bij: list[float] = []
    pair_qij: list[float] = []
    cmap_rows: list[list[int]] = [[] for _ in range(8)]
    cmap_energy: list[list[list[float]]] = []
    cmap_origin: list[list[float]] = []

    import math

    from rdkit.Chem import AddHs, MolFromSmiles, RemoveAllHs

    for atom_lo, atom_hi in _iter_ligand_residues(batch, 0, atom_chain[0]):
        smiles = smiles_by_chain.get(int(atom_chain[0, atom_lo].item()))
        if smiles is None:
            continue
        reason = unsupported_ligand_reason(smiles, GAFF_ELEMENTS)
        if reason is not None:
            logger.info("GAFF: leaving %r unsteered (%s)", smiles, reason)
            continue
        # Cache first, compute if boonza and AmberTools are here. Warming the
        # cache beforehand is preferable -- antechamber is ~12 s of
        # single-threaded CPU per ligand, holds a GPU idle, and AM1-BCC charges
        # depend on the conformer fed to sqm, so caching also pins them across
        # runs -- but refusing to compute when it is possible would make GAFF
        # silently do nothing on a cold cache.
        data = parameters_for(smiles, protein_ff=PROTEIN_FF, ligand_ff=ligand_ff)
        if data is None:
            continue
        mol = MolFromSmiles(smiles)
        mol_w_hs = AddHs(mol)
        _assign_mol_atom_names(mol_w_hs)
        mol_wo_hs = RemoveAllHs(mol_w_hs)
        idx_map = _atom_name_idx_map(batch, 0, atom_lo, atom_hi, mol_wo_hs)
        if not idx_map:
            logger.warning("GAFF: no atom-name match for %r", smiles)
            continue
        # Two hops: canonical index -> this molecule's index (graph match),
        # then this molecule's index -> the batch's global atom index (the
        # template builder's own atom names).
        canon_to_local = map_to(mol_wo_hs, data)
        if canon_to_local is None:
            logger.warning("GAFF: cached entry does not match the ligand for %r",
                           smiles)
            continue
        local_to_global = dict(idx_map)

        def resolve(*canon_ids: int) -> tuple[int, ...] | None:
            out = []
            for canon_id in canon_ids:
                local = canon_to_local.get(canon_id)
                if local is None or local not in local_to_global:
                    return None
                out.append(local_to_global[local])
            return tuple(out)

        for i, j, r0, fc in data["bonds"]:
            got = resolve(i, j)
            if got:
                bond_rows[0].append(got[0])
                bond_rows[1].append(got[1])
                bond_r0.append(r0)
                bond_fc.append(fc)

        for i, k, c, theta0, fc in data["angles"]:
            got = resolve(i, k, c)
            if got:
                for row, value in zip(angle_rows, got):
                    row.append(value)
                angle_theta0.append(theta0)
                angle_fc.append(fc)

        # Hydrogens are not predicted by the model. They are reconstructed
        # from heavy atoms at evaluation time and appended after the real
        # atoms, so a term can refer to one by an index past the atom count.
        n_heavy = data.get("n_heavy")
        extended: dict[int, int] = {}
        n_atoms = int(batch.atom_mask.shape[1])
        for entry in data.get("hydrogens", []):
            got = resolve(entry["parent"], entry["b"], entry["a"])
            if got is None:
                continue
            extended[entry["index"]] = n_atoms + len(hydrogen_geom)
            for row, value in zip(hydrogen_rows, got):
                row.append(value)
            hydrogen_geom.append([entry["r"], math.radians(entry["theta"]),
                                  math.radians(entry["phi"])])

        def resolve_ext(*ids: int) -> tuple[int, ...] | None:
            """Resolve indices that may name a hydrogen as well as a heavy atom."""
            out = []
            for i in ids:
                if n_heavy is not None and i >= n_heavy:
                    slot = extended.get(i)
                    if slot is None:
                        return None
                    out.append(slot)
                    continue
                got = resolve(i)
                if got is None:
                    return None
                out.append(got[0])
            return tuple(out)

        for entry in data.get("cmaps", []):
            got = resolve(*entry["atom_idxs"])
            if got is None:
                continue
            size = entry["size"]
            flat = entry["energy"]
            grid = [flat[r * size:(r + 1) * size] for r in range(size)]
            for row, value in zip(cmap_rows, got):
                row.append(value)
            cmap_energy.append(grid)
            cmap_origin.append([math.radians(entry["phi_min"]),
                                math.radians(entry["psi_min"])])

        # Full nonbonded: every pair of mapped atoms except the exclusions.
        # Per-atom parameters first, then the combining rules.
        nb_param: dict[int, tuple[float, float, float]] = {}
        for canon, sig, eps_i, q in data.get("nonbonded", []):
            got = resolve_ext(canon)
            if got is not None:
                nb_param[got[0]] = (float(sig), float(eps_i), float(q))
        excluded: set[tuple[int, int]] = set()
        for a_i, b_i in data.get("exclusions", []):
            got = resolve_ext(a_i, b_i)
            if got is not None:
                excluded.add((min(got), max(got)))
        present = sorted(nb_param)
        for x in range(len(present)):
            for y in range(x + 1, len(present)):
                i_idx, j_idx = present[x], present[y]
                if (i_idx, j_idx) in excluded:
                    continue
                s_i, e_i, q_i = nb_param[i_idx]
                s_j, e_j, q_j = nb_param[j_idx]
                nb_rows[0].append(i_idx)
                nb_rows[1].append(j_idx)
                nb_sigma.append(0.5 * (s_i + s_j))
                nb_eps.append((max(e_i, 0.0) * max(e_j, 0.0)) ** 0.5)
                nb_qq.append(q_i * q_j)

        for a, b, aij, bij, qij in data.get("pairs", []):
            got = resolve_ext(a, b)
            if got is None:
                continue
            pair_rows[0].append(got[0])
            pair_rows[1].append(got[1])
            pair_aij.append(aij)
            pair_bij.append(bij)
            pair_qij.append(qij)

        for i, j, k, l, phi0, fcs in data["torsions"]:
            got = resolve_ext(i, j, k, l)
            if got:
                for row, value in zip(torsion_rows, got):
                    row.append(value)
                torsion_phi0.append(phi0)
                torsion_fc.append(list(fcs))

    def index(rows: list[list[int]]) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.long, device=device).reshape(1, len(rows), -1)

    def values(data: list[float]) -> torch.Tensor:
        return torch.tensor([data], dtype=torch.float32, device=device)

    radians = [math.radians(v) for v in angle_theta0]
    phis = [math.radians(v) for v in torsion_phi0]
    fc_tensor = torch.tensor([torsion_fc] if torsion_fc else [[]],
                             dtype=torch.float32, device=device)
    if fc_tensor.numel() == 0:
        fc_tensor = fc_tensor.reshape(1, 0, 7)

    return GAFFTerms(
        bond_index=index(bond_rows), bond_r0=values(bond_r0), bond_fc=values(bond_fc),
        angle_index=index(angle_rows), angle_theta0=values(radians),
        angle_fc=values(angle_fc),
        torsion_index=index(torsion_rows), torsion_phi0=values(phis),
        torsion_fc=fc_tensor,
        hydrogen_ref=index(hydrogen_rows),
        hydrogen_geom=(torch.tensor([hydrogen_geom], dtype=torch.float32,
                                    device=device)
                       if hydrogen_geom else torch.zeros((1, 0, 3),
                                                         dtype=torch.float32,
                                                         device=device)),
        nb_index=index(nb_rows), nb_sigma=values(nb_sigma),
        nb_eps=values(nb_eps), nb_qq=values(nb_qq),
        pair_index=index(pair_rows), pair_aij=values(pair_aij),
        pair_bij=values(pair_bij), pair_qij=values(pair_qij),
        cmap_index=index(cmap_rows),
        cmap_energy=(torch.tensor([cmap_energy], dtype=torch.float32, device=device)
                     if cmap_energy else torch.zeros((1, 0, 1, 1), dtype=torch.float32,
                                                     device=device)),
        cmap_origin=(torch.tensor([cmap_origin], dtype=torch.float32, device=device)
                     if cmap_origin else torch.zeros((1, 0, 2), dtype=torch.float32,
                                                     device=device)),
    )
