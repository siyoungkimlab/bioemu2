# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Inference-time steering potentials.

See ``docs/potentials.md`` and ``docs/steering.md`` for more information.

Single-system topology assumption: indices are taken from ``feats[...][0]``
(the first batch element). Every particle or replicate in the broadcast batch
shares one topology.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F

from bioemu2.biomoa.ops.atom_token_ops import broadcast_tokens_to_atoms
from bioemu2.data_interface import System
from bioemu2.denoising.rigid_align import weighted_rigid_align
from bioemu2.denoising.steering.feats import (
    convert_densebatch_to_potential_features,
    potential_features_from_system,
)
from bioemu2.denoising.steering.geometry_topology import (
    BondedGeometryGraph,
    StereoTables,
    build_bonded_geometry_graph,
    build_polymer_steric_graph,
    compute_geometry_bounds,
    get_angle_reference,
    get_bond_reference,
    load_stereo_tables,
    pack_angle_constraints,
    pack_bond_constraints,
)
from bioemu2.denoising.steering.gaff import gaff_terms_from_system
from bioemu2.denoising.steering.mmff import mmff_terms_from_system
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate


def _connected_zero(coords: torch.Tensor) -> torch.Tensor:
    """Per-particle zero with a differentiable path to coordinates, including empty atoms."""
    return coords[..., :0, :].sum(dim=(-2, -1))


def _atom_chain_id(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-atom chain id ``(N,)`` for the topology-shared batch element 0."""
    return (
        broadcast_tokens_to_atoms(
            feats["asym_id"].unsqueeze(-1),  # (B, T) -> (B, T, 1)
            feats["atom_to_token_ids"],  # (B, N)
            ~feats["atom_pad_mask"].bool(),  # (B, N)
        )  # (B, N, 1)
        .squeeze(-1)  # (B, N)
        .long()[0]  # (N,) topology-shared batch element 0
    )


def _atom_vdw_radius(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-atom VDW radii ``[N]`` for the topology-shared batch element 0."""
    radii = torch.zeros(num_elements, dtype=torch.float32, device=feats["ref_element"].device)
    radii[1:119] = torch.tensor(vdw_radii, dtype=torch.float32, device=radii.device)
    return (feats["ref_element"].float() @ radii[:, None]).squeeze(-1)[0]  # [N]


def _steric_overlap_penalty(
    coords: torch.Tensor,  # [B, N, 3]
    index: torch.Tensor,  # [2, P]
    minimum_distance: torch.Tensor | float,  # [P] or scalar
) -> torch.Tensor:  # [B]
    """Return overlap energies with distances smoothed at 1e-6 angstroms.

    Use ``sqrt(||x_i - x_j||**2 + eps**2)`` to soften gradients near coincident atoms.
    """
    i, j = index
    # Promote FP16 so eps**2 cannot underflow to zero.
    if coords.dtype == torch.float16:
        coords = coords.float()
    eps = 1e-6  # angstroms
    delta = coords.index_select(-2, i) - coords.index_select(-2, j)  # [B, P, 3]
    distance = torch.sqrt(delta.square().sum(dim=-1) + eps**2)  # [B, P]
    return torch.relu(minimum_distance - distance).sum(dim=-1)


def compute_atom_pairs_within_cutoff(
    coords: torch.Tensor,  # [B, N, 3]
    atom_mask: torch.Tensor,  # [N]
    atom_chain_id: torch.Tensor,  # [N]
    allowed_chain_pairs: torch.Tensor,  # [C, C]
    cutoff: float,
    block_size: int,
) -> torch.Tensor:  # [2, P]
    """Find allowed pairs close in any particle, using detached blockwise distances.

    For each pair, the search tests ``min_b ||x[b, i] - x[b, j]|| < cutoff``. Callers
    recompute differentiable distances only for returned pairs. Pairs satisfy ``i < j``.
    Distances use the direct, non-matmul path to avoid cancellation for nearby atoms
    far from the origin. FP16/BF16 search coordinates are promoted to FP32.

    Args:
        coords: Coordinates of shape ``[B, N, 3]`` in angstroms.
        atom_mask: Boolean mask of shape ``[N]``; ``True`` includes an atom in the search.
        atom_chain_id: Integer chain indices of shape ``[N]``, shared across particles.
        allowed_chain_pairs: Boolean mask of shape ``[C, C]`` indexed by chain indices;
            ``True`` permits pairs between those chains, including within a chain if
            its diagonal entry is ``True``.
        cutoff: Strict distance threshold in angstroms.
        block_size: Number of candidate first atoms processed per distance block.

    Returns:
        Atom-index pairs of shape ``[2, P]`` into the original ``N``-atom coordinates,
        shared across all particles. Each pair is within the cutoff in at least one
        particle and passes both masks.
    """
    atoms = atom_mask.nonzero().squeeze(-1)  # [M]
    coords = coords.detach()[:, atoms]  # [B, M, 3]
    if coords.dtype in (torch.float16, torch.bfloat16):
        coords = coords.float()
    chain = atom_chain_id[atoms]  # [M]
    pairs = [atoms.new_zeros((2, 0))]
    # The last atom has no later partner, so the blocks stop before it.
    for start in range(0, len(atoms) - 1, block_size):
        stop = min(start + block_size, len(atoms))
        with torch.autocast(device_type=coords.device.type, enabled=False):
            closest = torch.cdist(
                coords[:, start:stop],
                coords[:, start:],
                compute_mode="donot_use_mm_for_euclid_dist",
            ).amin(0)  # [R, M-start]
        should_check = allowed_chain_pairs[
            chain[start:stop, None], chain[None, start:]
        ]  # [R, M-start]
        row, col = torch.triu((closest < cutoff) & should_check, diagonal=1).nonzero(as_tuple=True)
        pairs.append(torch.stack([atoms[start + row], atoms[start + col]]))  # [2, P_block]
    return torch.cat(pairs, dim=1)  # [2, P]


#: Where a buffered 14-7 crosses zero, as a fraction of R*.
#: Solving 1.12 R*^7/(r^7 + 0.12 R*^7) = 2 gives r = (0.44)^(1/7) R*.
#: R* itself is the MINIMUM of the 14-7, not its zero -- the distinction
#: matters because the cosine wall must vanish where the real potential
#: does, as GAFF's does at sigma. Cutting at R* instead put the branch
#: crossing in the middle of the repulsive wall, with a 15.6 kcal/mol/A
#: force discontinuity against GAFF's 0.69.


_MMFF_COULOMB = 332.0716       # kcal.A/mol/e^2
_MMFF_ELEC_BUFFER = 0.05       # angstrom, MMFF's buffered distance


#: Coulomb's constant for GAFF/AMBER, kcal.A/mol/e^2.
_GAFF_COULOMB = 332.0637
#: Smallest switch distance of the core forms, angstrom: pairs whose sigma (or
#: R*) is ~0 -- e.g. GAFF's hydroxyl hydrogen, which has no Lennard-Jones --
#: would otherwise have no core at all.
_CORE_FLOOR = 0.5


def _core_switch(size: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    """s = max(frac * size, 0.5 A), size being the pair's LJ sigma (GAFF) or R* (MMFF).

    For the Coulomb core frac is the term's coulomb_core_switch (0.5 by default
    everywhere, so short hydrogen-bond contacts keep real Coulomb). For the vdW
    core it is the term's lj_core_switch: 0.7 for GAFF2/OpenFF's 12-6 (at sigma/2 the r^-13 wall already
    pushes ~6000 kcal/mol/A on a C...C pair, and at guidance weight 0.005 that
    kicks atoms ~30 A per step against a bond pull-back of ~0.1-0.3 A, so
    compressed early x0_hat exploded), 0.5 for MMFF's buffered 14-7 (R*/2, whose
    wall is already ~20x softer there)."""
    return torch.clamp(frac * size, min=_CORE_FLOOR)


def _parabola_core(r, s, e_s, de_s, outer):
    """``outer`` for r >= s; inside, E = A - B r^2 with the value e_s and slope de_s
    of the outer form at s, so energy and force are continuous and E(0) = A is
    finite. ``outer`` must already be evaluated at max(r, s), never below s, so
    no masked branch carries an infinite gradient."""
    b = -de_s / (2.0 * s)
    a = e_s + b * s * s
    return torch.where(r < s, a - b * r * r, outer)


def _coulomb_core(r, k, s):
    """Real Coulomb k / r for r >= s, with k = C q_i q_j / D (scaling included);
    a parabola matching value and slope inside: (k / 2s)(3 - r^2 / s^2)."""
    rr = torch.maximum(r, s)
    return _parabola_core(r, s, k / s, -k / (s * s), k / rr)


def _lj126_core(r, a, b, s):
    """a / r^12 - b / r^6 for r >= s, parabola inside (see _parabola_core)."""
    rr = torch.maximum(r, s)
    s6 = s ** 6
    e_s = a / (s6 * s6) - b / s6
    de_s = -12.0 * a / (s6 * s6 * s) + 6.0 * b / (s6 * s)
    r6 = rr ** 6
    return _parabola_core(r, s, e_s, de_s, a / (r6 * r6) - b / r6)


def _mmff147(r, rstar, eps):
    """MMFF's buffered 14-7 and its derivative in r."""
    rs7 = rstar ** 7
    u = r + 0.07 * rstar
    f = (1.07 * rstar / u) ** 7
    w = r ** 7 + 0.12 * rs7
    g = 1.12 * rs7 / w - 2.0
    df = -7.0 * f / u
    dg = -1.12 * rs7 * 7.0 * r ** 6 / (w * w)
    return eps * f * g, eps * (df * g + f * dg)


def _mmff147_core(r, rstar, eps, s):
    """MMFF's buffered 14-7 for r >= s, parabola inside (see _parabola_core)."""
    e_s, de_s = _mmff147(s, rstar, eps)
    outer, _ = _mmff147(torch.maximum(r, s), rstar, eps)
    return _parabola_core(r, s, e_s, de_s, outer)


def _late_schedule(weight, *t_ons):
    """weight while t <= the smallest given t_on (late in denoising, t = 0 clean),
    0 before; the weight unchanged when every t_on is None."""
    t_ons = [float(t) for t in t_ons if t is not None]
    if not t_ons:
        return weight
    return PiecewiseStepFunction(thresholds=[min(t_ons)], values=[weight, 0.0])


def _apply_t_on(terms, t_on, nonbonded_t_on, guidance_ramp: bool = False) -> None:
    """Every force-field term acts (guidance and resampling) only for t <= t_on;
    the 1-4 and nonbonded terms also only for t <= nonbonded_t_on when given.

    With guidance_ramp the guidance weight instead rises linearly from 0 at t = 1
    to its full value at t = 0, over the whole schedule (the steering acts on
    x0_hat, which is a formed molecule from the start for small ligands, with
    noisy bonds early); resampling keeps the t_on switch.

    Early in denoising x0_hat is a blurred average with overlapping atoms and
    wrong bond lengths: the bonded terms then mostly react to noise, and in FK
    resampling they dominated the log-weights (bond 10.7, angle 6.7 log units of
    spread across 1000 particles at t ~ 1, phenolate), collapsing the population
    to an effective sample size of ~1 at nearly every step.
    """
    for p in terms:
        nb = isinstance(p, (GAFF14Potential, NonbondedPotential,
                            MMFF14Potential, MMFFNonbondedPotential))
        limits = (t_on, nonbonded_t_on) if nb else (t_on,)
        params = dict(p.parameters or {})
        for key in ("guidance_weight", "resampling_weight"):
            if key not in params:
                continue
            if key == "guidance_weight" and guidance_ramp:
                params[key] = LinearInterpolation(start_t=1.0, end_t=0.0,
                                                  start_val=0.0, end_val=params[key])
            else:
                params[key] = _late_schedule(params[key], *limits)
        p.parameters = params
        p.guidance_ramp = bool(guidance_ramp)
        known = [float(x) for x in limits if x is not None]
        p.t_on = min(known) if known else None
        if nb:
            p.nonbonded_t_on = nonbonded_t_on


#: The MMFF guidance weight, shared by every MMFF term.
#:
#: One weight for all of them keeps MMFF's internal balance exactly: both
#: bonded terms scale as w * 71.9663 * (kb or ka), so the bond:angle ratio of
#: 5.004 is MMFF's own, and torsion, stretch-bend, vdW and Coulomb are already
#: in kcal/mol. The force field is therefore scaled as a unit rather than
#: retuned term by term.
#:
#: The scale is set against PoseBusters, whose flat bottom and this
#: Huber both saturate to a constant force -- 2*w*k_CC*cutoff against
#: pose_weight*k_pose.
#: For reference, matching PoseBusters exactly on an sp3 C-C would give
#:     0.10 / (2 * 306.43 * 0.2) = 0.00081584
#: The value below is 0.61x that, so a C-C saturates at 0.061 kcal/mol/A
#: against PoseBusters' 0.100. One weight still covers every MMFF term, so
#: only the overall scale changes and MMFF's internal balance is untouched.
#: Worst gw*k is the carbonyl at 0.466, well inside the bound of 1.
MMFF_WEIGHT = 0.0005

#: One shared weight for every GAFF2 term, matching MMFF_WEIGHT.
#:
#: Matching PoseBusters' saturated bond force on the sp3 C-C bond would give
#: 0.10 / (2 * 234.1 * 0.2) = 0.001068, and that was the first choice. A
#: 100-task weight sweep then put a sharp ensemble-collapse threshold between
#: 0.001 and 0.0005 in BOTH force fields -- pyridylphenolate's O...N range
#: over 200 samples goes 0.003 -> 0.330 for GAFF2 and 0.001 -> 0.718 for MMFF
#: across that step -- so 0.001 returns a single geometry with numerical
#: jitter rather than an ensemble. Matching the force is the wrong criterion
#: on its own; 0.0005 gives gw*k = 0.117 on the sp3 C-C bond, about half
#: PoseBusters, and a real ensemble. One number keeps GAFF2's own term
#: balance, since every term enters as w * fc.
GAFF2_WEIGHT = 0.0005

#: The per-term weights the ligand presets steer with, measured on a 15-ligand
#: benchmark against 30 ns of GAFF2 MD (2026-11-23). The earlier values were
#: tuned against force magnitudes and pass rates; these are tuned against MD
#: geometry.
#:
#: Guidance is fixed-step gradient descent on the weighted energy, so a weight
#: IS that term's step length, not a strength. Descent on a bond of force
#: constant k diverges once w * 8k > 2, i.e. w* = 0.00038 for a GAFF2 C=O and
#: 0.00027 for an MMFF one, so the previous bond/angle 0.001 was past the bound
#: for every bond but sp3 C-C and was held together only by the Huber cutoff.
#: The damage sorted by force constant, not by molecule: at bond/angle 0.00125,
#: benzene reached 80.6% bad bonds while the three pure-sp3 rings stayed at
#: 0.00%.
#:
#: Bond, angle and torsion are a quarter of the earlier set. A uniform scale
#: preserves the ratios -- the fixed point of sum_i w_i grad U_i = 0, and so the
#: target geometry -- and shrinks only the step; scaling bond, or bond+angle,
#: ALONE measured worse than the old default, because the ratio is what sets the
#: geometry. pair14 is raised to the torsion weight because the benchmark held
#: the three soft terms together at 0.00125.
#:
#: Measured over 14 molecules x 3 seeds x 840 frames: bad bonds 6.3% -> 0.00%,
#: bond deviation 4.5% -> 2.0% against MD's own 1.6% floor, ring pucker Q 0.465
#: against MD's 0.459.
LIGAND_STEER_BOND_WEIGHT = 0.00025
LIGAND_STEER_ANGLE_WEIGHT = 0.00025
LIGAND_STEER_PAIR14_WEIGHT = 0.00125
LIGAND_STEER_TORSION_WEIGHT = 0.00125

#: Held at the torsion weight for every preset, OpenFF included through the
#: fallback. That is a uniform quarter of GAFF2's old 0.005 but only a sixth of
#: MMFF's old 0.0075, so MMFF's nonbonded term is driven 1.5x less than a
#: uniform rescale of its own preset would give. This value is the tested one
#: (0.04-0.05% bad bonds under MMFF); the uniform-quarter alternative for MMFF,
#: 0.001875, has not been run.
LIGAND_STEER_NONBONDED_WEIGHT = {"gaff2": 0.00125, "mmff": 0.00125}
LIGAND_STEER_NONBONDED_WEIGHT_DEFAULT = 0.00125

#: Guidance applies only below this diffusion time: torsions are decided around
#: t = 0.3-0.15, and steering earlier fights the model while the ligand is still
#: forming.
LIGAND_STEER_T_ON = 0.4

#: Langevin guidance: x0 <- x0 - sum w grad E + sqrt(2 w kT) xi, with kT at
#: room temperature in kcal/mol. The kick is added once per descent step, INSIDE
#: the inner loop, so the noise accumulates as sqrt(N) while the drift
#: accumulates linearly: at LIGAND_STEER_GD_STEPS this weight is about 0.17 A of
#: thermal displacement, where the old 0.005 would be 0.35 A. (Those were 0.39
#: and 0.77 A when this constant was 100; both scale as sqrt(steps).)
#:
#: This is a trade, not an ordering. Measured at bond/angle 0.00025, gd 80,
#: as bioemu2 / boltz1:
#:
#:   langevin   bad bonds       pucker Q (MD 0.459)   boltz1 mean |dtors|
#:   0.005      0.27% / 0.60%   0.311 / 0.331         10.0 deg
#:   0.00125    0.00% / 0.00%   0.460 / 0.465         20.4 deg
#:
#: The cooler value is the default because it is the one that leaves bonds and
#: ring geometry intact; raise it to 0.005 when conformational spread matters
#: more than bond lengths. One counterintuitive detail: where the descent is
#: ALREADY unstable, more noise does less bond damage (19.8% vs 45.7% at bond
#: 0.00125), because the kick decoheres the overshoot instead of compounding it.
LIGAND_STEER_LANGEVIN_KT = 0.593
LIGAND_STEER_LANGEVIN_WEIGHT = 0.00125

#: Particles carried per requested sample, and also the size of each independent
#: resampling group: the presets set the group size TO this value, so each output
#: sample's particles compete only with each other instead of in one global pool
#: over the whole batch. A single global pool measured better at large pool
#: sizes, but the benchmarked setting is k = 3 grouped, which keeps the cost per
#: sample fixed and still carries the FK selection.
LIGAND_STEER_PARTICLES_PER_SAMPLE = 3

#: Inner gradient-descent steps per diffusion step. The gradient is re-evaluated
#: at the displaced geometry on every iteration, so this is genuine iterative
#: descent, and the soft terms need the steps that the x0.25 weights cost them:
#: it is w * steps that reaches a fixed point. 80 and 20 were benchmarked -- in
#: boltz1 they tie, in bioemu2 20 loses the open conformers (bipyridine N-C-C-N
#: 122 deg -> 45, against MD's 166). 100 extrapolated the better of the two and
#: was never itself benchmarked.
#:
#: Set to 20 by request, 2026-11-25, with that bioemu2 result known and
#: accepted. Two things follow, neither of them only a cost saving:
#:   - bioemu2 is expected to close the open conformers again, as measured.
#:   - the Langevin kick is applied once per descent step, so its accumulated
#:     thermal displacement falls with sqrt(N): about 0.17 A here against the
#:     0.39 A of 100 steps. Fewer steps is also colder.
#: A later sweep over gd 20-100 found the aggregate error flat (mean 48.9 ->
#: 41.5 deg) with 0% broken bonds throughout, which is why 20 is defensible
#: despite the above; the aggregate is dominated by wrong-basin cells that gd
#: does not move either way.
#: Raise it back to 100 with -S num_gd_steps=100, or revert this constant.
LIGAND_STEER_GD_STEPS = 20

class ParameterSchedule(ABC):
    def compute(self, t: float) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        # Readable in parameters.yaml: the class and its constructor arguments.
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"

    def describe(self) -> str:
        """One line in words, for parameters.yaml; empty when repr says it all."""
        return ""


class ExponentialInterpolation(ParameterSchedule):
    def __init__(self, start, end, alpha):
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, t: float) -> float:
        if self.alpha != 0:
            return self.start + (self.end - self.start) * (math.exp(self.alpha * t) - 1) / (
                math.exp(self.alpha) - 1
            )
        else:
            return self.start + (self.end - self.start) * t


class PiecewiseStepFunction(ParameterSchedule):
    def __init__(self, thresholds, values):
        self.thresholds = thresholds
        self.values = values

    def describe(self) -> str:
        # t runs from 1 (noise) to 0 (clean); values[i] holds up to thresholds[i].
        edges = [0.0, *self.thresholds, 1.0]
        parts = [f"{v!r} for {'0 <= ' if i == 0 else f'{edges[i]!r} < '}t <= {edges[i + 1]!r}"
                 for i, v in enumerate(self.values)]
        return ", ".join(parts)

    def compute(self, t: float) -> float:
        assert len(self.thresholds) > 0
        assert len(self.values) == len(self.thresholds) + 1

        idx = 0
        while idx < len(self.thresholds) and t > self.thresholds[idx]:
            idx += 1
        return self.values[idx]


class LinearInterpolation(ParameterSchedule):
    def __init__(self, start_t, end_t, start_val, end_val):
        self.start_t = start_t
        self.end_t = end_t
        self.start_val = start_val
        self.end_val = end_val

    def describe(self) -> str:
        return (f"linear from {self.start_val!r} at t = {self.start_t!r} to "
                f"{self.end_val!r} at t = {self.end_t!r}, 0 outside")

    def compute(self, t: float) -> float:
        if (t > self.start_t) or (t < self.end_t):
            return 0.0
        else:
            # Linear interpolation between start and end values
            return self.start_val - (self.start_val - self.end_val) * (self.start_t - t) / (
                self.start_t - self.end_t
            )


# ``vdw_radii`` is indexed by atomic number 1..118.
num_elements = 128
vdw_radii = [
    1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54,
    2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4,
    2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1,
    2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3,
    2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25,
    2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45,
    2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28,
    2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05,
    2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4,
    2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
]  # fmt: skip


class Potential(ABC):
    """Public interface for an inference-time steering potential."""

    @abstractmethod
    def prepare(self, system: System) -> None:
        """Cache System-derived state (atom indices, features) once per sampling run."""

    def __call__(
        self,
        coords_angstrom: torch.Tensor,
        *,
        t: float,
    ) -> torch.Tensor:
        """Return unweighted per-sample energy at normalised diffusion time ``t``.

        The output must be differentiable with respect to ``coords_angstrom`` when used
        with gradient-based steering, such as FK gradient guidance or FKC's
        ``weight_update="guidance"`` mode.

        Args:
            coords_angstrom: Float32 or float64 coordinates in angstroms,
                shape ``[particles, atoms, 3]``.
                Padding may add trailing atoms; use the indices from ``prepare(system)``
                rather than assuming that every atom is real.
            t: Normalised diffusion time, from 1 at high noise to 0 at the clean endpoint.

        Returns:
            Energies of shape ``[particles]`` on the coordinates' device.
        """
        raise NotImplementedError

    def compute_parameters(self, t: float) -> dict[str, Any] | None:
        """Return parameters controlling whether/how strongly this potential is
        applied at diffusion time ``t`` (e.g. ``resampling_weight``,
        ``guidance_weight``, ``guidance_interval``)."""
        return None


class FeatsPotential(Potential):
    """Potential evaluated from a topology feature dictionary.

    Subclasses either split evaluation into ``_compute_args``, ``compute_variable``,
    and ``compute_function``, following the Boltz potential design, or override
    ``compute`` directly.
    """

    def __init__(
        self,
        parameters: dict[str, ParameterSchedule | float | int | bool] | None = None,
    ):
        self.parameters = parameters

    def prepare(self, system: System) -> None:
        """Build the topology feature cache used by subsequent evaluations."""
        self.feats = potential_features_from_system(system)

    def __call__(
        self,
        coords_angstrom: torch.Tensor,
        *,
        t: float,
    ) -> torch.Tensor:
        """Return differentiable, unweighted per-sample energy.

        ``t`` resolves parameter schedules at normalized diffusion time.
        """
        if coords_angstrom.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "Steering potentials require float32 or float64 coordinates; "
                f"got {coords_angstrom.dtype}."
            )
        parameters = self.compute_parameters(t) or {}
        self.feats = {name: value.to(coords_angstrom.device) for name, value in self.feats.items()}
        # Atom padding is trailing on both sides: truncate (CP pads wider than the
        # collate window) then pad (unpadded coords) to the feats width.
        n = self.feats["atom_pad_mask"].shape[1]
        coords = F.pad(coords_angstrom[:, :n], (0, 0, 0, max(0, n - coords_angstrom.shape[1])))
        return self.compute(coords, self.feats, parameters)

    def compute(
        self,
        coords: torch.Tensor,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Computes the value of the potential for the given coordinates.
        Args:
            coords: The coordinates of the sample atoms, shape (batch_size, n_atoms, 3).
            feats: The features of the input data.
            parameters: Additional parameters for the potential.
        Returns:
            A tensor containing the computed potential value, of shape (batch_size, ).
        """
        index, args, com_args = self._compute_args(feats, parameters)
        return self._compute_from_args(coords, index, args, com_args)

    def _compute_from_args(
        self,
        coords: torch.Tensor,
        index: torch.Tensor,
        args: tuple,
        com_args: tuple | None,
    ) -> torch.Tensor:
        if index.shape[1] == 0:
            return _connected_zero(coords)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[..., atom_pad_mask, :]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
                dtype=coords.dtype,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
                include_self=False,  # true COM: divide by N atoms, not N + 1
            )
        value = self.compute_variable(coords, index)
        energy = self.compute_function(value, *args)
        return energy.sum(dim=-1)  # type: ignore[union-attr]

    def compute_parameters(self, t: float) -> dict[str, Any] | None:
        """
        Computes the parameters for applying potential at a given timestep,
        e.g. the weight of a guidance step.
        Args:
            t: The current diffusion timestep.
        Returns:
            A dictionary of parameters for the potential, or None if no parameters are set.
        """
        if self.parameters is None:
            return None
        parameters = {
            name: parameter
            if not isinstance(parameter, ParameterSchedule)
            else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        return parameters

    def compute_function(self, value: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Computes the potential function given an input value, e.g. compute
        flat-bottom potentials given bond lengths.
        Args:
            value: The input value for the potential function, of shape (batch_size, n_values).
            *args: Additional arguments for the potential function.
        Returns:
            A tensor containing the computed potential, of shape (batch_size, n_values).
        """
        raise NotImplementedError

    def compute_variable(self, coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantity whose value will be used in the potential function,
        e.g. bond lengths or angles.
        Args:
            coords: The coordinates of the sample atoms, shape (batch_size, n_atoms, 3).
            index: The indices of the atoms involved in the potential.
        Returns:
            A tensor containing the computed variable, of shape (batch_size, n_values).
        """
        raise NotImplementedError

    def _compute_args(
        self, feats: dict, parameters: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, tuple, tuple | None]:
        """
        Computes the arguments required for the potential computation.
        Args:
            feats: The features of the input data.
            parameters: Additional parameters for the potential.
        Returns:
            A tuple containing:
                - index: The indices of the atoms involved in the potential.
                - args: A tuple of additional arguments required for the potential computation.
                - com_args: Optional arguments for center of mass calculations.
        """
        raise NotImplementedError


class FlatBottomPotential(FeatsPotential):
    """Linear hinge outside a lower/upper interval.

    For each measured value ``v``, the energy is
    ``k * (relu(lower - v) + relu(v - upper))``. A missing bound is treated as
    negative or positive infinity, respectively. This is linear, not quadratic.
    """

    def compute_function(self, value, k, lower_bounds, upper_bounds):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))
        return k * (torch.relu(lower_bounds - value) + torch.relu(value - upper_bounds))


class DistancePotential(FeatsPotential):
    def compute_variable(self, coords, index):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        return torch.linalg.norm(r_ij, dim=-1)


class DihedralPotential(FeatsPotential):
    _DIHEDRAL_EPS = 1e-8

    def compute_variable(self, coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)

        # atan2 form instead of sign(.) * arccos(clamp(.)): clamping to 1 - 1e-8 is
        # a no-op in float32, so an exactly planar dihedral (phi in {0, pi}) gives
        # arccos'(1) = inf, times sign(0) = 0 -> NaN under autograd. atan2 is exact
        # there with finite gradients.
        bond_length = torch.linalg.norm(r_kj, dim=-1)
        valid = (
            (bond_length > self._DIHEDRAL_EPS)
            & (torch.linalg.norm(n_ijk, dim=-1) > self._DIHEDRAL_EPS)
            & (torch.linalg.norm(n_jkl, dim=-1) > self._DIHEDRAL_EPS)
        )
        sin_phi = (r_kj * torch.cross(n_ijk, n_jkl, dim=-1)).sum(-1) / bond_length.clamp_min(
            self._DIHEDRAL_EPS
        )
        cos_phi = (n_ijk * n_jkl).sum(-1)
        return torch.atan2(
            torch.where(valid, sin_phi, 0.0),
            torch.where(valid, cos_phi, 1.0),
        )


class AbsDihedralPotential(DihedralPotential):
    def compute_variable(self, coords, index):
        return torch.abs(super().compute_variable(coords, index))


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def _compute_args(self, feats, parameters):
        pair_index = feats["rdkit_bounds_index"][0]
        lower_bounds = feats["rdkit_lower_bounds"][0].clone()
        upper_bounds = feats["rdkit_upper_bounds"][0].clone()
        bond_mask = feats["rdkit_bounds_bond_mask"][0]
        angle_mask = feats["rdkit_bounds_angle_mask"][0]

        lower_bounds[bond_mask * ~angle_mask] *= 1.0 - parameters["bond_buffer"]
        upper_bounds[bond_mask * ~angle_mask] *= 1.0 + parameters["bond_buffer"]
        lower_bounds[~bond_mask * angle_mask] *= 1.0 - parameters["angle_buffer"]
        upper_bounds[~bond_mask * angle_mask] *= 1.0 + parameters["angle_buffer"]
        lower_bounds[bond_mask * angle_mask] *= 1.0 - min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        upper_bounds[bond_mask * angle_mask] *= 1.0 + min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        lower_bounds[~bond_mask * ~angle_mask] *= 1.0 - parameters["clash_buffer"]
        upper_bounds[~bond_mask * ~angle_mask] = float("inf")

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None


class InterChainVDWOverlapPotential(FeatsPotential):
    """Clash penalty on atom pairs in different chains.

    Pairs at most ``NONBONDED_EXCLUSION_GRAPH_DISTANCE`` covalent bonds apart, which only
    occur across declared cross-chain bonds such as inter-chain disulfides, are excluded.
    For an inter-chain disulfide, the SG-SG bond and its immediate neighbours (atom pairs
    up to ``NONBONDED_EXCLUSION_GRAPH_DISTANCE`` bonds apart through it) are dropped from
    the inter-chain clash potential. Across chains such a path must pass through a declared
    bond, so only pairs next to it are skipped. For an inter-chain disulfide these are
    SG–SG′, SG–CB′, CB–SG′, CB–CB′, SG–CA′ and CA–SG′; all other pairs between the two
    chains are scored. :class:`PolymerLocalClashPotential` scores the skipped pairs other
    than SG–SG′.
    The clash distance is the sum of the two VDW radii, reduced by ``buffer``. At every
    evaluation, score a pair only if it is within ``cutoff`` in any particle, so all
    atom pairs are never held at once. The cutoff must exceed every clash distance so
    that skipped pairs would have zero penalty and gradient.
    """

    cutoff: float = 10.0  # angstroms; must exceed all clash distances
    block_size: int = 512  # atoms per search block, limiting peak memory

    def prepare(self, system: System) -> None:
        """Cache atom radii, interchain eligibility and bonded cross-chain pairs for the system."""
        super().prepare(system)
        feats = self.feats
        n_atoms = feats["atom_pad_mask"].shape[1]  # N, padded atom count
        chain = _atom_chain_id(feats)  # [N]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()  # [N]
        chain_sizes = torch.bincount(chain[atom_pad_mask])  # [C]
        steric_atom = atom_pad_mask & (chain_sizes > 1)[chain]  # [N]
        atom_radius = _atom_vdw_radius(feats)  # [N]

        # [C, C]; False on the diagonal, so only pairs of different chains are scored.
        interchain = ~torch.eye(
            len(chain_sizes), dtype=torch.bool, device=chain.device
        )  # [C, C]:bool ; false on diagonal
        graph = build_bonded_geometry_graph(AllAtomStructure.from_system(system, mode="structure"))

        # graph.nonbonded_exclusions(): Return atom pairs (i, j) separated by at most ``max_bonds`` covalent bonds.
        # Compute bonds we exclude from inter-chain VDW scoring: those separated by at most NONBONDED_EXCLUSION_GRAPH_DISTANCE covalent bonds.
        excluded = (
            torch.tensor(
                sorted(graph.nonbonded_exclusions()), dtype=torch.long, device=chain.device
            )  # [E_all, 2]
            .reshape(-1, 2)  # [E_all, 2], also when there are no exclusions
            .T
        )  # [2, E_all]
        # Filter excluded bonds to only keep excluded cross-chain pairs
        excluded = excluded[:, chain[excluded[0]] != chain[excluded[1]]]  # [2, E], cross-chain

        # Largest VDW radius per chain. Summed over scored chain pairs, it bounds every clash distance,
        # so compute() can check that pairs skipped by the cutoff search could not clash.
        chain_radius = atom_radius.new_zeros(len(chain_sizes)).scatter_reduce(
            0, chain[atom_pad_mask], atom_radius[atom_pad_mask], "amax", include_self=False
        )  # [C]
        allowed = interchain & (chain_sizes > 1)[:, None] & (chain_sizes > 1)[None, :]  # [C, C]
        radius_sum = (chain_radius[:, None] + chain_radius[None, :])[allowed]  # [A] allowed pairs
        # Keep the bound on the host, with the same float32 arithmetic as clash distances.
        self._max_interchain_radius_sum: torch.Tensor | None = (
            radius_sum.max().cpu() if radius_sum.numel() else None
        )  # [] scalar, or None without allowed pairs
        # The interchain bonded pair id's are stored as single numbers, first * N + second, for quick lookup during compute().
        feats.update(
            atom_chain_id=chain.unsqueeze(0),  # [1, N]
            atom_vdw_radius=atom_radius.unsqueeze(0),  # [1, N]
            interchain_steric_atom_mask=steric_atom.unsqueeze(0),  # [1, N]
            interchain_pair_mask=interchain.unsqueeze(0),  # [1, C, C]
            interchain_bonded_pair_key=(excluded[0] * n_atoms + excluded[1]).unsqueeze(0),  # [1, E]
        )

    def compute(self, coords, feats, parameters=None):
        """Return ``E_b = sum_p [d_min,p - ||x_bi - x_bj||]_+``."""
        assert parameters is not None and "buffer" in parameters, (
            "InterChainVDWOverlapPotential requires parameters['buffer']."
        )
        # Require pair discovery to include every pair that can have positive energy.
        if self._max_interchain_radius_sum is not None:
            maximum = (self._max_interchain_radius_sum * (1.0 - parameters["buffer"])).item()
            if maximum >= self.cutoff:
                raise ValueError(
                    f"cutoff ({self.cutoff}) must exceed every clash distance (maximum {maximum})"
                )
        # Atom pairs from different chains that are closer than the cutoff in any particle.
        pairs = compute_atom_pairs_within_cutoff(  # [2, P_found]: first atoms, second atoms
            coords,
            atom_mask=feats["interchain_steric_atom_mask"][0],
            atom_chain_id=feats["atom_chain_id"][0],
            allowed_chain_pairs=feats["interchain_pair_mask"][0],
            cutoff=self.cutoff,
            block_size=self.block_size,
        )
        first, second = pairs

        # Skip pairs within three bonds of a declared cross-chain bond, such as the two sulfurs
        # of an inter-chain disulfide: the bond sets their distance, so it is not a clash.
        # prepare() stored these pairs as one number each, first * N + second.
        n_atoms = feats["atom_vdw_radius"].shape[1]  # N
        near_bond = torch.isin(first * n_atoms + second, feats["interchain_bonded_pair_key"][0])
        pairs = pairs[:, ~near_bond]  # [2, P]
        first, second = pairs

        # A pair clashes when it is closer than its summed VDW radii, shrunk by the buffer.
        radius = feats["atom_vdw_radius"][0]  # [N]
        clash_distance = (radius[first] + radius[second]) * (1.0 - parameters["buffer"])  # [P]
        return _steric_overlap_penalty(coords, pairs, clash_distance)  # [B]


class IntraChainVDWOverlapPotential(FeatsPotential):
    """Clash penalty on nonbonded protein and nucleic-acid atoms within one chain.

    Includes canonical protein/DNA/RNA atoms and unknown protein residue X atoms,
    excluding modified residues, unknown nucleotides and ligands. Nearby covalent
    neighbors are excluded; :class:`PolymerLocalClashPotential` covers clashes
    between such neighbors in different residues.

    Pass ``buffer`` through ``parameters`` to scale the summed VDW radii by
    ``1 - buffer``. The ``cutoff`` in angstroms must exceed every clash distance.
    """

    cutoff: float = 10.0  # angstroms; must exceed all clash distances
    block_size: int = 512  # atoms per search block, limiting peak memory

    def prepare(self, system: System) -> None:
        """Cache eligible atoms and same-chain pair filters for the system.

        Exclude pairs separated by at most ``NONBONDED_EXCLUSION_GRAPH_DISTANCE``
        covalent bonds, including explicit links. Atom eligibility and connectivity
        are defined by :func:`build_polymer_steric_graph`.
        """
        super().prepare(system)
        structure = AllAtomStructure.from_system(system, mode="structure")
        graph, eligible = build_polymer_steric_graph(structure)
        feats = self.feats
        n_atoms = feats["atom_pad_mask"].shape[1]
        chain = _atom_chain_id(feats)
        atom_radius = _atom_vdw_radius(feats)
        steric_atom = torch.zeros(n_atoms, dtype=torch.bool, device=chain.device)
        steric_atom[: structure.num_nodes] = eligible
        excluded = (
            torch.tensor(
                sorted(graph.nonbonded_exclusions()), dtype=torch.long, device=chain.device
            )
            .reshape(-1, 2)
            .T
        )  # [2, E]
        keys = excluded[0] * n_atoms + excluded[1]  # [E]
        feats.update(
            atom_chain_id=chain.unsqueeze(0),  # [1, N]
            atom_vdw_radius=atom_radius.unsqueeze(0),  # [1, N]
            polymer_steric_atom_mask=steric_atom.unsqueeze(0),  # [1, N]; True means eligible
            polymer_bonded_pair_key=keys.unsqueeze(0),  # [1, E]
            intrachain_pair_mask=torch.eye(
                int(chain.max()) + 1, dtype=torch.bool, device=chain.device
            ).unsqueeze(0),  # [1, C, C]; True permits same-chain pairs
        )

        eligible_radii = atom_radius[steric_atom]
        # Twice the largest eligible radius bounds all pair sums, even if no nonbonded
        # pairs remain. Retain this float32 bound on the host to avoid GPU synchronization.
        self._max_intrachain_radius_sum = (
            2 * eligible_radii.max().cpu() if eligible_radii.numel() else None
        )

    def compute(self, coords, feats, parameters=None):
        """Return the overlap penalty for eligible nonbonded same-chain pairs."""
        assert parameters is not None and "buffer" in parameters, (
            "IntraChainVDWOverlapPotential requires parameters['buffer']."
        )
        # Validate the buffer-scaled bound before searching; skip only if no atoms are eligible.
        if self._max_intrachain_radius_sum is not None:
            maximum = (self._max_intrachain_radius_sum * (1.0 - parameters["buffer"])).item()
            if maximum >= self.cutoff:
                raise ValueError(
                    f"cutoff ({self.cutoff}) must exceed every clash distance "
                    f"(conservative bound {maximum})"
                )
        steric_atom = feats["polymer_steric_atom_mask"][0]  # [N]
        chain = feats["atom_chain_id"][0]  # [N]
        index = compute_atom_pairs_within_cutoff(  # [2, P_cutoff]
            coords,
            atom_mask=steric_atom,
            atom_chain_id=chain,
            allowed_chain_pairs=feats["intrachain_pair_mask"][0],
            cutoff=self.cutoff,
            block_size=self.block_size,
        )
        i, j = index
        nonbonded = ~torch.isin(  # [P_cutoff]
            i * len(steric_atom) + j, feats["polymer_bonded_pair_key"][0]
        )
        index = index[:, nonbonded]  # [2, P]
        i, j = index
        radius = feats["atom_vdw_radius"][0]  # [N]
        clash_distance = (radius[i] + radius[j]) * (1.0 - parameters["buffer"])  # [P]
        return _steric_overlap_penalty(coords, index, clash_distance)


class PolymerLocalClashPotential(FeatsPotential):
    """Penalize short-range clashes between protein or nucleic-acid residues.

    Uses the same eligible atoms as :class:`IntraChainVDWOverlapPotential` and
    complements it and :class:`InterChainVDWOverlapPotential` by covering nearby
    covalent neighbors excluded from those terms, including across declared bonds.
    Pass the minimum allowed distance in angstroms through
    ``parameters={"minimum_distance": 2.0}``; it can also be a ``ParameterSchedule``.
    This distance is a lower bound, not a VDW radius or an equilibrium target.
    """

    def prepare(self, system: System) -> None:
        """Cache inter-residue pairs two or three covalent bonds apart."""
        super().prepare(system)
        structure = AllAtomStructure.from_system(system, mode="structure")
        graph, eligible = build_polymer_steric_graph(structure)
        # Starting from all pairs separated by at most three covalent bonds, remove direct
        # bonds and same-residue pairs. The result contains the inter-residue 1-3 and 1-4
        # pairs scored by the local-clash potential; cross-chain pairs arise only across
        # declared bonds, such as inter-chain disulfides.
        direct = set(graph.bonds)
        index = (
            torch.tensor(
                sorted(graph.nonbonded_exclusions() - direct),
                dtype=torch.long,
                device=structure.coords.device,
            )
            .reshape(-1, 2)
            .T
        )  # [2, P]
        i, j = index
        keep = (
            eligible[i] & eligible[j] & (structure.residue_idx[i] != structure.residue_idx[j])
        )  # [P]; True means eligible inter-residue endpoints
        self.feats["polymer_local_clash_index"] = index[:, keep].unsqueeze(0)  # [1, 2, P_local]

    def compute(self, coords, feats, parameters=None):
        """Return the overlap penalty using ``parameters["minimum_distance"]`` in angstroms."""
        assert parameters is not None and "minimum_distance" in parameters, (
            "PolymerLocalClashPotential requires parameters['minimum_distance']."
        )
        return _steric_overlap_penalty(
            coords, feats["polymer_local_clash_index"][0], parameters["minimum_distance"]
        )


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def _compute_args(self, feats, parameters):
        atom_chain_id = _atom_chain_id(feats)
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats["symmetric_chain_index"][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return (
            pair_index,
            (k, lower_bounds, upper_bounds),
            (atom_chain_id, atom_pad_mask),
        )


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def _compute_args(self, feats, parameters):
        stereo_bond_index = feats["stereo_bond_index"][0]
        stereo_bond_orientations = feats["stereo_bond_orientations"][0].bool()

        lower_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        upper_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters["buffer"]
        upper_bounds[stereo_bond_orientations] = float("inf")
        lower_bounds[~stereo_bond_orientations] = float("-inf")
        upper_bounds[~stereo_bond_orientations] = parameters["buffer"]

        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def _compute_args(self, feats, parameters):
        chiral_atom_index = feats["chiral_atom_index"][0]
        chiral_atom_orientations = feats["chiral_atom_orientations"][0].bool()

        lower_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        upper_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        lower_bounds[chiral_atom_orientations] = parameters["buffer"]
        upper_bounds[chiral_atom_orientations] = float("inf")
        upper_bounds[~chiral_atom_orientations] = -1 * parameters["buffer"]
        lower_bounds[~chiral_atom_orientations] = float("-inf")

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def _compute_args(self, feats, parameters):
        double_bond_index = feats["planar_bond_index"][0].T
        double_bond_improper_index = torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 3],
            ],
            device=double_bond_index.device,
        ).T
        improper_index = (
            double_bond_index[:, double_bond_improper_index].swapaxes(0, 1).flatten(start_dim=1)
        )
        lower_bounds = None
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None


class ContactPotential(FlatBottomPotential, DistancePotential):
    """Flat-bottom upper on the distance between pairs of atoms.

    Reads feats (single-system topology, indexed at ``[0]``):

    - ``contact_pair_index``  ``[B, 2, K]`` int64 — atom pair indices
    - ``contact_thresholds``  ``[B, K]`` float32 — per-pair upper bound

    Supports independent atom pairs. It does not implement union semantics for
    multi-atom pockets.
    """

    def _compute_args(self, feats, parameters):
        pair_index = feats["contact_pair_index"][0]
        upper_bounds = feats["contact_thresholds"][0].clone()
        k = torch.ones_like(upper_bounds)
        return pair_index, (k, None, upper_bounds), None


class TemplateReferencePotential(FlatBottomPotential):
    """Flat-bottom upper on per-token distance between sampled coords and
    a rigid-aligned reference template.

    Reads feats (single-template, single-system topology; leading dim = 1):

    - ``template_cb``               ``[1, T, 3]`` float32 — reference Cβ coords
    - ``template_mask_cb``          ``[1, T]`` bool — valid ref atoms
    - ``template_force_threshold``  scalar-shaped float — flat-bottom upper
    - ``token_center_atom_index``   ``[1, T]`` int64 — rep-atom index per token
      using direct integer indices.

    Supports one reference template. Overrides :meth:`compute` directly because
    reference coordinates and masks do not fit the scalar-potential argument
    contract used by :meth:`_compute_args`.
    """

    def _compute_args(self, feats, parameters):
        raise NotImplementedError("TemplateReferencePotential overrides compute() directly")

    def compute_variable(self, coords, index):
        raise NotImplementedError("TemplateReferencePotential overrides compute() directly")

    def compute(
        self,
        coords: torch.Tensor,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        # coords: [B_particles, N_atoms_pad, 3]. Single-system topology → feats
        # tensors carry a leading len-1 dim that we drop with [0].
        ref_coords = feats["template_cb"][0].float()  # [T, 3]
        ref_mask = feats["template_mask_cb"][0].bool()  # [T]

        if not ref_mask.any():
            return _connected_zero(coords)

        # Rep-atom index per token (Cβ for protein tokens; the per-atom token for ligand).
        rep_atom_idx = feats["token_center_atom_index"][0]  # [T]

        # Gather sampled coords at each token's rep atom.
        sampled = coords[..., rep_atom_idx, :]  # [B_particles, T, 3]

        B = sampled.shape[0]
        ref_bc = ref_coords[None].expand(B, -1, -1).contiguous()  # [B, T, 3]
        mask_bc = ref_mask[None].float().expand(B, -1).contiguous()  # [B, T]

        # weighted_rigid_align requires float32 and autocast off (see edm_steered.py:351).
        with torch.autocast("cuda", enabled=False):
            aligned_ref, _, _ = weighted_rigid_align(
                ref_bc.float(), sampled.float(), mask_bc, mask_bc
            )
        aligned_ref = aligned_ref.to(sampled.dtype)

        # Per-token Euclidean distance between sample and rigid-aligned ref.
        d = torch.linalg.norm(sampled - aligned_ref, dim=-1)  # [B, T]

        # Flat-bottom upper only where the ref token is valid; elsewhere unbounded.
        threshold = feats["template_force_threshold"].float().reshape(-1)[0]
        upper = torch.where(
            ref_mask[None].expand_as(d),
            threshold,
            torch.full_like(d, float("inf")),
        )
        k = torch.ones_like(d)
        energy = self.compute_function(d, k, None, upper)
        return energy.sum(dim=-1)  # [B_particles]


class AnglePotential(FeatsPotential):
    """Compute unsigned bond angles in radians, between zero and pi.

    Index rows are the two outer atoms followed by their shared center.
    """

    _ANGLE_EPS = 1e-6

    def compute_variable(self, coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        first = coords.index_select(-2, index[0]) - coords.index_select(-2, index[2])  # [B, P, 3]
        second = coords.index_select(-2, index[1]) - coords.index_select(-2, index[2])  # [B, P, 3]
        # Avoid division by zero when an outer atom coincides with the center.
        first = first / torch.linalg.norm(first, dim=-1, keepdim=True).clamp_min(self._ANGLE_EPS)
        second = second / torch.linalg.norm(second, dim=-1, keepdim=True).clamp_min(self._ANGLE_EPS)
        # Keep acos gradients finite for nearly straight or folded-back bonds.
        cosine = (first * second).sum(dim=-1).clamp(-1.0 + self._ANGLE_EPS, 1.0 - self._ANGLE_EPS)
        return torch.acos(cosine)


class BondPotential(FlatBottomPotential):
    """Shared geometry preparation for the bond-length and bond-angle restraints.

    Args:
        sigma_tolerance: Required finite, nonnegative number of reference standard
            deviations used to expand the bounds. This is dimensionless, not a
            distance or angle, and must be fixed because bounds are prepared once.
        parameters: Steering weights and schedules, separate from the fixed tolerance.

    See :func:`bioemu2.denoising.steering.geometry_topology.get_bond_reference`
    and :func:`bioemu2.denoising.steering.geometry_topology.get_angle_reference`
    for reference selection, and
    :func:`bioemu2.denoising.steering.geometry_topology.compute_geometry_bounds`
    for tolerance scaling and clipping.
    """

    def __init__(
        self,
        *,
        sigma_tolerance: float,
        parameters: dict[str, ParameterSchedule | float | int | bool] | None = None,
    ) -> None:
        super().__init__(parameters=parameters)
        if not math.isfinite(sigma_tolerance) or sigma_tolerance < 0:
            raise ValueError("sigma_tolerance must be finite and nonnegative.")
        self.sigma_tolerance = sigma_tolerance

    def _prepare_geometry(
        self, system: System
    ) -> tuple[AllAtomStructure, BondedGeometryGraph, StereoTables]:
        structure = AllAtomStructure.from_system(system, mode="structure")
        batch = batch_collate([structure])
        self.feats = convert_densebatch_to_potential_features(batch)
        bond_graph = build_bonded_geometry_graph(structure)
        tables = load_stereo_tables()
        return structure, bond_graph, tables


class BondLengthPotential(BondPotential, DistancePotential):
    """Penalize bond lengths outside reference or heuristic intervals.

    Reference values, stored bounds, and linear length violations are in angstroms.
    """

    def prepare(self, system: System) -> None:
        """Build bond topology, resolve reference values, and store indices and lower/upper bounds."""
        structure, bond_graph, tables = self._prepare_geometry(system)
        constraints = []
        pooled: dict[tuple[tuple[str, ...], bool], tuple[float, float, float] | None] = {}
        for i, j in bond_graph.bonds:
            reference = get_bond_reference(bond_graph, i, j, tables, pooled)
            bounds = compute_geometry_bounds(reference, self.sigma_tolerance)
            constraints.append((i, j, bounds))
        index, lower, upper = pack_bond_constraints(constraints, structure.coords.device)
        # One system: indices [1, 2, P] and lower/upper bounds [1, P], with P bonds.
        self.feats.update(bond_index=index, bond_lower=lower, bond_upper=upper)

    def _compute_args(
        self, feats: dict[str, torch.Tensor], parameters: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        """Return prepared bond indices, unit penalty slopes, and bounds for flat-bottom evaluation."""
        index = feats["bond_index"][0]
        lower, upper = feats["bond_lower"][0], feats["bond_upper"][0]
        return index, (torch.ones_like(lower), lower, upper), None


class BondAnglePotential(BondPotential, AnglePotential):
    """Penalize bond angles outside reference or heuristic intervals.

    Reference values are in degrees; stored bounds and linear angular violations
    are in radians.
    """

    def prepare(self, system: System) -> None:
        """Build angle topology, resolve reference values, and store indices and radian bounds."""
        structure, bond_graph, tables = self._prepare_geometry(system)
        constraints = []
        pooled: dict[tuple[tuple[str, ...], bool], tuple[float, float, float] | None] = {}
        for i, k, center in bond_graph.angles:
            reference = get_angle_reference(bond_graph, i, center, k, tables, pooled)
            bounds = compute_geometry_bounds(reference, self.sigma_tolerance, angle=True)
            constraints.append((i, k, center, bounds))
        index, lower, upper = pack_angle_constraints(constraints, structure.coords.device)
        # One system: indices [1, 3, P] and lower/upper bounds [1, P], with P angles.
        self.feats.update(angle_index=index, angle_lower=lower, angle_upper=upper)

    def _compute_args(
        self, feats: dict[str, torch.Tensor], parameters: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        """Return prepared angle indices, unit penalty slopes, and bounds for flat-bottom evaluation."""
        index = feats["angle_index"][0]
        lower, upper = feats["angle_lower"][0], feats["angle_upper"][0]
        return index, (torch.ones_like(lower), lower, upper), None


class MMFFPotential(FeatsPotential):
    """Shared preparation for the two MMFF ligand terms.

    ``prepare`` runs RDKit's MMFF94 typing once per sampling run and caches the
    packed indices and parameters alongside the usual topology features.
    """

    def prepare(self, system: System) -> None:
        super().prepare(system)
        terms = mmff_terms_from_system(system)
        self.feats.update(
            mmff_pair_index=terms.pair_index,
            mmff_pair_rstar=terms.pair_rstar,
            mmff_pair_eps=terms.pair_eps,
            mmff_pair_qq=terms.pair_qq,
            mmff_nb_index=terms.nb_index,
            mmff_nb_rstar=terms.nb_rstar,
            mmff_nb_eps=terms.nb_eps,
            mmff_nb_qq=terms.nb_qq,
            mmff_hydrogen_ref=terms.hydrogen_ref,
            mmff_hydrogen_geom=terms.hydrogen_geom,
            mmff_bond_index=terms.bond_index,
            mmff_bond_r0=terms.bond_r0,
            mmff_bond_kb=terms.bond_kb,
            mmff_angle_index=terms.angle_index,
            mmff_angle_theta0=terms.angle_theta0,
            mmff_angle_ka=terms.angle_ka,
            sb_index=terms.sb_index,
            sb_theta0=terms.sb_theta0,
            sb_r0_ij=terms.sb_r0_ij,
            sb_r0_kj=terms.sb_r0_kj,
            sb_kba_ijk=terms.sb_kba_ijk,
            sb_kba_kji=terms.sb_kba_kji,
            mmff_torsion_index=terms.torsion_index,
            mmff_torsion_v1=terms.torsion_v1,
            mmff_torsion_v2=terms.torsion_v2,
            mmff_torsion_v3=terms.torsion_v3,
        )


class HydrogenMixin:
    """Builds ligand hydrogens before evaluating, when asked to.

    The models predict heavy atoms only, but an all-atom force field spreads a
    bond's torsional barrier over every i-j-k-l path through it. Keeping only
    the heavy path does not weaken the term, it inverts it: measured on
    cyclohexane, all nine paths minimise at 60 and 180 degrees while the heavy
    path alone minimises at 180 and puts a MAXIMUM at the chair's 60.

    Hydrogens are placed from each parent's current heavy neighbours
    (_place_hydrogens), with rotatable ones (OH, SH) at the rotamer of lowest
    energy summed over the potentials sharing this hydrogen group.

    With ``use_hydrogens`` false the terms naming a hydrogen are skipped and the
    old heavy-only behaviour is recovered, which is what the two are compared
    against.
    """

    _H_REF_KEY = ""
    _H_GEOM_KEY = ""

    def _hydrogen_plan_for(self, feats):
        ref = feats[self._H_REF_KEY][0]
        cache = self.__dict__.setdefault("_hydrogen_plans", {})
        key = (int(ref.shape[-1]), int(feats["atom_pad_mask"].shape[1]))
        if key not in cache:
            n = key[1]                       # hydrogens are indexed from the padded width
            bonds = feats[self._H_REF_KEY.replace("hydrogen_ref", "bond_index")][0].cpu()
            bonds = bonds[:, (bonds < n).all(0)]
            element = feats["ref_element"][0].argmax(-1).cpu().tolist()
            cache[key] = _hydrogen_plan([int(x) for x in ref[0].cpu().tolist()],
                                        [tuple(map(int, e)) for e in bonds.t().tolist()],
                                        element)
        return cache[key]

    def _with_hydrogens(self, coords, feats):
        if not getattr(self, "use_hydrogens", False):
            return coords
        if _HYDROGEN_OVERRIDE is not None:
            return _HYDROGEN_OVERRIDE
        ref = feats.get(self._H_REF_KEY)
        geom = feats.get(self._H_GEOM_KEY)
        if ref is None or geom is None or ref[0].shape[-1] == 0:
            return coords
        plan = self._hydrogen_plan_for(feats)
        r = geom[0][:, 0]                    # bond lengths; the stored angles are unused
        phis = None
        if plan.get("rotor"):
            group = self.__dict__.get("_hydrogen_group")
            if group is None:
                group = self._hydrogen_group = _HydrogenGroup([self])
            phis = group.rotamers(coords, plan, r)
        return _place_hydrogens(coords, plan, r, phis)

    def _select_terms(self, feats, index, *values):
        """Keep every term when hydrogens are rebuilt; otherwise only the
        terms whose atoms are all real atoms.

        Both extractions keep hydrogen-containing torsions now, so without this
        the heavy-only mode indexes past the end of the coordinate tensor.
        """
        if getattr(self, "use_hydrogens", False):
            return index, values
        n_atoms = int(feats["atom_pad_mask"].shape[1])
        keep = (index < n_atoms).all(dim=0)
        return index[:, keep], tuple(v[keep] for v in values)


class MMFFBondPotential(MMFFPotential, DistancePotential):
    """MMFF-typed restraint on every ligand bond length.

    Huber in the length deviation: ``k dr^2`` within ``cutoff`` of r0 and
    linear beyond, so the curvature at the minimum is MMFF's own force constant
    while the force stays bounded by ``2 k cutoff``.

    The bound is the point. Guidance takes num_gd_steps descent steps per
    denoising step, and descent on a quadratic of curvature k diverges for a
    step above 2/k; an earlier version of this term used MMFF's force constant
    with no bound (plain quadratic) and blew every ligand out to 14-24 A bonds.
    The constant itself is MMFF's own (143.9325/2 * kb, ~306 kcal/mol/A^2 for an
    sp3 C-C): with uniform_k=False, as get_mmff_potentials builds it, k is not
    normalized, and the guidance weight (e.g. 0.001 -> w*k ~ 0.3) keeps descent
    stable.

    MMFF's quartic correction is dropped for simplicity, not for correctness:
    with cs = -2 the shape factor dr^2 - 2 dr^3 + (7/3) dr^4 has no stationary
    point but dr = 0 (the discriminant of its derivative's quadratic factor is
    9 - 4(14/3) = -9.67), so it is monotonic either side of the minimum and
    would have served. It simply has no bounded-force region.

    This exists because BioEmu2 draws ring C-C bonds at roughly one length
    whatever the hybridisation: 1.413 A for cyclohexane against an ideal 1.53,
    and 1.412 for benzene against 1.39.
    """

    def __init__(self, parameters=None, cutoff: float = 0.2,
                 uniform_k: bool = False) -> None:
        super().__init__(parameters=parameters)
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive.")
        self.cutoff = cutoff
        # uniform_k=False (the default here and in get_mmff_potentials): MMFF's
        # own kb, 143.9325/2 * kb in kcal/mol/A^2, not normalized. True gives
        # k = 1 for every bond.
        self.uniform_k = uniform_k

    def compute_function(self, value, k, r0):
        delta = (value - r0).abs()
        d = self.cutoff
        return k * torch.where(delta <= d, delta * delta, d * (2.0 * delta - d))

    def _compute_args(self, feats, parameters=None):
        index = feats["mmff_bond_index"][0]
        r0 = feats["mmff_bond_r0"][0]
        k = torch.ones_like(r0) if self.uniform_k else feats["mmff_bond_kb"][0]
        return index, (k, r0), None


class MMFFAnglePotential(MMFFPotential):
    """MMFF-typed restraint on every ligand bond angle.

    ``E = k (1 - cos(theta - theta0))``, with theta0 taken from MMFF94's own
    angle-bend parameters. The force is zero at theta0 and bounded by k
    everywhere, and at theta0 = 180 degrees the form reduces to MMFF's linear
    bend, so linear geometries need no special case.

    This is a well, not a flat bottom. A flat-bottom angle term cannot forbid a
    planar six-ring at any tolerance: for a planar hexagon the 1-3 distance is
    exactly sqrt(3) times the bond, so bond and 1-3 windows that admit the real
    ring also admit the flat one.
    """

    _ANGLE_EPS = 1e-9

    def compute_variable(self, coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        u = coords.index_select(-2, index[0]) - coords.index_select(-2, index[2])
        v = coords.index_select(-2, index[1]) - coords.index_select(-2, index[2])
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(self._ANGLE_EPS)
        v = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(self._ANGLE_EPS)
        cosine = (u * v).sum(dim=-1)
        sine = torch.linalg.norm(torch.cross(u, v, dim=-1), dim=-1)
        return torch.atan2(sine, cosine)

    def compute_function(self, value, k, theta0):
        return k * (1.0 - torch.cos(value - theta0))

    def __init__(self, parameters=None, uniform_k: bool = False) -> None:
        super().__init__(parameters=parameters)
        self.uniform_k = uniform_k

    def _compute_args(self, feats, parameters=None):
        index = feats["mmff_angle_index"][0]
        theta0 = feats["mmff_angle_theta0"][0]
        if self.uniform_k:
            return index, (torch.ones_like(theta0), theta0), None
        # MMFF's own ka as mmff.py stores it: 143.9325 * ka in kcal/mol/rad^2
        # (the cosine form k (1 - cos dtheta) has curvature k, so this matches
        # MMFF's harmonic bend), NOT normalized -- there is no KA_REF. Up to
        # ~174 kcal/mol/rad^2; the guidance loop takes unit steps, so stability
        # needs guidance_weight * k < 2 (e.g. 0.001 * 174 = 0.17).
        ka = feats.get("mmff_angle_ka")
        k = torch.ones_like(theta0) if ka is None else ka[0]
        return index, (k, theta0), None


class MMFFStretchBendPotential(MMFFPotential):
    """MMFF's stretch-bend cross term.

        EBA = 2.51210 * (kba_ijk * dr_ij + kba_kji * dr_kj) * dth_deg

    Couples each bond angle to the two bonds that form it: narrowing an angle
    crowds the outer atoms, and the bonds lengthen to relieve it. Measured on
    acetamide at dr = 0.02 A and dth = 3 deg this is +0.14 to +0.17 kcal/mol,
    against 0.19 for the angle term and 0.12 for the bond term at the same
    deformation -- the same size as the terms it couples.

    The two force constants are not interchangeable: kba_ijk multiplies the
    i-centre bond deviation and kba_kji the k-centre one, so sb_index must
    keep the order the parameters were read in.

    Scope follows the angle term: heavy centre, heavy neighbours.
    """

    #: MMFF's stretch-bend conversion, kcal/mol per (md/rad * A * degree).
    CONV = 2.51210
    _EPS = 1e-12

    def compute(self, coords, feats, parameters=None):
        index = feats["sb_index"][0]
        if index.shape[-1] == 0:
            return _connected_zero(coords)
        theta0 = feats["sb_theta0"][0]
        r0_ij = feats["sb_r0_ij"][0]
        r0_kj = feats["sb_r0_kj"][0]
        k_ijk = feats["sb_kba_ijk"][0]
        k_kji = feats["sb_kba_kji"][0]

        i, k, j = index[0], index[1], index[2]        # j is the centre
        u = coords.index_select(-2, i) - coords.index_select(-2, j)
        v = coords.index_select(-2, k) - coords.index_select(-2, j)
        r_ij = torch.sqrt((u * u).sum(-1) + self._EPS)
        r_kj = torch.sqrt((v * v).sum(-1) + self._EPS)

        un = u / r_ij.unsqueeze(-1)
        vn = v / r_kj.unsqueeze(-1)
        cosine = (un * vn).sum(dim=-1)
        sine = torch.linalg.norm(torch.cross(un, vn, dim=-1), dim=-1)
        theta = torch.atan2(sine, cosine)

        # MMFF states this term with the angle deviation in DEGREES; the
        # 2.51210 prefactor carries that convention.
        dtheta_deg = (theta - theta0) * (180.0 / math.pi)
        e = self.CONV * (k_ijk * (r_ij - r0_ij)
                         + k_kji * (r_kj - r0_kj)) * dtheta_deg
        return e.sum(dim=-1)


class MMFFTorsionPotential(HydrogenMixin, MMFFPotential, DihedralPotential):
    _H_REF_KEY = "mmff_hydrogen_ref"
    _H_GEOM_KEY = "mmff_hydrogen_geom"

    def __init__(self, parameters=None, use_hydrogens: bool = False) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens

    def compute(self, coords, feats, parameters=None):
        return super().compute(self._with_hydrogens(coords, feats), feats, parameters)

    """Three-term MMFF94 torsion on ligand dihedrals.

    ``E = 1/2 [V1 (1 + cos phi) + V2 (1 - cos 2 phi) + V3 (1 + cos 3 phi)]``

    Scoped at preparation time to torsions about ring bonds plus non-ring
    torsions with V2 > 0; see ``mmff._torsion_terms`` for why.
    """

    def compute_function(self, value, v1, v2, v3):
        return 0.5 * (
            v1 * (1.0 + torch.cos(value))
            + v2 * (1.0 - torch.cos(2.0 * value))
            + v3 * (1.0 + torch.cos(3.0 * value))
        )

    def _compute_args(self, feats, parameters=None):
        index, values = self._select_terms(
            feats, feats["mmff_torsion_index"][0],
            feats["mmff_torsion_v1"][0], feats["mmff_torsion_v2"][0],
            feats["mmff_torsion_v3"][0])
        return index, values, None


class MMFF14Potential(HydrogenMixin, MMFFPotential):
    """MMFF 1-4 pairs: buffered 14-7 vdW, unscaled, plus 0.75 electrostatics.

    MMFF's torsion parameters were fitted with these present, the same way
    GAFF's were, so the torsion term alone is incomplete in both. Adding GAFF's
    took cyclopentane from 0.010 to 0.154 and methylcyclohexane from 0.229 to
    0.295 against a true 0.293, which is why this is worth testing here.

    Electrostatics are real Coulomb, C q_i q_j / (D r) with D = ``dielectric``,
    outside s = max(frac * size, 0.5 A) and a parabola matching value and slope
    inside (_coulomb_core); the van der Waals is the force field's own form
    outside its own switch and a parabola inside. size is the pair's LJ sigma
    for GAFF2 and R* for MMFF.
    """

    _H_REF_KEY = "mmff_hydrogen_ref"
    _H_GEOM_KEY = "mmff_hydrogen_geom"

    # Bounded cosine electrostatics, shared with the full nonbonded
    # term. The force field's own constant (332) is 28x this one and is
    # deliberately not used: see the class docstring.
    _NORM_EPS = 1e-12       # keeps d||r||/dr finite at coincidence

    def __init__(self, parameters=None, use_hydrogens: bool = False,
                 dielectric: float = 1.0, lj_core_switch: float = 0.5,
                 coulomb_core_switch: float = 0.5) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens
        if not math.isfinite(lj_core_switch) or lj_core_switch <= 0:
            raise ValueError("lj_core_switch must be finite and positive.")
        # vdW core switch s = max(lj_core_switch * size, 0.5 A); see _core_switch.
        self.lj_core_switch = float(lj_core_switch)
        if not math.isfinite(coulomb_core_switch) or coulomb_core_switch <= 0:
            raise ValueError("coulomb_core_switch must be finite and positive.")
        # Coulomb core switch s = max(coulomb_core_switch * size, 0.5 A).
        self.coulomb_core_switch = float(coulomb_core_switch)
        if not math.isfinite(dielectric) or dielectric <= 0:
            raise ValueError("dielectric must be finite and positive.")
        self.dielectric = float(dielectric)
        # False gives MMFF's own vdW and electrostatics: no cosine wall and
        # no cosine Coulomb. Both remain finite at coincidence thanks to
        # MMFF's 0.07 R* and 0.05 buffers, but the forces are large.
        # independently because capping the vdW and replacing the
        # Coulomb are separate decisions with opposite verdicts.
        # cosine 1-4 electrostatics: the same bounded form the full nonbonded
        # force fields.

    def compute(self, coords, feats, parameters=None):
        coords = self._with_hydrogens(coords, feats)
        index = feats["mmff_pair_index"][0]
        rstar = feats["mmff_pair_rstar"][0]
        eps = feats["mmff_pair_eps"][0]
        qq = feats["mmff_pair_qq"][0]
        index, (rstar, eps, qq) = self._select_terms(feats, index, rstar, eps, qq)
        if index.shape[-1] == 0:
            return _connected_zero(coords)

        delta = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r = torch.sqrt((delta * delta).sum(-1) + self._NORM_EPS)

        # Buffered 14-7. The 0.07 R* keeps the first factor finite at r = 0,
        # which is why this form needs less protection than a 12-6.
        rs = rstar.clamp_min(1e-6)
        # MMFF's buffered 14-7 outside the core switch, a parabola matching
        # value and slope inside it.
        vdw = _mmff147_core(r, rs, eps, _core_switch(rs, self.lj_core_switch))
        coulomb = _coulomb_core(
            r, _MMFF_COULOMB * qq / self.dielectric,
            _core_switch(rs, self.coulomb_core_switch))
        return (vdw + coulomb).sum(dim=-1)


class MMFFNonbondedPotential(HydrogenMixin, MMFFPotential):
    """Full intramolecular nonbonded for MMFF: capped buffered 14-7 + Coulomb.

        vdw  = eps * (1.07 R*/(r + 0.07 R*))^7
                   * (1.12 R*^7/(r^7 + 0.12 R*^7) - 2)
        coul = C q_i q_j cos^2(pi r / 2 r_c)

    The vdW is deliberately the same form as ``MMFF14Potential``, the 1-4 term
    for this force field: modelling MMFF's vdW two different ways inside one
    potential set would be wrong whatever it bought elsewhere.

    MMFF's bare 332 q q / (r + 0.05) reaches ~6600 q q kcal/mol at coincidence,
    which is a hazard across the full nonbonded set because early diffusion
    structures do have overlapping atoms. The core switch bounds it instead:
    real Coulomb outside s and a parabola matching value and slope inside.

    The 0.07 R* buffer keeps the first factor finite at coincidence, so this
    needs less protection than a 12-6, but it still reaches ~9.7e7 kcal/mol at
    r = 0, which is why the van der Waals carries a core switch too, exactly
    as the 1-4 term caps it; the attractive well is left alone.

    Scope: 1-2, 1-3 and 1-4 pairs are excluded, the 1-4s because they are a
    separate term carrying MMFF's own 0.75 electrostatic scaling. Charges here
    are unscaled.

    Electrostatics are real Coulomb, C q_i q_j / (D r) with D = ``dielectric``,
    outside s = max(frac * size, 0.5 A) and a parabola matching value and slope
    inside (_coulomb_core); the van der Waals is the force field's own form
    outside its own switch and a parabola inside. size is the pair's LJ sigma
    for GAFF2 and R* for MMFF.
    """

    _H_REF_KEY = "mmff_hydrogen_ref"
    _H_GEOM_KEY = "mmff_hydrogen_geom"

    KJ_TO_KCAL = 0.2390057
    _NORM_EPS = 1e-12

    def __init__(self, parameters=None, use_hydrogens: bool = False,
                 dielectric: float = 1.0, lj_core_switch: float = 0.5,
                 coulomb_core_switch: float = 0.5) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens
        if not math.isfinite(lj_core_switch) or lj_core_switch <= 0:
            raise ValueError("lj_core_switch must be finite and positive.")
        # vdW core switch s = max(lj_core_switch * size, 0.5 A); see _core_switch.
        self.lj_core_switch = float(lj_core_switch)
        if not math.isfinite(coulomb_core_switch) or coulomb_core_switch <= 0:
            raise ValueError("coulomb_core_switch must be finite and positive.")
        # Coulomb core switch s = max(coulomb_core_switch * size, 0.5 A).
        self.coulomb_core_switch = float(coulomb_core_switch)
        if not math.isfinite(dielectric) or dielectric <= 0:
            raise ValueError("dielectric must be finite and positive.")
        self.dielectric = float(dielectric)
        # MMFF's own forms when False: no cosine wall,
        # no cosine Coulomb.
        # independently because capping the vdW and replacing the
        # Coulomb are separate decisions with opposite verdicts.
        # The electrostatic part relative to the steric part. They do
        # different jobs, and round 3 put the GAFF optimum at 3.0 -- a setting
        # that does not exist with a single knob.
        # a vacuum range: in water these charges are screened by a
        # dielectric of ~78, and at 12 A a zwitterion's two ends
        # attract strongly enough to force a ring substituent axial
        # (tranexamic_z, 33.7% axial against a truth of 0.5%).

    def compute(self, coords, feats, parameters=None):
        coords = self._with_hydrogens(coords, feats)
        index = feats["mmff_nb_index"][0]
        rstar = feats["mmff_nb_rstar"][0]
        eps = feats["mmff_nb_eps"][0]
        qq = feats["mmff_nb_qq"][0]
        index, (rstar, eps, qq) = self._select_terms(feats, index, rstar,
                                                     eps, qq)
        if index.shape[-1] == 0:
            return _connected_zero(coords)

        delta = (coords.index_select(-2, index[0])
                 - coords.index_select(-2, index[1]))
        r = torch.sqrt((delta * delta).sum(-1) + self._NORM_EPS)

        rs = rstar.clamp_min(1e-6)
        # MMFF's buffered 14-7 outside the core switch, a parabola matching
        # value and slope inside it.
        vdw = _mmff147_core(r, rs, eps, _core_switch(rs, self.lj_core_switch))
        coul = _coulomb_core(
            r, _MMFF_COULOMB * qq / self.dielectric,
            _core_switch(rs, self.coulomb_core_switch))
        return (vdw + coul).sum(dim=-1)


#: Default late stage for a two-stage bond/angle weight: the last quarter of
#: the guided window (t_on 0.4 -> 0.1).
LIGAND_BOND_ANGLE_LATE_T_ON = 0.1

#: Resolved lazily: these classes are defined further down this module than
#: the factories that call the helper below.
def _late_types():
    return ((GAFFBondPotential, MMFFBondPotential),
            (GAFFAnglePotential, MMFFAnglePotential))


def _apply_late_bond_angle(terms, t_on, late_t_on, late_bond, late_angle) -> None:
    """Raise the bond and angle GUIDANCE weights for t <= late_t_on.

    Bond and angle are stiff, local and nearly convex, so clamping them hard
    only at the end -- once the torsions that decide chemical state have
    settled, and with little noise left to wash the correction out -- pulls
    back the bond stretch a large 1-4 weight induces without disturbing the
    conformation that weight was raised to get.

    Resampling weights are deliberately left alone: changing the Feynman-Kac
    reward partway through a run would reweight particles against a different
    objective than the one they were grown under.
    """
    bond_types, angle_types = _late_types()
    for p in terms:
        if isinstance(p, bond_types):
            late = late_bond
        elif isinstance(p, angle_types):
            late = late_angle
        else:
            continue
        if late is None:
            continue
        params = dict(p.parameters or {})
        early = params.get("guidance_weight")
        if early is None:
            continue
        if isinstance(early, PiecewiseStepFunction):
            # _apply_t_on already made a step at t_on; rebuild from its value.
            early = early.values[0]
        elif not isinstance(early, (int, float)):
            raise ValueError(
                "a late bond/angle weight needs a plain weight to start from, "
                f"got {type(early).__name__}"
            )
        thresholds = [late_t_on] if t_on is None else [late_t_on, t_on]
        values = [late, early] if t_on is None else [late, early, 0.0]
        params["guidance_weight"] = PiecewiseStepFunction(
            thresholds=thresholds, values=values
        )
        p.parameters = params


def _late_stage(t_on, guidance_ramp, bond_angle_late_weight,
                bond_late_weight, angle_late_weight, bond_angle_late_t_on):
    """(late_t_on, late_bond, late_angle), or None when no late stage is asked
    for. Validates the stage against t_on and guidance_ramp."""
    late_bond = (
        bond_late_weight if bond_late_weight is not None else bond_angle_late_weight
    )
    late_angle = (
        angle_late_weight if angle_late_weight is not None else bond_angle_late_weight
    )
    if late_bond is None and late_angle is None:
        return None
    late_t_on = bond_angle_late_t_on
    if late_t_on is None:
        late_t_on = LIGAND_BOND_ANGLE_LATE_T_ON
    if guidance_ramp:
        raise ValueError(
            "guidance_ramp replaces the whole guidance schedule; it cannot be "
            "combined with a late bond/angle weight"
        )
    if t_on is not None and late_t_on >= t_on:
        raise ValueError(
            f"bond_angle_late_t_on ({late_t_on}) must be below t_on ({t_on}); "
            "the late stage is the tail of the guided window, not all of it"
        )
    return late_t_on, late_bond, late_angle


def get_mmff_potentials(
    *, angle: bool = True, torsion: bool = True, bond: bool = True,
    pairs14: bool = True,
    # One shared weight: see MMFF_WEIGHT for why it is one number and where
    # its value comes from.
    angle_weight: float = MMFF_WEIGHT, torsion_weight: float = MMFF_WEIGHT,
    bond_weight: float = MMFF_WEIGHT, pair14_weight: float = MMFF_WEIGHT,
    # MMFF's real force constants, in kcal/mol/A^2 and kcal/mol/rad^2
    # (143.9325/2 * kb and 143.9325 * ka). Not normalized: MMFF_WEIGHT is
    # set against them.
    bond_uniform_k: bool = False, angle_uniform_k: bool = False,
    # Stretch-bend is off by default (-F stretchbend.enabled=1 turns it on).
    # k_ijk (dr_ij + dr_kj) dtheta is linear in the bond stretch and unbounded
    # below, so far from equilibrium it outweighs the bond term: stretching a
    # bond lowers the energy, guidance follows it, and FK resampling (which
    # rewards energy drops) spreads the exploding particle. Seen 2026-11-21:
    # charged picolinamide, every sample exploded (stretch-bend -140,000 vs
    # bond +108,000 kcal/mol); with the term off, no broken samples.
    stretchbend: bool = False, stretchbend_weight: float = MMFF_WEIGHT,
    hydrogen: bool = False,
    # MMFF has a vdW and an electrostatic term, so they are on. Off
    # meant the force field stopped at 1-4 and nothing acted between
    # two molecules in the same system.
    nonbonded: bool = True, nonbonded_weight: float = MMFF_WEIGHT,
    pose_weight: float = 0.10,
    pose_angle_buffer: float = 0.20,
    chiral_weight=None,
    stereo_weight=None,
    planar_weight=None,
    physicality: bool = True,
    chiral_buffer=None,
    pose_bond_buffer: float = 0.20,
    pose_clash_buffer: float = 0.15,
    # Realistic nonbonded (off by default): real Coulomb with a parabola core,
    # the force field's own LJ with a parabola core, and the dielectric D.
    dielectric: float = 1.0,
    # Dielectric for the 1-4 term and for the nonbonded term separately (-F pairs14.dielectric,
    # -F nonbonded.dielectric); None uses dielectric (-F dielectric), which sets both. E.g. 1-4 at the bond weight with
    # pairs14_dielectric 0.2: the 1-4 LJ no longer stretches rings, the 1-4 Coulomb keeps 5x that weight.
    pairs14_dielectric: float | None = None,
    nonbonded_dielectric: float | None = None,
    # vdW core switch s = max(lj_core_switch * R*, 0.5 A) for MMFF's buffered 14-7
    # (the Coulomb core stays at R*/2); -F lj_core.switch.
    lj_core_switch: float = 0.5,
    # Coulomb core switch s = max(coulomb_core_switch * R*, 0.5 A); -F coulomb_core.switch.
    coulomb_core_switch: float = 0.5,
    # Bond Huber cutoff in A: quadratic within, linear beyond (force bounded at
    # 2 k cutoff); -F bond.cutoff. 0.2: guidance is plain descent, unstable for
    # stiff (aromatic, polar) bonds at w * 8k > 2, and the cutoff bounds that
    # oscillation; 0.5 let bonds swing to the break limit (2026-11-21: Langevin
    # without resampling, 15 ligands, broken samples 77-98 at 0.2 vs 2554-4586 at 0.5).
    bond_cutoff: float = 0.2,
    # Every force-field term acts only for t <= t_on (t = 1 noise, 0 clean), in
    # guidance and resampling. 0.4 by default: early structures overlap (pairs
    # inside max(sigma/2, 0.5 A) up to t ~ 0.6, none from t = 0.5) and their
    # bond and angle energies collapsed FK resampling. 1 is always on, the
    # behaviour before 2026-11-20. nonbonded_t_on optionally delays the 1-4 and
    # nonbonded terms further.
    t_on: float | None = 0.4,
    nonbonded_t_on: float | None = None,
    # Guidance weight w * (1 - t) instead of the t_on switch (resampling keeps it).
    guidance_ramp: bool = False,
    # Two-stage bond/angle guidance: the weight above holds for
    # bond_angle_late_t_on < t <= t_on, then these for t <= late_t_on.
    bond_angle_late_weight: float | None = None,
    bond_late_weight: float | None = None,
    angle_late_weight: float | None = None,
    bond_angle_late_t_on: float | None = None,
) -> Sequence[Potential]:
    """The default potentials plus the requested MMFF ligand terms.

    Weights transfer from the Boltz fork in units but not always in effect:
    both take num_gd_steps = 20 descent steps on x0_hat per denoising step, so a
    weight means the same displacement, and the angle term's 0.10 reproduced its
    Boltz result here without retuning (cyclohexane 0.212 against 0.186).

    The torsion weight is the exception, and is 0.002 here against Boltz's
    0.001. At 0.001 this model leaves cyclopentane nearly flat (0.043 A against
    MMFF's 0.177) where Boltz reaches 0.179, because a five-ring puckers by
    torsion and below that threshold the angle term overcomes it. 0.002 gives
    0.184, costs the six-rings about 0.008 A, and leaves the aromatics at 0.001;
    they break between 0.005 and 0.008.

    Stability, not chemistry, bounds these: the inner loop is plain descent with
    step size guidance_weight, so for a restraint of curvature k it converges
    only while guidance_weight * k < 1. An earlier bond term carried MMFF's
    kb (~306) as a plain quadratic at weight 0.10, putting the product at 30.6,
    and every ligand blew out to 14-24 A bonds. The bond term now keeps MMFF's
    own kb but is Huber-bounded (MMFFBondPotential), and the weights used are
    ~0.001, so the product is ~0.3.
    """
    extra: list[Potential] = []
    if nonbonded:
        extra.append(
            MMFFNonbondedPotential(
                use_hydrogens=hydrogen,
                dielectric=(dielectric if nonbonded_dielectric is None else nonbonded_dielectric),
                lj_core_switch=lj_core_switch,
                coulomb_core_switch=coulomb_core_switch,
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": nonbonded_weight,
                    "resampling_weight": 0.1,
                },
            )
        )
    if pairs14:
        extra.append(
            MMFF14Potential(
                use_hydrogens=hydrogen,
                # Same cosine electrostatics as the full
                # nonbonded term, so s and the cutoff mean one
                # thing across every pair.
                dielectric=(dielectric if pairs14_dielectric is None else pairs14_dielectric),
                lj_core_switch=lj_core_switch,
                coulomb_core_switch=coulomb_core_switch,
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": pair14_weight,
                    "resampling_weight": 0.1,
                },
            )
        )
    if bond:
        extra.append(
            MMFFBondPotential(
                cutoff=bond_cutoff,
                uniform_k=bond_uniform_k,
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": bond_weight,
                    "resampling_weight": 0.1,
                },
            )
        )
    if angle:
        extra.append(
            MMFFAnglePotential(
                uniform_k=angle_uniform_k,
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": angle_weight,
                    "resampling_weight": 0.1,
                }
            )
        )
    if stretchbend:
        extra.append(
            MMFFStretchBendPotential(
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": stretchbend_weight,
                    "resampling_weight": 0.1,
                }
            )
        )
    if torsion:
        extra.append(
            MMFFTorsionPotential(
                use_hydrogens=hydrogen,
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": torsion_weight,
                    "resampling_weight": 0.1,
                }
            )
        )
    defaults = get_default_potentials(
        pose_weight=pose_weight,
        pose_angle_buffer=pose_angle_buffer,
        bond_buffer=pose_bond_buffer,
        clash_buffer=pose_clash_buffer,
        chiral_weight=chiral_weight,
        chiral_buffer=chiral_buffer,
        stereo_weight=stereo_weight,
        planar_weight=planar_weight,
    ) if physicality else ()
    _apply_t_on(extra, t_on, nonbonded_t_on, guidance_ramp)
    _late = _late_stage(t_on, guidance_ramp, bond_angle_late_weight,
                        bond_late_weight, angle_late_weight,
                        bond_angle_late_t_on)
    if _late is not None:
        _apply_late_bond_angle(extra, t_on, *_late)
    _share_hydrogen_group(extra)
    return (*defaults, *extra)


class GAFFPotential(FeatsPotential):
    #: Ligand force field the terms come from; get_gaff_potentials sets it.
    #: OpenFF shares GAFF2's functional forms, so these classes carry either.
    ligand_ff: str = "gaff2"

    """Shared preparation for the GAFF2 ligand terms.

    ``prepare`` runs boonza's GAFF2 parameterization once per sampling run.
    That calls antechamber for AM1-BCC charges and takes roughly 12 s per
    ligand, so results are cached by SMILES rather than recomputed per sample.
    """

    def prepare(self, system: System) -> None:
        super().prepare(system)
        terms = gaff_terms_from_system(system, ligand_ff=self.ligand_ff)
        self.feats.update(
            gaff_bond_index=terms.bond_index,
            gaff_bond_r0=terms.bond_r0,
            gaff_angle_index=terms.angle_index,
            gaff_angle_theta0=terms.angle_theta0,
            gaff_angle_fc=terms.angle_fc,
            gaff_bond_fc=terms.bond_fc,
            gaff_torsion_index=terms.torsion_index,
            gaff_torsion_phi0=terms.torsion_phi0,
            gaff_torsion_fc=terms.torsion_fc,
            gaff_hydrogen_ref=terms.hydrogen_ref,
            gaff_hydrogen_geom=terms.hydrogen_geom,
            gaff_nb_index=terms.nb_index,
            gaff_nb_sigma=terms.nb_sigma,
            gaff_nb_eps=terms.nb_eps,
            gaff_nb_qq=terms.nb_qq,
            gaff_pair_index=terms.pair_index,
            gaff_pair_aij=terms.pair_aij,
            gaff_pair_bij=terms.pair_bij,
            gaff_pair_qij=terms.pair_qij,
            gaff_cmap_index=terms.cmap_index,
            gaff_cmap_energy=terms.cmap_energy,
            gaff_cmap_origin=terms.cmap_origin,
        )


#: Ideal angles for hydrogens placed from neighbours, radians. BENT is the X-P-H
#: angle of a two-coordinate parent (hydroxyl, thiol), by atomic number.
_H_TET = math.radians(109.4712)
_H_TRI = math.radians(120.0)
_H_BENT = {8: math.radians(108.5), 16: math.radians(96.0)}
#: Rotations tried for a rotatable hydrogen (OH, SH), as the dihedral H-P-B-D in
#: degrees, by the coordination of B (the atom the O or S is bonded to,
#: hydrogens included). On a trigonal B (phenol, enol) the minima are in the
#: plane; on a tetrahedral one (an alcohol) they are staggered. Other B keep a
#: full scan.
_H_ROTAMERS_BY_COORD = {3: (0.0, 180.0), 4: (60.0, 180.0, 300.0)}
_H_ROTAMERS = tuple(float(d) for d in range(0, 360, 30))


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)


def _hydrogen_plan(parent: list[int], bonds: list[tuple[int, int]],
                   element: list[int]) -> dict[str, list[tuple]]:
    """Group hydrogens by the rule that places them, from topology alone.

    ``parent[k]`` is hydrogen k's heavy parent, ``bonds`` the heavy-heavy bonds and
    ``element`` the atomic number of every heavy atom. Coordination is heavy
    neighbours plus hydrogens:

      3 heavy + 1 H   (CH, NH+)          tet3: opposite the sum of the bond directions
      2 heavy + 2 H   (CH2)              tet2: +-54.7 deg out of the X-P-X plane
      2 heavy + 1 H   (=CH-, ring NH)    tri2: in the plane, opposite the bisector
      1 heavy, 4-coordinate (CH3, NH3+)  term: staggered, dihedral 180 / +60 / -60
      1 heavy, 3-coordinate (NH2, =CH2)  term: planar, dihedral 0 / 180
      1 heavy, 2-coordinate sp C (#CH)   linear: along the B->P axis
      1 heavy, 2-coordinate (OH, SH)     rotor: lowest-energy rotamer, of 0 / 180
                                         on a trigonal B (in plane) or 60 / 180 / 300
                                         on a tetrahedral B (staggered)

    Dihedrals are H-P-B-D with B the parent's heavy neighbour and D a heavy
    neighbour of B. A hydrogen no rule covers raises: there is no fallback.
    """
    nb: dict[int, set[int]] = {}
    for i, j in bonds:
        nb.setdefault(i, set()).add(j)
        nb.setdefault(j, set()).add(i)
    hs: dict[int, list[int]] = {}
    for k, p in enumerate(parent):
        hs.setdefault(p, []).append(k)
    plan: dict[str, list[tuple]] = {}
    uncovered: list[int] = []
    for p, ks in hs.items():
        heavy = sorted(nb.get(p, ()))
        coord = len(heavy) + len(ks)
        if len(heavy) == 3 and len(ks) == 1:
            plan.setdefault("tet3", []).append((ks[0], p, *heavy))
        elif len(heavy) == 2 and len(ks) == 2 and coord == 4:
            plan.setdefault("tet2", []).append((ks[0], ks[1], p, *heavy))
        elif len(heavy) == 2 and len(ks) == 1:
            plan.setdefault("tri2", []).append((ks[0], p, *heavy))
        elif len(heavy) == 1:
            b = heavy[0]
            d = sorted(x for x in nb.get(b, ()) if x != p)
            if coord == 2 and len(ks) == 1 and element[p] == 6:
                # sp carbon: one heavy neighbour plus one hydrogen can only be
                # a terminal alkyne, whose C-H lies on the extension of the
                # B->P axis. It needs its own rule rather than an _H_BENT
                # entry of 180 deg, because at 180 the bend-and-dihedral
                # construction is degenerate -- e1 is the unit part of a
                # vector that has become parallel to a -- and would give NaN.
                # No dihedral is needed, so this is placed before the `d`
                # guard and covers acetylene itself, where B has no other
                # heavy neighbour. Carbon only: a 2-coordinate N-H (an imine)
                # is bent, not linear, and divalent carbenes are not in scope.
                plan.setdefault("linear", []).append((ks[0], p, b))
            elif not d:
                uncovered += ks
            elif coord == 4 and len(ks) == 3:
                for k, phi in zip(ks, (180.0, 60.0, -60.0)):
                    plan.setdefault("term", []).append((k, p, b, d[0], _H_TET, math.radians(phi)))
            elif coord == 3 and len(ks) == 2:
                for k, phi in zip(ks, (0.0, 180.0)):
                    plan.setdefault("term", []).append((k, p, b, d[0], _H_TRI, math.radians(phi)))
            elif coord == 2 and len(ks) == 1 and element[p] in _H_BENT:
                b_coord = len(nb.get(b, ())) + len(hs.get(b, ()))
                cands = _H_ROTAMERS_BY_COORD.get(b_coord, _H_ROTAMERS)
                plan.setdefault("rotor", []).append(
                    (ks[0], p, b, d[0], _H_BENT[element[p]], cands))
            else:
                uncovered += ks
        else:
            uncovered += ks
    if uncovered:
        raise NotImplementedError(
            f"no placement rule for hydrogens {sorted(uncovered)} (parents "
            f"{sorted({parent[k] for k in uncovered})})")
    return plan


def _dihedral_hydrogens(coords, p, b, d, r, theta, phi):
    """Hydrogens at angle theta from the P->B bond and dihedral phi against D:

        P + r [a cos(theta) + sin(theta) (cos(phi) e1 + sin(phi) e2)]

    a = unit(B - P), e1 the unit part of (D - B) perpendicular to a, e2 = a x e1.
    """
    def at(i):
        return coords.index_select(-2, torch.as_tensor(i, device=coords.device))
    P0, B0, D0 = at(p), at(b), at(d)
    a = _unit(B0 - P0)
    w = D0 - B0
    e1 = _unit(w - (w * a).sum(-1, keepdim=True) * a)
    e2 = torch.cross(a, e1, dim=-1)
    theta, phi = theta.unsqueeze(-1), phi.unsqueeze(-1)
    return P0 + r.unsqueeze(-1) * (a * torch.cos(theta)
                                   + torch.sin(theta) * (torch.cos(phi) * e1 + torch.sin(phi) * e2))


def _place_hydrogens(coords: torch.Tensor, plan: dict[str, list[tuple]],
                     r: torch.Tensor, rotor_phi: torch.Tensor | None = None) -> torch.Tensor:
    """Append hydrogens to ``coords``, each built from its parent's CURRENT heavy
    neighbours (see ``_hydrogen_plan``). Only the bond length ``r`` [H] comes from
    the parameterization; no reference conformer is used.

    ``rotor_phi`` [..., n_rotors] sets the dihedral of each rotatable hydrogen,
    radians; anti (pi) when None. Differentiable in ``coords`` throughout.

    This replaces a NeRF rebuild from (r, theta, phi) measured on a reference
    conformer. phi is conformational: a ring CH2 hydrogen fixed against one ring
    neighbour ignores the other, so for cyclohexane, whose reference was a
    twist-boat, hydrogens on GAFF2 MD chairs came out 0.5 A from MD's and down to
    0.6 A from a ring carbon, and the steering energy put the chair 62 kcal/mol
    above the twist-boat. Placed from neighbours they are 0.10 A from MD's.
    """
    n_h = int(r.shape[-1])
    dev, dt = coords.device, coords.dtype
    out = torch.cat((coords, coords.new_zeros((*coords.shape[:-2], n_h, 3))), dim=-2)
    if n_h == 0:
        return out
    r = r.to(dt)
    idx: list[int] = []
    pos: list[torch.Tensor] = []

    def at(i):
        return coords.index_select(-2, torch.as_tensor(i, device=dev))

    if plan.get("tet3"):
        k, p, a, b, c = map(list, zip(*plan["tet3"]))
        P0 = at(p)
        s = _unit(at(a) - P0) + _unit(at(b) - P0) + _unit(at(c) - P0)
        idx += k
        pos.append(P0 - r[k].unsqueeze(-1) * _unit(s))
    if plan.get("linear"):
        k, p, b = map(list, zip(*plan["linear"]))
        P0 = at(p)
        idx += k
        pos.append(P0 - r[k].unsqueeze(-1) * _unit(at(b) - P0))
    if plan.get("tet2"):
        k0, k1, p, a, b = map(list, zip(*plan["tet2"]))
        P0 = at(p)
        u, v = _unit(at(a) - P0), _unit(at(b) - P0)
        bis, nrm = _unit(u + v), _unit(torch.cross(u, v, dim=-1))
        cos_h, sin_h = math.cos(_H_TET / 2), math.sin(_H_TET / 2)
        idx += k0 + k1
        pos.append(P0 + r[k0].unsqueeze(-1) * (-bis * cos_h + nrm * sin_h))
        pos.append(P0 + r[k1].unsqueeze(-1) * (-bis * cos_h - nrm * sin_h))
    if plan.get("tri2"):
        k, p, a, b = map(list, zip(*plan["tri2"]))
        P0 = at(p)
        bis = _unit(_unit(at(a) - P0) + _unit(at(b) - P0))
        idx += k
        pos.append(P0 - r[k].unsqueeze(-1) * bis)
    if plan.get("term"):
        k, p, b, d, th, ph = map(list, zip(*plan["term"]))
        idx += k
        pos.append(_dihedral_hydrogens(coords, p, b, d, r[k],
                                       torch.tensor(th, device=dev, dtype=dt),
                                       torch.tensor(ph, device=dev, dtype=dt)))
    if plan.get("rotor"):
        k, p, b, d, th = (list(x) for x in list(zip(*plan["rotor"]))[:5])
        if rotor_phi is None:
            rotor_phi = torch.full((*coords.shape[:-2], len(k)), math.pi, device=dev, dtype=dt)
        idx += k
        pos.append(_dihedral_hydrogens(coords, p, b, d, r[k],
                                       torch.tensor(th, device=dev, dtype=dt), rotor_phi.to(dt)))
    pos_t = torch.cat(pos, dim=-2).to(out.dtype)
    n = coords.shape[-2]
    return out.index_copy(-2, torch.as_tensor(idx, device=dev) + n, pos_t)


class _HydrogenGroup:
    """Potentials that see the same hydrogens, so a rotatable hydrogen is placed
    at the rotamer of lowest summed energy over all of them rather than each
    term choosing its own. The factories share one group across their terms;
    a potential built on its own forms a group by itself.
    """

    def __init__(self, members=()):
        self.members = list(members)
        self._choice: tuple | None = None     # (fingerprint, rotor dihedrals)

    @staticmethod
    def _fingerprint(c: torch.Tensor) -> tuple:
        c = c.detach().double()
        w = torch.arange(1, c.numel() + 1, device=c.device, dtype=c.dtype).reshape(c.shape)
        return (tuple(c.shape), str(c.device), round(float(c.sum()), 6),
                round(float((c * w).sum()), 3))

    def rotamers(self, coords, plan, r):
        """[..., n_rotors] dihedrals of lowest summed energy, one rotor at a time
        with the others held (anti to start), each from its own candidates
        (_H_ROTAMERS_BY_COORD). Chosen without gradients."""
        global _HYDROGEN_OVERRIDE
        fp = self._fingerprint(coords)
        if self._choice is not None and self._choice[0] == fp:
            return self._choice[1]
        c = coords.detach()
        n_rot = len(plan["rotor"])
        phis = torch.full((*c.shape[:-2], n_rot), math.pi, device=c.device, dtype=c.dtype)
        with torch.no_grad():
            for j in range(n_rot):
                cand = torch.deg2rad(torch.tensor(plan["rotor"][j][5], device=c.device, dtype=c.dtype))
                energies = []
                for phi in cand:
                    trial = phis.clone()
                    trial[..., j] = phi
                    _HYDROGEN_OVERRIDE = _place_hydrogens(c, plan, r, trial)
                    try:
                        energies.append(sum(m(c, t=0.0) for m in self.members))
                    finally:
                        _HYDROGEN_OVERRIDE = None
                phis[..., j] = cand[torch.stack(energies, -1).argmin(-1)]
        self._choice = (fp, phis)
        return phis


#: Coordinates with hydrogens, set while a _HydrogenGroup scores rotamers.
_HYDROGEN_OVERRIDE: torch.Tensor | None = None


def _share_hydrogen_group(potentials) -> None:
    """One _HydrogenGroup for every hydrogen-using term of a force field."""
    members = [p for p in potentials
               if isinstance(p, HydrogenMixin) and getattr(p, "use_hydrogens", False)]
    group = _HydrogenGroup(members)
    for p in members:
        p._hydrogen_group = group


class GAFFBondPotential(GAFFPotential, DistancePotential):
    """GAFF2 bond lengths, Huber in |r - r0|.

    Quadratic within ``cutoff`` (0.2 A; see get_gaff_potentials' bond_cutoff) and linear beyond, so the force is
    bounded by ``2 k cutoff``. With uniform_k=True (this class's default) k is 1;
    with uniform_k=False -- what the gaff2 and openff-2.2.1 presets build -- k is
    the force field's own constant in kcal/mol/A^2 (AMBER's fc, which already
    folds in the 1/2; e.g. 232.5 for a GAFF2 sp3 C-C), not normalized.
    """

    def __init__(self, parameters=None, cutoff: float = 0.2,
                 uniform_k: bool = True) -> None:
        super().__init__(parameters=parameters)
        # After super(): these are nn.Modules, which refuse attribute
        # assignment before Module.__init__ has run.
        self.uniform_k = uniform_k
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive.")
        self.cutoff = cutoff

    def compute_function(self, value, k, r0):
        delta = (value - r0).abs()
        d = self.cutoff
        return k * torch.where(delta <= d, delta * delta, d * (2.0 * delta - d))

    def _compute_args(self, feats, parameters=None):
        index = feats["gaff_bond_index"][0]
        r0 = feats["gaff_bond_r0"][0]
        if self.uniform_k:
            k = torch.ones_like(r0)
        else:
            # GAFF2's own constant, in kcal/mol/A^2. No conversion: AMBER
            # folds the 1/2 into fc and our Huber is k d^2, so fc is k.
            fc = feats.get("gaff_bond_fc")
            k = torch.ones_like(r0) if fc is None else fc[0]
        return index, (k, r0), None


class GAFFAnglePotential(GAFFPotential):
    """GAFF2 bond angles, ``k (theta - theta0)^2``.

    With uniform_k=True (this class's default) k is 1 for every angle, which
    discards the force field's own constants and with them the BALANCE of
    stiffnesses around a ring -- and a five-ring's pucker is set by that
    balance. With uniform_k=False -- what the gaff2 and openff-2.2.1 presets
    build -- k is the force field's own constant in kcal/mol/rad^2, NOT
    normalized (FC_REF is no longer used).

    theta is computed with atan2 rather than acos: acos' is unbounded at a
    straight angle, where a nitrile or an alkyne sits exactly. The energy is
    bounded because theta itself is; ``cutoff`` (0.35 rad in the presets)
    additionally makes the term Huber, bounding the force at 2 k cutoff.
    """

    _ANGLE_EPS = 1e-9

    FC_REF = 50.0      # kcal/mol/rad^2; unused since the constants are no longer normalized

    def __init__(self, parameters=None, uniform_k: bool = True,
                 cutoff: float | None = None) -> None:
        super().__init__(parameters=parameters)
        self.uniform_k = uniform_k
        if cutoff is not None and (not math.isfinite(cutoff)
                                   or cutoff <= 0):
            raise ValueError("cutoff must be finite and positive.")
        # None keeps the plain quadratic; a value bounds the force at
        # 2 k cutoff, as GAFFBondPotential does for bonds.
        self.cutoff = cutoff

    def compute_variable(self, coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        u = coords.index_select(-2, index[0]) - coords.index_select(-2, index[2])
        v = coords.index_select(-2, index[1]) - coords.index_select(-2, index[2])
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(self._ANGLE_EPS)
        v = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(self._ANGLE_EPS)
        cosine = (u * v).sum(dim=-1)
        sine = torch.linalg.norm(torch.cross(u, v, dim=-1), dim=-1)
        return torch.atan2(sine, cosine)

    def compute_function(self, value, k, theta0):
        delta = value - theta0
        if self.cutoff is None:
            return k * delta * delta
        # Huber: quadratic inside the cutoff, linear outside, so the force
        # saturates at 2 k cutoff instead of growing with the deviation.
        # This is the only term of ours whose force would otherwise be
        # unbounded -- MMFF's angle is a cosine well and both bonds are
        # already Huber.
        d = self.cutoff
        ad = delta.abs()
        return k * torch.where(ad <= d, ad * ad, d * (2.0 * ad - d))

    def _compute_args(self, feats, parameters=None):
        index = feats["gaff_angle_index"][0]
        theta0 = feats["gaff_angle_theta0"][0]
        if self.uniform_k:
            k = torch.ones_like(theta0)
        else:
            # The force field's own constants, not normalized. These carry
            # the BALANCE of stiffness around a ring, which uniform k erases --
            # and a five-ring's pucker is set by that balance.
            fc = feats.get("gaff_angle_fc")
            if fc is None:
                k = torch.ones_like(theta0)
            else:
                # GAFF2's own constant, kcal/mol/rad^2. Our angle
                # is k d^2, the same form AMBER uses, so no
                # conversion and no FC_REF normalization.
                k = fc[0]
        return index, (k, theta0), None


class GAFFTorsionPotential(HydrogenMixin, GAFFPotential, DihedralPotential):
    """GAFF2 torsions: ``fc0 + sum_n fc_n cos(n phi - phi0)`` for n up to 6.

    Seven Fourier coefficients per torsion against MMFF's three, and the phase
    enters as ``cos(n phi - phi0)`` rather than ``cos(n (phi - phi0))``.

    Unlike bond and angle this keeps GAFF2's own force constants: their spread
    across torsion types is exactly the chemistry the term is for.
    """

    def compute_function(self, value, phi0, fc):
        # value [B, P]; fc [P, 7]. n = 0 is a constant offset and is dropped --
        # it shifts the energy without changing any gradient.
        energy = value.new_zeros(value.shape)
        for n in range(1, 7):
            energy = energy + fc[:, n] * torch.cos(n * value - phi0)
        return energy

    _H_REF_KEY = "gaff_hydrogen_ref"
    _H_GEOM_KEY = "gaff_hydrogen_geom"

    def __init__(self, parameters=None, use_hydrogens: bool = False) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens

    def compute(self, coords, feats, parameters=None):
        return super().compute(self._with_hydrogens(coords, feats), feats, parameters)

    def _compute_args(self, feats, parameters=None):
        index, values = self._select_terms(
            feats, feats["gaff_torsion_index"][0],
            feats["gaff_torsion_phi0"][0], feats["gaff_torsion_fc"][0])
        return index, values, None


class GAFFCMAPPotential(GAFFPotential):
    """The ff19SB backbone CMAP, as a steering term.

    Each term names eight atoms: the phi quadruple followed by the psi
    quadruple. The surface is a square grid over both angles, stored with the
    cached parameters so nothing external is needed to evaluate it.

    Bilinear interpolation, wrapped periodically in both angles. phi and psi are
    computed with the same atan2 dihedral the torsion term uses, so a planar
    dihedral costs no precision.
    """

    _EPS = 1e-8

    def _dihedral(self, coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])
        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)
        bond = torch.linalg.norm(r_kj, dim=-1)
        sin_phi = (r_kj * torch.cross(n_ijk, n_jkl, dim=-1)).sum(-1) / bond.clamp_min(
            self._EPS
        )
        cos_phi = (n_ijk * n_jkl).sum(-1)
        return torch.atan2(sin_phi, cos_phi)

    def compute(self, coords, feats, parameters=None):
        index = feats["gaff_cmap_index"][0]          # [8, P]
        if index.shape[-1] == 0:
            return _connected_zero(coords)
        grids = feats["gaff_cmap_energy"][0]         # [P, size, size]
        origin = feats["gaff_cmap_origin"][0]        # [P, 2], radians

        phi = self._dihedral(coords, index[0:4])     # [B, P]
        psi = self._dihedral(coords, index[4:8])

        size = grids.shape[-1]
        spacing = 2.0 * math.pi / size

        # Grid coordinates, wrapped: the surface is periodic in both angles, so
        # a sample just past the last column interpolates back onto the first.
        u = (phi - origin[:, 0]) / spacing
        v = (psi - origin[:, 1]) / spacing
        u0 = torch.floor(u)
        v0 = torch.floor(v)
        fu = u - u0
        fv = v - v0
        i0 = (u0.long() % size)
        j0 = (v0.long() % size)
        i1 = (i0 + 1) % size
        j1 = (j0 + 1) % size

        term = torch.arange(index.shape[-1], device=coords.device)
        g00 = grids[term, i0, j0]
        g10 = grids[term, i1, j0]
        g01 = grids[term, i0, j1]
        g11 = grids[term, i1, j1]

        energy = (
            g00 * (1 - fu) * (1 - fv)
            + g10 * fu * (1 - fv)
            + g01 * (1 - fu) * fv
            + g11 * fu * fv
        )
        return energy.sum(dim=-1)


class GAFF14Potential(HydrogenMixin, GAFFPotential):
    """The 1-4 nonbonded pairs, with a bounded repulsion.

    A/r^12 - B/r^6 + 332.0637 q_i q_j / r over the pairs the force field marks
    as 1-4, with its own scaling already applied. These are part of what GAFF's
    torsion parameters were fitted against, so the torsion term is incomplete
    without them.

    The repulsion is capped: a bare r^-12 would make guidance unusable on an
    intermediate where two atoms are nearly coincident.

    Electrostatics are real Coulomb, C q_i q_j / (D r) with D = ``dielectric``,
    outside s = max(frac * size, 0.5 A) and a parabola matching value and slope
    inside (_coulomb_core); the van der Waals is the force field's own form
    outside its own switch and a parabola inside. size is the pair's LJ sigma
    for GAFF2 and R* for MMFF.
    """

    _H_REF_KEY = "gaff_hydrogen_ref"
    _H_GEOM_KEY = "gaff_hydrogen_geom"

    # Bounded cosine electrostatics, shared with the full nonbonded
    # term. The force field's own constant (332) is 28x this one and is
    # deliberately not used: see the class docstring.
    _NORM_EPS = 1e-12       # keeps d||r||/dr finite at coincidence

    def __init__(self, parameters=None, use_hydrogens: bool = False,
                 dielectric: float = 1.0, lj_core_switch: float = 0.5,
                 coulomb_core_switch: float = 0.5) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens
        if not math.isfinite(lj_core_switch) or lj_core_switch <= 0:
            raise ValueError("lj_core_switch must be finite and positive.")
        # vdW core switch s = max(lj_core_switch * size, 0.5 A); see _core_switch.
        self.lj_core_switch = float(lj_core_switch)
        if not math.isfinite(coulomb_core_switch) or coulomb_core_switch <= 0:
            raise ValueError("coulomb_core_switch must be finite and positive.")
        # Coulomb core switch s = max(coulomb_core_switch * size, 0.5 A).
        self.coulomb_core_switch = float(coulomb_core_switch)
        if not math.isfinite(dielectric) or dielectric <= 0:
            raise ValueError("dielectric must be finite and positive.")
        self.dielectric = float(dielectric)
        # False gives MMFF's own vdW and electrostatics: no cosine wall and
        # no cosine Coulomb. Both remain finite at coincidence thanks to
        # MMFF's 0.07 R* and 0.05 buffers, but the forces are large.
        # independently because capping the vdW and replacing the
        # Coulomb are separate decisions with opposite verdicts.
        # cosine 1-4 electrostatics: the same bounded form the full nonbonded
        # force fields.

    def compute(self, coords, feats, parameters=None):
        coords = self._with_hydrogens(coords, feats)
        index = feats["gaff_pair_index"][0]
        aij = feats["gaff_pair_aij"][0]
        bij = feats["gaff_pair_bij"][0]
        qij = feats["gaff_pair_qij"][0]
        index, (aij, bij, qij) = self._select_terms(feats, index, aij, bij, qij)
        if index.shape[-1] == 0:
            return _connected_zero(coords)

        delta = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        # Softened norm: ||delta|| has gradient delta/||delta||, which is 0/0 at
        # coincidence and gives NaN. Hydrogens are placed deterministically, so
        # two of them sharing internal coordinates would coincide exactly rather
        # than merely improbably.
        r = torch.sqrt((delta * delta).sum(-1) + self._NORM_EPS)

        # sigma from the pair's own A and B: A/B = sigma^6 where both are set.
        ratio = aij / bij.clamp_min(1e-12)
        sigma = torch.where(bij > 0, ratio.clamp_min(1e-12) ** (1.0 / 6.0),
                            torch.full_like(ratio, 3.0))
        # Pairs without Lennard-Jones (B = 0) get the floor as their switch.
        size = torch.where(bij > 0, sigma, torch.zeros_like(sigma))
        # The force field's own 12-6 outside the core switch, a parabola inside.
        repulsive = _lj126_core(r, aij, bij, _core_switch(size, self.lj_core_switch))
        coulomb = _coulomb_core(
                r, _GAFF_COULOMB * qij / self.dielectric, _core_switch(size, self.coulomb_core_switch))
        return (repulsive + coulomb).sum(dim=-1)


def get_gaff_potentials(
    *, angle: bool = True, torsion: bool = True, bond: bool = True,
    # CMAP OFF by default: it inverts the alanine dipeptide backbone
    # (2.0% phi<0 with it, 92.0% without, truth ~95). Cause not yet
    # localized -- the atom indices are correct, so the suspects are
    # the sign, the grid reshape, or the weight.
    cmap: bool = False, pairs14: bool = True, nonbonded: bool = False,
    nonbonded_weight: float = 0.001,
    angle_uniform_k: bool = True, bond_uniform_k: bool = True,
    angle_cutoff: float | None = None,
    angle_weight: float = 0.10, torsion_weight: float = 0.002,
    bond_weight: float = 0.10, cmap_weight: float = 0.05,
    pair14_weight: float = 0.0005,
    hydrogen: bool = False,
    pose_weight: float = 0.10,
    pose_angle_buffer: float = 0.20,
    chiral_weight=None,
    stereo_weight=None,
    planar_weight=None,
    physicality: bool = True,
    chiral_buffer=None,
    pose_bond_buffer: float = 0.20,
    pose_clash_buffer: float = 0.15,
    # Realistic nonbonded (off by default): real Coulomb with a parabola core,
    # the force field's own LJ with a parabola core, and the dielectric D.
    dielectric: float = 1.0,
    # Dielectric for the 1-4 term and for the nonbonded term separately (-F pairs14.dielectric,
    # -F nonbonded.dielectric); None uses dielectric (-F dielectric), which sets both. E.g. 1-4 at the bond weight with
    # pairs14_dielectric 0.2: the 1-4 LJ no longer stretches rings, the 1-4 Coulomb keeps 5x that weight.
    pairs14_dielectric: float | None = None,
    nonbonded_dielectric: float | None = None,
    # vdW core switch s = max(lj_core_switch * sigma, 0.5 A) for the 12-6 LJ
    # (the Coulomb core stays at sigma/2); -F lj_core.switch. 0.7 since
    # 2026-11-21 (was 0.5): see _core_switch.
    lj_core_switch: float = 0.7,
    # Coulomb core switch s = max(coulomb_core_switch * sigma, 0.5 A); -F
    # coulomb_core.switch. 0.5 keeps short hydrogen-bond contacts on real Coulomb.
    coulomb_core_switch: float = 0.5,
    # Bond Huber cutoff in A: quadratic within, linear beyond (force bounded at
    # 2 k cutoff); -F bond.cutoff. 0.2: guidance is plain descent, unstable for
    # stiff (aromatic, polar) bonds at w * 8k > 2, and the cutoff bounds that
    # oscillation; 0.5 let bonds swing to the break limit (2026-11-21: Langevin
    # without resampling, 15 ligands, broken samples 77-98 at 0.2 vs 2554-4586 at 0.5).
    bond_cutoff: float = 0.2,
    # Every force-field term acts only for t <= t_on (t = 1 noise, 0 clean), in
    # guidance and resampling. 0.4 by default: early structures overlap (pairs
    # inside max(sigma/2, 0.5 A) up to t ~ 0.6, none from t = 0.5) and their
    # bond and angle energies collapsed FK resampling. 1 is always on, the
    # behaviour before 2026-11-20. nonbonded_t_on optionally delays the 1-4 and
    # nonbonded terms further.
    t_on: float | None = 0.4,
    nonbonded_t_on: float | None = None,
    # Guidance weight w * (1 - t) instead of the t_on switch (resampling keeps it).
    guidance_ramp: bool = False,
    # Two-stage bond/angle guidance: the weight above holds for
    # bond_angle_late_t_on < t <= t_on, then these for t <= late_t_on.
    bond_angle_late_weight: float | None = None,
    bond_late_weight: float | None = None,
    angle_late_weight: float | None = None,
    bond_angle_late_t_on: float | None = None,
    # The ligand force field the terms are parameterized with: "gaff2", or an
    # OpenFF release such as "openff-2.2.1" (same functional forms).
    ligand_ff: str = "gaff2",
) -> Sequence[Potential]:
    """The default potentials plus the requested GAFF2 (or OpenFF) ligand terms.

    Weights start from the MMFF values, which were calibrated on bare rings.
    They are a starting point here, not a result: GAFF2's torsion constants are
    on a different scale from MMFF's V1/V2/V3, so the torsion weight in
    particular needs recalibrating against the same ring set.

    Requires boonza and AmberTools. Use get_mmff_potentials() for the
    RDKit-only path.
    """
    extra: list[Potential] = []
    if bond:
        extra.append(
            GAFFBondPotential(
                cutoff=bond_cutoff,
                uniform_k=bond_uniform_k,
parameters={
                    "guidance_interval": 1,
                    "guidance_weight": bond_weight,
                    "resampling_weight": 0.1,
                }
            )
        )
    if angle:
        extra.append(
            GAFFAnglePotential(cutoff=angle_cutoff, 
                uniform_k=angle_uniform_k,
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": angle_weight,
                    "resampling_weight": 0.1,
                }
            )
        )
    if pairs14:
        extra.append(
            GAFF14Potential(
                use_hydrogens=hydrogen,
                # Same cosine electrostatics as the full
                # nonbonded term, so s and the cutoff mean one
                # thing across every pair.
                dielectric=(dielectric if pairs14_dielectric is None else pairs14_dielectric),
                lj_core_switch=lj_core_switch,
                coulomb_core_switch=coulomb_core_switch,
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": pair14_weight,
                    "resampling_weight": 0.1,
                },
            )
        )
    if nonbonded:
        extra.append(
            NonbondedPotential(
                use_hydrogens=hydrogen,
                dielectric=(dielectric if nonbonded_dielectric is None else nonbonded_dielectric),
                lj_core_switch=lj_core_switch,
                coulomb_core_switch=coulomb_core_switch,
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": nonbonded_weight,
                    "resampling_weight": 0.1,
                },
            )
        )
    if cmap:
        extra.append(
            GAFFCMAPPotential(
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": cmap_weight,
                    "resampling_weight": 0.1,
                }
            )
        )
    if torsion:
        extra.append(
            GAFFTorsionPotential(
                use_hydrogens=hydrogen,
                parameters={
                    "guidance_interval": 1,
                    "guidance_weight": torsion_weight,
                    "resampling_weight": 0.1,
                }
            )
        )
    defaults = get_default_potentials(
        pose_weight=pose_weight,
        pose_angle_buffer=pose_angle_buffer,
        bond_buffer=pose_bond_buffer,
        clash_buffer=pose_clash_buffer,
        chiral_weight=chiral_weight,
        chiral_buffer=chiral_buffer,
        stereo_weight=stereo_weight,
        planar_weight=planar_weight,
    ) if physicality else ()
    for p in extra:
        if isinstance(p, GAFFPotential):
            p.ligand_ff = ligand_ff
    _apply_t_on(extra, t_on, nonbonded_t_on, guidance_ramp)
    _late = _late_stage(t_on, guidance_ramp, bond_angle_late_weight,
                        bond_late_weight, angle_late_weight,
                        bond_angle_late_t_on)
    if _late is not None:
        _apply_late_bond_angle(extra, t_on, *_late)
    _share_hydrogen_group(extra)
    return (*defaults, *extra)


def get_default_potentials(*, pose_weight: float = 0.10,
                           pose_angle_buffer: float = 0.20,
                           bond_buffer: float = 0.20,
                           clash_buffer: float = 0.15,
                           chiral_weight=None,
                           chiral_buffer=None,
                           stereo_weight=None,
                           stereo_buffer=None,
                           planar_weight=None,
                           planar_buffer=None,
                           bond_length_weight: float = 0.05,
                           bond_length_sigma: float = 10.0,
                           bond_angle_weight: float = 0.05,
                           bond_angle_sigma: float = 10.0,
                           clash_min_distance: float = 2.0,
                           ) -> Sequence[Potential]:
    """BioEmu2's physicality set.

    pose_weight and pose_angle_buffer are exposed because this set, with no
    force field at all, collapses hydroxyproline from 81.7% exo to 1.2% and
    fluoroproline from 75.3% to 0.6%. They are the two parameters that can
    plausibly do that: the angle buffer decides WHICH angles are pushed on,
    the weight decides how hard. Defaults reproduce the stock behaviour
    exactly, so nothing changes unless asked.
    """
    return (
        SymmetricChainCOMPotential(
            parameters={
                "guidance_interval": 4,
                "guidance_weight": 0.5,
                "resampling_weight": 0.5,
                "buffer": ExponentialInterpolation(start=1.0, end=5.0, alpha=-2.0),
            }
        ),
        InterChainVDWOverlapPotential(
            parameters={
                "guidance_interval": 5,
                "guidance_weight": PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0]),
                "resampling_weight": PiecewiseStepFunction(thresholds=[0.6], values=[0.01, 0.0]),
                "buffer": 0.225,
            }
        ),
        PoseBustersPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": pose_weight,
                "resampling_weight": 0.1,
                "bond_buffer": bond_buffer,
                "angle_buffer": pose_angle_buffer,
                "clash_buffer": clash_buffer,
            }
        ),
        ChiralAtomPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": chiral_weight if chiral_weight is not None else (0.20),
                "resampling_weight": 1.0,
                "buffer": (chiral_buffer
                           if chiral_buffer is not None
                           else 0.52360),
            }
        ),
        StereoBondPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": stereo_weight if stereo_weight is not None else (0.10),
                "resampling_weight": 1.0,
                "buffer": (stereo_buffer
                           if stereo_buffer is not None
                           else 0.52360),
            }
        ),
        PlanarBondPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": planar_weight if planar_weight is not None else (0.10),
                "resampling_weight": 1.0,
                "buffer": (planar_buffer
                           if planar_buffer is not None
                           else 0.26180),
            }
        ),
        IntraChainVDWOverlapPotential(
            parameters={
                "guidance_interval": 5,
                "guidance_weight": PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0]),
                "resampling_weight": PiecewiseStepFunction(thresholds=[0.6], values=[0.01, 0.0]),
                "buffer": 0.225,
            }
        ),
        PolymerLocalClashPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0]),
                "resampling_weight": 0.1,
                "minimum_distance": clash_min_distance,
            }
        ),
        BondLengthPotential(
            sigma_tolerance=bond_length_sigma,
            parameters={
                "guidance_interval": 1,
                "guidance_weight": bond_length_weight,
                "resampling_weight": 0.1,
            },
        ),
        BondAnglePotential(
            sigma_tolerance=bond_angle_sigma,
            parameters={
                "guidance_interval": 1,
                "guidance_weight": bond_angle_weight,
                "resampling_weight": 0.1,
            },
        ),
    )

class NonbondedPotential(HydrogenMixin, GAFFPotential):
    """Full intramolecular nonbonded: the force field's own LJ and Coulomb.

        E(r) = min(rep, lj) + coul
        rep  = A cos^2(pi r / 2 sigma)
        lj   = 4 eps ((sig/rr)^12 - (sig/rr)^6),  rr = max(r, 0.5 sig)
        coul = C q_i q_j cos^2(pi r / 2 r_c)

    Both terms are bounded at coincidence by their core switch, so a clashing
    early structure gives a finite energy and a finite force.

    Scope: 1-2, 1-3 and 1-4 pairs are all excluded, the 1-4s because they are
    a separate term with the force field's own scaling. For a small ligand
    that leaves very few pairs -- fluoroethylamine cation has 9 of 55 -- so
    this matters for macrocycles and peptides rather than for small rings.

    Electrostatics are real Coulomb, C q_i q_j / (D r) with D = ``dielectric``,
    outside s = max(frac * size, 0.5 A) and a parabola matching value and slope
    inside (_coulomb_core); the van der Waals is the force field's own form
    outside its own switch and a parabola inside. size is the pair's LJ sigma
    for GAFF2 and R* for MMFF.
    """

    _H_REF_KEY = "gaff_hydrogen_ref"
    _H_GEOM_KEY = "gaff_hydrogen_geom"

    KJ_TO_KCAL = 0.2390057
    _EPS = 1e-12

    def __init__(self, parameters=None, use_hydrogens: bool = False,
                 dielectric: float = 1.0, lj_core_switch: float = 0.5,
                 coulomb_core_switch: float = 0.5) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens
        if not math.isfinite(lj_core_switch) or lj_core_switch <= 0:
            raise ValueError("lj_core_switch must be finite and positive.")
        # vdW core switch s = max(lj_core_switch * size, 0.5 A); see _core_switch.
        self.lj_core_switch = float(lj_core_switch)
        if not math.isfinite(coulomb_core_switch) or coulomb_core_switch <= 0:
            raise ValueError("coulomb_core_switch must be finite and positive.")
        # Coulomb core switch s = max(coulomb_core_switch * size, 0.5 A).
        self.coulomb_core_switch = float(coulomb_core_switch)
        if not math.isfinite(dielectric) or dielectric <= 0:
            raise ValueError("dielectric must be finite and positive.")
        self.dielectric = float(dielectric)
        # MMFF's own forms when False: no cosine wall,
        # no cosine Coulomb.
        # independently because capping the vdW and replacing the
        # Coulomb are separate decisions with opposite verdicts.
        # The electrostatic part relative to the steric part. They do
        # different jobs: sterics fix clashes and packing, while the charge
        # term is the only thing here that can produce a charge-DEPENDENT
        # conformation. One knob for both means raising the electrostatics to
        # fix histamine also raises the sterics, which distorts other systems.
        # a vacuum range: in water these charges are screened by a
        # dielectric of ~78, and at 12 A a zwitterion's two ends
        # attract strongly enough to force a ring substituent axial
        # (tranexamic_z, 33.7% axial against a truth of 0.5%).

    def compute(self, coords, feats, parameters=None):
        coords = self._with_hydrogens(coords, feats)
        index = feats["gaff_nb_index"][0]          # [2, P] the pairs to compute
        sigma = feats["gaff_nb_sigma"][0]          # [P] combined
        eps = feats["gaff_nb_eps"][0]              # [P] combined
        qq = feats["gaff_nb_qq"][0]                # [P] q_i * q_j
        index, (sigma, eps, qq) = self._select_terms(feats, index, sigma,
                                                     eps, qq)
        if index.shape[-1] == 0:
            return _connected_zero(coords)

        delta = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        # Softened norm: d/dr of ||delta|| is 0/0 at coincidence, and
        # hydrogens are placed deterministically.
        r = torch.sqrt((delta * delta).sum(-1) + self._EPS)
        sig = sigma.clamp_min(1e-6)

        # Pairs without Lennard-Jones (eps = 0) get the floor as their switch.
        size = torch.where(eps > 0, sig, torch.zeros_like(sig))
        # The force field's own 12-6 outside the core switch, a parabola inside.
        s6 = sig ** 6
        vdw = _lj126_core(r, 4.0 * eps * s6 * s6, 4.0 * eps * s6,
                          _core_switch(size, self.lj_core_switch))
        coul = _coulomb_core(
                r, _GAFF_COULOMB * qq / self.dielectric, _core_switch(size, self.coulomb_core_switch))
        return (vdw + coul).sum(dim=-1)

