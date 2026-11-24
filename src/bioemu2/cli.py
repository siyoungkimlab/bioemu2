"""Command-line interface for basic usage."""

import logging
import shlex
import sys
from collections.abc import Callable, Sequence
from copy import copy
from functools import partial
from pathlib import Path
from typing import cast, get_args

import click
import mdtraj
import numpy as np
import torch.distributed as dist
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from bioemu2.checkpoints import DEFAULT_CHECKPOINT, SUPPORTED_CHECKPOINTS
from bioemu2.conditioning.method import SamplingMode
from bioemu2.context_parallel import api as cp_api
from bioemu2.data_interface import Contact, System, convert_boltz, load_system_and_contacts
from bioemu2.denoising.samplers import Sampler
from bioemu2.denoising.steering.potentials import (
    Potential,
    get_default_potentials,
    get_mmff_potentials,
)
from bioemu2.mae import save_mae
from bioemu2.model import load_model
from bioemu2.msa.alignment import get_msa
from bioemu2.physicality import PhysicalityFilterMode, PhysicalityReport
from bioemu2.propagation import propagate, propagate_with_embeddings
from bioemu2.sampling import (
    SamplerPreset,
    _drop_overriding_physicality,
    _ligand_potential_kw,
    check_physicality,
    expected_extra_bonds_from_system,
    predict_structure,
    sample_equilibrium,
    sampler_from_preset,
    save_topology_cif,
    write_sampling_system,
)
from bioemu2.structure import AllAtomStructure

logger = logging.getLogger(__name__)

SAMPLER_PRESETS = list(get_args(SamplerPreset))
PHYSICALITY_FRAME_CHUNK_SIZE = 32


def _sampler(
    preset: str, potentials: Path | None, params: tuple[str, ...] = (),
    sampler_params: tuple[str, ...] = (),
    ff_params: tuple[str, ...] = (), *,
    mode: SamplingMode = "equilibrium"
) -> Sampler:
    """Build the sampler for ``--sampler PRESET [--potentials YAML]``.

    ``edm`` is unsteered. The other presets use the default physicality potentials;
    ``potentials`` points to a YAML with a top-level ``potentials:`` list and
    replaces that default list.
    """
    return sampler_from_preset(preset, mode=mode, potentials=_load_potentials(preset, potentials, params,
                                                   ff_params),
        sampler_params=_parse_sampler_params(sampler_params))


#: Presets that carry potentials, so -P and --potentials apply to them.
_STEERING_PRESETS = ("edm_gd", "edm_fk", "edm_fkc", "mmff", "gaff2", "openff-2.2.1")

#: Presets that carry a force field, so -F applies to them.
_FORCEFIELD_PRESETS = ("mmff", "gaff2", "openff-2.2.1")
#: Presets built from get_gaff_potentials; the name is the ligand force field.
_GAFF_PRESETS = ("gaff2", "openff-2.2.1")

#: -P drop_overriding=NAME says which physicality terms the ligand presets
#: leave out. The three of them hold bond and angle geometry at 10-50x the
#: force field's own weights, and they carry resampling_weight as well, so
#: they steer FK selection and not only the gradient. It is not a potential
#: parameter, so it is taken out of the -P list before the rest are parsed.
#:
#: bond_angle (the default, and what _LIGAND_PRESET_DROP ships) keeps
#: PoseBusters as a guard rail -- a flat bottom on RDKit's distance bounds,
#: no force within +-20% of a bound -- while dropping the two that compete
#: with the force field outright.
_DROP_OVERRIDING_SETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "bond_angle": ("BondLengthPotential", "BondAnglePotential"),
    "all": ("PoseBustersPotential", "BondLengthPotential",
            "BondAnglePotential"),
}
_DROP_OVERRIDING_KEY = "drop_overriding"


def _split_drop_overriding(
    values: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """``("drop_overriding=all",)`` -> ``((), the three class names)``."""
    rest: list[str] = []
    drop: tuple[str, ...] | None = None
    for item in values:
        key, sep, raw = item.partition("=")
        if key.strip() != _DROP_OVERRIDING_KEY:
            rest.append(item)
            continue
        token = raw.strip().lower()
        if not sep or token not in _DROP_OVERRIDING_SETS:
            raise ValueError(
                f"--potential-param {_DROP_OVERRIDING_KEY} takes one of "
                f"{', '.join(_DROP_OVERRIDING_SETS)}; got {raw.strip()!r}"
            )
        drop = _DROP_OVERRIDING_SETS[token]
    return tuple(rest), drop



#: Sampler fields that may be overridden. Unlike the potential parameters
#: these ARE preset-dependent, so the binding check is against the built
#: sampler's dataclass fields inside sampler_from_preset; this table is for
#: the help text and for catching typos before the model loads.
_SAMPLER_FIELDS: dict[str, str] = {
    "num_steps": "all",
    "step_scale": "edm, edm_gd, edm_fk",
    "noise_scale": "edm, edm_gd, edm_fk",
    "resampling_interval": "edm_gd, edm_fk, edm_fkc (none disables)",
    "resampling_group_size": "edm_gd, edm_fk, edm_fkc (particles per independent "
                             "group; none = one pool over the whole batch. The "
                             "ligand force-field presets default to k, one group "
                             "per output sample)",
    "fk_lambda": "edm_gd, edm_fk",
    "num_gd_steps": "edm_gd, edm_fk",
    "max_step": "edm_gd, edm_fk (A per atom per guidance step; none = uncapped)",
    "langevin_kT": "edm_gd, edm_fk (kcal/mol; Langevin guidance, none = plain descent)",
    "langevin_weight": "edm_gd, edm_fk (guidance weight w of the noise sqrt(2 w kT))",
    "langevin_t_on": "edm_gd, edm_fk (Langevin noise only for t <= this)",
    "particles_per_sample": "edm_gd, edm_fk (particles carried per requested sample; "
                            "--num-samples still sets how many are written. 3 for the "
                            "ligand force-field presets, 1 otherwise)",
    "ess_threshold": "edm_fkc",
    "beta_power": "edm_fkc",
    "gamma_0": "edm_fkc",
    "gamma_min": "edm_fkc",
}


def _parse_sampler_params(values: tuple[str, ...]) -> dict:
    """``("resampling_interval=5",)`` -> ``{"resampling_interval": 5}``.

    ``none`` becomes None, which is meaningful for resampling_interval: it
    disables FK weighting and resampling, so ``--sampler edm_fk -S
    resampling_interval=none`` is edm_gd.
    """
    out: dict = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--sampler-param expects field=value, got {item!r}")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in _SAMPLER_FIELDS:
            raise ValueError(f"unknown sampler field {key!r}. Valid: "
                             + ", ".join(sorted(_SAMPLER_FIELDS)))
        if raw.lower() in ("none", "null"):
            out[key] = None
            continue
        try:
            out[key] = int(raw) if raw.lstrip("-").isdigit() else float(raw)
        except ValueError:
            raise ValueError(f"--sampler-param {key} needs a number or "
                             f"'none', got {raw!r}") from None
    return out


#: ``term.param`` -> the get_mmff_potentials keyword it sets.
#: ``all.weight`` is special-cased: it sets every term's weight at once,
#: which is what preserves the force field's internal balance.
_FORCEFIELD_PARAMS: dict[str, str] = {
    "bond.weight": "bond_weight",
    "angle.weight": "angle_weight",
    "torsion.weight": "torsion_weight",
    "pairs14.weight": "pair14_weight",
    "stretchbend.weight": "stretchbend_weight",
    "nonbonded.weight": "nonbonded_weight",
    # Global, not per-term: one kwarg feeds every HydrogenMixin
    # consumer. The models predict heavy atoms only, and the
    # heavy-only torsion path has the wrong shape, not merely a
    # smaller amplitude, so this is an ablation rather than a
    # speed knob.
    "hydrogen": "hydrogen",
    # On/off per term. get_mmff_potentials gates construction on
    # these, so a disabled term is never built or prepared.
    "bond.enabled": "bond",
    "angle.enabled": "angle",
    "torsion.enabled": "torsion",
    "pairs14.enabled": "pairs14",
    "stretchbend.enabled": "stretchbend",
    "nonbonded.enabled": "nonbonded",
    "bond.uniform_k": "bond_uniform_k",
    "angle.uniform_k": "angle_uniform_k",
    # Realistic nonbonded, opt-in: real Coulomb and the force field's own
    # LJ, each with a parabola core inside max(sigma/2, 0.5 A) (R* for
    # MMFF); D, the dielectric of that Coulomb; and t_on, so the 1-4 and
    # nonbonded terms act only late in denoising (t <= t_on, t = 0 clean).
    # vdW core switch, as a fraction of sigma (GAFF2/OpenFF, default 0.7) or R*
    # (MMFF, default 0.5); the Coulomb core stays at half.
    "lj_core.switch": "lj_core_switch",
    # Coulomb core switch, as a fraction of sigma / R* (default 0.5).
    "coulomb_core.switch": "coulomb_core_switch",
    # Bond Huber cutoff in A (default 0.2): quadratic within, linear beyond.
    "bond.cutoff": "bond_cutoff",
    # Dielectric D of the real Coulomb: dielectric sets the 1-4 and nonbonded terms together;
    # pairs14.dielectric / nonbonded.dielectric set one term alone (default: dielectric).
    # Before 2026-11-21 nonbonded.dielectric set both terms (now: dielectric).
    "dielectric": "dielectric",
    "pairs14.dielectric": "pairs14_dielectric",
    "nonbonded.dielectric": "nonbonded_dielectric",
    "nonbonded.t_on": "nonbonded_t_on",
    # Every force-field term acts only for t <= t_on (default 0.4; 1 = always).
    "t_on": "t_on",
    # Guidance weight ramps linearly from 0 at t = 1 to its value at t = 0.
    "guidance.ramp": "guidance_ramp",
    # Two-stage bond/angle guidance: bond.weight / angle.weight hold for
    # bond_angle.late_t_on < t <= t_on, then these for t <= late_t_on.
    "bond_angle.late_weight": "bond_angle_late_weight",
    "bond.late_weight": "bond_late_weight",
    "angle.late_weight": "angle_late_weight",
    "bond_angle.late_t_on": "bond_angle_late_t_on",
}

#: Keys that select a term's force constants rather than scale it.
_FORCEFIELD_FLAGS = frozenset({
    "bond.uniform_k", "angle.uniform_k", "hydrogen",
    "bond.enabled", "angle.enabled", "torsion.enabled",
    "pairs14.enabled", "stretchbend.enabled", "nonbonded.enabled",
    "guidance.ramp",
})

#: The six MMFF energy functions, for ``-F terms=``.
_ALL_TERMS = ("bond", "angle", "torsion", "pairs14", "stretchbend",
              "nonbonded")


#: Every weight, for ``all.weight``. The superset over both force fields:
#: stretchbend_weight is MMFF-only, so the expansion is filtered against the
#: target factory by _weights_for() rather than passed blind.
_ALL_WEIGHTS = ("bond_weight", "angle_weight", "torsion_weight",
                "pair14_weight", "stretchbend_weight", "nonbonded_weight")


def _weights_for(factory) -> tuple[str, ...]:
    """The _ALL_WEIGHTS a given factory actually accepts.

    Read from the signature, not a second list: GAFF has no stretch-bend, and
    hard-coding which weights each factory has is how all.weight came to pass
    stretchbend_weight to get_gaff_potentials.
    """
    import inspect

    params = inspect.signature(factory).parameters
    return tuple(w for w in _ALL_WEIGHTS if w in params)


def _parse_forcefield_params(values: tuple[str, ...],
                             factory=None) -> dict:
    """``("bond.weight=0.001",)`` -> ``{"bond_weight": 0.001}``.

    ``factory`` restricts an ``all.weight`` expansion to the
    weights that factory has; without it the full superset is
    used, which is MMFF's shape.
    """
    out: dict = {}
    values_seen: list[str] = []
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"--forcefield-param expects term.param=value, got {item!r}")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        values_seen.append(key)
        if key == "terms":
            # A whitelist: named terms on, everything else off, so one key
            # fully determines the set.
            want = {w.strip() for w in raw.split(",") if w.strip()}
            unknown = want - set(_ALL_TERMS)
            if unknown:
                raise ValueError(
                    f"unknown term(s) {sorted(unknown)}. Valid: "
                    + ", ".join(_ALL_TERMS))
            for term in _ALL_TERMS:
                out[term] = term in want
            continue
        if key == "all.weight":
            try:
                v = float(raw)
            except ValueError:
                raise ValueError(
                    f"--forcefield-param all.weight needs a number, "
                    f"got {raw!r}") from None
            if v < 0.0:
                raise ValueError("all.weight must not be negative")
            # The target factory's weights, not the superset: GAFF
            # has no stretch-bend term.
            for name in (_weights_for(factory) if factory is not None
                         else _ALL_WEIGHTS):
                out[name] = v
            continue
        if key not in _FORCEFIELD_PARAMS:
            raise ValueError(
                f"unknown force-field parameter {key!r}. Valid: terms, all.weight, "
                + ", ".join(sorted(_FORCEFIELD_PARAMS))
            )
        if key in _FORCEFIELD_FLAGS:
            low = raw.lower()
            if low in ("1", "true", "yes", "on"):
                out[_FORCEFIELD_PARAMS[key]] = True
            elif low in ("0", "false", "no", "off"):
                out[_FORCEFIELD_PARAMS[key]] = False
            else:
                raise ValueError(
                    f"--forcefield-param {key} is a flag: use 0/1 or "
                    f"true/false, got {raw!r}")
            continue
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(
                f"--forcefield-param {key} needs a number, got {raw!r}"
            ) from None
        if v < 0.0:
            raise ValueError(f"--forcefield-param {key} must not be negative")
        if key.endswith("dielectric") and v == 0.0:
            raise ValueError(f"--forcefield-param {key} must be positive")
        if key in ("nonbonded.t_on", "t_on", "bond_angle.late_t_on") and v > 1.0:
            raise ValueError(f"--forcefield-param {key} is a diffusion time in [0, 1]")
        out[_FORCEFIELD_PARAMS[key]] = v
    # The factory builds the nonbonded term only when nonbonded is
    # true, so setting nonbonded.weight on its own would have been a
    # silent no-op. Asking for any nonbonded parameter switches it on.
    # Asking for any nonbonded parameter implies the term, unless
    # nonbonded.enabled said otherwise -- setdefault leaves an
    # explicit value alone.
    if any(k.startswith("nonbonded.") for k in values_seen):
        out.setdefault("nonbonded", True)
    return out


#: ``Potential.param`` -> the get_default_potentials keyword it sets.
#: Keys are the potential CLASS NAMES as they appear in that function, so the
#: CLI vocabulary matches the code's. Only scalar parameters appear here: the
#: PiecewiseStepFunction weights and SymmetricChainCOM's
#: ExponentialInterpolation buffer are noise-level schedules, and taking a
#: float for them would quietly replace a schedule with a constant.
_POTENTIAL_PARAMS: dict[str, str] = {
    "PoseBusters.weight": "pose_weight",
    "PoseBusters.bond_buffer": "bond_buffer",
    "PoseBusters.angle_buffer": "pose_angle_buffer",
    "PoseBusters.clash_buffer": "clash_buffer",
    "ChiralAtom.weight": "chiral_weight",
    "ChiralAtom.buffer": "chiral_buffer",
    "StereoBond.weight": "stereo_weight",
    "StereoBond.buffer": "stereo_buffer",
    "PlanarBond.weight": "planar_weight",
    "PlanarBond.buffer": "planar_buffer",
    "BondLength.weight": "bond_length_weight",
    "BondLength.sigma": "bond_length_sigma",
    "BondAngle.weight": "bond_angle_weight",
    "BondAngle.sigma": "bond_angle_sigma",
}

#: Shipped values, for error messages and documentation.
_POTENTIAL_DEFAULTS: dict[str, str] = {
    "PoseBusters.weight": "0.10",
    "PoseBusters.bond_buffer": "0.20",
    "PoseBusters.angle_buffer": "0.20",
    "PoseBusters.clash_buffer": "0.15",
    "ChiralAtom.weight": "0.20",
    "ChiralAtom.buffer": "0.52360 rad (30 deg)",
    "StereoBond.weight": "0.10",
    "StereoBond.buffer": "0.52360 rad (30 deg)",
    "PlanarBond.weight": "0.10",
    "PlanarBond.buffer": "0.26180 rad (15 deg)",
    "BondLength.weight": "0.05",
    "BondLength.sigma": "10.0",
    "BondAngle.weight": "0.05",
    "BondAngle.sigma": "10.0",
}


def _parse_potential_params(values: tuple[str, ...]) -> dict:
    """``("ChiralAtom.buffer=0",)`` -> ``{"chiral_buffer": 0.0}``."""
    out: dict = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"--potential-param expects Name.param=value, got {item!r}"
            )
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in _POTENTIAL_PARAMS:
            matches = [k for k in _POTENTIAL_PARAMS
                       if k.split(".")[0].lower() == key.split(".")[0].lower()]
            hint = (f" Did you mean one of: {', '.join(sorted(matches))}?"
                    if matches else
                    f" Valid keys: {', '.join(sorted(_POTENTIAL_PARAMS))}.")
            raise ValueError(f"unknown potential parameter {key!r}.{hint}")
        try:
            number = float(raw)
        except ValueError:
            raise ValueError(
                f"--potential-param {key} needs a number, got {raw!r}"
            ) from None
        if number < 0.0:
            raise ValueError(f"--potential-param {key} must not be negative")
        out[_POTENTIAL_PARAMS[key]] = number
    return out


def _load_potentials(preset: str, path: Path | None,
                     params: tuple[str, ...] = (),
                     ff_params: tuple[str, ...] = (),
                     ) -> Sequence[Potential] | None:
    params, drop_overriding = _split_drop_overriding(params)
    overrides = _parse_potential_params(params)
    # Which factory the flags are headed for decides what all.weight
    # expands to: GAFF has no stretch-bend term.
    if preset in _GAFF_PRESETS:
        from bioemu2.denoising.steering.potentials import (
            get_gaff_potentials as _ff_factory,
        )
    else:
        from bioemu2.denoising.steering.potentials import (
            get_mmff_potentials as _ff_factory,
        )
    ff = _parse_forcefield_params(ff_params, factory=_ff_factory)
    if ff and preset not in _FORCEFIELD_PRESETS:
        raise ValueError(
            "--forcefield-param requires a force-field sampler "
            f"(mmff, gaff2); --sampler {preset} carries no force field"
        )
    if path is None:
        if not overrides and not ff and drop_overriding is None:
            return None          # unchanged: the sampler uses its own default
        if preset not in _STEERING_PRESETS:
            raise ValueError(
                "--potential-param requires a steering sampler (edm_gd, "
                f"edm_fk, edm_fkc); --sampler {preset} is unsteered, so it "
                "has no potentials to configure"
            )
        # The preset's own potentials with the named parameters replaced, so
        # an -F/-P flag changes ONLY what it names. Both the weights and the
        # drop come from sampling.py, the single source of truth: a second
        # copy of the weights here silently reset every unnamed term to the
        # pre-retuning GAFF2_WEIGHT (0.0005) and skipped the drop, so -F also
        # restored the three overriding physicality terms. Measured: the same
        # nonbonded weight scored 77.8% vs 12.2% bad bonds depending only on
        # whether an unrelated -F flag was present.
        # mmff carries ligand terms as well, and get_mmff_potentials
        # forwards the physicality parameters, so -P applies to both.
        if preset == "mmff":
            kw = {**_ligand_potential_kw("mmff"), **overrides, **ff}
            potentials = tuple(get_mmff_potentials(**kw))
        elif preset in _GAFF_PRESETS:
            # openff-2.2.1 is the same preset with OpenFF's parameters.
            from bioemu2.denoising.steering.potentials import (
                get_gaff_potentials,
            )
            kw = {**_ligand_potential_kw(preset), "ligand_ff": preset,
                  **overrides, **ff}
            potentials = tuple(get_gaff_potentials(**kw))
        else:
            # Not a ligand preset, so there is nothing to drop.
            return tuple(get_default_potentials(**overrides))
        if drop_overriding is None:
            # Unset: exactly what the preset does, via sampling.py's own set.
            return _drop_overriding_physicality(potentials)
        # Explicit: filter here rather than through
        # _drop_overriding_physicality, which closes over _LIGAND_PRESET_DROP
        # and would ignore the chosen set.
        return tuple(p for p in potentials
                     if type(p).__name__ not in drop_overriding)
    if overrides:
        raise ValueError(
            "--potentials replaces the physicality set entirely, so it cannot "
            "be combined with --potential-param; set the values inside the "
            "YAML instead"
        )
    if preset not in _STEERING_PRESETS:
        raise ValueError("--potentials requires a steering sampler (edm_gd, edm_fk, edm_fkc)")
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig) or "potentials" not in config:
        raise ValueError(f"{path} must have a top-level `potentials:` list")
    return tuple(instantiate(config.potentials))


def _common_options(function: Callable) -> Callable:
    options = [
        click.option(
            "--system",
            type=click.Path(path_type=Path, dir_okay=False),
            metavar="YAML",
            help="Path to a system YAML file describing the molecular entities and optional "
            "contacts. Packaged examples are available under bioemu2.paths.EXAMPLES_DIR. "
            "Sampling commands require --system or "
            "--pdb-id; propagate can infer the system from --x0-cif when both are omitted.",
        ),
        click.option(
            "--pdb-id",
            metavar="ID",
            help="PDB ID to download from RCSB as biological assembly 1, including all chains "
            "and ligands, for example 1UBQ or 101M. Sampling commands require --system or "
            "--pdb-id; propagate can infer the system from --x0-cif when both are omitted.",
        ),
        click.option(
            "--checkpoint",
            type=click.Choice(SUPPORTED_CHECKPOINTS),
            help="Checkpoint name in the microsoft/bioemu2 Hugging Face repository. "
            f"Defaults to {DEFAULT_CHECKPOINT} when --checkpoint-path is not given.",
        ),
        click.option(
            "--checkpoint-path",
            type=click.Path(path_type=Path, dir_okay=False),
            metavar="CKPT",
            help="Path to a local checkpoint file instead of a Hugging Face checkpoint. "
            "The matching config.yaml must be in the same directory.",
        ),
        click.option(
            "--checkpoint-cache-dir",
            type=click.Path(path_type=Path, file_okay=False),
            metavar="DIR",
            help="Directory for downloaded Hugging Face checkpoints. Defaults to the standard "
            "Hugging Face cache, which HF_HUB_CACHE or HF_HOME can relocate. Cannot be used "
            "with --checkpoint-path.",
        ),
        click.option(
            "--out",
            required=True,
            type=click.Path(path_type=Path, file_okay=False),
            metavar="DIR",
            help="Directory for generated trajectories, topology.mae, system.yaml, and "
            "command.txt. It is created if it does not exist; command output files from an "
            "earlier run may be overwritten.",
        ),
        click.option(
            "--batch-size",
            type=click.IntRange(min=1),
            help="Number of structures denoised together. Defaults to --num-samples for "
            "sampling commands and the number of starting conformations for propagation.",
        ),
        click.option(
            "--seed",
            type=int,
            help="Seed for random number generation. By default no seed is set, so repeated "
            "runs generally produce different random samples.",
        ),
        click.option(
            "--num-steps",
            type=click.IntRange(min=1),
            help="Number of denoising steps. By default each sampler uses its configured "
            "value, currently 200 or 500 depending on the sampler.",
        ),
        click.option(
            "--sampler",
            type=click.Choice(SAMPLER_PRESETS),
            default="edm",
            show_default=True,
            help="Sampling algorithm. EDM is unsteered; edm_gd uses gradient guidance, "
            "edm_fk adds Feynman-Kac resampling, and edm_fkc uses the Feynman-Kac corrector.",
        ),
        click.option(
            "--potentials",
            type=click.Path(path_type=Path, dir_okay=False),
            metavar="YAML",
            help="Path to a YAML file containing a top-level 'potentials' list. This replaces "
            "the default physicality potentials and requires edm_gd, edm_fk, or edm_fkc.",
        ),
        click.option(
            "-S",
            "--sampler-param",
            "sampler_param",
            multiple=True,
            metavar="FIELD=VALUE",
            help="Change one sampler field, keeping the preset's other "
            "values. Repeatable. These are preset-specific: "
            "resampling_interval and resampling_group_size on edm_gd, "
            "edm_fk and edm_fkc; fk_lambda, num_gd_steps, max_step (A per atom per guidance step) and "
            "langevin_kT / langevin_weight / langevin_t_on (Langevin guidance) on edm_gd "
            "and edm_fk; ess_threshold, beta_power, gamma_0 and "
            "gamma_min on edm_fkc. 'none' is accepted where None is "
            "meaningful, so -S resampling_interval=none turns edm_fk "
            "into edm_gd.",
        ),
        click.option(
            "-F",
            "--forcefield-param",
            "forcefield_param",
            multiple=True,
            metavar="TERM.PARAM=VALUE",
            help="Change one force-field parameter. Repeatable, and "
            "requires a force-field sampler. terms=bond,angle,torsion "
            "chooses the energy functions in one key: named terms on, "
            "unnamed off. "
            "requires a force-field sampler such as --sampler mmff. "
            "Keys: bond.weight, angle.weight, torsion.weight, "
            "pairs14.weight, stretchbend.weight, nonbonded.weight, "
            "bond.enabled, angle.enabled, torsion.enabled, "
            "pairs14.enabled, stretchbend.enabled, "
            "nonbonded.enabled to choose terms (MMFF's stretch-bend is off "
            "by default because it is unbounded below; "
            "stretchbend.enabled=1 turns it on), and "
            "bond.uniform_k / angle.uniform_k which take 0 or 1 and "
            "choose between the force field's own constants and "
            "k = 1. The 1-4 and nonbonded terms are real Coulomb and the "
            "force field's own Lennard-Jones, each with a smooth core inside "
            "max(F sigma, 0.5 A). dielectric=D sets both terms, pairs14.dielectric "
            "and nonbonded.dielectric one term alone (default: dielectric); "
            "lj_core.switch=F moves the vdW core to "
            "max(F sigma, 0.5 A) (F sigma = F R* for MMFF; default 0.7 for "
            "gaff2/openff, 0.5 for mmff); coulomb_core.switch=F does the same "
            "for the Coulomb core (default 0.5). bond.cutoff=C sets the bond "
            "terms' Huber cutoff in A (default 0.2). t_on=T applies every force-field term "
            "(guidance and resampling) only for t <= T (t = 1 noise, 0 "
            "clean; default 0.4, 1 for always); nonbonded.t_on=T delays "
            "the 1-4 and nonbonded terms further. guidance.ramp=1 makes "
            "every force-field guidance weight rise linearly from 0 at t = 1 "
            "to its value at t = 0 (resampling keeps t_on). "
            "all.weight sets every term at once, which is "
            "what keeps the force field's internal balance: setting "
            "terms individually changes it.",
        ),
        click.option(
            "-P",
            "--potential-param",
            "potential_param",
            multiple=True,
            metavar="NAME.PARAM=VALUE",
            help="Change one physicality parameter, keeping every other shipped "
            "value. Repeatable. Keys are the potential class names: "
            "PoseBusters.weight/.bond_buffer/.angle_buffer/.clash_buffer, "
            "ChiralAtom.weight/.buffer, StereoBond.weight/.buffer, "
            "PlanarBond.weight/.buffer, BondLength.weight/.sigma, "
            "BondAngle.weight/.sigma. Buffers are radians for the dihedral "
            "terms and fractions for PoseBusters; sigma is sigma_tolerance. "
            "Requires a steering sampler, and cannot be combined with "
            "--potentials. Example: -P ChiralAtom.buffer=0 enforces the sign "
            "of each centre without demanding a minimum improper.",
        ),
        click.option(
            "--mae",
            is_flag=True,
            help="Also write samples.mae: one structure per sample with every bond's order "
            "and every atom's formal charge, for preparation and viewing in Maestro, PyMOL "
            "or boonza. samples.xtc and topology.mae are written as usual.",
        ),
        click.option(
            "--cif",
            is_flag=True,
            help="Also write topology.cif, the topology as CIF, which MDTraj reads as the "
            "topology of samples.xtc and propagate takes as --x0-cif. The topology is "
            "always written as topology.mae.",
        ),
        click.option(
            "--physicality-filter",
            type=click.Choice([mode.value for mode in PhysicalityFilterMode]),
            default=PhysicalityFilterMode.HIGH_QUALITY.value,
            show_default=True,
            metavar="MODE",
            help="Physicality checks reported per frame for invalid bond lengths, clashes, "
            "chirality, or chain breaks. Sampling outputs retain all frames; use no_filter "
            "to skip checks.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def _speed_options(function: Callable) -> Callable:
    options = [
        click.option(
            "--precision",
            type=click.Choice(["32-true", "bf16-mixed"]),
            default="bf16-mixed",
            show_default=True,
            help="Precision for trunk embedding. Static-context computation and denoising "
            "always run in full precision; the denoiser's float32 matmul precision is set "
            "by --matmul-precision.",
        ),
        click.option(
            "--compile-denoiser",
            is_flag=True,
            help="Not recommended for light workload because compilation takes time. For large number of "
            "samples and very small system size ( < 128 token), turn it on. "
            "If use `high` or `medium` matmul precision, turn it on.",
        ),
        click.option(
            "--matmul-precision",
            type=click.Choice(["highest", "high", "medium"]),
            default="high",
            show_default=True,
            help="Float32 matmul precision during denoising. "
            "'high' uses TF32 tensor cores on supported GPUs at slightly lower precision.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def _resolve_common(
    *,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    x0_cif: Path | None = None,
) -> tuple[System, list[Contact] | None, str | None]:
    """Resolve and validate shared CLI inputs before loading the model."""
    if (system is None and pdb_id is None and x0_cif is None) or (
        system is not None and pdb_id is not None
    ):
        raise click.UsageError("Provide exactly one of --system or --pdb-id.")
    if checkpoint is None and checkpoint_path is None:
        checkpoint = DEFAULT_CHECKPOINT
    if (checkpoint is None) == (checkpoint_path is None):
        raise click.UsageError("Provide exactly one of --checkpoint or --checkpoint-path.")
    if checkpoint_path is not None and checkpoint_cache_dir is not None:
        raise click.UsageError("--checkpoint-cache-dir cannot be used with --checkpoint-path.")
    if system is not None:
        resolved_system, contacts = load_system_and_contacts(system)
    elif pdb_id is not None:
        resolved_system, contacts = System.from_pdb_id(pdb_id), None
    else:
        assert x0_cif is not None
        resolved_system, contacts = System.from_cif(x0_cif), None
    return resolved_system, contacts, checkpoint


def _write_physicality_report(
    mask: np.ndarray,
    reasons: list[str | None],
    physicality_filter: PhysicalityFilterMode,
    out: Path,
    checked: bool,
) -> None:
    PhysicalityReport(passed=mask, reasons=reasons, mode=physicality_filter, checked=checked).save(
        out / "physicality.json"
    )


def _write_propagation_output(
    system: System,
    *,
    x0: mdtraj.Trajectory,
    lag_time_ns: float,
    out: Path,
    endpoints: mdtraj.Trajectory,
    contacts: list[Contact] | None = None,
    physicality_filter: PhysicalityFilterMode = PhysicalityFilterMode.HIGH_QUALITY,
    mae: bool = False,
    cif: bool = False,
) -> Path:
    """Write every sampled endpoint to ``samples.xtc`` in input-frame order.

    ``physicality.json`` records the status of each endpoint without removing
    rejected frames. Only the sampled endpoints are saved; the input ``x0``
    frames are not included in the XTC.
    """
    write_sampling_system(system, mode="equilibrium", contacts=contacts, out=out)
    structure = AllAtomStructure.from_system(system, mode="equilibrium", contacts=contacts)
    save_mae(x0[0], out / "topology.mae", structure=structure, titles=["topology"])
    if cif:
        save_topology_cif(structure, system, x0.xyz[0] * 10.0, out / "topology.cif")
    expected_extra_bonds = expected_extra_bonds_from_system(system, endpoints.topology)
    passes_checks = np.empty(endpoints.n_frames, dtype=bool)
    reasons: list[str | None] = []
    checked = False
    for start in range(0, endpoints.n_frames, PHYSICALITY_FRAME_CHUNK_SIZE):
        chunk = endpoints[start : start + PHYSICALITY_FRAME_CHUNK_SIZE]
        report = check_physicality(
            chunk, physicality_filter, expected_extra_bonds=expected_extra_bonds
        )
        checked = report.checked
        passes_checks[start : start + chunk.n_frames] = report.passed
        reasons.extend(report.reasons)
    endpoints.time = np.full(endpoints.n_frames, lag_time_ns * 1000.0)
    endpoints.save_xtc(str(out / "samples.xtc"))
    if mae:
        save_mae(endpoints, out / "samples.mae", structure=structure)
    _write_physicality_report(passes_checks, reasons, physicality_filter, out, checked)
    if not checked:
        logger.info("Wrote %d unchecked endpoints to %s", endpoints.n_frames, out)
    else:
        logger.info(
            "%d/%d endpoints passed physicality checks; results saved in %s",
            int(passes_checks.sum()),
            endpoints.n_frames,
            out,
        )
    return out


def validate_x0(trajectory: mdtraj.Trajectory, system: System) -> mdtraj.Trajectory:
    """Check that starting conformations have the system's atoms and topology."""
    atoms = [
        (a.name, a.element.symbol, a.residue.name, a.residue.chain.index)
        for a in trajectory.topology.atoms
    ]
    expected = [
        (a.name, a.element.symbol, a.residue.name, a.residue.chain.index)
        for a in system.topology.atoms
    ]
    if atoms != expected:
        k = next((k for k, (a, e) in enumerate(zip(atoms, expected)) if a != e), None)
        detail = (
            f"{len(atoms)} atoms, system has {len(expected)}"
            if k is None
            else f"atom {k} is {atoms[k]}, system has {expected[k]}"
        )
        raise ValueError(f"--x0-cif atoms do not match the system: {detail}")
    # Same atoms, so use the system's topology: the CIF writer adds ligand bonds that
    # System.topology lacks, and propagate() compares bonds too.
    trajectory.topology = system.topology
    return trajectory


def _write_parameters(out: Path, sampler) -> None:
    """Record every resolved parameter, with the CLI key that sets it.

    Values are read back off the constructed objects, not re-derived from the
    flags, so what is written is what will actually be used.
    """
    import dataclasses

    out.mkdir(parents=True, exist_ok=True)
    pot_key = {v: k for k, v in _POTENTIAL_PARAMS.items()}
    ff_key = {v: k for k, v in _FORCEFIELD_PARAMS.items()}

    lines: list[str] = [
        "# Every parameter in effect for this run, read off the constructed",
        "# objects. The 'cli' field is the flag that sets it; a blank one is",
        "# not reachable from the command line.",
        "sampler:",
        f"  class: {type(sampler).__name__}",
    ]
    try:
        for f in sorted(dataclasses.fields(sampler), key=lambda x: x.name):
            if f.name == "potentials":
                continue
            v = getattr(sampler, f.name)
            cli = f"-S {f.name}" if f.name in _SAMPLER_FIELDS else ""
            lines.append(f"  {f.name}: {v!r}"
                         + (f"            # {cli}" if cli else ""))
    except TypeError:
        lines.append("  (not a dataclass)")

    lines.append("potentials:")
    for p in getattr(sampler, "potentials", ()) or ():
        lines.append(f"  - class: {type(p).__name__}")
        params = getattr(p, "parameters", None) or {}
        # MMFFBondPotential -> "bond", so the -F keys (bond.weight,
        # torsion.weight, ...) match. MMFF14Potential -> "pairs14".
        stem = type(p).__name__.replace("Potential", "").lower()
        for ff in ("mmff", "gaff"):
            if stem.startswith(ff) and stem != ff:
                stem = stem[len(ff):]
        if stem == "14":
            stem = "pairs14"
        for k in sorted(params):
            # Keys are Class.param. Several potentials store a parameter under
            # the same name -- ChiralAtom, StereoBond and PlanarBond all use
            # "buffer" -- so the class has to pick the key, not the name.
            suffix = "weight" if k == "guidance_weight" else k
            cli = next((c for c in list(_POTENTIAL_PARAMS)
                        + list(_FORCEFIELD_PARAMS)
                        if c.split(".")[0].lower() == stem
                        and c.split(".", 1)[1] == suffix), "")
            if not cli:
                cli = pot_key.get(k, "")
            flag = "-F" if cli in _FORCEFIELD_PARAMS else "-P"
            notes = [f"{flag} {cli}"] if cli else []
            describe = getattr(params[k], "describe", None)
            if callable(describe) and describe():
                notes.append(describe())
            lines.append(f"      {k}: {params[k]!r}"
                         + (f"        # {'; '.join(notes)}" if notes else ""))
        for attr in ("uniform_k", "use_hydrogens", "cutoff", "sigma_tolerance",
                     "lj_core_switch", "coulomb_core_switch", "dielectric", "t_on",
                     "nonbonded_t_on", "guidance_ramp", "ligand_ff",
                     "bond_angle_late_weight", "bond_late_weight",
                     "angle_late_weight", "bond_angle_late_t_on"):
            if hasattr(p, attr):
                cli = ff_key.get(attr, "")
                lines.append(f"      {attr}: {getattr(p, attr)!r}"
                             + (f"        # -F {cli}" if cli else ""))

    lines.append("available_keys:")
    lines.append("  potential (-P):")
    for k in sorted(_POTENTIAL_PARAMS):
        lines.append(f"    - {k}")
    lines.append("  forcefield (-F):")
    lines.append("    - all.weight")
    for k in sorted(_FORCEFIELD_PARAMS):
        lines.append(f"    - {k}")
    lines.append("  sampler (-S):")
    for k, where in sorted(_SAMPLER_FIELDS.items()):
        lines.append(f"    - {k}    # {where}")

    (out / "parameters.yaml").write_text("\n".join(lines) + "\n")


def _write_command(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.txt").write_text(shlex.join(sys.argv) + "\n")


@click.group()
def main() -> None:
    """Sample structures and trajectories for a molecular system with BioEmu2."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # huggingface_hub logs every HTTP request through httpx at INFO, including the
    # revision checks made for files that are already cached.
    logging.getLogger("httpx").setLevel(logging.WARNING)


@main.command("convert-boltz")
@click.argument("source", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    metavar="YAML",
    help="Destination BioEmu2 system YAML. Parent directories are created; existing files "
    "are not overwritten.",
)
def convert_boltz_command(source: Path, out: Path) -> None:
    """Convert Boltz-1 or Boltz-2 input YAML to BioEmu2 system YAML.

    Both use input schema version 1. Conversion does not load a model.
    Boltz MSAs and unsupported non-contact fields are ignored with warnings.
    Boltz contact force settings are ignored with warnings; unsupported contact
    fields raise errors. See docs/boltz.md for details.
    """
    try:
        convert_boltz(source, out)
    except (OSError, TypeError, ValueError, KeyError, NotImplementedError, yaml.YAMLError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote BioEmu2 system YAML to {out}")


@main.command("sample-equilibrium")
@_common_options
@_speed_options
@click.option(
    "--num-samples",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Number of structures to generate.",
)
def sample_equilibrium_command(
    num_samples: int,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    out: Path,
    batch_size: int | None,
    seed: int | None,
    num_steps: int | None,
    sampler: str,
    potentials: Path | None,
    potential_param: tuple[str, ...],
    forcefield_param: tuple[str, ...],
    sampler_param: tuple[str, ...],
    precision: str,
    physicality_filter: str,
    mae: bool,
    cif: bool,
    compile_denoiser: bool,
    matmul_precision: str | None,
) -> None:
    """Sample the equilibrium ensemble."""
    resolved_system, contacts, resolved_checkpoint = _resolve_common(
        system=system,
        pdb_id=pdb_id,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_cache_dir=checkpoint_cache_dir,
    )
    resolved_potentials = _load_potentials(sampler, potentials, potential_param,
                                       forcefield_param)
    _write_command(out)
    try:
        _write_parameters(out, _sampler(sampler, potentials, potential_param,
                                        sampler_param, forcefield_param,
                                        mode="equilibrium"))
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "could not write parameters.yaml: %s", exc)
    model = load_model(
        checkpoint=resolved_checkpoint,
        checkpoint_path=checkpoint_path,
        cache_dir=checkpoint_cache_dir,
        precision=precision,
        compile_denoiser=compile_denoiser,
        matmul_precision=matmul_precision,
    )
    sample_equilibrium(
        resolved_system,
        model=model,
        contacts=contacts,
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=cast(SamplerPreset, sampler),
        num_steps=num_steps,
        potentials=resolved_potentials,
        sampler_params=_parse_sampler_params(sampler_param),
        seed=seed,
        output_dir=out,
        physicality_mode=physicality_filter,
        mae=mae,
        cif=cif,
    )


@main.command("predict-structure")
@_common_options
@_speed_options
@click.option(
    "--num-samples",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Number of structures to generate.",
)
def predict_structure_command(
    num_samples: int,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    out: Path,
    batch_size: int | None,
    seed: int | None,
    num_steps: int | None,
    sampler: str,
    potentials: Path | None,
    precision: str,
    physicality_filter: str,
    mae: bool,
    cif: bool,
    compile_denoiser: bool,
    matmul_precision: str | None,
    potential_param: tuple[str, ...] = (),
    forcefield_param: tuple[str, ...] = (),
    sampler_param: tuple[str, ...] = (),
) -> None:
    """Predict the folded structure."""
    resolved_system, contacts, resolved_checkpoint = _resolve_common(
        system=system,
        pdb_id=pdb_id,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_cache_dir=checkpoint_cache_dir,
    )
    resolved_potentials = _load_potentials(sampler, potentials, potential_param,
                                       forcefield_param)
    _write_command(out)
    try:
        _write_parameters(out, _sampler(sampler, potentials, potential_param,
                                        sampler_param, forcefield_param,
                                        mode="structure"))
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "could not write parameters.yaml: %s", exc)
    model = load_model(
        checkpoint=resolved_checkpoint,
        checkpoint_path=checkpoint_path,
        cache_dir=checkpoint_cache_dir,
        precision=precision,
        compile_denoiser=compile_denoiser,
        matmul_precision=matmul_precision,
    )
    predict_structure(
        resolved_system,
        model=model,
        contacts=contacts,
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=cast(SamplerPreset, sampler),
        num_steps=num_steps,
        potentials=resolved_potentials,
        sampler_params=_parse_sampler_params(sampler_param),
        seed=seed,
        output_dir=out,
        physicality_mode=physicality_filter,
        mae=mae,
        cif=cif,
    )


@main.command("propagate")
@_common_options
@_speed_options
@click.option(
    "--x0-cif",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    metavar="CIF",
    help="Topology of the starting conformations; atoms must match the system.",
)
@click.option(
    "--x0-xtc",
    type=click.Path(path_type=Path, dir_okay=False),
    metavar="XTC",
    help="Sample one endpoint per input frame. Pass exactly one of --x0-xtc and --num-samples.",
)
@click.option(
    "--num-samples",
    type=click.IntRange(min=1),
    metavar="N",
    help="Number of endpoints to sample from the structure in --x0-cif. "
    "Pass exactly one of --num-samples and --x0-xtc.",
)
@click.option(
    "--lag-time-ns",
    required=True,
    type=click.FloatRange(min=0, min_open=True),
    metavar="T",
    help="Time lag in nanoseconds. We recommend 1 to 1000 (1 ns to 1 µs), the range "
    "the model mostly saw during training. Lags longer than a few milliseconds "
    "effectively disable conditioning on the starting structure.",
)
def propagate_command(
    x0_cif: Path,
    x0_xtc: Path | None,
    num_samples: int | None,
    lag_time_ns: float,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    out: Path,
    batch_size: int | None,
    seed: int | None,
    num_steps: int | None,
    sampler: str,
    potentials: Path | None,
    precision: str,
    physicality_filter: str,
    mae: bool,
    cif: bool,
    compile_denoiser: bool,
    matmul_precision: str | None,
    potential_param: tuple[str, ...] = (),
    forcefield_param: tuple[str, ...] = (),
    sampler_param: tuple[str, ...] = (),
) -> None:
    """Sample endpoints at a time lag from a CIF or each frame of an XTC.

    When neither --system nor --pdb-id is provided, derive the system from
    --x0-cif; explicit system inputs remain available for custom definitions.
    Writes every endpoint in input order to samples.xtc (without starting
    frames) and per-frame physicality results to physicality.json.
    """
    if (x0_xtc is None) == (num_samples is None):
        raise click.UsageError("Pass exactly one of --x0-xtc and --num-samples.")
    existing_outputs = [
        path
        for path in (
            *out.glob("samples_*.xtc"),
            out / "samples.xtc",
            out / "physicality.json",
        )
        if path.exists()
    ]
    if existing_outputs:
        raise click.UsageError(
            "Propagation output already exists; use a new --out directory: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    resolved_system, contacts, resolved_checkpoint = _resolve_common(
        system=system,
        pdb_id=pdb_id,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_cache_dir=checkpoint_cache_dir,
        x0_cif=x0_cif,
    )
    resolved_sampler = _sampler(sampler, potentials, potential_param, sampler_param,
                               forcefield_param)
    if num_steps is not None:
        resolved_sampler.num_steps = num_steps
    if x0_xtc is not None:
        x0 = mdtraj.load(str(x0_xtc), top=str(x0_cif))
    else:
        x0 = mdtraj.load(str(x0_cif))
        if x0.n_frames != 1:
            raise click.UsageError("--x0-cif must contain exactly one structure without --x0-xtc.")
        assert num_samples is not None
        x0 = x0[[0] * num_samples]
    x0 = validate_x0(x0, resolved_system)
    _write_command(out)
    try:
        _write_parameters(out, _sampler(sampler, potentials, potential_param,
                                        sampler_param, forcefield_param,
                                        mode="equilibrium"))
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "could not write parameters.yaml: %s", exc)
    model = load_model(
        checkpoint=resolved_checkpoint,
        checkpoint_path=checkpoint_path,
        cache_dir=checkpoint_cache_dir,
        precision=precision,
        compile_denoiser=compile_denoiser,
        matmul_precision=matmul_precision,
    )
    if num_steps is None and potentials is None:
        endpoints = propagate(
            resolved_system,
            model=model,
            x0=x0,
            lag_time_ns=lag_time_ns,
            contacts=contacts,
            batch_size=batch_size,
            sampler=cast(SamplerPreset, sampler),
            seed=seed,
        )
    else:
        msa = get_msa(resolved_system)
        trunk_embeddings = model.embed(
            resolved_system,
            msa=msa,
            mode="equilibrium",
            contacts=contacts,
        )
        endpoints = propagate_with_embeddings(
            resolved_system,
            trunk_embeddings,
            x0=x0,
            lag_time_ns=lag_time_ns,
            model=model,
            batch_size=batch_size,
            sampler=resolved_sampler,
            seed=seed,
        )
    _write_propagation_output(
        resolved_system,
        x0=x0,
        lag_time_ns=lag_time_ns,
        out=out,
        endpoints=endpoints,
        contacts=contacts,
        physicality_filter=PhysicalityFilterMode(physicality_filter),
        mae=mae,
        cif=cif,
    )


@main.group("cp")
def context_parallel_commands() -> None:
    """Run context-parallel (CP) sampling on a square mesh of at least 4 GPUs.

    Example: torchrun --standalone --nproc-per-node=4
    --module bioemu2.cp sample-equilibrium --system system.yaml --out output
    """


def _run_cp_sampling_command(
    *,
    mode: SamplingMode,
    num_samples: int,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    out: Path,
    batch_size: int | None,
    seed: int | None,
    num_steps: int | None,
    sampler: str,
    potentials: Path | None,
    precision: str,
    matmul_precision: str | None,
    physicality_filter: str,
    potential_param: tuple[str, ...] = (),
    forcefield_param: tuple[str, ...] = (),
    sampler_param: tuple[str, ...] = (),
) -> None:
    with cp_api.session():
        resolved: list[
            tuple[System, list[Contact] | None, Sequence[Potential] | None, str | None] | None
        ] = [None]
        if dist.get_rank() == 0:
            rank0_resolved = _resolve_common(
                system=system,
                pdb_id=pdb_id,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                checkpoint_cache_dir=checkpoint_cache_dir,
            )
            resolved_system, contacts, resolved_checkpoint = rank0_resolved
            resolved_potentials = _load_potentials(sampler, potentials, potential_param,
                                       forcefield_param)
            resolved[0] = (resolved_system, contacts, resolved_potentials, resolved_checkpoint)
            _write_command(out)
            try:
                _write_parameters(out, _sampler(sampler, potentials, potential_param,
                                                sampler_param, forcefield_param,
                                                mode="equilibrium"))
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "could not write parameters.yaml: %s", exc)
        dist.broadcast_object_list(resolved, src=0)
        assert resolved[0] is not None
        resolved_system, contacts, resolved_potentials, resolved_checkpoint = resolved[0]
        msa = get_msa(resolved_system) if dist.get_rank() == 0 else None
        model = cp_api.load_model(
            checkpoint=resolved_checkpoint,
            checkpoint_path=checkpoint_path,
            cache_dir=checkpoint_cache_dir,
            precision=precision,
            matmul_precision=matmul_precision,
        )
        sample = cp_api.sample_equilibrium if mode == "equilibrium" else cp_api.predict_structure
        sample(
            resolved_system,
            model=model,
            msa=msa,
            contacts=contacts,
            num_samples=num_samples,
            batch_size=batch_size,
            sampler=cast(SamplerPreset, sampler),
            num_steps=num_steps,
            potentials=resolved_potentials,
            sampler_params=_parse_sampler_params(sampler_param),
            seed=seed,
            output_dir=out,
            physicality_mode=physicality_filter,
        )


_cp_commands: list[tuple[click.Command, SamplingMode]] = [
    (sample_equilibrium_command, "equilibrium"),
    (predict_structure_command, "structure"),
]
for _command, _mode in _cp_commands:
    _params = []
    for _option in _command.params:
        # The CP model loader does not support denoiser compilation.
        if _option.name == "compile_denoiser":
            continue
        _cp_option = copy(_option)
        if isinstance(_cp_option, click.Option):
            if _cp_option.name == "sampler":
                _cp_option.type = click.Choice(["edm", "edm_gd", "edm_fk"])
                _cp_option.help = (
                    "Sampling algorithm. CP supports EDM, gradient guidance, and FK resampling."
                )
            elif _cp_option.name == "potentials":
                _cp_option.help = (
                    "YAML containing a top-level 'potentials' list; replaces the default "
                    "physicality potentials and requires edm_gd or edm_fk."
                )
            elif _cp_option.name == "precision":
                _cp_option.help = (
                    "Precision for trunk embedding. Denoising runs in full precision; "
                    "its float32 matmul precision is set by --matmul-precision."
                )
        _params.append(_cp_option)
    context_parallel_commands.add_command(
        click.Command(
            name=_command.name,
            callback=partial(_run_cp_sampling_command, mode=_mode),
            params=_params,
            help=_command.help,
        )
    )


if __name__ == "__main__":
    main()
