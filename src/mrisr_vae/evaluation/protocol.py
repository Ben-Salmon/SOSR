r"""An evaluation protocol: one yaml (e.g. configs/evaluation/kaggle_cmr_test.yaml)
that fixes how every method is scored, so their scores compare row by row.

    out_dir: <dir>            every method's predictions and the scores
    data_dir: <dir>           the test subjects
    geometry: <yaml>          the latent grid; every checkpoint must match it
    heldout_views: [la_4ch]   the views predicted and scored
    region: sa_first          `<group>_first` or null (see `conditioning`)
    conditioning_views: [all, sa, sa_first, la_2ch]
    frame: null               null: a frame fixed per subject, the same for every method
    seed: 0                   posterior samples are reseeded per subject and set from it
    split: all                all subjects of data_dir, or its train or val split
    split_seed: 42            (for train and val)
    lanczos_a: 3              the forward model's Lanczos order
    methods:                  models, each giving one prediction
      <name>: {checkpoint: <path>, n_samples: 32}   the mean of n posterior samples
      <name>: {checkpoint: <path>}                  n_samples 0: its direct predictor
    baselines:                interpolation and SMORE (`baselines`)
      sources: [linear, cubic, smore]
      work_dir: <dir>         exported stacks and SMORE's outputs

Paths are relative to the repository root.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from ..data import CMRObservationDataset, split_subjects
from ..geometry import LatentGeometry
from .conditioning import check_region, norm_views_for, resolve_conditioning

BASELINE_SOURCES = ("linear", "cubic", "smore")
# The baselines' conditioning set: each subject's lowest-numbered short-axis stack.
BASELINE_CONDITIONING = "sa_first"
# What protocol.yaml in out_dir records; a run with other values is refused.
SHARED_KEYS = (
    "data_dir", "heldout_views", "region", "conditioning_views", "frame", "seed", "split", "split_seed",
    "lanczos_a",
)
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass
class Method:
    name: str
    checkpoint: str
    n_samples: int = 0


@dataclass
class Baselines:
    work_dir: str
    sources: list[str] = field(default_factory=lambda: list(BASELINE_SOURCES))


@dataclass
class Protocol:
    out_dir: str
    data_dir: str
    geometry_path: str
    geometry_config: dict
    heldout_views: list[str]
    region: str | None = "sa_first"
    conditioning_views: list[str] = field(default_factory=lambda: ["all"])
    frame: int | None = None
    seed: int = 0
    split: str = "all"
    split_seed: int = 42
    lanczos_a: int = 3
    methods: dict[str, Method] = field(default_factory=dict)
    baselines: Baselines | None = None

    @property
    def geometry(self) -> LatentGeometry:
        return LatentGeometry.from_config(self.geometry_config)

    def conditioning_sets(self) -> dict[str, list[str]]:
        views = self.geometry_config["VIEWS"]
        return {spec: resolve_conditioning(spec, views) for spec in self.conditioning_views}

    def subjects(self) -> list[str] | None:
        r"""The subject names to evaluate; None for every subject of data_dir."""
        if self.split == "all":
            return None
        train, val = split_subjects(self.data_dir, self.geometry_config["VIEWS"], seed=self.split_seed)
        return {"train": train, "val": val}[self.split]

    def dataset(
        self,
        view: str,
        spec: str,
        subjects: list[str] | None = None,
        max_subjects: int | None = None,
        extra_noise_std: dict | None = None,
    ) -> CMRObservationDataset:
        r"""The items scored on held-out `view` from conditioning set `spec`.

        Every training view is loaded with the held-out one, so the frame
        (chosen from the frames all of them have) is the same for every
        conditioning set and method; only the conditioning views set the
        normalisation. `extra_noise_std` is the model's, as trained.
        """
        geometry = self.geometry
        names = subjects if subjects is not None else self.subjects()
        dataset = CMRObservationDataset(
            self.data_dir,
            geometry,
            subjects=names,
            extra_noise_std=extra_noise_std,
            views=list(geometry.views) + [view],
            norm_views=norm_views_for(spec, resolve_conditioning(spec, geometry.views)),
            train=False,
            frame=self.frame,
            lanczos_a=self.lanczos_a,
        )
        if max_subjects is not None:
            dataset.subjects = dataset.subjects[:max_subjects]
        return dataset

    def shared(self) -> dict:
        r"""The settings every method in out_dir must share, with the geometry itself."""
        return {**{key: getattr(self, key) for key in SHARED_KEYS}, "geometry": self.geometry_config}

    def record(self):
        r"""Writes <out_dir>/protocol.yaml, or checks this protocol matches it."""
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "protocol.yaml"
        shared = self.shared()
        if path.is_file():
            with open(path) as f:
                recorded = yaml.safe_load(f)
            changed = sorted(key for key in shared if recorded.get(key) != shared[key])
            if changed:
                raise SystemExit(
                    f"{path} records other values of {', '.join(changed)}: its scores would not compare with "
                    "this run's. Use a fresh out_dir."
                )
            return
        with open(path, "w") as f:
            yaml.safe_dump(shared, f, sort_keys=False)


def load_protocol(path) -> Protocol:
    r"""Reads and checks a protocol yaml."""
    with open(path) as f:
        given = yaml.safe_load(f) or {}
    known = {"out_dir", "data_dir", "geometry", "methods", "baselines", *SHARED_KEYS}
    unknown = sorted(set(given) - known)
    if unknown:
        raise ValueError(f"{path}: unknown option(s) {', '.join(unknown)}")
    for key in ("out_dir", "data_dir", "geometry", "heldout_views"):
        if given.get(key) is None:
            raise ValueError(f"{path}: set `{key}`")
    with open(given["geometry"]) as f:
        geometry_config = yaml.safe_load(f)

    methods = {}
    for name, options in (given.get("methods") or {}).items():
        if not NAME.match(str(name)) or name in BASELINE_SOURCES:
            raise ValueError(f"{path}: `{name}` cannot name a method (letters, digits, _ and -; not a baseline's name)")
        options = dict(options or {})
        unknown = sorted(set(options) - {"checkpoint", "n_samples"})
        if unknown or "checkpoint" not in options:
            raise ValueError(f"{path}: method `{name}` needs `checkpoint` (and optionally `n_samples`) only")
        methods[name] = Method(name=name, checkpoint=options["checkpoint"], n_samples=int(options.get("n_samples", 0)))
        if methods[name].n_samples < 0:
            raise ValueError(f"{path}: method `{name}`: n_samples must not be negative")
    baselines = None
    if given.get("baselines") is not None:
        options = dict(given["baselines"])
        if options.get("work_dir") is None:
            raise ValueError(f"{path}: set `baselines: work_dir`")
        sources = list(options.get("sources") or BASELINE_SOURCES)
        bad = sorted(set(sources) - set(BASELINE_SOURCES))
        if bad:
            raise ValueError(f"{path}: unknown baseline source(s) {', '.join(bad)}")
        baselines = Baselines(work_dir=options["work_dir"], sources=sources)

    defaults = Protocol(out_dir="", data_dir="", geometry_path="", geometry_config={}, heldout_views=[])
    protocol = Protocol(
        out_dir=given["out_dir"],
        data_dir=given["data_dir"],
        geometry_path=given["geometry"],
        geometry_config=geometry_config,
        heldout_views=list(given["heldout_views"]),
        **{key: given.get(key, getattr(defaults, key)) for key in SHARED_KEYS if key not in ("data_dir", "heldout_views")},
        methods=methods,
        baselines=baselines,
    )
    protocol.conditioning_views = list(protocol.conditioning_views)
    if protocol.region in ("none", "null"):
        protocol.region = None

    views = geometry_config["VIEWS"]
    trained = sorted(set(protocol.heldout_views) & set(views))
    if trained:
        raise ValueError(f"{path}: {trained} are in the geometry's VIEWS, so they are not held out")
    protocol.conditioning_sets()  # raises ValueError for an unknown set
    check_region(protocol.region, views)
    if protocol.split not in ("all", "train", "val"):
        raise ValueError(f"{path}: split must be all, train or val")
    if protocol.frame is not None and protocol.frame < 0:
        raise ValueError(f"{path}: the frame must not be negative")
    return protocol


def same_geometry(a: LatentGeometry, b: LatentGeometry) -> bool:
    return (
        a.views == b.views
        and a.size_lvox == b.size_lvox
        and torch.allclose(a.su_per_lvox, b.su_per_lvox)
        and torch.allclose(a.min_coords_su, b.min_coords_su)
    )
