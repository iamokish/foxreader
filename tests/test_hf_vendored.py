"""The vendored `fox_reader.hf` package keeps working as upstream moves.

`fox_reader/hf/models/*` holds files copied byte-for-byte from
``transformers/models/pp_*``, and PaddlePaddle ships new PP-OCR models and
revises old ones regularly. Re-copying a folder is meant to be the whole job --
these checks are what make a *failed* re-copy say so.

Two kinds of check, and they fail for different reasons:

* :class:`TestImportSurface` reads the vendored files without importing them
  and asserts every relative import resolves against the shim. A newly copied
  model that needs something `fox_reader.hf` does not provide fails here, with
  the missing symbol named -- rather than at a user's first OCR run, eleven
  frames into a model constructor.
* :class:`TestConfigRoundTrip` builds each config from the real ``config.json``
  on disk. This is what catches the shim's one genuinely fragile dependency:
  the configs are ``huggingface_hub`` strict dataclasses, so an hf-hub upgrade
  that moves ``dataclasses.strict`` or changes its field semantics breaks
  loading, and nothing else here would notice.

Everything that needs the ~190 MB of weights skips when they are absent, so
this file runs on a bare checkout. The bit-exact comparison against paddleocr
lives in ``tests/manual/smoke_ocr_parity.py``: it needs the dependency being
removed, so it can only ever run mid-migration.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

HF_ROOT = Path(__file__).resolve().parents[1] / "src" / "fox_reader" / "hf"
MODELS_ROOT = HF_ROOT / "models"
WEIGHTS_ROOT = Path(__file__).resolve().parents[1] / "models" / "paddleocr"

#: Checkpoint directory -> the ``model_type`` its ``config.json`` declares.
#: Spelled out because the two differ: the PP-OCRv6 *medium* recogniser
#: declares ``pp_ocrv6_small_rec``, and the Korean checkpoint declares
#: ``pp_ocrv5_mobile_rec``. A test that derived one from the other would pass
#: while proving nothing.
CHECKPOINTS = {
    "PP-OCRv6_medium_det_safetensors": "pp_ocrv6_medium_det",
    "PP-OCRv6_medium_rec_safetensors": "pp_ocrv6_small_rec",
    "korean_PP-OCRv5_mobile_rec_safetensors": "pp_ocrv5_mobile_rec",
}


def vendored_files() -> list[Path]:
    """Every vendored model file, excluding the hand-written ``__init__``s."""
    return sorted(p for p in MODELS_ROOT.rglob("*.py") if p.name != "__init__.py")


def model_packages() -> list[Path]:
    return sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir() and p.name != "auto")


def _relative_imports(path: Path) -> list[tuple[str, str, int]]:
    """``(module, symbol, lineno)`` for each ``from ..x import y`` in a file.

    ``level`` counts the leading dots: 1 is the model's own package, 2 is
    ``hf.models`` (``from ..auto import AutoConfig``), 3 is ``hf`` itself
    (``from ...modeling_utils import PreTrainedModel``).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            continue  # absolute: torch, numpy, huggingface_hub -- real deps
        anchor = {1: "fox_reader.hf.models.PACKAGE", 2: "fox_reader.hf.models", 3: "fox_reader.hf"}
        if node.level not in anchor:
            pytest.fail(f"{path.name}:{node.lineno} imports {node.level} levels up, which is outside fox_reader.hf")
        base = anchor[node.level]
        if node.level == 1:
            base = f"fox_reader.hf.models.{path.parent.name}"
        module = f"{base}.{node.module}" if node.module else base
        for alias in node.names:
            found.append((module, alias.name, node.lineno))
    return found


class TestImportSurface:
    """Every relative import in a vendored file resolves against the shim."""

    def test_there_are_vendored_files_to_check(self):
        # Guards against the globs silently matching nothing after a move,
        # which would make every other check in this class vacuously pass.
        assert len(vendored_files()) >= 13, vendored_files()
        assert len(model_packages()) >= 7, model_packages()

    @pytest.mark.parametrize("path", vendored_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
    def test_relative_imports_resolve(self, path: Path):
        missing = []
        for module, symbol, lineno in _relative_imports(path):
            try:
                imported = importlib.import_module(module)
            except ImportError as exc:
                missing.append(f"  line {lineno}: no module {module} ({exc})")
                continue
            if not hasattr(imported, symbol):
                missing.append(f"  line {lineno}: {module} has no {symbol!r}")
        assert not missing, (
            f"{path.relative_to(HF_ROOT.parent)} needs symbols fox_reader.hf does not provide:\n"
            + "\n".join(missing)
            + "\n\nThis is the expected failure when a model folder copied from a newer "
            "transformers needs more of the shim. Add the symbol to fox_reader/hf/, "
            "or see fox_reader/hf/README.md."
        )

    @pytest.mark.parametrize("package", model_packages(), ids=lambda p: p.name)
    def test_package_init_exports_every_public_class(self, package: Path):
        """A model's ``__init__.py`` re-exports what its files declare.

        The hand-written ``__init__.py`` is the one file not copied from
        upstream, so it is the one that gets forgotten when a model gains a
        class. The registry resolves classes through the package, so anything
        missing here is invisible to `AutoConfig` and friends.
        """
        declared: set[str] = set()
        for path in package.glob("*.py"):
            if path.name == "__init__.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "__all__" for t in node.targets
                ):
                    declared.update(e.value for e in node.value.elts)

        module = importlib.import_module(f"fox_reader.hf.models.{package.name}")
        exported = set(getattr(module, "__all__", ()))
        assert declared <= exported, (
            f"fox_reader/hf/models/{package.name}/__init__.py does not export "
            f"{sorted(declared - exported)}, which its own files declare in __all__."
        )
        for name in exported:
            assert hasattr(module, name), f"{package.name}/__init__.py exports a missing {name!r}"


class TestRegistry:
    """The hand-maintained tables in ``hf/models/auto`` stay consistent."""

    def test_every_model_type_has_a_config(self):
        from fox_reader.hf.models import auto

        assert set(auto.MODEL_TYPE_TO_MODULE) == set(auto.CONFIG_MAPPING_NAMES)

    @pytest.mark.parametrize(
        "mapping_name",
        [
            "MODEL_FOR_BACKBONE_MAPPING_NAMES",
            "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES",
            "MODEL_FOR_TEXT_RECOGNITION_MAPPING_NAMES",
            "IMAGE_PROCESSOR_MAPPING_NAMES",
        ],
    )
    def test_task_mappings_name_known_model_types(self, mapping_name: str):
        from fox_reader.hf.models import auto

        unknown = set(getattr(auto, mapping_name)) - set(auto.MODEL_TYPE_TO_MODULE)
        assert not unknown, f"{mapping_name} names model types not in MODEL_TYPE_TO_MODULE: {sorted(unknown)}"

    def test_every_registered_class_can_be_imported(self):
        """Resolve every table entry, so a typo fails here and not at load."""
        from fox_reader.hf.models import auto

        for model_type in auto.CONFIG_MAPPING_NAMES:
            assert auto.AutoConfig.for_model_type(model_type) is not None
        for model_type in auto.MODEL_FOR_BACKBONE_MAPPING_NAMES:
            assert auto.AutoBackbone.for_model_type(model_type) is not None
        for model_type in auto.MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES:
            assert auto.AutoModelForObjectDetection.for_model_type(model_type) is not None
        for model_type in auto.MODEL_FOR_TEXT_RECOGNITION_MAPPING_NAMES:
            assert auto.AutoModelForTextRecognition.for_model_type(model_type) is not None
        for module, class_name in auto.IMAGE_PROCESSOR_MAPPING_NAMES.values():
            package = importlib.import_module(f"fox_reader.hf.models.{module}")
            assert hasattr(package, class_name), f"{module} has no {class_name}"

    def test_unknown_model_type_names_what_is_known(self):
        from fox_reader.hf.models import auto

        with pytest.raises(ValueError, match="pp_ocrv9_imaginary"):
            auto.AutoConfig.for_model_type("pp_ocrv9_imaginary")


class TestConfigRoundTrip:
    """Real ``config.json`` files still build real config objects."""

    @pytest.mark.parametrize("directory,model_type", sorted(CHECKPOINTS.items()))
    def test_config_loads_from_the_real_checkpoint(self, directory: str, model_type: str):
        path = WEIGHTS_ROOT / directory
        if not (path / "config.json").is_file():
            pytest.skip(f"{directory} is not downloaded")

        from fox_reader.hf.models.auto import AutoConfig

        config = AutoConfig.from_pretrained(path)
        assert type(config).model_type == model_type

        # The sub-config recursion: a nested dict must become a real config
        # object, or `load_backbone` has nothing to build a backbone from.
        backbone = config.backbone_config
        assert not isinstance(backbone, dict), "backbone_config stayed a plain dict"
        assert backbone.stage_names, "the backbone config has no stage_names"
        assert backbone.out_features, "the backbone config resolved no out_features"
        assert len(backbone.out_features) == len(backbone.out_indices)

    def test_unknown_config_keys_are_kept_not_rejected(self):
        """A future weights release may add fields; that must not be fatal.

        PP-OCRv6_medium_det already ships four such keys (``mode``,
        ``use_lab``, ``class_num``, ``class_expand``), which is why this is a
        property worth pinning rather than a hypothetical.
        """
        path = WEIGHTS_ROOT / "PP-OCRv6_medium_det_safetensors"
        if not (path / "config.json").is_file():
            pytest.skip("PP-OCRv6_medium_det is not downloaded")

        from fox_reader.hf.models.auto import AutoConfig

        config = AutoConfig.from_pretrained(path)
        assert config.mode == "large"
        assert config.class_num == 1000

    def test_a_config_built_from_scratch_still_validates(self):
        """Construction with no JSON at all works -- the defaults are coherent.

        Needs no weights, so this is the part of the round-trip that still
        guards hf-hub's ``strict`` on a bare checkout. Four stages because
        `PPLCNetV4Config.validate_architecture` insists on it, which is itself
        worth knowing: hf-hub's *class* validators run through the shim, not
        just its per-field type checks.
        """
        from fox_reader.hf.models.pp_lcnet_v4 import PPLCNetV4Config

        config = PPLCNetV4Config(
            stem_channels=[3, 16, 32],
            block_configs=[
                [[3, 32, 64, 1, False]],
                [[3, 64, 128, 2, False]],
                [[3, 128, 256, 2, False]],
                [[3, 256, 512, 2, False]],
            ],
        )
        assert config.stage_names == ["stem", "stage1", "stage2", "stage3", "stage4"]
        # Neither out_features nor out_indices given: the mixin must derive
        # both, and agree with itself.
        assert config.out_features == ["stage4"]
        assert config.out_indices == [4]

    def test_a_class_validator_still_rejects_a_bad_architecture(self):
        from fox_reader.hf.models.pp_lcnet_v4 import PPLCNetV4Config

        with pytest.raises(Exception, match="block_configs"):
            PPLCNetV4Config(stem_channels=[3, 16, 32], block_configs=[[[3, 32, 64, 1, False]]])

    def test_a_bad_out_feature_is_rejected(self):
        from fox_reader.hf.models.pp_lcnet_v4 import PPLCNetV4Config

        with pytest.raises(ValueError, match="out_features"):
            PPLCNetV4Config(
                stem_channels=[3, 16, 32],
                block_configs=[
                    [[3, 32, 64, 1, False]],
                    [[3, 64, 128, 2, False]],
                    [[3, 128, 256, 2, False]],
                    [[3, 256, 512, 2, False]],
                ],
                out_features=["stage7"],
            )
