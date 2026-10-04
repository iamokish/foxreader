"""Tests for :mod:`fox_reader.clean.unetplusplus`, the written-out UNet++.

This module exists to let the app load the Manga-Text-Segmentation-2025
checkpoint without ``segmentation-models-pytorch`` and ``timm`` installed. That
only works while it stays a *replica*: ``load_state_dict(strict=True)`` matches
parameters by name, so a tensor that is renamed, resized, reordered or dropped
does not degrade the masks -- it refuses to load at all, on a user's machine,
for a checkpoint they have already downloaded.

So the architecture is pinned three ways here, in decreasing order of strength:

* against ``segmentation_models_pytorch`` itself, bit for bit, when it happens
  to be installed (:class:`TestAgainstThePackage`). This is the real check and
  the one that was run when the module was written; it is skipped in the
  ordinary environment, which is exactly the environment that no longer has the
  package.
* against a digest of the reference state dict -- every key with its shape, in
  order -- recorded from smp 0.5.0 / timm 1.0.28 when the two agreed bit for
  bit. This needs nothing installed and catches everything the first check would,
  short of arithmetic.
* against the handful of numbers a human can check by eye, so that a digest
  mismatch says *where*: the parameter count, the feature widths, the decoder's
  block names.

The digest is deliberately not recomputed from the module under test. Its
authority comes from having been read off the package.
"""
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402 - only importable once torch is known present

from fox_reader.clean.unetplusplus import (  # noqa: E402
    ENCODER_NAME,
    UnetPlusPlus,
)

#: The configuration the checkpoint was trained at. Anything else is a different
#: set of keys, so there is only one of these to test.
TEXT_SEG_KWARGS = dict(
    encoder_name=ENCODER_NAME,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
    decoder_attention_type="scse",
)

#: sha256 over ``"<key> <shape>"`` for every state-dict entry, in state-dict
#: order, of
#: ``smp.UnetPlusPlus(**TEXT_SEG_KWARGS)`` under segmentation-models-pytorch
#: 0.5.0 and timm 1.0.28.
REFERENCE_DIGEST = (
    "11c9c82400b0430b48d72744c9e4a0fe957a5cc63c079b03667bc658f2f6c198")
REFERENCE_TENSORS = 1462
REFERENCE_PARAMS = 53_787_485

#: What the encoder hands the decoder, including the leading input image that
#: smp's wrapper prepends and the decoder then drops.
REFERENCE_OUT_CHANNELS = [3, 32, 56, 80, 192, 328]

#: UNet++'s ladder: eleven blocks, ``x_<depth>_<layer>``, in construction order.
REFERENCE_BLOCK_NAMES = [
    "x_0_0",
    "x_0_1", "x_1_1",
    "x_0_2", "x_1_2", "x_2_2",
    "x_0_3", "x_1_3", "x_2_3", "x_3_3",
    "x_0_4",
]


def _manifest(model):
    """``"<key> <shape>"`` per state-dict entry, in order -- the digest input."""
    return "\n".join(f"{k} {tuple(v.shape)}" for k, v in
                     model.state_dict().items())


@pytest.fixture(scope="module")
def model():
    """One model for the whole file: 54M parameters is not free to build."""
    return UnetPlusPlus(**TEXT_SEG_KWARGS)


class TestStateDictMatchesTheCheckpoint:
    """The names and shapes ``strict=True`` will be matching against."""

    def test_the_manifest_digest_is_unchanged(self, model):
        # A one-line failure with no detail, on purpose: the tests below are
        # what localise it, and a 1462-line expected value in the source would
        # be read by nobody.
        assert hashlib.sha256(
            _manifest(model).encode()).hexdigest() == REFERENCE_DIGEST

    def test_the_tensor_and_parameter_counts_are_unchanged(self, model):
        assert len(model.state_dict()) == REFERENCE_TENSORS
        assert sum(p.numel() for p in model.parameters()) == REFERENCE_PARAMS

    def test_every_key_is_prefixed_the_way_the_checkpoint_expects(self, model):
        # smp's own three top-level names, and timm's `model.` hop inside the
        # encoder. Nothing may sit outside them: a key that does is a key the
        # checkpoint has no value for.
        prefixes = ("encoder.model.", "decoder.blocks.", "segmentation_head.")
        stray = [k for k in model.state_dict()
                 if not k.startswith(prefixes)]
        assert stray == []

    def test_the_decoder_norm_keys_survive_the_groupnorm_swap(self, model):
        # `seg._build_model` replaces the decoder's BatchNorms with GroupNorms
        # before loading, because that is what the checkpoint carries. The swap
        # drops the three running-statistics buffers per norm and keeps weight
        # and bias, so the count has to land exactly here or the real load
        # fails on a machine, not in CI.
        from fox_reader.clean import seg

        swapped = UnetPlusPlus(**TEXT_SEG_KWARGS)
        seg._convert_batchnorm_to_groupnorm(swapped.decoder, nn)

        norms = [m for m in swapped.decoder.modules()
                 if isinstance(m, nn.GroupNorm)]
        assert len(norms) == 22
        assert not [m for m in swapped.decoder.modules()
                    if isinstance(m, nn.BatchNorm2d)]
        # 22 norms x 3 buffers dropped.
        assert len(swapped.state_dict()) == REFERENCE_TENSORS - 66

    def test_the_encoder_keeps_its_batchnorms(self, model):
        # The swap above walks the decoder only. The encoder's norm-act layers
        # are BatchNorm2d subclasses, so pointing it at the whole model would
        # take these too -- and their activations with them.
        found = [m for m in model.encoder.modules()
                 if isinstance(m, nn.BatchNorm2d)]
        assert len(found) == 168


class TestShape:
    """Widths and resolutions, which is where a transcription error lands."""

    def test_encoder_out_channels(self, model):
        assert model.encoder.out_channels == REFERENCE_OUT_CHANNELS

    def test_decoder_block_names_and_order(self, model):
        assert list(model.decoder.blocks.keys()) == REFERENCE_BLOCK_NAMES

    def test_decoder_channel_bookkeeping(self, model):
        # Derived in the decoder from the encoder widths; spelled out here so a
        # change to either side has to be deliberate.
        assert model.decoder.in_channels == [328, 256, 128, 64, 32]
        assert model.decoder.skip_channels == [192, 80, 56, 32, 0]
        assert model.decoder.out_channels == [256, 128, 64, 32, 16]

    def test_the_features_halve_at_every_stage(self, model):
        model.eval()
        x = torch.zeros(1, 3, 64, 96)

        with torch.inference_mode():
            features = model.encoder(x)

        # Six maps for five stages: the first is the input image itself.
        assert [tuple(f.shape) for f in features] == [
            (1, 3, 64, 96),
            (1, 32, 32, 48),
            (1, 56, 16, 24),
            (1, 80, 8, 12),
            (1, 192, 4, 6),
            (1, 328, 2, 3),
        ]

    def test_the_mask_comes_back_at_the_input_size(self, model):
        model.eval()

        with torch.inference_mode():
            out = model(torch.zeros(1, 3, 64, 96))

        assert tuple(out.shape) == (1, 1, 64, 96)

    def test_logits_not_probabilities(self, model):
        # `seg._forward` applies the sigmoid, under inference_mode, on the one
        # tensor that needs it. A model that had already done so would be
        # squashed twice and every mask would come out flat.
        model.eval()
        torch.manual_seed(0)

        with torch.inference_mode():
            out = model(torch.randn(1, 3, 64, 64))

        assert (out < 0).any() or (out > 1).any()


class TestUnsupportedConfigurations:
    """Refusing beats building something the checkpoint does not fit."""

    def test_a_ragged_input_says_which_size_to_pad_to(self, model):
        # Each decoder rung doubles the resolution, so an odd side fails deep
        # inside a concatenation with a shape error about numbers the caller
        # never chose.
        with pytest.raises(RuntimeError, match=r"divisible by 32"):
            model(torch.zeros(1, 3, 60, 96))

    def test_another_encoder_is_refused(self):
        with pytest.raises(ValueError, match=ENCODER_NAME):
            UnetPlusPlus(**{**TEXT_SEG_KWARGS, "encoder_name": "resnet34"})

    def test_pretrained_weights_are_refused(self):
        # There is no download here, and silently ignoring the argument would
        # hand back a randomly-initialised net that looks configured.
        with pytest.raises(ValueError, match="encoder_weights"):
            UnetPlusPlus(**{**TEXT_SEG_KWARGS, "encoder_weights": "imagenet"})

    def test_a_baked_in_activation_is_refused(self):
        with pytest.raises(ValueError, match="activation"):
            UnetPlusPlus(**{**TEXT_SEG_KWARGS, "activation": "sigmoid"})

    def test_an_unknown_attention_is_refused(self):
        with pytest.raises(ValueError, match="not implemented"):
            UnetPlusPlus(
                **{**TEXT_SEG_KWARGS, "decoder_attention_type": "cbam"})


class TestAgainstThePackage:
    """The real equivalence check, when smp is installed to check against.

    Skipped in the environment this module was written for -- which is the
    point of the module -- so this is the test to reach for after touching any
    of the block bodies: install the packages once, run it, remove them again.
    """

    @staticmethod
    def _reference():
        smp = pytest.importorskip("segmentation_models_pytorch")
        pytest.importorskip("timm")
        return smp.UnetPlusPlus(**TEXT_SEG_KWARGS)

    def test_the_manifests_are_identical(self):
        ref = self._reference()

        assert _manifest(UnetPlusPlus(**TEXT_SEG_KWARGS)) == _manifest(ref)

    def test_the_forward_pass_is_bitwise_identical(self):
        ref = self._reference()
        mine = UnetPlusPlus(**TEXT_SEG_KWARGS)
        # Randomise the running statistics too: left at their defaults the
        # norms are near enough to identity that a wrongly-wired one still
        # produces plausible numbers.
        state = ref.state_dict()
        torch.manual_seed(0)
        for key, value in state.items():
            if not value.dtype.is_floating_point:
                continue
            noise = torch.randn_like(value) * 0.1
            state[key] = (noise.abs() + 0.2 if key.endswith("running_var")
                          else noise)
        ref.load_state_dict(state, strict=True)
        mine.load_state_dict(state, strict=True)
        ref.eval()
        mine.eval()
        x = torch.randn(1, 3, 128, 160)

        with torch.inference_mode():
            assert torch.equal(ref(x), mine(x))

    def test_it_is_still_identical_through_the_real_clean_path(self):
        # GroupNorm decoder, channels_last layout, `seg._forward`'s padding --
        # the combination an actual clean runs, none of which the plain forward
        # pass above exercises.
        from fox_reader.clean import seg

        ref = self._reference()
        mine = UnetPlusPlus(**TEXT_SEG_KWARGS)
        seg._convert_batchnorm_to_groupnorm(ref.decoder, nn)
        seg._convert_batchnorm_to_groupnorm(mine.decoder, nn)
        state = ref.state_dict()
        mine.load_state_dict(state, strict=True)
        for net in (ref, mine):
            net.eval()
            net.to(memory_format=torch.channels_last)
        rgb = (np.random.default_rng(3).random((97, 131, 3)) * 255).astype(
            np.uint8)
        tensor = seg._to_tensor(rgb, True)

        assert torch.equal(seg._forward(ref, tensor, True),
                           seg._forward(mine, tensor, True))
