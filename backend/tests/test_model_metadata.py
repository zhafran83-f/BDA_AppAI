"""Metadata tests use in-memory hashes and fake bundles, never AI artifacts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

from backend.app.core import model_metadata as metadata
from backend.app.core.model_metadata import (
    ARTIFACT_KEYS, BASELINE_MANIFEST, ArtifactSpec, BundleSnapshot, BundleSpec,
    VerifiedBundleProvider,
)


def setup_provider(**changes):
    manifest = {
        module: BundleSpec("dummy-v1", "dummy-clean-v1", tuple(
            ArtifactSpec(key, module + "/" + key + ".bin", "a" * 64)
            for key in keys
        ))
        for module, keys in ARTIFACT_KEYS.items()
    }
    loader = Mock()
    bundles = {name: object() for name in ARTIFACT_KEYS}
    loader.fashion.return_value = bundles["fashion"]
    loader.review.return_value = bundles["reviews"]
    loader.purchase.return_value = bundles["purchase"]
    factory = Mock(return_value=loader)
    hasher = Mock(return_value="a" * 64)
    options = dict(loader_factory=factory, hasher=hasher, manifest=manifest)
    options.update(changes)
    provider = VerifiedBundleProvider(Path.cwd() / "dummy-model-root", **options)
    return provider, loader, factory, hasher, bundles


@pytest.fixture(autouse=True)
def forbid_real_artifact_access(monkeypatch):
    monkeypatch.setattr(metadata, "_hash_file", Mock(
        side_effect=AssertionError("Real model hashing forbidden in unit tests.")
    ))
    monkeypatch.setattr(metadata, "ModelLoader", Mock(
        side_effect=AssertionError("Real model loading forbidden in unit tests.")
    ))


def test_constructor_and_snapshot_lookup_do_not_load_or_hash():
    provider, loader, factory, hasher, bundles = setup_provider()
    assert provider.snapshot_for("fashion", bundles["fashion"]) is None
    assert provider.cached_bundle("fashion") is None
    factory.assert_not_called()
    hasher.assert_not_called()
    assert not loader.method_calls


@pytest.mark.parametrize("module,method", [
    ("fashion", "fashion"), ("reviews", "review"), ("purchase", "purchase"),
])
def test_exact_bundle_identity_and_no_hashing_on_repeated_requests(module, method):
    provider, loader, factory, hasher, bundles = setup_provider()
    assert getattr(provider, method)() is bundles[module]
    verified = provider.snapshot_for(module, bundles[module])
    assert verified.module == module
    assert tuple(key for key, _ in verified.artifact_sha256) == ARTIFACT_KEYS[module]
    assert hasher.call_count == 2 * len(ARTIFACT_KEYS[module])
    for _ in range(5):
        assert getattr(provider, method)() is bundles[module]
        assert provider.snapshot_for(module, bundles[module]) is verified
    assert provider.snapshot_for(module, object()) is None
    assert provider.snapshot_for("unsupported", bundles[module]) is None
    assert provider.cached_bundle(module) is bundles[module]
    assert hasher.call_count == 2 * len(ARTIFACT_KEYS[module])
    getattr(loader, method).assert_called_once_with()
    factory.assert_called_once()


@pytest.mark.parametrize("mode", ["wrong_hash", "changed_during_load", "read_error"])
def test_unverified_bundle_still_returns_without_rehashing(mode):
    hasher = Mock(return_value="b" * 64)
    if mode == "changed_during_load":
        hasher.side_effect = ["a" * 64, "b" * 64]
    elif mode == "read_error":
        hasher.side_effect = OSError("dummy-sensitive-path")
    provider, loader, _, _, bundles = setup_provider(hasher=hasher)
    assert provider.fashion() is bundles["fashion"]
    assert provider.snapshot_for("fashion", bundles["fashion"]) is None
    calls = hasher.call_count
    assert provider.fashion() is bundles["fashion"]
    assert hasher.call_count == calls
    loader.fashion.assert_called_once()


def test_loaded_snapshot_does_not_change_when_file_hash_changes_later():
    provider, _, _, hasher, bundles = setup_provider()
    provider.fashion()
    verified = provider.snapshot_for("fashion", bundles["fashion"])
    hasher.return_value = "b" * 64
    assert provider.fashion() is bundles["fashion"]
    assert provider.snapshot_for("fashion", bundles["fashion"]) is verified
    assert verified.artifact_sha256 == (("vgg16", "a" * 64),)
    assert hasher.call_count == 2


def test_failed_load_cannot_claim_verified_metadata_on_later_cached_file_state():
    provider, loader, _, hasher, bundles = setup_provider()
    primary = RuntimeError("dummy load failure")
    loader.fashion.side_effect = [primary, bundles["fashion"]]
    with pytest.raises(RuntimeError) as caught:
        provider.fashion()
    assert caught.value is primary
    assert provider.cached_bundle("fashion") is None
    calls = hasher.call_count
    assert provider.fashion() is bundles["fashion"]
    assert provider.snapshot_for("fashion", bundles["fashion"]) is None
    assert hasher.call_count == calls


def test_loading_concurrency_is_single_and_keeps_vgg_lazy():
    provider, loader, factory, hasher, bundles = setup_provider()
    entered, release = Event(), Event()

    def load():
        entered.set()
        assert release.wait(5)
        return bundles["reviews"]

    loader.review.side_effect = load
    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(provider.review)
        try:
            assert entered.wait(5)
            others = [executor.submit(provider.review) for _ in range(3)]
        finally:
            release.set()
        results = [first.result(timeout=5), *(f.result(timeout=5) for f in others)]
    assert all(bundle is bundles["reviews"] for bundle in results)
    loader.review.assert_called_once()
    loader.fashion.assert_not_called()
    assert hasher.call_count == 4
    factory.assert_called_once()


def test_snapshot_is_deeply_immutable():
    value = BundleSnapshot("fashion", "dummy", "dummy-clean", (("vgg16", "a" * 64),))
    with pytest.raises(FrozenInstanceError):
        value.model_version = "changed"
    with pytest.raises(TypeError):
        value.artifact_sha256[0] = ("vgg16", "b" * 64)


@pytest.mark.parametrize("hashes", [
    (("wrong", "a" * 64),), (("vgg16", "A" * 64),),
    (("vgg16", "short"),), [("vgg16", "a" * 64)],
])
def test_invalid_snapshot_is_rejected(hashes):
    with pytest.raises(ValueError):
        BundleSnapshot("fashion", "dummy", "dummy-clean", hashes)


def test_manifest_has_seven_real_baseline_hashes_without_reading_files():
    assert set(BASELINE_MANIFEST) == set(ARTIFACT_KEYS)
    artifacts = [artifact for spec in BASELINE_MANIFEST.values() for artifact in spec.artifacts]
    assert len(artifacts) == 7
    assert len({artifact.sha256 for artifact in artifacts}) == 7
    assert BASELINE_MANIFEST["fashion"].artifacts[0].sha256 == (
        "dd6edc3347458c34a7a3342108c570074d29577e13f5b8a17ada0ea73ffb0a54"
    )
    for module, spec in BASELINE_MANIFEST.items():
        BundleSnapshot(module, spec.model_version, spec.preprocessing_version,
                       tuple((a.key, a.sha256) for a in spec.artifacts))


def test_path_outside_model_root_is_not_hashed():
    manifest = {"fashion": BundleSpec("dummy", "dummy", (
        ArtifactSpec("vgg16", "../outside.bin", "a" * 64),
    ))}
    provider, _, _, hasher, bundles = setup_provider(manifest=manifest)
    assert provider.fashion() is bundles["fashion"]
    assert provider.snapshot_for("fashion", bundles["fashion"]) is None
    hasher.assert_not_called()
