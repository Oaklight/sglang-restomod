"""`scripts/check_site.py` must classify a site before a run uses it.

The check exists because three measurement sessions were invalidated by site
state: a missing `0002` (NCCL fallback), a missing tuned MoE config (default
heuristic), and a half-patched site nobody noticed. Each is a state this test
constructs directly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_site.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_site", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


check_site = _module()


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _site(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A minimal site with one patch target, and a manifest that pins its hashes."""
    target = tmp_path / "site" / "python" / "sglang" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("UPSTREAM = 1\n")
    unpatched = _sha(target)
    target.write_text("PATCHED = 1\n")
    patched = _sha(target)
    target.write_text("UPSTREAM = 1\n")

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "upstream": "0" * 40,
                "patches": {
                    "0009-example.patch": {
                        "paths": ["python/sglang/module.py"],
                        "site_hashes": {
                            "python/sglang/module.py": {
                                "unpatched": unpatched,
                                "patched": patched,
                            }
                        },
                        "reason": "test",
                        "source": "test",
                        "upstream_status": "pending",
                    }
                },
            }
        )
    )
    return tmp_path / "site", manifest


def test_the_three_patch_states_and_a_missing_file(tmp_path):
    site, manifest = _site(tmp_path)
    hashes = json.loads(manifest.read_text())["patches"]["0009-example.patch"][
        "site_hashes"
    ]["python/sglang/module.py"]

    assert check_site.patch_state(site, "python/sglang/module.py", hashes)[0] == "ABSENT"
    (site / "python" / "sglang" / "module.py").write_text("PATCHED = 1\n")
    assert check_site.patch_state(site, "python/sglang/module.py", hashes)[0] == "PRESENT"
    (site / "python" / "sglang" / "module.py").write_text("EDITED = 1\n")
    state, actual = check_site.patch_state(site, "python/sglang/module.py", hashes)
    assert state == "MISMATCH"
    assert actual == _sha(site / "python" / "sglang" / "module.py")
    assert check_site.patch_state(site, "python/sglang/absent.py", hashes)[0] == "MISSING"


def test_the_command_fails_on_an_unpatched_site_and_passes_on_a_patched_one(tmp_path):
    site, manifest = _site(tmp_path)
    assert check_site.main([str(site), "--manifest", str(manifest)]) == 1

    (site / "python" / "sglang" / "module.py").write_text("PATCHED = 1\n")
    assert check_site.main([str(site), "--manifest", str(manifest)]) == 0


def test_tuned_configs_are_reported_per_version_directory(tmp_path):
    site, _ = _site(tmp_path)
    config_dir = (
        site
        / "python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1"
    )
    config_dir.mkdir(parents=True)
    (config_dir / "E=512,N=160,device_name=Tesla_V100-SXM2-32GB.json").write_text("{}")

    assert ("triton_3_5_1", "E=512,N=160,device_name=Tesla_V100-SXM2-32GB.json") in (
        check_site.tuned_configs(site)
    )


def test_the_tree_hash_ignores_pycache(tmp_path):
    site, _ = _site(tmp_path)
    before = check_site.tree_hash(site)
    (site / "python" / "sglang" / "__pycache__").mkdir()
    (site / "python" / "sglang" / "__pycache__" / "module.pyc").write_text("junk")
    assert check_site.tree_hash(site) == before


def test_an_installed_site_reports_its_version(tmp_path):
    site, _ = _site(tmp_path)
    metadata = site / "lib" / "python3.12" / "site-packages" / "sglang-0.5.23.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: sglang\nVersion: 0.5.23\n")

    version, source = check_site.site_version(site)
    assert version == "0.5.23"
    assert "dist-info" in source


def test_a_missing_directory_is_a_usage_error(tmp_path):
    assert check_site.main([str(tmp_path / "nope")]) == 2


def test_the_manifest_records_a_hash_pair_for_every_patch_target():
    """A target without `site_hashes` would pass the check by not being checked."""
    manifest = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "patches" / "manifest.json").read_text()
    )
    for name, entry in manifest["patches"].items():
        recorded = set(entry.get("site_hashes") or {})
        assert recorded == set(entry["paths"]), (
            f"{name}: {sorted(recorded)} != {sorted(entry['paths'])}"
        )
        for hashes in entry["site_hashes"].values():
            assert set(hashes) == {"unpatched", "patched"}
            assert all(len(value) == 64 for value in hashes.values())
