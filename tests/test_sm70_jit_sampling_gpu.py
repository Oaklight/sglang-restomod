"""GPU test for the speculative tree-sampling JIT kernel on SM70.

The pinned ``flashinfer/sampling.cuh`` needs ``cuda::maximum`` and
``cuda::minimum``, which only exist in CCCL 3.x (CUDA 13). Building the kernel
under the CUDA 12.8 image fails before compilation starts unless the component
puts ``sglang_restomod/spec/cccl`` on the JIT include path. This test builds and
runs the real kernel, so a toolkit or pin change that breaks the shim shows up
as a failure rather than as a crash at the first verify step.

Marked ``gpu`` and excluded from ``restomod-ci``; run it with
``scripts/test_gpu.sh``.
"""

import pytest

pytestmark = pytest.mark.gpu


def test_tree_sampling_jit_builds_and_runs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    sampling = pytest.importorskip("sglang.kernels.ops.speculative.sampling")

    from sglang_restomod.spec.sm70_sampling import install_jit_cccl_compat

    assert install_jit_cccl_compat()

    device = "cuda"
    batch, drafts, spec, vocab = 2, 2, 2, 16
    candidates = torch.zeros((batch, drafts), dtype=torch.int64, device=device)
    retrive_index = (
        torch.arange(drafts, device=device, dtype=torch.int64).expand(batch, -1).contiguous()
    )
    retrive_next_token = torch.full((batch, drafts), -1, dtype=torch.int64, device=device)
    retrive_next_sibling = torch.full((batch, drafts), -1, dtype=torch.int64, device=device)
    predicts = torch.full((batch * drafts,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((batch, spec), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((batch,), dtype=torch.int32, device=device)
    uniform = torch.rand((batch, drafts), dtype=torch.float32, device=device)
    uniform_final = torch.rand((batch,), dtype=torch.float32, device=device)

    # One-hot target distribution on token 0 with an all-zero draft: the bonus
    # token must come back as 0, whatever the tree walk does with the drafts.
    target_probs = torch.zeros((batch, drafts, vocab), dtype=torch.float32, device=device)
    target_probs[:, :, 0] = 1.0
    draft_probs = torch.zeros_like(target_probs)

    sampling.tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform,
        uniform_samples_for_final_sampling=uniform_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    torch.cuda.synchronize()

    lengths = accept_token_num.tolist()
    assert all(0 <= length < spec for length in lengths), lengths
    rows = torch.arange(batch, device=device)
    bonus_positions = accept_index[rows, accept_token_num.to(torch.int64)].to(torch.int64)
    assert predicts[bonus_positions].tolist() == [0] * batch


def test_pin_default_include_has_no_cuda_functional():
    """The shim has to be the first <cuda/functional> the JIT build sees.

    ``load_jit`` builds ``DEFAULT_INCLUDE + extra_include_paths``, so a copy
    of the header inside the pin's own include directory would be found before
    the shim, and the build would fail with the original
    "namespace cuda has no member maximum". This holds on a real install
    rather than on the stub module the CPU tests use.
    """
    import os

    compile_module = pytest.importorskip("sglang.kernels.jit.utils.compile")
    for entry in compile_module.DEFAULT_INCLUDE:
        header = os.path.join(entry, "cuda", "functional")
        assert not os.path.exists(header), f"the pin's include path already ships {header}"
