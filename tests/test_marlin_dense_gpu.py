"""GPU test: the SM70 dense W4A16 repack must leave the empty pinned kernel.

The pinned ``gptq_marlin_repack`` is compiled to ``return;`` below SM80, so its
output stays zero. marlin_v100's repack writes real values. The difference
between the two outputs on the same input is the whole point of the redirect,
and it is what a W4A16 checkpoint would have silently loaded as zeros.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

K, N, BITS = 128, 64, 4


def _packed_weights(device):
    pack = 32 // BITS
    code = torch.randint(1, 15, (K, N), dtype=torch.int32, device=device)
    qweight = torch.zeros((K // pack, N), dtype=torch.int32, device=device)
    for k in range(K):
        qweight[k // pack, :] |= code[k] << (BITS * (k % pack))
    return qweight


@pytest.fixture(scope="module")
def marlin():
    module = pytest.importorskip("sglang_restomod.quant.sm70_fp4_moe.marlin_dense")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
        pytest.skip("SM70 is required")
    if not module.install():
        pytest.skip("the wiring is already installed by another test")
    return module


def test_repack_is_no_longer_a_stub(marlin):
    device = torch.device("cuda")
    qweight = _packed_weights(device)
    perm = torch.arange(N, dtype=torch.int32, device=device)

    routed = marlin.gptq_marlin_repack(qweight, perm, K, N, BITS)
    pinned = marlin._ORIGINALS["repack"](qweight, perm, K, N, BITS)

    assert int((routed != 0).sum()) > 0, "the routed repack produced nothing"
    assert int((pinned != 0).sum()) == 0, (
        "the pinned repack is expected to be an SM80+ stub; if this fails the "
        "upstream kernel grew an SM70 path and the redirect can be retired"
    )
    assert not torch.equal(routed, pinned)
