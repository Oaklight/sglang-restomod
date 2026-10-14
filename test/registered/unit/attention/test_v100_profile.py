import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a")


class TestV100Profile(CustomTestCase):
    def _view(self, **overrides):
        values = dict(
            attention_backend="v100_plus",
            dtype="float16",
            kv_cache_dtype="auto",
            enable_prefill_cp=False,
            dcp_size=1,
            dp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            max_running_requests=1,
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
            linear_attn_backend="triton",
            linear_attn_decode_backend=None,
            linear_attn_prefill_backend=None,
            linear_attn_verify_backend=None,
            mm_attention_backend=None,
            mamba_ssm_dtype="float16",
            mamba_radix_cache_strategy="no_buffer",
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_resolve_modes_without_mutating_input(self):
        from sglang_v100_plus.attention import resolve_profile

        view = self._view()
        before = vars(view).copy()
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"]
            )
        )
        original = Mock(side_effect=AssertionError("legacy resolution reached"))
        with patch(
            "sglang.srt.arg_groups.overrides.model_config_of", return_value=model
        ):
            decisions = resolve_profile(original, view)
        self.assertEqual(set(decisions.values()), {"v100_plus"})
        self.assertEqual(len(decisions), 7)
        self.assertEqual(vars(view), before)

    def test_conflicts_fail_before_startup(self):
        from sglang_v100_plus.attention import resolve_profile

        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"]
            )
        )
        for overrides in (
            {"kv_cache_dtype": "fp8_e4m3"},
            {"dtype": "bfloat16"},
            {"dsa_decode_backend": "triton"},
            {"mm_attention_backend": "sdpa"},
            {"linear_attn_verify_backend": "triton"},
            {"dcp_size": 2},
            {"attn_dp_size": 2},
            {"attn_cp_size": 2},
            {"max_running_requests": 2},
            {"mamba_ssm_dtype": "float32"},
            {"mamba_radix_cache_strategy": "extra_buffer_lazy"},
        ):
            with (
                self.subTest(overrides=overrides),
                patch(
                    "sglang.srt.arg_groups.overrides.model_config_of",
                    return_value=model,
                ),
            ):
                with self.assertRaises(ValueError):
                    resolve_profile(Mock(), self._view(**overrides))

    def test_paged_prefix_cache_keeps_every_native_mode(self):
        """The SM70 profile used to reject the state snapshots radix needs."""
        from sglang_v100_plus.attention import resolve_profile, supports_cache_snapshots

        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"]
            )
        )
        view = self._view(mamba_radix_cache_strategy="extra_buffer")
        with patch(
            "sglang.srt.arg_groups.overrides.model_config_of", return_value=model
        ):
            self.assertEqual(set(resolve_profile(Mock(), view).values()), {"v100_plus"})
        view.linear_attn_backend = "v100_plus"
        self.assertTrue(
            supports_cache_snapshots(Mock(return_value=False), view, model.hf_config)
        )
        model.hf_config.architectures = ["Qwen3NextForCausalLM"]
        self.assertFalse(
            supports_cache_snapshots(Mock(return_value=True), view, model.hf_config)
        )

    def test_other_profiles_keep_their_resolution(self):
        from sglang_v100_plus.attention import resolve_profile

        original = Mock(return_value={"sentinel": 1})
        view = self._view(attention_backend="qsa")
        self.assertEqual(resolve_profile(original, view), {"sentinel": 1})
        original.assert_called_once_with(view)

    def test_safe_gate_dispatch_uses_capability(self):
        from sglang_v100_plus.attention import V100KDAKernel

        from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher

        dispatcher = object.__new__(KDAKernelDispatcher)
        kernel = V100KDAKernel()
        dispatcher.decode_kernel = dispatcher.verify_kernel = kernel
        with patch.object(kernel, "decode", return_value="native"):
            self.assertEqual(
                dispatcher.decode(
                    None,
                    None,
                    None,
                    None,
                    None,
                    A_log=None,
                    dt_bias=None,
                    ssm_states=None,
                    cache_indices=None,
                    query_start_loc=None,
                    lower_bound=-5.0,
                ),
                "native",
            )
        with patch.object(kernel, "target_verify", return_value="native"):
            self.assertEqual(
                dispatcher.target_verify(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    ssm_states=None,
                    cache_indices=None,
                    query_start_loc=None,
                    intermediate_states_buffer=None,
                    intermediate_state_indices=None,
                    cache_steps=4,
                    retrieve_parent_token=None,
                    lower_bound=-5.0,
                ),
                "native",
            )
        kernel.supports_safe_gate_decode = False
        with self.assertRaises(NotImplementedError):
            dispatcher.decode(
                None,
                None,
                None,
                None,
                None,
                A_log=None,
                dt_bias=None,
                ssm_states=None,
                cache_indices=None,
                query_start_loc=None,
                lower_bound=-5.0,
            )

    def test_frozen_plugin_rounds_glm_verification_state(self):
        import torch
        from sglang_v100_plus.runtime import round_verify_state

        original = Mock(return_value="reference")
        snapshots = torch.empty(1, 4, 16, 128, 128, dtype=torch.float16)
        result = round_verify_state(
            original,
            intermediate_states_buffer=snapshots,
            HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=False,
            grid=(1, 32, 16),
            HV=16,
            H=16,
            B=1,
            K=128,
            V=128,
            T=4,
            IS_KDA=True,
            DISABLE_STATE_UPDATE=True,
            q=torch.empty(1, dtype=torch.float16),
        )
        self.assertEqual(result, "reference")
        self.assertTrue(original.call_args.kwargs["QUANTIZE_STATE_EACH_STEP"])


if __name__ == "__main__":
    unittest.main()
