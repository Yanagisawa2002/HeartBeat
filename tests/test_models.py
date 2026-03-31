import unittest

import torch

from src.comparison_models import (
    BENCHMARK_MODEL_NAMES,
    SEQUENCE_FIRST_MODEL_NAMES,
    create_comparison_model,
)


class TestModelInstantiation(unittest.TestCase):
    def test_supported_models_instantiate_and_run_forward(self) -> None:
        batch_size = 2
        input_dim = 12
        seq_len = 64
        model_names = BENCHMARK_MODEL_NAMES

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                model = create_comparison_model(
                    model_name=model_name,
                    input_dim=input_dim,
                    seq_len=seq_len,
                    num_classes=2,
                )
                model.eval()

                if model_name in SEQUENCE_FIRST_MODEL_NAMES:
                    sample = torch.randn(batch_size, seq_len, input_dim)
                else:
                    sample = torch.randn(batch_size, input_dim, seq_len)

                with torch.no_grad():
                    output = model(sample)

                self.assertEqual(tuple(output.shape), (batch_size, 2))
                self.assertGreater(sum(p.numel() for p in model.parameters()), 0)
