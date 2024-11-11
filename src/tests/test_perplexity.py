import unittest
import torch
from perpelexity import Perplexity


class TestPerplexityCalculator(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        return super().setUp()
    
    def test_calculate_perpexity(self):
        hf_model_id = "openai-community/gpt2-large"
        stride = 512
        device = self.device

        expected_perplexity = 16.45410919189453
        calculated_perplexity = Perplexity(hf_model_id, stride, device).calculate_perplexity()
        delta = 3.0

        self.assertAlmostEqual(calculated_perplexity, expected_perplexity, None, "", delta)
        self.assertEqual(calculated_perplexity, expected_perplexity)


if __name__ == "__main__":
    unittest.main()
