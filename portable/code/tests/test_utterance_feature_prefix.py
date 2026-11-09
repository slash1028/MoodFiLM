"""CPU contract tests for the simple per-utterance prefix experiment."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utterance_feature_prefix import UtteranceFeaturePrefix, prepend_masked_prefix
from data_utils.data_utils import Seq2SeqCollator


class UtteranceFeaturePrefixTest(unittest.TestCase):
    def test_empty_rank_keeps_projection_gradients_with_masked_padding(self):
        module = UtteranceFeaturePrefix(4, 4, 8, True, True)
        empty = torch.zeros(1, 0, 4)
        empty_mask = torch.zeros(1, 0, dtype=torch.bool)
        prefix, mask = module(empty, empty, empty_mask, empty_mask)
        self.assertEqual(tuple(prefix.shape), (1, 1, 8))
        self.assertEqual(int(mask.sum()), 0)
        prefix.sum().backward()
        self.assertIsNotNone(module.text_projection.weight.grad)
        self.assertIsNotNone(module.audio_projection.weight.grad)
        self.assertEqual(float(module.text_projection.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(module.audio_projection.weight.grad.abs().sum()), 0.0)

    def test_context_window_selection_is_history_plus_target(self):
        collator = Seq2SeqCollator.__new__(Seq2SeqCollator)
        collator.utterance_feature_enable = True
        collator.iemocap_num_turns_by_conv = {'Ses01F_impro01': 40}
        for history in (0, 6, 12, 18):
            collator.utterance_feature_history_window = history
            ids = collator._effective_qwen_turn_sample_ids({
                'sample_id': 'Ses01F_impro01_20', 'meta': {}
            })
            self.assertEqual(len(ids), history + 1)
            self.assertEqual(ids[-1], 'Ses01F_impro01_20')

    def test_dynamic_windows_missing_audio_order_and_padding(self):
        for history in (0, 6, 12, 18):
            turns = history + 1
            module = UtteranceFeaturePrefix(4, 4, 4, True, True)
            text = torch.arange(2 * turns * 4, dtype=torch.float32).reshape(2, turns, 4)
            audio = text + 1000
            text_mask = torch.zeros(2, turns, dtype=torch.bool)
            audio_mask = torch.zeros(2, turns, dtype=torch.bool)
            text_mask[0, :turns] = True
            text_mask[1, :1] = True
            audio_mask[0, :turns] = True
            audio_mask[0, 0] = False  # missing audio: no fake valid vector
            audio_mask[1, :1] = True
            prefix, mask = module(text, audio, text_mask, audio_mask)
            self.assertEqual(int(mask[0].sum()), 2 * turns - 1)
            self.assertEqual(int(mask[1].sum()), 2)
            self.assertTrue(torch.equal(prefix[0, 0], text[0, 0]))
            if turns > 1:
                self.assertTrue(torch.equal(prefix[0, 1], text[0, 1]))
                self.assertTrue(torch.equal(prefix[0, 2], audio[0, 1]))
            self.assertTrue(torch.equal(mask[1, 2:], torch.zeros_like(mask[1, 2:])))

    def test_prepend_loss_gradient_reload_and_inference_shape(self):
        module = UtteranceFeaturePrefix(4, 4, 8, True, True)
        token_embedding = torch.nn.Embedding(32, 8)
        output_head = torch.nn.Linear(8, 32)
        text = torch.randn(2, 3, 4)
        audio = torch.randn(2, 3, 4)
        text_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
        audio_mask = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.bool)
        prefix, mask = module(text, audio, text_mask, audio_mask)
        ids = torch.tensor([[3, 4, 5], [0, 3, 4]])
        labels = torch.tensor([[-100, -100, 5], [-100, -100, 4]])
        batch = prepend_masked_prefix({
            'input_ids': ids, 'inputs_embeds': token_embedding(ids),
            'attention_mask': torch.tensor([[1, 1, 1], [0, 1, 1]]),
            'labels': labels,
        }, prefix, mask, 0)
        self.assertEqual(batch['input_ids'].shape[1], ids.shape[1] + prefix.shape[1])
        self.assertTrue(torch.equal(batch['labels'][:, :prefix.shape[1]], torch.full_like(batch['labels'][:, :prefix.shape[1]], -100)))
        self.assertTrue(torch.equal(batch['attention_mask'][1, :prefix.shape[1]], mask[1]))
        logits = output_head(batch['inputs_embeds'])
        target = batch['labels'][:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32), target.reshape(-1), ignore_index=-100)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        # A classifier on the last prefix position supplies a direct gradient
        # check independent of the toy token-by-token loss above.
        module.zero_grad(set_to_none=True)
        prefix2, _ = module(text, audio, text_mask, audio_mask)
        prefix2.sum().backward()
        self.assertGreater(float(module.text_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(module.audio_projection.weight.grad.abs().sum()), 0)
        restored = UtteranceFeaturePrefix(4, 4, 8, True, True)
        restored.load_state_dict(module.state_dict())
        with torch.no_grad():
            reloaded, _ = restored(text, audio, text_mask, audio_mask)
            shuffled, _ = restored(text, torch.roll(audio, 1, dims=1), text_mask, audio_mask)
            zeroed, _ = restored(text, torch.zeros_like(audio), text_mask, audio_mask)
        self.assertTrue(torch.allclose(prefix2, reloaded))
        self.assertFalse(torch.allclose(reloaded, shuffled))
        self.assertFalse(torch.allclose(reloaded, zeroed))
        # Generation begins with the same prefixed attention/embedding length.
        generation = prepend_masked_prefix({
            'input_ids': ids, 'inputs_embeds': token_embedding(ids),
            'attention_mask': torch.ones_like(ids),
        }, prefix2.detach(), mask, 0)
        self.assertEqual(generation['attention_mask'].shape[1], generation['inputs_embeds'].shape[1])


if __name__ == '__main__':
    unittest.main()
