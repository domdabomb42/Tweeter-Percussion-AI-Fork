"""Safe training synchronization and shared decoder regressions."""
import math
import unittest
from pathlib import Path
import tempfile

from percussion_generation_v3.model import consume_loss_group
from percussion_generation_v3.profiling import StageProfiler


class EfficiencyTests(unittest.TestCase):
    def test_grouped_metric_read_preserves_optimizer_update(self):
        import torch
        torch.manual_seed(7)
        baseline = torch.nn.Linear(2, 1)
        optimized = torch.nn.Linear(2, 1)
        optimized.load_state_dict(baseline.state_dict())
        inputs = [torch.tensor([[0.3, -0.5]]), torch.tensor([[0.8, 0.1]])]
        targets = [torch.tensor([[0.2]]), torch.tensor([[-0.4]])]
        old_optimizer = torch.optim.AdamW(baseline.parameters(), lr=2e-5)
        new_optimizer = torch.optim.AdamW(optimized.parameters(), lr=2e-5)
        old_weighted = 0.0
        for x, y, count in zip(inputs, targets, (3, 5)):
            loss = torch.nn.functional.mse_loss(baseline(x), y)
            (loss / 2).backward()
            old_weighted += float(loss.detach()) * count
        old_optimizer.step()
        pending = []
        for i, (x, y, count) in enumerate(zip(inputs, targets, (3, 5))):
            loss = torch.nn.functional.mse_loss(optimized(x), y)
            (loss / 2).backward()
            pending.append((loss.detach(), count,
                            {"source": f"s{i}", "target_track_id": i}))
        new_weighted, new_tokens = consume_loss_group(pending)
        new_optimizer.step()
        self.assertEqual(new_weighted, old_weighted)
        self.assertEqual(new_tokens, 8)
        self.assertFalse(pending)
        for old, new in zip(baseline.parameters(), optimized.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_nonfinite_group_stops_before_optimizer(self):
        import torch
        row = {"source": "bad", "target_track_id": 4}
        with self.assertRaisesRegex(ValueError, "bad track 4"):
            consume_loss_group([(torch.tensor(float("nan")), 2, row)])

    def test_optional_stage_profile_is_json_and_reports_backend(self):
        import json
        import torch
        model = torch.nn.Linear(2, 2)
        profiler = StageProfiler(torch.device("cpu"))
        with profiler.stage("prefix_forward"):
            model(torch.ones(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            report = profiler.save(path, model, examples=1,
                                   extra={"model_id": "synthetic-test"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
            self.assertEqual(report["stages"]["prefix_forward"]["calls"], 1)
            self.assertGreater(report["examples_per_second"], 0)


if __name__ == "__main__":
    unittest.main()
