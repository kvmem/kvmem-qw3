import unittest
import torch
from kv_baselines.memory import storage_inventory


class StorageInventoryTest(unittest.TestCase):
    def test_views_and_shared_history_count_once(self):
        tensor = torch.empty(100, dtype=torch.float16)
        copied_state = torch.empty(40, dtype=torch.float32)
        cyclic = []
        cyclic.append(cyclic)
        result = storage_inventory({"live": [tensor, tensor[:2], cyclic],
                                    "snapshot": [tensor, copied_state]})
        self.assertEqual(result["total_storage_bytes"], 360)
        self.assertEqual(result["categories"]["live:cpu"]["storage_bytes"], 200)
        self.assertEqual(result["categories"]["snapshot:cpu"]["storage_bytes"], 160)


if __name__ == "__main__":
    unittest.main()
