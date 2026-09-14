"""Storage inventory for calibration, outside the timed generation interval.

Count backing storages, not tensor views: InfLLM's CPU history and representative
prefixes are shared by snapshots. CUDA allocator slack/driver memory are separate.
"""
import inspect
import torch


def storage_inventory(roots):
    objects, storages, rows = set(), {}, {}

    def visit(value, category):
        if id(value) in objects:
            return
        objects.add(id(value))
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key in storages or not storage.nbytes():
                return
            storages[key] = category
            label = category + ":" + str(value.device)
            row = rows.setdefault(label, {"storage_bytes": 0, "pinned_bytes": 0, "storages": 0})
            row["storage_bytes"] += storage.nbytes()
            row["storages"] += 1
            if value.device.type == "cpu" and value.is_pinned():
                row["pinned_bytes"] += storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, category)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, category)
        elif (not inspect.isclass(value) and not inspect.isroutine(value)
              and hasattr(value, "__dict__")):
            # Roots must be cache objects, never a model or optimizer. RoPE
            # modules under cache managers contain only their position buffers.
            for item in vars(value).values():
                visit(item, category)

    for category, root in roots.items():
        visit(root, category)
    return {"categories": rows,
            "total_storage_bytes": sum(row["storage_bytes"] for row in rows.values()),
            "shared_storage_attribution": "first root encountered; each backing storage counted once"}


def session_memory(session):
    return storage_inventory({"live_cache": session.cache, "checkpoints": session.checkpoints})
