"""Shared bounded transport capacity for retained visual model histories."""

# A 1M-token desktop history can contain several hundred full screenshots.
# This is a wire limit; the engine still enforces its separate context and KV
# selection budgets. The authenticated relay and local Responses bridge must
# accept the same bound so they do not fail at different layers.
MAX_MODEL_REQUEST_BYTES = 512 * 1024 * 1024


def model_request_limit(value):
    if type(value) is not int or not 1 <= value <= MAX_MODEL_REQUEST_BYTES:
        raise ValueError('Model request limit must be an integer within 1..512 MiB')
    return value
