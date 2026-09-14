#!/usr/bin/env python3
"""Append converted MTP tensors while preserving every base GGUF tensor byte."""
import argparse
import hashlib
import json
from pathlib import Path

import gguf


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for part in iter(lambda: data.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("mtp", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve() in (args.base.resolve(), args.mtp.resolve()):
        raise ValueError("output must be a separate file")
    base = gguf.GGUFReader(args.base)
    draft = gguf.GGUFReader(args.mtp)
    arch = base.get_field("general.architecture").contents()
    layers = base.get_field(f"{arch}.block_count").contents()
    for key in ("general.architecture", f"{arch}.embedding_length",
                f"{arch}.attention.head_count", f"{arch}.attention.head_count_kv",
                f"{arch}.rope.dimension_count"):
        if base.get_field(key).contents() != draft.get_field(key).contents():
            raise ValueError(f"MTP/base model shape mismatch: {key}")
    if base.get_field(f"{arch}.nextn_predict_layers"):
        raise ValueError("base already declares MTP layers")
    if draft.get_field(f"{arch}.block_count").contents() != layers + 1 or draft.get_field(f"{arch}.nextn_predict_layers").contents() != 1:
        raise ValueError("expected one appended MTP block")
    added = [t for t in draft.tensors if t.name.startswith(f"blk.{layers}.")]
    if len(added) != 15:
        raise ValueError(f"expected all 15 Qwen MTP tensors, found {len(added)}")
    staging = args.output.with_suffix(".partial")
    writer = gguf.GGUFWriter(staging, arch, endianess=base.endianess)
    for field in base.fields.values():
        if field.name.startswith("GGUF.") or field.name in ("general.architecture", f"{arch}.block_count"):
            continue
        kind = field.types[0]
        subtype = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), kind, sub_type=subtype)
    writer.add_block_count(layers + 1)
    writer.add_nextn_predict_layers(1)
    tensors = list(base.tensors) + added
    source_hashes = {}
    for tensor in tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
        source_hashes[tensor.name] = hashlib.sha256(tensor.data).hexdigest()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=base.endianess)
    writer.close()
    output = gguf.GGUFReader(staging)
    for tensor in output.tensors:
        if hashlib.sha256(tensor.data).hexdigest() != source_hashes[tensor.name]:
            raise RuntimeError(f"tensor byte preservation failed: {tensor.name}")
    staging.replace(args.output)
    report = dict(base=str(args.base), base_sha256=sha(args.base),
                  mtp=str(args.mtp), mtp_sha256=sha(args.mtp),
                  output=str(args.output), output_sha256=sha(args.output),
                  base_tensor_bytes_unchanged=True, added_tensor_names=[t.name for t in added],
                  tensor_sha256=source_hashes)
    args.output.with_suffix(".augmentation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tensor_sha256"}), flush=True)


if __name__ == "__main__":
    main()
