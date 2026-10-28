"""Shrink the textures inside a .glb (binary glTF) so it is small enough for git.

Meshy exports a 4096x4096 texture (~21 MB) that the dashboard doesn't need.
This resizes every embedded image to at most --size pixels on its longest side
(PNG stays PNG) and rewrites the binary chunk; meshes and animations are copied
byte for byte.

    python animation/shrink_glb.py animation/avatar.glb            # in place, keeps avatar.original.glb
    python animation/shrink_glb.py in.glb --out small.glb --size 1024
"""

import argparse
import io
import json
import shutil
import struct
from pathlib import Path

from PIL import Image

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = b"glTF", b"JSON", b"BIN\x00"


def read_glb(path):
    data = Path(path).read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise SystemExit(f"{path}: not a glTF 2.0 binary file")
    chunks, pos = {}, 12
    while pos < len(data):
        length, kind = struct.unpack_from("<I4s", data, pos)
        chunks[kind] = data[pos + 8:pos + 8 + length]
        pos += 8 + length
    return json.loads(chunks[JSON_CHUNK]), chunks.get(BIN_CHUNK, b"")


def pad4(b, fill):
    return b + fill * (-len(b) % 4)


def write_glb(path, gltf, binary):
    js = pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    binary = pad4(binary, b"\x00")
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", GLB_MAGIC, 2, total))
        f.write(struct.pack("<I4s", len(js), JSON_CHUNK) + js)
        f.write(struct.pack("<I4s", len(binary), BIN_CHUNK) + binary)


def shrink(gltf, binary, size):
    views = gltf["bufferViews"]
    new_bytes = {}                      # bufferView index -> replacement bytes
    for img in gltf.get("images", []):
        if "bufferView" not in img:
            continue
        i = img["bufferView"]
        v = views[i]
        raw = binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
        im = Image.open(io.BytesIO(raw))
        before = im.size
        if max(im.size) > size:
            im.thumbnail((size, size), Image.LANCZOS)
        out = io.BytesIO()
        fmt = "JPEG" if img.get("mimeType") == "image/jpeg" else "PNG"
        im.save(out, fmt, **({"quality": 90} if fmt == "JPEG" else {"optimize": True}))
        new_bytes[i] = out.getvalue()
        print(f"  {img.get('name', i)}: {before[0]}x{before[1]} {len(raw) / 1e6:.1f} MB"
              f" -> {im.size[0]}x{im.size[1]} {len(new_bytes[i]) / 1e6:.1f} MB")

    # Rebuild the binary chunk view by view (4-byte aligned), in the original order.
    rebuilt = bytearray()
    for i in sorted(range(len(views)), key=lambda k: views[k].get("byteOffset", 0)):
        v = views[i]
        chunk = new_bytes.get(i)
        if chunk is None:
            start = v.get("byteOffset", 0)
            chunk = binary[start:start + v["byteLength"]]
        rebuilt += b"\x00" * (-len(rebuilt) % 4)
        v["byteOffset"] = len(rebuilt)
        v["byteLength"] = len(chunk)
        rebuilt += chunk
    gltf["buffers"][0]["byteLength"] = len(rebuilt)
    return bytes(rebuilt)


def main():
    ap = argparse.ArgumentParser(description="Resize the textures inside a .glb.")
    ap.add_argument("glb")
    ap.add_argument("--out", help="output file (default: overwrite, keeping <name>.original.glb)")
    ap.add_argument("--size", type=int, default=1024, help="max texture side in pixels")
    args = ap.parse_args()

    src = Path(args.glb)
    gltf, binary = read_glb(src)
    if len(gltf.get("buffers", [])) != 1:
        raise SystemExit("expected exactly one buffer (a normal .glb)")
    out = Path(args.out) if args.out else src
    if not args.out:
        backup = src.with_suffix(".original.glb")
        if not backup.exists():
            shutil.copy2(src, backup)
            print(f"kept the original as {backup}")
    before = src.stat().st_size
    binary = shrink(gltf, binary, args.size)
    write_glb(out, gltf, binary)
    print(f"{out}: {before / 1e6:.1f} MB -> {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
