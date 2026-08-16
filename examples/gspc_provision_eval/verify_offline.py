#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Contributed by CSOAI (csoai.org) — Council of AI (CSOAI LTD, UK #16939677).
"""Offline verifier for Council of AI signed measurement cards.

Zero network. Zero secrets. Public key only.

A signed measurement card is a JSON object with:
  body      - the measured payload (any JSON object)
  id        - SHA-256 hex of the canonical body (see canonical() below)
  prev      - SHA-256 hex id of the previous card in the chain (or 64 zeros)
  pubkey    - Ed25519 public key, hex
  signature - Ed25519 signature over the ASCII bytes of `id`, hex
  status    - "SIGNED"

Verification proves: the body hashes to `id`, and `signature` is a valid
Ed25519 signature of `id` under `pubkey`. Anyone can check a card without
calling us, without an account, and without trusting the transport.

Usage:
  python verify_offline.py --card sample_card.json
  python verify_offline.py --card sample_card.json --pubkey <hex>   # pin key
  python verify_offline.py --chain chain.jsonl                      # whole chain

Exit code 0 = every check passed, 1 = any check failed.
"""
import argparse
import hashlib
import json
import sys


def canonical(obj) -> bytes:
    """Canonical JSON: sorted keys, compact separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def content_id(body) -> str:
    return hashlib.sha256(canonical(body)).hexdigest()


def ed25519_verify(pubkey_hex: str, sig_hex: str, message: bytes) -> bool:
    """Verify with PyNaCl if present, else cryptography, else fail loudly."""
    try:
        from nacl.signing import VerifyKey  # PyNaCl
        try:
            VerifyKey(bytes.fromhex(pubkey_hex)).verify(message, bytes.fromhex(sig_hex))
            return True
        except Exception:
            return False
    except ImportError:
        pass
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex)).verify(
                bytes.fromhex(sig_hex), message)
            return True
        except Exception:
            return False
    except ImportError:
        raise SystemExit(
            "Need PyNaCl or cryptography: pip install pynacl  (still fully offline)")


def check_card(card: dict, pin_pubkey: str | None = None) -> list[str]:
    errors = []
    for field in ("body", "id", "pubkey", "signature"):
        if field not in card:
            errors.append(f"missing field: {field}")
    if errors:
        return errors
    if content_id(card["body"]) != card["id"]:
        errors.append("digest MISMATCH: body does not hash to id (tampered?)")
    if pin_pubkey and card["pubkey"] != pin_pubkey:
        errors.append("pubkey does not match the pinned key")
    if not ed25519_verify(card["pubkey"], card["signature"], card["id"].encode("ascii")):
        errors.append("signature INVALID")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--card", help="single signed card JSON")
    ap.add_argument("--chain", help="JSONL chain of signed cards (also checks prev linkage)")
    ap.add_argument("--pubkey", help="pin: require this hex public key")
    args = ap.parse_args()
    if not args.card and not args.chain:
        ap.error("give --card or --chain")

    cards = []
    if args.card:
        cards = [json.load(open(args.card))]
    if args.chain:
        cards += [json.loads(l) for l in open(args.chain) if l.strip()]

    ok = True
    prev_id = None
    for i, card in enumerate(cards):
        errs = check_card(card, args.pubkey)
        if args.chain and prev_id is not None and card.get("prev") != prev_id:
            errs.append("chain link BROKEN: prev does not match previous card id")
        prev_id = card.get("id")
        label = card.get("body", {}).get("card_type", args.card or f"entry[{i}]")
        if errs:
            ok = False
            for e in errs:
                print(f"INVALID {label}: {e}")
        else:
            print(f"VALID {label} id={card['id'][:16]}... pubkey={card['pubkey'][:16]}...")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
