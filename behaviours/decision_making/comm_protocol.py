"""
Packs (option, quality, confidence) into a single integer so opinions can be
sent over the Thymio's single-channel `prox.comm` link.

The original ARGoS controller used range-and-bearing messages with 3 data
bytes (opinion, quality byte, confidence byte) and could see several
neighbours' messages per tick. robot.py's `send`/`receive` only carry one
int via `prox.comm.tx` / `prox.comm.rx`, so this is a single message per
tick rather than a multi-neighbour broadcast — closer to the "voter model"
update than majority voting.

Layout (15 bits, fits the signed 16-bit VM variable behind prox.comm):

    bits 14-12 : option      (3 bits, 0-7)
    bits 11-6  : quality     (6 bits, 0-63)
    bits 5-0   : confidence  (6 bits, 0-63)
"""

QUALITY_BITS = 6
CONF_BITS = 6
QUALITY_MAX = (1 << QUALITY_BITS) - 1  # 63
CONF_MAX = (1 << CONF_BITS) - 1        # 63

from utils.utils import clamp01


def encode_message(option: int, quality01: float, confidence01: float) -> int:
    op = max(0, min(7, int(option)))
    q = round(clamp01(quality01) * QUALITY_MAX)
    c = round(clamp01(confidence01) * CONF_MAX)
    return (op << (QUALITY_BITS + CONF_BITS)) | (q << CONF_BITS) | c


def decode_message(value: int):
    """Returns (option: int, quality01: float, confidence01: float)."""
    value = int(value)
    op = (value >> (QUALITY_BITS + CONF_BITS)) & 0x7
    q = (value >> CONF_BITS) & QUALITY_MAX
    c = value & CONF_MAX
    return op, q / QUALITY_MAX, c / CONF_MAX
