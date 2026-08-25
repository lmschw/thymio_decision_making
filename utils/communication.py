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

OPT_BITS, QUAL_BITS, CONF_BITS = 2, 5, 3 # 10 bits total, max 1023 

def encode_message(option, quality01,confidence01): 
    op = max(0, min(3, int(option))) 
    q = round(clamp01(quality01) * 31) 
    c = round(clamp01(confidence01) * 7) 
    return ((op << 8) | (q << 3) | c) + 1 # +1 so 0 is reserved for "nothing" 

def decode_message(value): 
    v = int(value[0]) - 1
    if v < 0: 
        return None, None, None # rx was 0: no message 
    return (v >> 8) & 0x3, ((v >> 3) & 31) / 31.0, (v & 7) / 7.0

# smaller encoding
def encode_opinion_quality(opinion: int, quality: float) -> int:
    """2 bits opinion, 6 bits quality."""
    opinion = max(0, min(3, opinion))
    quality = clamp01(quality)

    quality_int = round(quality * 63)

    return (opinion << 6) | quality_int


def decode_opinion_quality(value: int):
    opinion = (value >> 6) & 0b11
    quality = (value & 0b00111111) / 63.0

    return opinion, quality
