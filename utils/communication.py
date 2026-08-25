"""
Packs (option, quality, confidence) into a single integer so opinions can be
sent over the Thymio's single-channel `prox.comm` link.

Relation to the ARGoS reference controller
------------------------------------------
CThymioBestOfTwo::BroadcastOpinionAndEstimate sends three independent
BYTES over range-and-bearing:

    Data[0] = option + 1          (0 reserved for "empty slot")
    Data[1] = ToByte01(mu_q[op])  8 bits over [0,1]
    Data[2] = ToByte01(max(0.05, EpistemicConfidence()))

That is 24 bits. The Thymio radio carries 11 (the API docs advise
staying within 10 in case firmware reserves one), so the three fields
must be re-quantised. The semantics are preserved exactly: option,
quality and a per-message sender confidence that scales tau_base in
BeliefUpdateFromMessages. Only the resolution changes.

Confidence is quantised over [CONF_LO, 1.0] rather than [0,1] because
EpistemicConfidence() = 1/(1+sqrt(1/tau)) saturates: with prior_var=0.25
its floor is 0.667, and a robot only broadcasts once it has sampled, so
the value on the wire is always in roughly [0.67, 1.0]. The reference
controller's 8-bit field uses ~13 of its 256 codewords for this reason.
Re-ranging recovers most of that resolution in 3 bits.

The `seq` field is a rotating nonce carrying no information. prox.comm.rx
LATCHES its last value (unlike ARGoS RAB, where GetReadings() returns
only messages received this tick), so without it a receiver cannot tell
a fresh message from the stale one it already consumed. Set SEQ_BITS=0
if you dedupe on rx._intensities instead.

Default layout (10 bits, max value 1023):

    bits 9-8 : option      (2 bits, 0-3)
    bits 7-4 : quality     (4 bits, 0-15)
    bits 3-1 : confidence  (3 bits, 0-7, over [CONF_LO, 1.0])
    bit  0   : seq         (1 bit)

Raw value 0 is reserved for "nothing has ever been received", mirroring
the reference controller's option+1 offset.
"""

from utils.utils import clamp01

# --- bit budget: adjust these four, PAYLOAD_BITS must stay <= 11 -------
OPT_BITS = 2      # 4 options max
QUALITY_BITS = 4  # step 1/15  = 0.067
CONF_BITS = 3     # step 0.40/7 = 0.057 over [CONF_LO, 1.0]
SEQ_BITS = 1      # 0 = no nonce (only if deduping on rx._intensities)

CONF_LO = 0.60    # 1/(1+sqrt(prior_var)) = 0.667 is the true floor

OPT_MAX = (1 << OPT_BITS) - 1
QUALITY_MAX = (1 << QUALITY_BITS) - 1
CONF_MAX = (1 << CONF_BITS) - 1
SEQ_MAX = (1 << SEQ_BITS) - 1 if SEQ_BITS else 0

PAYLOAD_BITS = OPT_BITS + QUALITY_BITS + CONF_BITS + SEQ_BITS
assert PAYLOAD_BITS <= 11, f"{PAYLOAD_BITS} bits will not fit prox.comm"

_Q_SHIFT = CONF_BITS + SEQ_BITS
_OPT_SHIFT = QUALITY_BITS + _Q_SHIFT


def _pack_conf(c):
    """max(0.05, c) -> CONF_BITS code, mirroring the C++ 0.05 floor."""
    c = max(0.05, clamp01(c))
    x = (min(1.0, max(CONF_LO, c)) - CONF_LO) / (1.0 - CONF_LO)
    return int(round(x * CONF_MAX))


def _unpack_conf(code):
    return CONF_LO + (1.0 - CONF_LO) * (code / CONF_MAX)


def encode_message(option, quality01, confidence01, seq=0):
    """Port of BroadcastOpinionAndEstimate's three SetData calls."""
    op = max(0, min(OPT_MAX, int(option)))
    q = int(round(clamp01(quality01) * QUALITY_MAX))
    c = _pack_conf(confidence01)
    packed = (op << _OPT_SHIFT) | (q << _Q_SHIFT) | (c << SEQ_BITS)
    if SEQ_BITS:
        packed |= int(seq) & SEQ_MAX
    return packed or 1  # never transmit 0


def decode_message(value):
    """Port of the per-message decode in BeliefUpdateFromMessages.

    Returns (option, quality01, confidence01), or (None, None, None) if
    nothing has been received (prox.comm.rx still 0).
    """
    packed = int(value[0])
    if packed <= 0:
        return None, None, None
    op = (packed >> _OPT_SHIFT) & OPT_MAX
    q = (packed >> _Q_SHIFT) & QUALITY_MAX
    c = (packed >> SEQ_BITS) & CONF_MAX
    return op, q / QUALITY_MAX, _unpack_conf(c)


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