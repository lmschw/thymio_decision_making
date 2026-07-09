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
