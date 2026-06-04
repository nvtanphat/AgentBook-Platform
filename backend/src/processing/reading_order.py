from __future__ import annotations


def order_blocks_by_reading(blocks: list) -> list:
    """Return blocks in true top-to-bottom reading order.

    The parser's ``reading_order`` is normally correct — and crucially handles
    multi-column layouts that a naive geometric sort would scramble — so we keep
    it by default. But some PDFs are emitted bottom-to-top, which reverses
    chapters/articles and corrupts chunk assembly (a heading gets glued to the
    *previous* section's body). We detect that *gross reversal* from bbox geometry
    and only then fall back to a geometric sort, leaving well-ordered (including
    multi-column) documents untouched.

    Convention-agnostic: ``bbox.y1`` is the block's top edge, but the coordinate
    origin differs (PDF = bottom-left → the top edge has the larger y; image OCR
    = top-left → the top edge has the smaller y). The origin is inferred per
    block-set so a smaller sort key always means "higher on the page".

    Duck-typed: works on any object exposing ``.bbox`` (with ``.x1/.y1/.y2``) and
    ``.reading_order`` — both ``ParsedBlock`` and the persisted ``MaterialBlock``.
    """
    by_ro = sorted(blocks, key=lambda b: b.reading_order)
    withbb = [b for b in blocks if getattr(b, "bbox", None) is not None]
    if len(withbb) < 3:
        return by_ro

    # Bottom-left origin when most blocks have top (y1) >= bottom (y2).
    bottom_left = sum(1 for b in withbb if b.bbox.y1 >= b.bbox.y2) >= len(withbb) / 2

    def vpos(b) -> float:  # smaller = higher on the page
        if getattr(b, "bbox", None) is None:
            return float("inf")
        return -b.bbox.y1 if bottom_left else b.bbox.y1

    # Does reading_order climb up the page (reversed) instead of down it?
    seq = [vpos(b) for b in by_ro if getattr(b, "bbox", None) is not None]
    down = sum(1 for i in range(len(seq) - 1) if seq[i] < seq[i + 1])  # correct direction
    up = sum(1 for i in range(len(seq) - 1) if seq[i] > seq[i + 1])    # reversed direction
    if up > down * 2:  # grossly reversed → trust geometry (top-to-bottom, then left-to-right)
        return sorted(
            blocks,
            key=lambda b: (vpos(b), b.bbox.x1 if getattr(b, "bbox", None) is not None else 0.0),
        )
    return by_ro
