"""Classical predict_pdf spine: extract → fee recovery → apply_safety_policy."""

from __future__ import annotations

from pathlib import Path

from .extract import FieldHit, extract_packet
from .fee_vision import recover_fee_from_pixmap
from .policy import build_prediction


def _scan_fee_hog(packet) -> tuple[str, float, int] | None:
    """Best HOG fee label across image-only pages."""
    try:
        import fitz
    except ImportError:
        return None
    span_by_page = {p.page_index: p.n_trusted_spans for p in packet.pages}
    doc = fitz.open(packet.pdf_path)
    best: tuple[str, float, int] | None = None
    for page_index, page in enumerate(doc):
        if span_by_page.get(page_index, 0) >= 8:
            continue
        for imginfo in page.get_images(full=True):
            try:
                pix = fitz.Pixmap(doc, imginfo[0])
            except Exception:
                continue
            if pix.width < 900 or pix.height < 900:
                continue
            fee_lab, fee_conf = recover_fee_from_pixmap(pix)
            if not fee_lab:
                continue
            if best is None or fee_conf > best[1]:
                best = (fee_lab, fee_conf, page_index)
            if best and best[1] >= 0.8:
                return best
    return best


def _ensure_fee_from_scans(packet) -> None:
    """If fee still missing, classify large embedded page scans."""
    if "fee_status" in packet.fields:
        return
    best = _scan_fee_hog(packet)
    if best and best[1] >= 0.62:
        packet.fields["fee_status"] = FieldHit(
            value=best[0], source="ocr", page=best[2], confidence=best[1]
        )
        if "fee_missing" in packet.evidence_issues:
            packet.evidence_issues = [x for x in packet.evidence_issues if x != "fee_missing"]


def _reconcile_noisy_fee(packet) -> None:
    """Override low-confidence OCR fee when HOG strongly disagrees.

    Noisy fee-receipt scans often OCR as the wrong status (e.g. waived vs paid).
    Prefer high-confidence HOG / amount cues on image-only pages only.
    """
    import re

    fee = packet.fields.get("fee_status")
    if not fee:
        return
    # Trusted digital text-layer fee receipts (high confidence) stay put.
    if fee.source == "fee_receipt" and fee.confidence >= 0.9:
        # Still allow override if the "fee_receipt" hit came from OCR of a scan
        # (confidence scaled to ~0.675) — those are < 0.9.
        return

    # Amount cue from OCR/trusted text on image pages.
    text = "\n".join(p.trusted_text for p in packet.pages)
    if re.search(r"\$\s*809", text) and fee.value != "paid":
        packet.fields["fee_status"] = FieldHit(
            value="paid", source="receipt_amount_waiver", page=fee.page, confidence=0.7
        )
        return
    if re.search(r"\$\s*0\.00", text) and fee.value == "paid" and fee.confidence < 0.85:
        # weak paid OCR against zero amount → waived
        packet.fields["fee_status"] = FieldHit(
            value="waived", source="receipt_amount_waiver", page=fee.page, confidence=0.65
        )
        return

    if fee.confidence >= 0.85:
        return

    best = _scan_fee_hog(packet)
    if not best:
        return
    hog_lab, hog_conf, page = best
    if hog_conf >= 0.75 and hog_lab != fee.value:
        packet.fields["fee_status"] = FieldHit(
            value=hog_lab, source="ocr", page=page, confidence=hog_conf
        )


def _purpose_from_sponsor(packet) -> None:
    """Sponsor letters often encode purpose as 'expected on Earth for X'."""
    import re

    if "declared_purpose" in packet.fields:
        val = packet.fields["declared_purpose"].value.strip(" .,;:")
        packet.fields["declared_purpose"].value = val
        return

    text = "\n".join(p.trusted_text for p in packet.pages)
    text_flat = re.sub(r"\s+", " ", text)
    m = re.search(
        r"expected on Earth for\s+([A-Za-z][A-Za-z\- ]{2,60})",
        text_flat,
        re.I,
    )
    if m:
        purpose = re.sub(r"\s+", " ", m.group(1)).strip(" .,;:")
        purpose = re.split(
            r"\b(?:compliance|and immediate|reporting)\b", purpose, maxsplit=1
        )[0].strip(" .,;:")
        if purpose:
            packet.fields["declared_purpose"] = FieldHit(
                value=purpose, source="sponsor_letter", page=-1, confidence=0.75
            )



def _cleanup_fields(packet) -> None:
    import re
    for key in ("applicant_name", "home_world", "declared_purpose", "species_code"):
        hit = packet.fields.get(key)
        if not hit:
            continue
        val = hit.value.strip(" .,;:-—_|")
        val = re.sub(r"\s+", " ", val)
        hit.value = val

    # Prefer non-OCR name when OCR name looks corrupted vs sponsor/registry text.
    name = packet.fields.get("applicant_name")
    if name and (name.source == "ocr" or name.confidence < 0.8):
        text = "\n".join(p.trusted_text for p in packet.pages)
        cands = []
        for pat in (
            r"attests that\s+([A-Z][A-Za-z\- ]+?)\s+is expected",
            r"Registry Name\s*\n\s*([A-Z][A-Za-z\- ]+)",
            r"Applicant:\s*([A-Z][A-Za-z\- ]+)",
        ):
            m = re.search(pat, text)
            if m:
                cands.append(m.group(1).strip(" .,;:"))
        if cands and name:
            # pick candidate with highest alpha overlap if OCR has unlikely chars
            import re as _re
            if _re.search(r"[0-9]|bxom|vv|rn", name.value.lower()) or name.confidence < 0.8:
                packet.fields["applicant_name"].value = cands[0]
                packet.fields["applicant_name"].source = "sponsor_letter"
                packet.fields["applicant_name"].confidence = 0.85


def predict_pdf(pdf_path: str | Path, case_id: str | None = None) -> dict:
    pdf_path = Path(pdf_path)
    packet = extract_packet(pdf_path, case_id=case_id or pdf_path.stem)
    _purpose_from_sponsor(packet)
    _ensure_fee_from_scans(packet)
    _reconcile_noisy_fee(packet)
    _cleanup_fields(packet)

    # If fee recovered after extract, drop fee_missing issue.
    if "fee_status" in packet.fields and "fee_missing" in packet.evidence_issues:
        packet.evidence_issues = [x for x in packet.evidence_issues if x != "fee_missing"]

    return build_prediction(
        case_id=packet.case_id,
        fields=packet.fields,
        risk_flags=packet.risk_flags,
        evidence_issues=packet.evidence_issues,
        manual_finding=packet.manual_finding,
        docs_present=packet.docs_present,
    )
