import re

from src.pipeline.index import tokenize

_SYNONYMS: dict[str, list[str]] = {
    "amount":        ["payment", "sum", "price", "cost", "fee", "charge", "total", "value", "consideration"],
    "payment":       ["amount", "sum", "fee", "remittance", "disbursement", "compensation"],
    "total":         ["sum", "aggregate", "overall", "combined", "gross", "net"],
    "price":         ["amount", "cost", "fee", "rate", "charge", "value"],
    "cost":          ["price", "fee", "expense", "amount", "charge"],
    "fee":           ["amount", "payment", "charge", "cost", "remuneration"],
    "date":          ["dated", "effective", "executed", "signed", "entered", "commencement"],
    "signed":        ["executed", "dated", "entered", "agreed"],
    "effective":     ["commencement", "start", "dated", "signed"],
    "term":          ["duration", "period", "length", "tenure", "life"],
    "duration":      ["term", "period", "length"],
    "terminate":     ["termination", "cancellation", "expiry", "expiration", "end", "cessation"],
    "termination":   ["terminate", "cancellation", "expiry", "end"],
    "party":         ["parties", "counterparty", "signatory", "vendor", "buyer", "seller", "contractor"],
    "parties":       ["party", "counterparty", "signatories"],
    "agreement":     ["contract", "deed", "arrangement", "covenant", "indenture", "document"],
    "contract":      ["agreement", "deed", "arrangement", "document"],
    "liability":     ["obligation", "indemnity", "exposure", "responsibility", "damages"],
    "obligation":    ["liability", "duty", "responsibility", "requirement"],
    "notice":        ["notification", "written", "inform", "advise", "alert"],
    "breach":        ["default", "violation", "failure", "noncompliance", "infringement"],
    "default":       ["breach", "violation", "failure", "noncompliance"],
    "penalty":       ["liquidated", "damages", "fine", "surcharge", "sanction"],
    "damages":       ["penalty", "compensation", "indemnification", "loss"],
    "interest":      ["rate", "accrual", "compounded", "annum", "yield"],
    "rate":          ["interest", "percentage", "proportion", "amount"],
    "warranty":      ["warrant", "representation", "guarantee", "assurance"],
    "guarantee":     ["warranty", "indemnity", "assurance", "undertaking"],
    "assign":        ["assignment", "transfer", "novation", "delegate"],
    "assignment":    ["transfer", "novation", "delegation"],
    "confidential":  ["proprietary", "nondisclosure", "secret", "privileged"],
    "govern":        ["governing", "jurisdiction", "applicable", "law"],
    "jurisdiction":  ["governing", "law", "venue", "forum"],
    "intellectual":  ["property", "patent", "copyright", "trademark"],
    "property":      ["intellectual", "asset", "right", "ownership"],
    "indemnify":     ["indemnification", "hold harmless", "defend", "compensate"],
    "force":         ["majeure", "circumstances", "extraordinary", "event"],
    "dispute":       ["arbitration", "litigation", "resolution", "disagreement"],
    "arbitration":   ["dispute", "resolution", "proceedings"],
    "payment":       ["installment", "tranche", "disbursement", "transfer"],
    "clause":        ["provision", "section", "article", "paragraph"],
    "provision":     ["clause", "section", "article", "term", "condition"],
    "represent":     ["representation", "warrant", "certify", "confirm"],
    "name":          ["named", "entitled", "called", "designated", "party", "company"],
    "company":       ["corporation", "entity", "organization", "firm", "business"],
    "person":        ["individual", "party", "entity", "organization"],
    "employee":      ["staff", "worker", "personnel", "contractor"],
    "service":       ["services", "work", "deliverable", "performance"],
    "work":          ["services", "deliverable", "performance", "task"],
    "deliver":       ["delivery", "provide", "supply", "perform"],
    "authorized":    ["approved", "permitted", "allowed", "entitled"],
    "approved":      ["authorized", "permitted", "confirmed", "accepted"],
    "prohibited":    ["forbidden", "restricted", "disallowed", "barred"],
    "include":       ["including", "comprise", "consist", "cover"],
    "exclude":       ["excluding", "except", "omit"],
}


def expand_query(tokens: list[str], max_per_token: int = 3,
                 original_query: str = "", answer_type: str = "") -> list[str]:
    expanded = list(tokens)
    seen = set(tokens)
    for token in tokens:
        for syn in _SYNONYMS.get(token, [])[:max_per_token]:
            if syn not in seen:
                expanded.append(syn)
                seen.add(syn)

    # Article number boost: when query mentions "Article N", add the bare number
    # as a token to boost pages containing that article (BM25 term frequency).
    if original_query:
        for m in re.finditer(r"\bArticle\s+(\d+)", original_query, re.IGNORECASE):
            num = m.group(1)
            if num not in seen:
                expanded.append(num)
                seen.add(num)
        # Also boost "Section" and "Part" references
        for m in re.finditer(r"\b(?:Section|Part)\s+(\d+)", original_query, re.IGNORECASE):
            num = m.group(1)
            if num not in seen:
                expanded.append(num)
                seen.add(num)

    # Type-aware expansion: add domain terms that help BM25 find relevant pages
    if answer_type == "date":
        for term in ["enacted", "commencement", "force", "effective"]:
            if term not in seen:
                expanded.append(term)
                seen.add(term)
    elif answer_type in ("bool", "boolean"):
        for term in ["shall", "must", "may", "permitted", "prohibited"]:
            if term not in seen:
                expanded.append(term)
                seen.add(term)

    return expanded
