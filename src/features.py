"""URL-only feature extraction.

The SAME functions are used in training and in the API, so features
are identical at train time and serve time (no train/serve skew).
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract

# Use only the bundled public-suffix snapshot: no network call at runtime.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")
_WORDS = ("login", "secure", "verify", "account", "update",
          "bank", "signin", "confirm", "password", "paypal")


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def extract_features(url: str) -> dict:
    """Features from the FULL url (scheme, www, path, query included)."""
    u = str(url).strip()
    parsed = urlparse(u if "://" in u else "http://" + u)
    host = (parsed.hostname or "").lower()
    ext = _EXTRACT(host)
    tld = ext.suffix.rsplit(".", 1)[-1] if ext.suffix else ""

    n = max(len(u), 1)
    letters = sum(ch.isalpha() for ch in u)
    digits = sum(ch.isdigit() for ch in u)
    other_special = sum(1 for ch in u if not ch.isalnum() and ch not in "=?&")
    low = u.lower()

    return {
        # length / structure
        "url_length": len(u),
        "domain_length": len(host),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "num_subdomains": len([s for s in ext.subdomain.split(".") if s]),
        "tld_length": len(tld),
        "tld": tld,  # categorical, encoded in training
        # host / protocol
        "is_ip": int(bool(_IP_RE.match(host))),
        "is_https": int(parsed.scheme == "https"),
        # character counts and ratios
        "num_letters": letters,
        "letter_ratio": letters / n,
        "num_digits": digits,
        "digit_ratio": digits / n,
        "num_equals": u.count("="),
        "num_qmark": u.count("?"),
        "num_amp": u.count("&"),
        "num_other_special": other_special,
        "special_ratio": other_special / n,
        "num_dots": u.count("."),
        "num_hyphens": u.count("-"),
        "num_at": u.count("@"),
        # obfuscation / randomness / keywords
        "num_pct_encoded": len(_PCT_RE.findall(u)),
        "entropy": _entropy(u),
        "num_suspicious_words": sum(w in low for w in _WORDS),
    }


def canonical_host(url: str) -> str:
    """Lower-cased host with scheme, path, query and a leading 'www.' removed."""
    u = str(url).strip()
    parsed = urlparse(u if "://" in u else "http://" + u)
    host = (parsed.hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def extract_host_features(url: str) -> dict:
    """Features computed from the host only (no scheme/www/path/query)."""
    host = canonical_host(url)
    ext = _EXTRACT(host)
    tld = ext.suffix.rsplit(".", 1)[-1] if ext.suffix else ""
    labels = [p for p in host.split(".") if p]
    n = max(len(host), 1)
    digits = sum(ch.isdigit() for ch in host)
    letters = sum(ch.isalpha() for ch in host)
    return {
        "host_length": len(host),
        "domain_label_length": len(ext.domain),
        "num_labels": len(labels),
        "num_subdomains": len([s for s in ext.subdomain.split(".") if s]),
        "longest_label": max((len(p) for p in labels), default=0),
        "tld_length": len(tld),
        "tld": tld,
        "is_ip": int(bool(_IP_RE.match(host))),
        "num_digits": digits,
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "num_hyphens": host.count("-"),
        "has_punycode": int("xn--" in host),
        "entropy": _entropy(host),
        "num_suspicious_words": sum(w in host for w in _WORDS),
    }