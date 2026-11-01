"""Fetch fact sheet text at run time, so the sheets themselves stay out of the repository."""
import html
import re
import urllib.request

UW = "https://pddc.wisc.edu"

SHEETS = {
    "early_blight": f"{UW}/2015/07/23/early-blight/",
    "late_blight": f"{UW}/2015/07/28/late-blight/",
    "septoria_leaf_spot": f"{UW}/2015/08/16/septoria-leaf-spot/",
    "bacterial_spot": f"{UW}/2017/02/17/bacterial-spot-tomato/",
    "mosaic_virus": f"{UW}/2021/02/19/tobacco-mosaic/",
    "leaf_mold": "https://ipm.cahnr.uconn.edu/?p=15437",
    "yellow_leaf_curl_virus": "https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus",
    "healthy": "https://hort.extension.wisc.edu/articles/home-grown-tomatoes-wisconsin/",
}


def page_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf8", errors="ignore")
    raw = re.sub(r"(?is)<(script|style|nav|header|footer).*?</\1>", " ", raw)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", raw))
    return re.sub(r"\n\s*\n+", "\n", text)


def section(text, start_phrase, end_phrase):
    """Return the text between two headings, matching the UW sheets' question headings."""
    start = text.lower().find(start_phrase.lower())
    if start < 0:
        return ""
    end = text.lower().find(end_phrase.lower(), start + len(start_phrase))
    return text[start : end if end > 0 else start + 2000].strip()


def uw_passages(label):
    """Return (symptoms, care steps) for a class with a UW sheet."""
    text = page_text(SHEETS[label])
    symptoms = section(text, "look like?", "Where does")
    care = section(text, "How do I save", "For more information")
    return symptoms, care
