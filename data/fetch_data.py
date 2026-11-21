"""Fetch the input tables that are not redistributed here, and check every input table against its SHA-256.

    python data/fetch_data.py

Shipped in this folder (AME2020, M. Wang et al., Chin. Phys. C 45, 030003 (2021)):
    rct1.mas20, mass.mas20

Downloaded, because their sources state no redistribution licence:
    required for the paper's main fit
        WS_mass_table.txt   N.-N. Ma et al., Chin. Phys. C 43, 044105 (2019)
    optional, for the mass-model check of Sec. II D (residual_analyzer/); skipped by run_all.py if absent
        WS4.txt             N. Wang et al., Phys. Lett. B 734, 215 (2014)
        hfb24-dat           S. Goriely, N. Chamel, J. M. Pearson, Phys. Rev. C 88, 024308 (2013)
        KTUY05_m246.dat     H. Koura et al., Prog. Theor. Phys. 113, 305 (2005)
        frdm_text.txt       P. Moller et al., At. Data Nucl. Data Tables 109-110, 1 (2016): the text of
                            arXiv:1508.06294v1, made here with `pdftotext -raw -enc UTF-8` (xpdf or poppler).
                            That command reproduces all 9318 rows of the file the paper used.
    optional, for the bound on the decay-rate benchmark of Sec. III C (validation/decay_rate_bound.py)
        nubase_4.mas20.txt  NUBASE2020, F. G. Kondev et al., Chin. Phys. C 45, 030001 (2021)

Exit code 0 if the required tables are present and correct; the optional ones only print their status.
"""
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "residual_analyzer" / "raw"

# (path, url or None if shipped, sha256, required)
FILES = [
    (HERE / "rct1.mas20", None, "e6ba1d2256f90053464c48e24d44691828dc49820ce64054aa418d834cc18e90", True),
    (HERE / "mass.mas20", None, "05b054a0538f2c308d061b22096f94a944f1603a5b0948a62e5171657601c674", True),
    (HERE / "WS_mass_table.txt",
     "https://raw.githubusercontent.com/lukeronger/NuclearData-LZU/master/NuclearData/data_table.txt",
     "a928998d982bf5c7768133676ee0a9975986fc25a89245029e3df45d66a22cf3", True),
    (RAW / "ws4" / "WS4.txt", "http://www.imqmd.com/mass/WS4.txt",
     "9492ef1451a72321d12d1ebe496570e0d8f3ed491de22a25b64b950859d8de3e", False),
    (RAW / "hfb24" / "hfb24-dat", "https://www.astro.ulb.ac.be/bruslib/nucdata/hfb24-dat",
     "23ced4dcf37a9ba7a06130dce16f9148a17e4591118857c6779e3ff06a76f63e", False),
    (RAW / "ktuy05" / "KTUY05_m246.dat", "https://wwwndc.jaea.go.jp/nucldata/mass/KTUY05_m246.dat",
     "bf4b22407902c38ffbdbb0b0659857f1aaf3a7407e0ae852c609ac67f1d93901", False),
    (RAW / "nubase2020" / "nubase_4.mas20.txt", "https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt",
     "1585a5eea86c5e17e90307c7e6e786d060049c4039e392a261ff6db977df9859", False),
]
FRDM_PDF_URL = "https://arxiv.org/pdf/1508.06294v1"
FRDM_PDF_SHA = "99ce7ece6830b0904f68d9a6342b9e66d0ebf0d235c2eb6c4de78716768c10e4"
FRDM_TEXT = RAW / "frdm2012" / "frdm_text.txt"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=120) as r:
        path.write_bytes(r.read())


def frdm():
    """arXiv PDF -> text. The text's hash depends on the pdftotext build, so the PDF is what is checked here;
    residual_analyzer/ra_ingest.py then checks every row against its own binding-energy column."""
    if FRDM_TEXT.exists():
        return "present"
    exe = shutil.which("pdftotext")
    if exe is None:
        return ("MISSING: needs pdftotext (xpdf or poppler). Or by hand: download " + FRDM_PDF_URL +
                f" and run  pdftotext -raw -enc UTF-8 <pdf> {FRDM_TEXT}")
    pdf = FRDM_TEXT.with_name("1508.06294v1.pdf")
    try:
        if not pdf.exists():
            print(f"downloading {FRDM_PDF_URL}")
            download(FRDM_PDF_URL, pdf)
    except OSError as err:
        return f"MISSING: download failed ({err})"
    if sha256(pdf) != FRDM_PDF_SHA:
        return "MISMATCH: the arXiv PDF is not the file the paper used"
    subprocess.run([exe, "-raw", "-enc", "UTF-8", str(pdf), str(FRDM_TEXT)], check=True)
    return "made from the arXiv PDF"


def main():
    ok = True
    for path, url, want, required in FILES:
        tag = "required" if required else "optional"
        if not path.exists():
            if url is None:
                print(f"MISSING  {path.name} ({tag}): it should ship with this archive")
                ok &= not required
                continue
            try:
                print(f"downloading {url}")
                download(url, path)
            except OSError as err:
                print(f"MISSING  {path.name} ({tag}): download failed ({err}); fetch it by hand from {url}")
                ok &= not required
                continue
        if sha256(path) == want:
            print(f"ok       {path.name}")
        else:
            print(f"MISMATCH {path.name} ({tag}): not the file the paper used; results may differ")
            ok &= not required
    status = frdm()
    print(f"{'ok      ' if not status.startswith(('MISSING', 'MISMATCH')) else ''} frdm_text.txt (optional): {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
