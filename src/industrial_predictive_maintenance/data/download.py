"""Download, verify and extract the raw bearing datasets.

FEMTO and IMS are downloaded automatically. SCA has no direct download
link on Mendeley Data, so its zip has to be downloaded by hand once.
Every archive is checked against its SHA-256 checksum before use. Steps
that are already done are skipped, so the script is safe to run again.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(os.environ.get("IPM_DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"

FEMTO_URL = "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"
FEMTO_ZIP = RAW_DIR / "femto_bearing.zip"
FEMTO_SHA256 = "e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f"

IMS_URL = "https://data.nasa.gov/docs/legacy/IMS.zip"
IMS_ZIP = RAW_DIR / "ims_bearing.zip"
IMS_SHA256 = "6cb42c263b0281c725abf99f4b9fcf49915c949f31dbd2333877dc2e06ce9ec2"

SCA_PAGE = "https://data.mendeley.com/datasets/tdn96mkkpt/2"
SCA_ZIP = RAW_DIR / "sca_bearing.zip"
SCA_SHA256 = "2ed753cf15be0ded1a7e723c74602b16fba11b5541ac966234aede0a50c80a70"

# Expected number of data files per folder after extraction.
FEMTO_EXPECTED = {"Learning_set": 8384, "Full_Test_Set": 19523}
IMS_EXPECTED = {"1st_test": 2156, "2nd_test": 984, "4th_test/txt": 6324}
SCA_EXPECTED = 22

CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid(path: Path, expected: str) -> bool:
    return path.exists() and sha256(path) == expected


def download(url: str, target: Path, expected: str) -> None:
    if is_valid(target, expected):
        print(f"ok        {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"download  {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as f:
        shutil.copyfileobj(response, f, CHUNK)
    if sha256(partial) != expected:
        partial.unlink()
        raise RuntimeError(f"checksum mismatch for {url}")
    partial.rename(target)
    print(f"verified  {target}")


def extract_member(archive: Path, member: str, target: Path) -> Path:
    """Copy one file out of a zip archive without its folder path."""
    if target.exists():
        return target
    partial = target.with_name(target.name + ".part")
    with zipfile.ZipFile(archive) as zf, zf.open(member) as src, partial.open("wb") as dst:
        shutil.copyfileobj(src, dst, CHUNK)
    partial.rename(target)
    return target


def count_files(folder: Path, pattern: str) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob(pattern) if p.is_file())


def check(folder: Path, pattern: str, expected: int) -> None:
    found = count_files(folder, pattern)
    if found != expected:
        raise RuntimeError(f"{folder}: expected {expected} files but found {found}")
    print(f"ok        {folder} ({found} files)")


def prepare_femto() -> None:
    download(FEMTO_URL, FEMTO_ZIP, FEMTO_SHA256)
    femto_dir = RAW_DIR / "femto"
    if not all(count_files(femto_dir / n, "*.csv") == c for n, c in FEMTO_EXPECTED.items()):
        femto_dir.mkdir(parents=True, exist_ok=True)
        # The download is a zip inside a zip. Test_set.zip holds the truncated
        # challenge version of the 11 test bearings, so it is not extracted.
        inner = extract_member(
            FEMTO_ZIP,
            "10. FEMTO Bearing/FEMTOBearingDataSet.zip",
            RAW_DIR / "FEMTOBearingDataSet.zip",
        )
        for name in ("Training_set.zip", "Validation_Set.zip"):
            with zipfile.ZipFile(extract_member(inner, name, femto_dir / name)) as zf:
                zf.extractall(femto_dir)
    for name, expected in FEMTO_EXPECTED.items():
        check(femto_dir / name, "*.csv", expected)


def prepare_ims() -> None:
    download(IMS_URL, IMS_ZIP, IMS_SHA256)
    ims_dir = RAW_DIR / "ims"
    if not all(count_files(ims_dir / n, "20*") == c for n, c in IMS_EXPECTED.items()):
        if shutil.which("unar") is None:
            raise RuntimeError(
                "IMS is packed as RAR files. Install unar first "
                "(macOS: brew install unar, Ubuntu: sudo apt install unar)."
            )
        ims_dir.mkdir(parents=True, exist_ok=True)
        extract_member(
            IMS_ZIP,
            "IMS/Readme Document for IMS Bearing Data.pdf",
            ims_dir / "Readme Document for IMS Bearing Data.pdf",
        )
        # 3rd_test.rar unpacks into 4th_test/txt and holds 1,876 recordings
        # after the documented end of test 3. All files are kept here. The
        # documented part is selected when the data is loaded.
        for name in ("1st_test.rar", "2nd_test.rar", "3rd_test.rar"):
            rar = extract_member(IMS_ZIP, f"IMS/{name}", ims_dir / name)
            subprocess.run(["unar", "-q", "-f", "-o", str(ims_dir), str(rar)], check=True)
    for name, expected in IMS_EXPECTED.items():
        check(ims_dir / name, "20*", expected)


def prepare_sca() -> None:
    if not is_valid(SCA_ZIP, SCA_SHA256):
        raise RuntimeError(
            f"{SCA_ZIP} is missing or does not match the expected checksum. "
            f"Download it by hand from {SCA_PAGE} with the Download All button "
            f"and save it as {SCA_ZIP}."
        )
    print(f"ok        {SCA_ZIP}")
    sca_dir = RAW_DIR / "sca"
    if count_files(sca_dir, "*.mat") != SCA_EXPECTED:
        with zipfile.ZipFile(SCA_ZIP) as zf:
            zf.extractall(sca_dir)
    check(sca_dir, "*.mat", SCA_EXPECTED)


def main() -> int:
    try:
        prepare_femto()
        prepare_ims()
        prepare_sca()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error     {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
