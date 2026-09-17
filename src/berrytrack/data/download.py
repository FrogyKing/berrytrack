"""Bounded downloads with publisher checksums and safe archive paths."""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: str | Path, checksum: str, max_bytes: int) -> Path:
    """Download an immutable file; checksum is e.g. ``md5:abc...``.

    Windows curl uses the system trust store as a TLS-verified fallback for servers
    whose intermediate certificate chain is absent from Python's certificate store.
    """
    destination = Path(destination)
    algorithm, expected = checksum.split(":", 1)
    if destination.exists() and file_digest(destination, algorithm) == expected:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError("Download exceeds configured maximum size")
                    out.write(block)
        except urllib.error.URLError as error:
            if not isinstance(error.reason, ssl.SSLError) or os.name != "nt":
                raise
            curl = shutil.which("curl.exe")
            if curl is None:
                raise
            subprocess.run(
                [
                    curl,
                    "--silent",
                    "--show-error",
                    "--fail",
                    "--location",
                    "--max-time",
                    "300",
                    "--max-filesize",
                    str(max_bytes),
                    url,
                    "--output",
                    str(temporary),
                ],
                check=True,
            )
        if temporary.stat().st_size > max_bytes:
            raise ValueError("Download exceeds configured maximum size")
        actual = file_digest(temporary, algorithm)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {destination.name}: {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def safe_archive_path(root: Path, name: str) -> Path:
    name = name.replace("\\", "/")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes destination: {name}")
    return target


def extract_archive(archive: Path, destination: Path, max_uncompressed_bytes: int) -> None:
    """Extract regular files only and bound the total declared unpacked size."""
    destination.mkdir(parents=True, exist_ok=True)
    size = 0
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                target = safe_archive_path(destination, member.filename)
                if member.external_attr >> 16 & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not permitted")
                size += member.file_size
                if size > max_uncompressed_bytes:
                    raise ValueError("Archive exceeds extraction budget")
                if not member.is_dir():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    elif archive.suffix == ".7z":
        import py7zr

        with py7zr.SevenZipFile(archive) as stream:
            for entry in stream.list():
                safe_archive_path(destination, entry.filename)
                if entry.is_symlink:
                    raise ValueError("Archive symlinks are not permitted")
                size += entry.uncompressed or 0
            if size > max_uncompressed_bytes:
                raise ValueError("Archive exceeds extraction budget")
            stream.extractall(destination)
    else:
        with tarfile.open(archive) as stream:
            for tar_member in stream:
                target = safe_archive_path(destination, tar_member.name)
                size += tar_member.size
                if size > max_uncompressed_bytes:
                    raise ValueError("Archive exceeds extraction budget")
                if not tar_member.isfile():
                    if tar_member.isdir():
                        continue
                    raise ValueError("Only regular archive files are permitted")
                target.parent.mkdir(parents=True, exist_ok=True)
                tar_source = stream.extractfile(tar_member)
                if tar_source is None:
                    raise ValueError("Archive file has no contents")
                with tar_source, target.open("wb") as dst:
                    shutil.copyfileobj(tar_source, dst)
