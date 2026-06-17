#!/usr/bin/env python3
"""Upload lidar model release directory to MinIO with MD5 metadata check."""

from minio import Minio
from minio.error import S3Error

import argparse
import hashlib
import mimetypes
import os
from pathlib import Path
import re
import sys


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = os.getenv("MINIO_BUCKET")

if not MINIO_ENDPOINT:
    print("❌ 错误：请设置环境变量 MINIO_ENDPOINT")
    sys.exit(1)
if not ACCESS_KEY:
    print("❌ 错误：请设置环境变量 MINIO_ACCESS_KEY")
    sys.exit(1)
if not SECRET_KEY:
    print("❌ 错误：请设置环境变量 MINIO_SECRET_KEY")
    sys.exit(1)
if not BUCKET_NAME:
    print("❌ 错误：请设置环境变量 MINIO_BUCKET")
    sys.exit(1)


RELEASE_PREFIX = "releases"
VERSION_PATTERN = re.compile(r"^v\d+\.\d+$")
CHUNK_SIZE = 1024 * 1024

# 上传时写入 MinIO object metadata。
# MinIO 实际保存时通常会变成 x-amz-meta-md5。
MD5_META_KEY = "md5"


client = Minio(
    MINIO_ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False,
)


def validate_version(version: str) -> None:
    if not VERSION_PATTERN.match(version):
        print(f"❌ 错误：版本号格式不合法：{version}")
        print("✅ 版本号必须符合 vx.y 格式，例如：v4.0、v1.2、v12.34")
        sys.exit(1)


def get_file_md5(file_path: str) -> str:
    """计算本地文件真实 MD5。"""
    if not os.path.exists(file_path):
        return ""

    md5_hash = hashlib.md5()

    with open(file_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def ensure_bucket_exists() -> None:
    try:
        if not client.bucket_exists(BUCKET_NAME):
            print(f"❌ 错误：Bucket 不存在：{BUCKET_NAME}")
            sys.exit(1)
    except S3Error as e:
        print(f"❌ 检查 Bucket 失败：{str(e)}")
        sys.exit(1)


def normalize_metadata_key(key: str) -> str:
    return key.lower().replace("x-amz-meta-", "")


def get_remote_md5(object_name: str) -> str:
    """读取远端对象的真实 MD5。

    优先读取 metadata md5，不再依赖 multipart 不可靠的 ETag。
    兼容旧对象：只有当 ETag 是标准 32 位 hex 时才当作 MD5。
    """
    try:
        stat_info = client.stat_object(BUCKET_NAME, object_name)

        metadata = stat_info.metadata or {}
        normalized = {
            normalize_metadata_key(k): v
            for k, v in metadata.items()
        }

        remote_md5 = normalized.get(MD5_META_KEY, "")
        if remote_md5:
            return remote_md5.strip('"').lower()

        etag = stat_info.etag.strip('"').lower()
        if re.fullmatch(r"[0-9a-f]{32}", etag):
            return etag

        return ""

    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchObject", "NoSuchBucket"):
            return ""

        if getattr(e, "response", None) is not None and e.response.status == 404:
            return ""

        raise


def iter_files(source_dir: Path):
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            yield path


def print_progress(done_bytes: int, total_bytes: int) -> None:
    if total_bytes <= 0:
        sys.stdout.write(f"\rUploaded {done_bytes} bytes")
        sys.stdout.flush()
        return

    progress = done_bytes / total_bytes * 100
    bar_length = 30
    block = int(bar_length * progress / 100)
    bar = "█" * block + "-" * (bar_length - block)

    sys.stdout.write(f"\r[{bar}] {progress:.1f}%")
    sys.stdout.flush()


class ProgressReader:
    """File-like reader with upload progress display."""

    def __init__(self, file_path: Path, total_size: int):
        self.file_path = file_path
        self.total_size = total_size
        self.uploaded = 0
        self.file = open(file_path, "rb")

    def read(self, size: int = -1) -> bytes:
        data = self.file.read(size)

        if data:
            self.uploaded += len(data)
            print_progress(self.uploaded, self.total_size)

        return data

    def close(self) -> None:
        self.file.close()


def upload_file_with_progress(
    local_path: Path,
    object_name: str,
    content_type: str,
    local_md5: str,
) -> None:
    """上传文件，并把自动计算得到的 MD5 写入 object metadata。"""
    total_size = local_path.stat().st_size
    reader = ProgressReader(local_path, total_size)

    try:
        client.put_object(
            BUCKET_NAME,
            object_name,
            reader,
            length=total_size,
            content_type=content_type,
            metadata={
                MD5_META_KEY: local_md5,
            },
        )
        print_progress(total_size, total_size)
        sys.stdout.write("\n")
        sys.stdout.flush()
    finally:
        reader.close()


def upload_one_file(
    local_path: Path,
    object_name: str,
    local_md5: str,
    remote_md5: str,
    dry_run: bool = False,
) -> bool:
    if remote_md5 and local_md5 == remote_md5:
        print(f"\n⏭️  MD5一致，跳过：{object_name}")
        return False

    if remote_md5:
        print(f"\n📤 MD5不一致，重新上传：{object_name}")
        print(f"   local : {local_md5}")
        print(f"   remote: {remote_md5}")
    else:
        print(f"\n📤 远端不存在或缺少真实MD5元数据，开始上传：{object_name}")
        print(f"   local : {local_md5}")

    if dry_run:
        print(f"   DRY-RUN：跳过实际上传：{local_path}")
        return False

    content_type, _ = mimetypes.guess_type(str(local_path))
    content_type = content_type or "application/octet-stream"

    upload_file_with_progress(
        local_path=local_path,
        object_name=object_name,
        content_type=content_type,
        local_md5=local_md5,
    )

    uploaded_md5 = get_remote_md5(object_name)

    if uploaded_md5 != local_md5:
        print(f"❌ 上传后校验失败：{object_name}")
        print(f"   local : {local_md5}")
        print(f"   remote metadata md5: {uploaded_md5}")
        raise RuntimeError(f"Upload MD5 check failed: {object_name}")

    print(f"✅ 上传完成并校验通过：{object_name}")
    return True


def upload_model_dir(version: str, source_dir: Path, dry_run: bool = False) -> None:
    source_dir = source_dir.expanduser().resolve()

    if not source_dir.exists():
        print(f"❌ 错误：源目录不存在：{source_dir}")
        sys.exit(1)

    if not source_dir.is_dir():
        print(f"❌ 错误：输入路径不是目录：{source_dir}")
        sys.exit(1)

    ensure_bucket_exists()

    files = list(iter_files(source_dir))

    if not files:
        print(f"❌ 错误：源目录为空：{source_dir}")
        sys.exit(1)

    print(f"🚀 开始上传模型版本：{version}")
    print(f"📁 源目录：{source_dir}")
    print(f"🪣 Bucket：{BUCKET_NAME}")
    print(f"☁️  目标前缀：{RELEASE_PREFIX}/{version}/")
    print(f"📦 文件数量：{len(files)}")

    uploaded_count = 0
    skipped_count = 0
    failed_count = 0

    for local_path in files:
        relative_path = local_path.relative_to(source_dir).as_posix()
        object_name = f"{RELEASE_PREFIX}/{version}/{relative_path}"

        try:
            local_md5 = get_file_md5(str(local_path))
            remote_md5 = get_remote_md5(object_name)

            uploaded = upload_one_file(
                local_path=local_path,
                object_name=object_name,
                local_md5=local_md5,
                remote_md5=remote_md5,
                dry_run=dry_run,
            )

            if uploaded:
                uploaded_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            failed_count += 1
            print(f"❌ 处理失败：{local_path}")
            print(f"   原因：{str(e)}")

    print()
    print("🎉 上传流程结束")
    print(f"版本：{version}")
    print(f"源目录：{source_dir}")
    print(f"目标：s3://{BUCKET_NAME}/{RELEASE_PREFIX}/{version}/")
    print(f"上传：{uploaded_count}")
    print(f"跳过：{skipped_count}")
    print(f"失败：{failed_count}")

    if failed_count > 0:
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="激光雷达模型上传工具（自动计算 MD5 并写入 MinIO metadata）"
    )

    parser.add_argument(
        "version_dir",
        type=str,
        help="本地版本目录，例如 ./v4.0。默认使用目录名作为远端版本名。",
    )

    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="可选：指定远端版本名，例如 v4.0。默认使用 version_dir 的目录名。",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要上传/跳过的文件，不真正上传。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.version_dir).expanduser()
    version = args.version or source_dir.name

    if not version:
        print("❌ 错误：version 不能为空")
        sys.exit(1)

    validate_version(version)

    upload_model_dir(
        version=version,
        source_dir=source_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()