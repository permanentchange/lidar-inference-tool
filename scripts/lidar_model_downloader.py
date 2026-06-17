from minio import Minio
from minio.error import S3Error
import argparse
import os
import re
import sys
import hashlib

# ====================== 从环境变量读取 MinIO 配置 ======================
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = os.getenv("MINIO_BUCKET")

# 检查必要配置是否存在
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
# ====================================================================

client = Minio(
    MINIO_ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False
)

MODEL_FILES = [
    "end2end.engine",
    "end2end.onnx",
    "model.yaml"
]

# ====================== MD5工具函数 ======================
def get_file_md5(file_path: str) -> str:
    """计算本地文件MD5值"""
    if not os.path.exists(file_path):
        return ""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()
# =========================================================

# ====================== 进度条函数 ======================
def download_with_progress(client, bucket, minio_path, local_path, remote_md5):
    try:
        total_size = client.stat_object(bucket, minio_path).size
        abs_local_path = os.path.abspath(local_path)

        print(f"\n📥 文件MD5不一致，开始下载: {os.path.basename(local_path)}")

        with open(local_path, "wb") as f:
            downloaded = 0
            response = client.get_object(bucket, minio_path)

            for data in response.stream(1024 * 1024):
                f.write(data)
                downloaded += len(data)
                progress = downloaded / total_size * 100
                bar_length = 30
                block = int(bar_length * progress / 100)
                bar = "█" * block + "-" * (bar_length - block)
                sys.stdout.write(f"\r[{bar}] {progress:.1f}%")
                sys.stdout.flush()

        # 下载完成二次校验MD5
        local_md5 = get_file_md5(local_path)
        if local_md5 != remote_md5:
            print(f"\n❌ 校验失败：本地MD5({local_md5}) != 远端MD5({remote_md5})")
            os.remove(local_path)
            return False

        sys.stdout.write(f"\n✅ 校验通过，已保存至: {abs_local_path}\n")
        return True
    except S3Error as e:
        print(f"❌ 下载失败: {str(e)}")
        return False
# =========================================================

def get_all_versions():
    versions = []
    for obj in client.list_objects(BUCKET_NAME, prefix="releases/", recursive=False):
        if obj.is_dir:
            dir_name = obj.object_name.strip("/").split("/")[-1]
            if re.match(r"v\d+\.\d+", dir_name):
                versions.append(dir_name)
    versions.sort(key=lambda x: list(map(int, x[1:].split("."))))
    return versions

def get_latest_version():
    versions = get_all_versions()
    return versions[-1] if versions else None

def download_model(version=None, save_dir=None):
    all_versions = get_all_versions()

    if not all_versions:
        print("❌ MinIO 中未找到任何模型版本！")
        return

    if version is None:
        version = get_latest_version()
        print(f"🔍 未指定版本，自动使用最新版：{version}")

    if version not in all_versions:
        print(f"❌ 错误：版本 {version} 不存在！")
        print(f"✅ 可用版本：{', '.join(all_versions)}")
        return

    # 默认保存目录 ~/lidar_perception_config
    if save_dir is None:
        save_dir = os.path.expanduser("~/lidar_perception_config")

    os.makedirs(save_dir, exist_ok=True)
    abs_save_dir = os.path.abspath(save_dir)
    print(f"🚀 开始处理模型版本：{version}，目标目录：{abs_save_dir}")

    for filename in MODEL_FILES:
        minio_path = f"releases/{version}/{filename}"
        local_path = os.path.join(save_dir, filename)

        # 获取远端文件ETag(MD5)
        try:
            stat_info = client.stat_object(BUCKET_NAME, minio_path)
            remote_md5 = stat_info.etag.strip('"')  # etag自带双引号，去除
        except S3Error as e:
            print(f"❌ 获取远端文件信息失败 {minio_path}: {str(e)}")
            continue

        # 对比本地MD5
        local_md5 = get_file_md5(local_path)
        if local_md5 == remote_md5:
            print(f"\n⏭️  {filename} MD5一致，跳过下载")
            continue

        # MD5不同/文件不存在，执行下载
        download_with_progress(client, BUCKET_NAME, minio_path, local_path, remote_md5)

    print(f"\n🎉 版本 {version} 文件处理完成！目录：{abs_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="激光雷达模型下载工具（支持MD5增量校验）")
    parser.add_argument("--version", type=str, default=None, help="指定版本，如 v1.0，不指定则自动最新")
    parser.add_argument("--save-dir", type=str, default=None, help="模型保存目录，默认 ~/lidar_perception_config")
    args = parser.parse_args()
    download_model(version=args.version, save_dir=args.save_dir)