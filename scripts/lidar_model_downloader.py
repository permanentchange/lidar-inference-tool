from minio import Minio
from minio.error import S3Error

import argparse
import hashlib
import os
import re
import sys


MINIO_ENDPOINT = "minio.zhongkezhichi.com"
ACCESS_KEY = "L01WU1Q4ISVER9HYC4KS"
SECRET_KEY = "qP+IaXOsJoHSjiw28CZ4m8Xk9DqMyliAMwZDj387"

CHUNK_SIZE = 1024 * 1024

client = Minio(
    MINIO_ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=True
)


def md5sum(file_path):
    if not os.path.exists(file_path):
        return ""

    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()


def parse_src(src):
    parts = src.strip("/").split("/")

    if len(parts) < 2:
        raise ValueError("src格式错误，应为 bucket/object")

    bucket = parts[0]
    object_path = "/".join(parts[1:])
    return bucket, object_path


def get_remote_md5(bucket, object_name):
    stat = client.stat_object(bucket, object_name)
    metadata = stat.metadata or {}

    # 优先读取上传时保存的MD5
    for key, value in metadata.items():
        if key.lower().endswith("md5"):
            return value.strip('"').lower()

    # 单part上传时ETag通常为MD5
    etag = stat.etag.strip('"').lower()
    if re.fullmatch(r"[0-9a-f]{32}", etag):
        return etag

    return ""


def download_one(bucket, object_name, local_path):
    try:
        stat = client.stat_object(bucket, object_name)
        remote_md5 = get_remote_md5(bucket, object_name)
    except S3Error as e:
        print(f"❌ 获取远端文件信息失败: {object_name}")
        print(e)
        return False

    local_md5 = md5sum(local_path)

    if remote_md5 and local_md5 and remote_md5 == local_md5:
        print(f"⏭️ MD5一致，跳过: {local_path}")
        return True

    print(f"⬇️ 下载: {object_name}")

    os.makedirs(
        os.path.dirname(os.path.abspath(local_path)),
        exist_ok=True
    )

    response = None

    try:
        response = client.get_object(bucket, object_name)

        downloaded = 0
        total_size = stat.size

        with open(local_path, "wb") as f:
            for data in response.stream(CHUNK_SIZE):
                f.write(data)
                downloaded += len(data)

                if total_size:
                    progress = downloaded / total_size * 100
                    sys.stdout.write(f"\r[{progress:6.2f}%]")
                    sys.stdout.flush()

        if total_size:
            print()

    except Exception as e:
        print(f"\n❌ 下载失败: {object_name}")
        print(e)

        if os.path.exists(local_path):
            os.remove(local_path)

        return False

    finally:
        if response:
            response.close()
            response.release_conn()

    if remote_md5:
        new_md5 = md5sum(local_path)

        if new_md5 != remote_md5:
            print(f"❌ MD5校验失败: {local_path}")
            os.remove(local_path)
            return False

    print(f"✅ 完成: {local_path}")
    return True


def download_file_mode(src, dst):
    bucket, object_name = parse_src(src)

    print(f"Bucket : {bucket}")
    print(f"Object : {object_name}")

    return download_one(bucket, object_name, dst)


def download_dir_mode(src, dst):
    bucket, prefix = parse_src(src)

    if not prefix.endswith("/"):
        prefix += "/"

    print(f"Bucket : {bucket}")
    print(f"Prefix : {prefix}")
    print(f"目标目录: {os.path.abspath(dst)}")

    count = 0
    success = 0

    try:
        objects = client.list_objects(
            bucket,
            prefix=prefix,
            recursive=True
        )

        for obj in objects:
            if obj.is_dir:
                continue

            relative_path = obj.object_name[len(prefix):]
            if not relative_path:
                continue

            local_path = os.path.join(dst, relative_path)

            if download_one(bucket, obj.object_name, local_path):
                success += 1

            count += 1

    except S3Error as e:
        print("❌ MinIO错误:")
        print(e)
        return False

    print(f"🎉 完成，共处理 {count} 个文件，成功 {success} 个")
    return success == count

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MinIO 模型下载工具\n"
            "\n"
            "默认以目录模式下载 MinIO 指定目录下的全部文件，并保持子目录结构。\n"
            "下载前会比较本地文件与远端文件的 MD5，MD5 一致时自动跳过。\n"
            "使用 --is-file 可切换为单文件模式，此模式不会调用 list_objects。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "\n"
            "  1. 使用默认参数下载 latest 目录:\n"
            "     %(prog)s\n"
            "\n"
            "  2. 下载指定模型版本目录:\n"
            "     %(prog)s \\\n"
            "       --src packages/lidar_detection_model/v6.0 \\\n"
            "       --dst ~/lidar_perception_config\n"
            "\n"
            "  3. 下载单个文件:\n"
            "     %(prog)s \\\n"
            "       --src packages/lidar_detection_model/v6.0/end2end.engine \\\n"
            "       --dst ~/lidar_perception_config/end2end.engine \\\n"
            "       --is-file\n"
            "\n"
            "路径说明:\n"
            "  --src 的第一段为 MinIO bucket 名，其余部分为 object/prefix。\n"
            "\n"
            "  例如:\n"
            "    packages/lidar_detection_model/v6.0\n"
            "\n"
            "  将解析为:\n"
            "    bucket = packages\n"
            "    prefix = lidar_detection_model/v6.0/\n"
            "\n"
            "目录模式下载结果示例:\n"
            "  MinIO:\n"
            "    packages/lidar_detection_model/v6.0/end2end.engine\n"
            "    packages/lidar_detection_model/v6.0/model.yaml\n"
            "\n"
            "  执行:\n"
            "    %(prog)s \\\n"
            "      --src packages/lidar_detection_model/v6.0 \\\n"
            "      --dst ~/lidar_perception_config\n"
            "\n"
            "  本地结果:\n"
            "    ~/lidar_perception_config/end2end.engine\n"
            "    ~/lidar_perception_config/model.yaml\n"
        )
    )

    parser.add_argument(
        "--src",
        default="packages/lidar_detection_model/latest",
        metavar="BUCKET/PATH",
        help=(
            "MinIO 源路径。\n"
            "格式: bucket/object 或 bucket/prefix\n"
            "默认: packages/lidar_detection_model/latest"
        )
    )

    parser.add_argument(
        "--dst",
        default="~/lidar_perception_config",
        type=os.path.expanduser,
        metavar="PATH",
        help=(
            "本地目标路径。\n"
            "目录模式下表示目标目录；--is-file 模式下表示目标文件。\n"
            "支持 ~ 用户目录展开。\n"
            "默认: ~/lidar_perception_config"
        )
    )

    parser.add_argument(
        "--is-file",
        action="store_true",
        help=(
            "启用单文件模式。\n"
            "--src 和 --dst 都按单个文件处理，不调用 list_objects。\n"
            "适用于没有 MinIO ListBucket 权限、但有指定对象读取权限的场景。"
        )
    )

    args = parser.parse_args()

    try:
        if args.is_file:
            success = download_file_mode(args.src, args.dst)
        else:
            success = download_dir_mode(args.src, args.dst)
    except ValueError as e:
        print(f"❌ {e}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()