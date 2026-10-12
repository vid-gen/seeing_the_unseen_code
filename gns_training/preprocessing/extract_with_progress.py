#!/usr/bin/env python3
"""带进度条的 tar.gz 解压工具"""
import subprocess
import os
import sys
from tqdm import tqdm

def extract_with_progress(archive_path, extract_dir=None):
    """解压 tar.gz 文件并显示进度条"""
    if not os.path.exists(archive_path):
        print(f"ERROR: {archive_path} not found")
        sys.exit(1)

    file_size = os.path.getsize(archive_path)
    archive_name = os.path.basename(archive_path)

    # 切换到解压目录
    if extract_dir:
        os.chdir(extract_dir)

    # 打开压缩文件，通过管道传给 tar
    with open(archive_path, 'rb') as f:
        process = subprocess.Popen(
            ['tar', '-xzf', '-'],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        chunk_size = 1024 * 1024  # 1MB chunks
        with tqdm(total=file_size, unit='B', unit_scale=True,
                  desc=f"Extracting {archive_name}") as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                process.stdin.write(chunk)
                pbar.update(len(chunk))

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            print(f"ERROR: tar failed: {stderr}")
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_with_progress.py <archive.tar.gz> [extract_dir]")
        sys.exit(1)

    archive = sys.argv[1]
    extract_dir = sys.argv[2] if len(sys.argv) > 2 else None
    extract_with_progress(archive, extract_dir)
