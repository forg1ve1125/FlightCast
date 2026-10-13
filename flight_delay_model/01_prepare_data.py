# -*- coding: utf-8 -*-
"""
步骤 1：数据清洗（文档推荐的方式一）
根据原始数据集 flights_sample_3m.csv，新建一个"去除掉不使用字段"的数据集
flights_sample_3m_clean.csv——既保留原始数据集，又得到干净的训练/测试数据集。

同时生成 data_manifest.json（原始文件MD5、行数、删除字段清单等），
用于后续验证"原始数据集从未被修改"。

用法：
    python 01_prepare_data.py                        # 使用默认路径
    python 01_prepare_data.py --original xx.csv --out-dir /some/dir
"""
import argparse
import hashlib
import json
from datetime import datetime

import pandas as pd

import config


def md5_of_file(path, chunk_size=1 << 20):
    """分块计算文件MD5（文件较大也不会占内存）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def prepare(original_csv, output_csv, manifest_json):
    original_csv = str(original_csv)
    print(f"[1/4] 读取原始数据集: {original_csv}")
    df = pd.read_csv(original_csv, low_memory=False)
    original_rows, original_cols = df.shape
    print(f"      原始数据: {original_rows:,} 行 x {original_cols} 列")

    print(f"[2/4] 校验并删除 {len(config.EXCLUDED_FIELDS)} 个不使用的字段")
    missing = [c for c in config.EXCLUDED_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"文档要求排除的字段在数据集中不存在: {missing}")
    df_clean = df.drop(columns=config.EXCLUDED_FIELDS)
    print(
        f"      清洗后: {df_clean.shape[0]:,} 行 x {df_clean.shape[1]} 列 "
        f"(共删除 {original_cols - df_clean.shape[1]} 列)"
    )

    print(f"[3/4] 写出新数据集: {output_csv}")
    df_clean.to_csv(output_csv, index=False)

    print("[4/4] 生成清洗清单 data_manifest.json")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "original_csv": str(original_csv),
        "original_md5": md5_of_file(original_csv),
        "original_rows": int(original_rows),
        "original_cols": int(original_cols),
        "clean_csv": str(output_csv),
        "clean_rows": int(df_clean.shape[0]),
        "clean_cols": int(df_clean.shape[1]),
        "excluded_fields": config.EXCLUDED_FIELDS,
        "kept_fields": list(df_clean.columns),
    }
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("完成。原始数据集未被改动；新数据集与清单已生成。")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="生成去除不使用字段的新数据集（方式一）")
    ap.add_argument("--original", default=str(config.ORIGINAL_CSV), help="原始CSV路径")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 Dataset/）")
    args = ap.parse_args()

    out_dir = config.Path(args.out_dir) if args.out_dir else config.DATASET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / config.CLEAN_CSV.name
    manifest_json = out_dir / config.MANIFEST_JSON.name

    prepare(args.original, output_csv, manifest_json)


if __name__ == "__main__":
    main()
