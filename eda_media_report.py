"""
Professional EDA pipeline for a multimodal deepfake dataset.

The dataset is expected to be arranged in folders such as:
    raw/audio/fake/*.mp4
    raw/audio/real/*.mp4
    processed/audio/fake/*.wav
    processed/audio/real/*.wav
    test/audio/fake/*.mp4
    test/audio/real/*.mp4

The script builds a tabular manifest from media files and then generates:
    - Markdown EDA report
    - CSV manifest
    - Statistical summary tables
    - Missing-value, univariate, bivariate, multivariate, and target plots

Usage:
    python eda_media_report.py
    python eda_media_report.py --data-dir path/to/dataset --output-dir reports/eda
    python eda_media_report.py --metadata-limit 5000 --hash-files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import warnings
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".wav", ".flac", ".mp3", ".m4a",
}
ARCHIVE_EXTENSIONS = {".zip", ".gz", ".tar", ".tgz", ".rar", ".7z"}
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dock"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports" / "eda"


@dataclass
class EDAConfig:
    data_dir: Path
    output_dir: Path
    metadata_limit: int | None = None
    hash_files: bool = False
    max_pairplot_rows: int = 2000
    max_plot_categories: int = 20
    alpha: float = 0.05


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return cleaned.strip("_")[:80] or "value"


def to_markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No data available._"
    shown = df.head(max_rows).copy()
    return shown.to_markdown(index=True)


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=True)


def infer_split_modality_label(path: Path, data_dir: Path) -> dict[str, str | None]:
    parts = [p.lower() for p in path.relative_to(data_dir).parts]
    split = next((p for p in parts if p in {"raw", "processed", "test", "train", "val", "valid", "validation"}), None)
    modality = next((p for p in parts if p in {"audio", "video", "image", "frames"}), None)
    label = next((p for p in parts if p in {"fake", "real", "bonafide", "spoof"}), None)
    if label == "bonafide":
        label = "real"
    elif label == "spoof":
        label = "fake"
    return {"split": split, "modality": modality, "label": label}


def parse_filename_features(path: Path) -> dict[str, Any]:
    stem = path.stem
    lower = stem.lower()
    tokens = [token for token in stem.replace("-", "_").split("_") if token]
    ids = [token for token in tokens if token.lower().startswith("id")]
    return {
        "stem": stem,
        "name_length": len(stem),
        "token_count": len(tokens),
        "identity_token_count": len(ids),
        "has_wavtolip_name": "wavtolip" in lower,
        "has_faceswap_name": "faceswap" in lower,
        "has_copy_name": "copy" in lower,
        "has_fake_name": "fake" in lower,
    }


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    try:
        digest = hashlib.md5()
        with path.open("rb") as fh:
            while chunk := fh.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def ffprobe_metadata(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        data = json.loads(result.stdout)
    except Exception:
        return {}

    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})

    def parse_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def parse_fps(value: str | None) -> float | None:
        if not value or "/" not in value:
            return parse_float(value)
        num, den = value.split("/", 1)
        den_f = parse_float(den)
        num_f = parse_float(num)
        if not den_f:
            return None
        return num_f / den_f

    return {
        "duration_s": parse_float(fmt.get("duration")),
        "bit_rate": parse_float(fmt.get("bit_rate")),
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        "frame_count": parse_float(video_stream.get("nb_frames")),
        "audio_sample_rate": parse_float(audio_stream.get("sample_rate")),
        "audio_channels": audio_stream.get("channels"),
    }


def wav_metadata(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            channels = wf.getnchannels()
            duration = frames / sr if sr else None
            return {
                "duration_s": duration,
                "audio_sample_rate": sr,
                "audio_channels": channels,
                "frame_count": frames,
                "audio_codec": "pcm",
            }
    except Exception:
        return {}


def collect_manifest(config: EDAConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    files = [p for p in config.data_dir.rglob("*") if p.is_file()]
    media_files = [p for p in files if p.suffix.lower() in MEDIA_EXTENSIONS]
    if config.metadata_limit is not None:
        media_files = media_files[: config.metadata_limit]

    for idx, path in enumerate(media_files, start=1):
        stat = path.stat()
        inferred = infer_split_modality_label(path, config.data_dir)
        parsed = parse_filename_features(path)
        ext = path.suffix.lower()
        metadata = wav_metadata(path) if ext == ".wav" else ffprobe_metadata(path)
        row = {
            "relative_path": str(path.relative_to(config.data_dir)),
            "file_name": path.name,
            "extension": ext,
            "file_size_bytes": stat.st_size,
            "file_size_mb": stat.st_size / (1024 ** 2),
            "directory_depth": len(path.relative_to(config.data_dir).parts) - 1,
            **inferred,
            **parsed,
            **metadata,
        }
        if config.hash_files:
            row["file_hash"] = file_hash(path)
        rows.append(row)

        if idx % 500 == 0:
            print(f"Indexed {idx:,} media files...")

    archive_rows = []
    for path in files:
        suffixes = "".join(path.suffixes).lower()
        if path.suffix.lower() in ARCHIVE_EXTENSIONS or suffixes.endswith(".tar.gz"):
            stat = path.stat()
            archive_rows.append({
                "relative_path": str(path.relative_to(config.data_dir)),
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "file_size_bytes": stat.st_size,
                "file_size_mb": stat.st_size / (1024 ** 2),
                "split": "archive",
                "modality": None,
                "label": None,
                "stem": path.stem,
                "name_length": len(path.stem),
                "token_count": len(path.stem.split("_")),
                "identity_token_count": 0,
                "has_wavtolip_name": "wavtolip" in path.stem.lower(),
                "has_faceswap_name": "faceswap" in path.stem.lower(),
                "has_copy_name": "copy" in path.stem.lower(),
                "has_fake_name": "fake" in path.stem.lower(),
                "is_archive": True,
            })

    df = pd.DataFrame(rows + archive_rows)
    if not df.empty:
        df["is_archive"] = df.get("is_archive", False).fillna(False)
        df["aspect_ratio"] = np.where(
            (df.get("height").notna() if "height" in df else False) & (df.get("height") != 0),
            df.get("width") / df.get("height"),
            np.nan,
        )
        df["pixels"] = df.get("width", np.nan) * df.get("height", np.nan)
        df["size_per_second_mb"] = df["file_size_mb"] / df["duration_s"].replace(0, np.nan)
        df["label_binary"] = df["label"].map({"real": 0, "fake": 1})
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number, "bool"]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_mb": df.memory_usage(deep=True).sum() / (1024 ** 2),
        "duplicate_records": int(df.duplicated().sum()),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "unique_counts": df.nunique(dropna=False).sort_values(ascending=False),
    }


def missing_analysis(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100,
    }).sort_values("missing_percent", ascending=False)

    plt.figure(figsize=(12, max(5, min(12, df.shape[1] * 0.25))))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="mako")
    plt.title("Missing Value Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Records")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    missing_nonzero = missing[missing["missing_count"] > 0].head(30)
    if missing_nonzero.empty:
        missing_nonzero = missing.head(min(10, len(missing)))
    sns.barplot(x=missing_nonzero["missing_percent"], y=missing_nonzero.index, color="#2a9d8f")
    plt.title("Missing Value Percentage by Column")
    plt.xlabel("Missing Percentage")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_percent_bar.png", dpi=160)
    plt.close()
    return missing


def descriptive_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    nums = numeric_columns(df)
    rows = []
    for col in nums:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        mode = s.mode()
        q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
        mean = s.mean()
        std = s.std()
        rows.append({
            "column": col,
            "count": s.count(),
            "mean": mean,
            "median": q2,
            "mode": mode.iloc[0] if not mode.empty else np.nan,
            "min": s.min(),
            "max": s.max(),
            "range": s.max() - s.min(),
            "variance": s.var(),
            "std": std,
            "skewness": s.skew(),
            "kurtosis": s.kurtosis(),
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "iqr": q3 - q1,
            "coefficient_of_variation": std / mean if mean != 0 else np.nan,
        })
    numeric_summary = pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    cat_tables = {}
    for col in categorical_columns(df):
        freq = df[col].fillna("Missing").value_counts(dropna=False).head(50).to_frame("count")
        freq["percent"] = freq["count"] / len(df) * 100
        cat_tables[col] = freq

    return numeric_summary, cat_tables


def outlier_analysis(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for col in numeric_columns(df):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.nunique() <= 1 or len(s) < 3:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_mask = (s < lower) | (s > upper)
        z = np.abs(stats.zscore(s, nan_policy="omit"))
        z_mask = z > 3
        rows.append({
            "column": col,
            "iqr_lower": lower,
            "iqr_upper": upper,
            "iqr_outliers": int(iqr_mask.sum()),
            "iqr_outlier_percent": iqr_mask.mean() * 100,
            "zscore_outliers": int(z_mask.sum()),
            "zscore_outlier_percent": z_mask.mean() * 100,
            "extreme_min": s[iqr_mask].min() if iqr_mask.any() else np.nan,
            "extreme_max": s[iqr_mask].max() if iqr_mask.any() else np.nan,
        })

        plt.figure(figsize=(8, 3.5))
        sns.boxplot(x=s, color="#e76f51")
        plt.title(f"Boxplot and IQR Outliers: {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_{safe_name(col)}.png", dpi=150)
        plt.close()
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def univariate_analysis(df: pd.DataFrame, output_dir: Path, config: EDAConfig, numeric_summary: pd.DataFrame) -> dict[str, list[str]]:
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    skewed_positive, skewed_negative, near_normal = [], [], []

    for col in nums:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 3 or s.nunique() <= 1:
            continue
        skewness = s.skew()
        if skewness > 0.75:
            skewed_positive.append(col)
        elif skewness < -0.75:
            skewed_negative.append(col)
        elif abs(skewness) < 0.5:
            near_normal.append(col)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(s, kde=True, ax=axes[0], color="#2a9d8f")
        axes[0].set_title(f"Histogram and KDE: {col}")
        axes[0].set_xlabel(col)
        sns.boxplot(x=s, ax=axes[1], color="#f4a261")
        axes[1].set_title(f"Boxplot: {col}")
        axes[1].set_xlabel(col)
        plt.tight_layout()
        plt.savefig(output_dir / f"univariate_{safe_name(col)}.png", dpi=150)
        plt.close()

    for col in cats:
        counts = df[col].fillna("Missing").value_counts().head(config.max_plot_categories)
        if counts.empty:
            continue
        plt.figure(figsize=(10, max(4, min(8, len(counts) * 0.35))))
        sns.barplot(x=counts.values, y=counts.index, color="#457b9d")
        plt.title(f"Count Plot: {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(output_dir / f"countplot_{safe_name(col)}.png", dpi=150)
        plt.close()

    return {
        "positively_skewed": skewed_positive,
        "negatively_skewed": skewed_negative,
        "near_normal_by_skew": near_normal,
    }


def bivariate_analysis(df: pd.DataFrame, output_dir: Path, config: EDAConfig) -> None:
    nums = [c for c in numeric_columns(df) if df[c].nunique(dropna=True) > 1]
    cats = [c for c in categorical_columns(df) if df[c].nunique(dropna=True) > 1]

    if len(nums) >= 2:
        selected_nums = nums[:6]
        sample = df[selected_nums + (["label"] if "label" in df else [])].dropna().sample(
            min(len(df.dropna(subset=selected_nums)), config.max_pairplot_rows),
            random_state=42,
        ) if len(df.dropna(subset=selected_nums)) else pd.DataFrame()
        if not sample.empty:
            sns.pairplot(sample, hue="label" if "label" in sample else None, diag_kind="kde")
            plt.suptitle("Pairplot Matrix", y=1.02)
            plt.savefig(output_dir / "pairplot_matrix.png", dpi=130, bbox_inches="tight")
            plt.close()

        for x, y in zip(selected_nums[:-1], selected_nums[1:]):
            plt.figure(figsize=(7, 5))
            sns.scatterplot(data=df, x=x, y=y, hue="label" if "label" in df else None, alpha=0.65)
            plt.title(f"Scatterplot: {x} vs {y}")
            plt.tight_layout()
            plt.savefig(output_dir / f"scatter_{safe_name(x)}_vs_{safe_name(y)}.png", dpi=150)
            plt.close()

    if "label" in df and nums:
        for col in nums[:12]:
            if df[col].notna().sum() < 3:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.boxplot(data=df, x="label", y=col, ax=axes[0], palette="Set2")
            axes[0].set_title(f"Grouped Boxplot: {col} by Label")
            sns.violinplot(data=df, x="label", y=col, ax=axes[1], palette="Set2", cut=0)
            axes[1].set_title(f"Violin Plot: {col} by Label")
            plt.tight_layout()
            plt.savefig(output_dir / f"label_vs_{safe_name(col)}.png", dpi=150)
            plt.close()

    if len(cats) >= 2:
        for col in cats[:5]:
            if col == "label" or "label" not in df:
                continue
            ct = pd.crosstab(df[col], df["label"], normalize="index").head(config.max_plot_categories)
            if not ct.empty:
                ct.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="viridis")
                plt.title(f"Stacked Category Distribution: {col} vs Label")
                plt.ylabel("Proportion")
                plt.tight_layout()
                plt.savefig(output_dir / f"stacked_{safe_name(col)}_label.png", dpi=150)
                plt.close()


def multivariate_analysis(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nums = [c for c in numeric_columns(df) if df[c].nunique(dropna=True) > 1]
    if not nums:
        return pd.DataFrame(), pd.DataFrame()

    corr = df[nums].corr(method="pearson")
    cov = df[nums].cov()

    plt.figure(figsize=(max(8, len(nums) * 0.6), max(6, len(nums) * 0.5)))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(nums) <= 12, fmt=".2f")
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(max(8, len(nums) * 0.6), max(6, len(nums) * 0.5)))
    sns.heatmap(cov, cmap="mako", annot=False)
    plt.title("Covariance Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "covariance_heatmap.png", dpi=160)
    plt.close()

    return corr, cov


def correlation_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    nums = [c for c in numeric_columns(df) if df[c].nunique(dropna=True) > 1]
    results = {}
    for method in ["pearson", "spearman", "kendall"]:
        if nums:
            results[method] = df[nums].corr(method=method)
        else:
            results[method] = pd.DataFrame()
    return results


def correlation_pairs(corr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if corr.empty:
        empty = pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
        return empty, empty, empty
    pairs = []
    cols = corr.columns
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            value = corr.loc[c1, c2]
            if pd.notna(value):
                pairs.append({"feature_1": c1, "feature_2": c2, "correlation": value})
    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        empty = pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
        return empty, empty, empty
    strongest_pos = pair_df.sort_values("correlation", ascending=False).head(10)
    strongest_neg = pair_df.sort_values("correlation", ascending=True).head(10)
    weak = pair_df[pair_df["correlation"].abs() < 0.1].sort_values("correlation").head(10)
    return strongest_pos, strongest_neg, weak


def target_analysis(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "label" not in df:
        return result

    counts = df["label"].fillna("Missing").value_counts()
    result["class_distribution"] = counts
    result["class_percent"] = counts / counts.sum() * 100
    if len(counts) >= 2:
        result["imbalance_ratio"] = counts.max() / counts.min()
    else:
        result["imbalance_ratio"] = np.nan

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="Set2")
    plt.title("Target Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close()

    if "label_binary" in df:
        nums = [c for c in numeric_columns(df) if c != "label_binary"]
        target_corr = df[nums + ["label_binary"]].corr()["label_binary"].drop("label_binary").sort_values(key=lambda x: x.abs(), ascending=False)
        result["target_correlation"] = target_corr
        result["feature_importance_proxy"] = target_corr.abs().sort_values(ascending=False)
    return result


def data_quality_checks(df: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["duplicate_rows"] = int(df.duplicated().sum())
    if "relative_path" in df:
        checks["duplicate_paths"] = int(df["relative_path"].duplicated().sum())
    if "file_hash" in df:
        checks["duplicate_file_hashes"] = int(df["file_hash"].dropna().duplicated().sum())

    nunique = df.nunique(dropna=False)
    checks["constant_columns"] = nunique[nunique <= 1].index.tolist()

    near_zero = []
    for col in numeric_columns(df):
        value_counts = df[col].value_counts(dropna=False)
        if len(value_counts) <= 1:
            continue
        top_ratio = value_counts.iloc[0] / len(df)
        if top_ratio > 0.95:
            near_zero.append(col)
    checks["near_zero_variance_columns"] = near_zero

    duplicated_feature_pairs = []
    cols = df.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            try:
                if df[c1].equals(df[c2]):
                    duplicated_feature_pairs.append((c1, c2))
            except Exception:
                continue
    checks["duplicated_feature_pairs"] = duplicated_feature_pairs

    invalid = {}
    for col in ["file_size_bytes", "file_size_mb", "duration_s", "width", "height", "fps", "audio_sample_rate"]:
        if col in df:
            invalid[col] = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
    if "fps" in df:
        invalid["fps_above_240"] = int((pd.to_numeric(df["fps"], errors="coerce") > 240).sum())
    if "duration_s" in df:
        invalid["duration_zero_or_negative"] = int((pd.to_numeric(df["duration_s"], errors="coerce") <= 0).sum())
    checks["invalid_value_counts"] = invalid
    return checks


def statistical_tests(df: pd.DataFrame, config: EDAConfig) -> dict[str, pd.DataFrame]:
    alpha = config.alpha
    nums = [c for c in numeric_columns(df) if df[c].nunique(dropna=True) > 1]
    tests: dict[str, list[dict[str, Any]]] = {
        "shapiro": [],
        "ks": [],
        "ttest": [],
        "anova": [],
        "chi_square": [],
    }

    for col in nums:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if 3 <= len(s) <= 5000:
            stat, p = stats.shapiro(s)
            tests["shapiro"].append({
                "column": col, "statistic": stat, "p_value": p,
                "interpretation": "non-normal" if p < alpha else "normality not rejected",
            })
        elif len(s) > 5000:
            sample = s.sample(5000, random_state=42)
            stat, p = stats.shapiro(sample)
            tests["shapiro"].append({
                "column": col, "statistic": stat, "p_value": p,
                "interpretation": "non-normal" if p < alpha else "normality not rejected",
                "note": "sampled 5000 rows",
            })

        if len(s) >= 3:
            z = (s - s.mean()) / s.std() if s.std() else s * 0
            stat, p = stats.kstest(z.dropna(), "norm")
            tests["ks"].append({
                "column": col, "statistic": stat, "p_value": p,
                "interpretation": "distribution differs from normal" if p < alpha else "normality not rejected",
            })

    if "label" in df:
        labels = df["label"].dropna().unique()
        if len(labels) == 2:
            for col in nums:
                groups = [pd.to_numeric(df.loc[df["label"] == label, col], errors="coerce").dropna() for label in labels]
                if all(len(g) >= 2 for g in groups):
                    stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    tests["ttest"].append({
                        "column": col, "group_1": labels[0], "group_2": labels[1],
                        "statistic": stat, "p_value": p,
                        "interpretation": "significant difference" if p < alpha else "no significant difference",
                    })
                    stat, p = stats.f_oneway(*groups)
                    tests["anova"].append({
                        "column": col, "statistic": stat, "p_value": p,
                        "interpretation": "significant group effect" if p < alpha else "no significant group effect",
                    })

        for cat in [c for c in categorical_columns(df) if c != "label"]:
            table = pd.crosstab(df[cat], df["label"])
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                stat, p, dof, _ = stats.chi2_contingency(table)
                tests["chi_square"].append({
                    "feature": cat, "statistic": stat, "p_value": p, "dof": dof,
                    "interpretation": "dependent on target" if p < alpha else "no significant association",
                })

    return {name: pd.DataFrame(rows) for name, rows in tests.items()}


def feature_engineering_insights(df: pd.DataFrame, numeric_summary: pd.DataFrame, missing: pd.DataFrame, outliers: pd.DataFrame) -> list[str]:
    insights = []
    nums = numeric_columns(df)
    cats = categorical_columns(df)

    if nums:
        insights.append("Apply robust scaling or standardization to numerical metadata such as file size, duration, bitrate, fps, and resolution-derived features.")
    skewed = numeric_summary[numeric_summary.get("skewness", pd.Series(dtype=float)).abs() > 1].index.tolist() if not numeric_summary.empty else []
    if skewed:
        insights.append(f"Consider log or Yeo-Johnson transforms for heavily skewed features: {', '.join(skewed[:10])}.")
    high_missing = missing[missing["missing_percent"] > 30].index.tolist()
    if high_missing:
        insights.append(f"Columns with high missingness should be imputed conditionally by media type or excluded: {', '.join(high_missing[:10])}.")
    if not outliers.empty:
        high_outlier_cols = outliers[outliers["iqr_outlier_percent"] > 5].index.tolist()
        if high_outlier_cols:
            insights.append(f"Use robust statistics or clipping for outlier-heavy features: {', '.join(high_outlier_cols[:10])}.")
    high_cardinality = [c for c in cats if df[c].nunique(dropna=True) > 50]
    if high_cardinality:
        insights.append(f"Use frequency encoding, hashing, or feature extraction for high-cardinality categorical fields: {', '.join(high_cardinality[:10])}.")
    if {"width", "height"}.issubset(df.columns):
        insights.append("Resolution-derived features such as aspect ratio, pixel count, and orientation are useful for media quality controls.")
    if {"duration_s", "file_size_mb"}.issubset(df.columns):
        insights.append("Size-per-second and bitrate-like features can capture compression differences but may also create dataset leakage if acquisition pipelines differ by class.")
    insights.append("For deepfake modelling, split by source identity/video/manipulation method rather than random file-level splits to reduce leakage.")
    return insights


def ml_readiness(df: pd.DataFrame, missing: pd.DataFrame, outliers: pd.DataFrame, target: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    score = 100
    risks = []
    if missing["missing_percent"].max() > 30:
        score -= 15
        risks.append("high missingness")
    if not outliers.empty and outliers["iqr_outlier_percent"].max() > 10:
        score -= 10
        risks.append("outlier-heavy features")
    if target.get("imbalance_ratio", 1) and target.get("imbalance_ratio", 1) > 2:
        score -= 15
        risks.append("target imbalance")
    if quality.get("duplicate_rows", 0) > 0 or quality.get("duplicate_file_hashes", 0) > 0:
        score -= 10
        risks.append("duplicates")
    if quality.get("constant_columns"):
        score -= 5
        risks.append("constant columns")
    score = max(0, min(100, score))

    algorithms = [
        "Logistic Regression / Linear SVM on engineered metadata for interpretable baselines",
        "Random Forest or Gradient Boosting for tabular metadata anomaly screening",
        "CNN/Transformer-based visual models for frame-level deepfake detection",
        "CNN + RNN/Transformer audio models for spectrogram-based fake audio detection",
        "Late fusion or calibrated weighted fusion for multimodal scoring",
    ]

    return {
        "model_readiness_score": score,
        "risks": risks,
        "preprocessing_checklist": [
            "Remove duplicates and verify split leakage.",
            "Impute missing metadata by media type.",
            "Scale numerical features.",
            "Encode categorical metadata.",
            "Calibrate thresholds on validation data.",
            "Use stratified and source-disjoint splits.",
        ],
        "recommended_algorithms": algorithms,
    }


def write_report(
    df: pd.DataFrame,
    config: EDAConfig,
    overview: dict[str, Any],
    missing: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    cat_tables: dict[str, pd.DataFrame],
    outliers: pd.DataFrame,
    distribution_notes: dict[str, list[str]],
    corr_results: dict[str, pd.DataFrame],
    target: dict[str, Any],
    quality: dict[str, Any],
    tests: dict[str, pd.DataFrame],
    insights: list[str],
    readiness: dict[str, Any],
) -> None:
    report_path = config.output_dir / "EDA_REPORT.md"
    strongest_pos, strongest_neg, weak = correlation_pairs(corr_results.get("pearson", pd.DataFrame()))

    lines = []
    lines.append("# Exploratory Data Analysis Report\n")
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"- Rows: **{overview['rows']:,}**")
    lines.append(f"- Columns: **{overview['columns']:,}**")
    lines.append(f"- Memory usage: **{overview['memory_mb']:.2f} MB**")
    lines.append(f"- Duplicate records: **{overview['duplicate_records']:,}**")
    lines.append(f"- Numeric columns: **{len(numeric_columns(df))}**")
    lines.append(f"- Categorical columns: **{len(categorical_columns(df))}**\n")
    lines.append("### Columns And Data Types\n")
    dtype_df = pd.DataFrame({"dtype": overview["dtypes"], "unique_values": overview["unique_counts"]})
    lines.append(to_markdown_table(dtype_df, 80))

    lines.append("\n## 2. Missing Value Analysis\n")
    lines.append(to_markdown_table(missing, 80))
    high_missing = missing[missing["missing_percent"] > 30]
    if high_missing.empty:
        lines.append("\nNo columns exceed 30% missingness.")
    else:
        lines.append("\nColumns above 30% missingness require conditional imputation or exclusion.")
    lines.append("\nGenerated plots: `missing_heatmap.png`, `missing_percent_bar.png`.\n")

    lines.append("\n## 3. Descriptive Statistics\n")
    lines.append("### Numerical Summary\n")
    lines.append(to_markdown_table(numeric_summary.round(4), 80))
    lines.append("\n### Categorical Summary\n")
    cat_summary = pd.DataFrame([
        {
            "column": col,
            "cardinality": df[col].nunique(dropna=True),
            "top_category": table.index[0] if not table.empty else None,
            "top_count": int(table.iloc[0]["count"]) if not table.empty else 0,
            "top_percent": float(table.iloc[0]["percent"]) if not table.empty else 0,
        }
        for col, table in cat_tables.items()
    ]).set_index("column") if cat_tables else pd.DataFrame()
    lines.append(to_markdown_table(cat_summary, 80))

    lines.append("\n## 4. Outlier Analysis\n")
    lines.append(to_markdown_table(outliers.round(4), 80))
    lines.append("\nBoxplots are saved as `boxplot_<column>.png`.\n")

    lines.append("\n## 5. Univariate Analysis\n")
    lines.append(f"- Positively skewed features: {', '.join(distribution_notes['positively_skewed'][:20]) or 'None'}")
    lines.append(f"- Negatively skewed features: {', '.join(distribution_notes['negatively_skewed'][:20]) or 'None'}")
    lines.append(f"- Near-normal by skewness: {', '.join(distribution_notes['near_normal_by_skew'][:20]) or 'None'}")
    lines.append("\nHistograms, KDE plots, boxplots, and count plots are saved in the output directory.\n")

    lines.append("\n## 6. Bivariate Analysis\n")
    lines.append("Generated scatterplots, grouped boxplots, violin plots, stacked categorical charts, and a pairplot matrix where applicable.\n")

    lines.append("\n## 7. Multivariate Analysis\n")
    lines.append("Generated `correlation_heatmap.png`, `covariance_heatmap.png`, and `pairplot_matrix.png` where applicable.")
    high_corr = strongest_pos[strongest_pos["correlation"].abs() >= 0.8] if not strongest_pos.empty else pd.DataFrame()
    if high_corr.empty:
        lines.append("\nNo strong positive Pearson correlations above 0.80 were found among analysed numeric features.")
    else:
        lines.append("\nStrong positive Pearson correlations:")
        lines.append(to_markdown_table(high_corr.round(4), 20))

    lines.append("\n## 8. Correlation Analysis\n")
    lines.append("### Strongest Positive Pearson Correlations\n")
    lines.append(to_markdown_table(strongest_pos.round(4), 10))
    lines.append("\n### Strongest Negative Pearson Correlations\n")
    lines.append(to_markdown_table(strongest_neg.round(4), 10))
    lines.append("\n### Weak Pearson Relationships\n")
    lines.append(to_markdown_table(weak.round(4), 10))

    lines.append("\n## 9. Target Variable Analysis\n")
    if target:
        lines.append("### Class Distribution\n")
        target_table = pd.DataFrame({
            "count": target["class_distribution"],
            "percent": target["class_percent"],
        })
        lines.append(to_markdown_table(target_table.round(4), 20))
        lines.append(f"\nImbalance ratio: **{target.get('imbalance_ratio', np.nan):.4f}**")
        if "target_correlation" in target:
            lines.append("\n### Top Target Correlations\n")
            lines.append(to_markdown_table(target["target_correlation"].head(20).to_frame("correlation").round(4), 20))
    else:
        lines.append("No target column was inferred.")

    lines.append("\n## 10. Feature Engineering Insights\n")
    for item in insights:
        lines.append(f"- {item}")

    lines.append("\n## 11. Data Quality Checks\n")
    lines.append(f"- Duplicate rows: **{quality.get('duplicate_rows', 0):,}**")
    lines.append(f"- Duplicate paths: **{quality.get('duplicate_paths', 0):,}**")
    if "duplicate_file_hashes" in quality:
        lines.append(f"- Duplicate file hashes: **{quality.get('duplicate_file_hashes', 0):,}**")
    lines.append(f"- Constant columns: {', '.join(quality.get('constant_columns', [])) or 'None'}")
    lines.append(f"- Near-zero variance columns: {', '.join(quality.get('near_zero_variance_columns', [])) or 'None'}")
    lines.append(f"- Duplicated feature pairs: {quality.get('duplicated_feature_pairs', []) or 'None'}")
    lines.append(f"- Invalid value counts: `{quality.get('invalid_value_counts', {})}`")

    lines.append("\n## 12. Statistical Tests\n")
    for name, table in tests.items():
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append(to_markdown_table(table.round(6) if not table.empty else table, 30))

    lines.append("\n## 13. Visualization Index\n")
    lines.append("The output directory contains heatmaps, histograms, KDE plots, count plots, scatterplots, grouped boxplots, violin plots, pairplots, stacked charts, and target distribution charts.")

    lines.append("\n## 14. Machine Learning Readiness\n")
    lines.append(f"- Model readiness score: **{readiness['model_readiness_score']}/100**")
    lines.append(f"- Main risks: {', '.join(readiness['risks']) or 'None detected'}")
    lines.append("\n### Preprocessing Checklist")
    for item in readiness["preprocessing_checklist"]:
        lines.append(f"- {item}")
    lines.append("\n### Recommended Algorithms")
    for item in readiness["recommended_algorithms"]:
        lines.append(f"- {item}")

    lines.append("\n## 15. Final Summary\n")
    lines.append("- The manifest provides a structured view of raw, processed, test, audio, video, real, and fake media files.")
    if target:
        lines.append(f"- Target imbalance ratio is **{target.get('imbalance_ratio', np.nan):.4f}**; this should guide sampling and evaluation strategy.")
    if not outliers.empty:
        top_outlier = outliers.sort_values("iqr_outlier_percent", ascending=False).head(1)
        lines.append(f"- Most outlier-heavy feature: **{top_outlier.index[0]}** with **{top_outlier.iloc[0]['iqr_outlier_percent']:.2f}%** IQR outliers.")
    if not strongest_pos.empty:
        first = strongest_pos.iloc[0]
        lines.append(f"- Strongest positive correlation: **{first['feature_1']}** vs **{first['feature_2']}** = **{first['correlation']:.4f}**.")
    if not strongest_neg.empty:
        first = strongest_neg.iloc[0]
        lines.append(f"- Strongest negative correlation: **{first['feature_1']}** vs **{first['feature_2']}** = **{first['correlation']:.4f}**.")
    lines.append("- Critical modelling risk: file-level random splits can leak identity, source video, codec, and manipulation-method artifacts.")
    lines.append("- Recommended next step: create source-disjoint train/validation/test manifests before model training.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_eda(config: EDAConfig) -> None:
    ensure_dir(config.output_dir)
    plots_dir = config.output_dir / "plots"
    ensure_dir(plots_dir)

    print("Building media manifest...")
    df = collect_manifest(config)
    if df.empty:
        raise RuntimeError("No supported media files were found.")

    manifest_path = config.output_dir / "media_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"Manifest saved: {manifest_path}")

    overview = dataset_overview(df)
    missing = missing_analysis(df, plots_dir)
    numeric_summary, cat_tables = descriptive_statistics(df)
    outliers = outlier_analysis(df, plots_dir)
    distribution_notes = univariate_analysis(df, plots_dir, config, numeric_summary)
    bivariate_analysis(df, plots_dir, config)
    corr, cov = multivariate_analysis(df, plots_dir)
    corr_results = correlation_analysis(df)
    target = target_analysis(df, plots_dir)
    quality = data_quality_checks(df)
    tests = statistical_tests(df, config)
    insights = feature_engineering_insights(df, numeric_summary, missing, outliers)
    readiness = ml_readiness(df, missing, outliers, target, quality)

    tables_dir = config.output_dir / "tables"
    ensure_dir(tables_dir)
    save_table(missing, tables_dir / "missing_values.csv")
    save_table(numeric_summary, tables_dir / "numeric_summary.csv")
    save_table(outliers, tables_dir / "outlier_summary.csv")
    save_table(corr, tables_dir / "pearson_correlation.csv")
    save_table(cov, tables_dir / "covariance_matrix.csv")
    for name, table in corr_results.items():
        save_table(table, tables_dir / f"{name}_correlation.csv")
    for name, table in tests.items():
        save_table(table, tables_dir / f"{name}_tests.csv")

    write_report(
        df=df,
        config=config,
        overview=overview,
        missing=missing,
        numeric_summary=numeric_summary,
        cat_tables=cat_tables,
        outliers=outliers,
        distribution_notes=distribution_notes,
        corr_results=corr_results,
        target=target,
        quality=quality,
        tests=tests,
        insights=insights,
        readiness=readiness,
    )
    print(f"EDA report saved: {config.output_dir / 'EDA_REPORT.md'}")
    print(f"Plots saved under: {plots_dir}")


def parse_args() -> EDAConfig:
    parser = argparse.ArgumentParser(description="Generate a complete EDA report for a multimodal media dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for report artifacts.")
    parser.add_argument("--metadata-limit", type=int, default=None, help="Optional limit for metadata extraction.")
    parser.add_argument("--hash-files", action="store_true", help="Compute file hashes for duplicate-content detection.")
    parser.add_argument("--max-pairplot-rows", type=int, default=2000)
    parser.add_argument("--max-plot-categories", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    return EDAConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metadata_limit=args.metadata_limit,
        hash_files=args.hash_files,
        max_pairplot_rows=args.max_pairplot_rows,
        max_plot_categories=args.max_plot_categories,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    run_eda(parse_args())
