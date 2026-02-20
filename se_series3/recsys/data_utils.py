from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests


MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


@dataclass
class MovieLensData:
    ratings: pd.DataFrame  # columns: user_id, item_id, rating, timestamp
    items: pd.DataFrame  # columns: item_id, title, genres_text


def download_movielens_100k(data_dir: Path) -> Path:
    """Скачивает и распаковывает MovieLens 100k в data_dir/ml-100k."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-100k.zip"
    out_dir = data_dir / "ml-100k"

    if out_dir.exists() and (out_dir / "u.data").exists():
        return out_dir

    if not zip_path.exists():
        r = requests.get(MOVIELENS_100K_URL, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    # В архиве папка ml-100k
    if not out_dir.exists():
        raise FileNotFoundError("Не удалось распаковать ml-100k")

    return out_dir


def load_movielens_100k(root_dir: Path) -> MovieLensData:
    """Загружает рейтинги и фильмы.

    Примечание: u.item имеет кодировку latin-1.
    """
    root_dir = Path(root_dir)
    ratings_path = root_dir / "u.data"
    items_path = root_dir / "u.item"

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    # u.item: item_id|title|release_date|video_release_date|imdb_url|genres(19)
    items_raw = pd.read_csv(
        items_path,
        sep="|",
        header=None,
        encoding="latin-1",
        engine="python",
    )

    item_id = items_raw[0]
    title = items_raw[1]
    genre_cols = list(range(5, 24))
    genres = items_raw[genre_cols]

    # Превращаем one-hot жанры в текст
    genre_names = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    genres_text = []
    for _, row in genres.iterrows():
        gs = [g for g, v in zip(genre_names, row.tolist()) if int(v) == 1]
        genres_text.append(" ".join(gs) if gs else "unknown")

    items = pd.DataFrame({"item_id": item_id, "title": title, "genres_text": genres_text})

    return MovieLensData(ratings=ratings, items=items)


def train_test_split_per_user(ratings: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Holdout split: для каждого пользователя часть взаимодействий в test."""
    rng = pd.RandomState(seed)

    train_parts = []
    test_parts = []

    for user_id, grp in ratings.groupby("user_id"):
        if len(grp) < 2:
            train_parts.append(grp)
            continue

        idx = grp.index.to_list()
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        test_parts.append(ratings.loc[test_idx])
        train_parts.append(ratings.loc[train_idx])

    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)

