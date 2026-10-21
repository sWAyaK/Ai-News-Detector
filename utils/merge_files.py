#!/usr/bin/env python3
import argparse
import glob
import sys
import pandas as pd
from pathlib import Path


def merge_csv_files_vertical(file_paths, output_path, add_source_col=False,
                             source_col_name='source_file', encoding='utf-8', **read_kwargs):
    """Объединяет CSV файлы вертикально."""
    if not file_paths:
        raise ValueError("Нет файлов для объединения")

    dfs = []
    for p in file_paths:
        df = pd.read_csv(p, encoding=encoding, **read_kwargs)
        if add_source_col:
            df[source_col_name] = Path(p).name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    merged.to_csv(output_path, index=False, encoding=encoding)

    print(f"✅ Объединено {len(dfs)} файлов")
    print(f"📊 Строк: {len(merged)}, столбцов: {len(merged.columns)}")
    print(f"💾 Сохранён: {output_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Объединение CSV файлов вертикально")
    parser.add_argument("--inputs", "-i", nargs="+", required=True,
                        help="Список CSV файлов или шаблонов (например data/*.csv)")
    parser.add_argument("--output", "-o", required=True,
                        help="Выходной CSV файл")
    parser.add_argument("--add-source", action="store_true",
                        help="Добавить колонку с именем исходного файла")
    parser.add_argument("--source-col", default="source_file",
                        help="Имя колонки-источника")
    parser.add_argument("--encoding", default="utf-8",
                        help="Кодировка файлов")
    parser.add_argument("--sep", default=",",
                        help="Разделитель CSV")

    args = parser.parse_args()

    # Разворачиваем возможные wildcard-шаблоны
    file_paths = []
    for pattern in args.inputs:
        matched = glob.glob(pattern, recursive=False)
        if not matched:
            # Если не нашли по шаблону – считаем, что это имя конкретного файла
            # (возможно, он не существует, но проверим позже)
            if Path(pattern).exists():
                matched = [pattern]
            else:
                print(f"⚠️  Файл не найден: {pattern}", file=sys.stderr)
        file_paths.extend(matched)

    file_paths = sorted(set(file_paths))

    if not file_paths:
        print("❌ Ошибка: не найдено ни одного существующего файла", file=sys.stderr)
        sys.exit(1)

    print(f"📄 Найдено файлов для объединения: {len(file_paths)}")
    for f in file_paths:
        print(f"   - {f}")

    try:
        merge_csv_files_vertical(
            file_paths,
            args.output,
            add_source_col=args.add_source,
            source_col_name=args.source_col,
            encoding=args.encoding,
            sep=args.sep
        )
    except Exception as e:
        print(f"❌ Ошибка при объединении: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()