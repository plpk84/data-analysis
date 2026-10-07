"""
Практическая работа №3: Работа с реальными данными
Вариант 6

1. Прочитать все CSV-файлы заказов с помощью glob.glob и pd.concat.
2. Задать оптимальные dtype для числовых и строковых столбцов.
3. Посчитать количество уникальных пользователей и товаров.
4. Сохранить агрегированную статистику (кол-во пользователей, товаров, заказов,
   общая выручка) в summary.json.
5. Попробовать уменьшить потребление памяти за счёт перевода текстовых столбцов в category.
"""

import glob
import json
from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "practice_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:,.2f}".format


def memory_mb(df):
    return df.memory_usage(deep=True).sum() / 1024 ** 2


print("=== Задание 1. Чтение orders_2024_*.csv через glob + concat ===")
files_pattern = str(DATA_DIR / "orders_2024_*.csv")
csv_files = sorted(glob.glob(files_pattern))
print("Найдены файлы:")
for f in csv_files:
    print("  ", Path(f).name)

df_list = [pd.read_csv(f) for f in csv_files]
df_orders_raw = pd.concat(df_list, ignore_index=True)
print("Форма объединённой таблицы:", df_orders_raw.shape)
print(df_orders_raw.head())
print("\nТипы без оптимизации:")
print(df_orders_raw.dtypes)

print("\n=== Задание 2. Подбор оптимальных dtype ===")
print(df_orders_raw[["order_id", "user_id", "product_id", "price", "quantity"]]
      .agg(["min", "max"]))

dtypes = {
    "order_id": "int32",
    "user_id": "int32",
    "product_id": "int16",
    "price": "float32",
    "quantity": "int16",
    "status": "string",
}
parse_dates = ["order_datetime"]

df_orders = pd.concat(
    [pd.read_csv(f, dtype=dtypes, parse_dates=parse_dates) for f in csv_files],
    ignore_index=True,
)
print("\nТипы после оптимизации:")
print(df_orders.dtypes)

print("\n=== Задание 3. Уникальные пользователи и товары ===")
n_users = df_orders["user_id"].nunique()
n_products = df_orders["product_id"].nunique()
print("Уникальных пользователей:", n_users)
print("Уникальных товаров:", n_products)

print("\n=== Задание 4. Сохранение summary.json ===")
revenue = df_orders["price"].astype("float64") * df_orders["quantity"]
summary = {
    "users_count": int(n_users),
    "products_count": int(n_products),
    "orders_count": int(df_orders["order_id"].nunique()),
    "rows_count": int(len(df_orders)),
    "total_revenue": round(float(revenue.sum()), 2),
    "period_start": str(df_orders["order_datetime"].min()),
    "period_end": str(df_orders["order_datetime"].max()),
}
summary_path = OUT_DIR / "summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)
print(json.dumps(summary, ensure_ascii=False, indent=4))
print("Сохранено:", summary_path)

print("\n=== Задание 5. Текстовые столбцы -> category ===")
text_cols = df_orders_raw.select_dtypes(include=["object", "string"]).columns.tolist()
print("Текстовые столбцы в исходной таблице:", text_cols)
for col in text_cols:
    print(f"  {col}: уникальных значений {df_orders_raw[col].nunique()} "
          f"из {len(df_orders_raw)} строк")

df_orders_cat = df_orders.copy()
df_orders_cat["status"] = df_orders_cat["status"].astype("category")

mem_by_col = pd.DataFrame({
    "raw (object)": df_orders_raw.memory_usage(deep=True) / 1024,
    "optimized dtype": df_orders.memory_usage(deep=True) / 1024,
    "optimized + category": df_orders_cat.memory_usage(deep=True) / 1024,
}).round(1)
print("\nПамять по столбцам, KB:")
print(mem_by_col)


mem_raw = memory_mb(df_orders_raw)
mem_opt = memory_mb(df_orders)
mem_cat = memory_mb(df_orders_cat)
memory_table = pd.DataFrame({
    "variant": ["исходные типы", "оптимальные dtype", "dtype + category"],
    "memory_mb": [mem_raw, mem_opt, mem_cat],
})
print("\nИтог по памяти:")
print(memory_table.round(2).to_string(index=False))
print(f"\nСтолбец status: {df_orders['status'].memory_usage(deep=True) / 1024:.1f} KB (string) -> "
      f"{df_orders_cat['status'].memory_usage(deep=True) / 1024:.1f} KB (category)")
