"""
Практическая работа №6: Подготовка фич и интеграция источников
Вариант 6

1. Для каждого пользователя посчитать средний чек и медианный чек.
2. Нормализовать эти показатели (Z-нормализация).
3. Найти пользователей с самым высоким и самым низким нормализованным средним чеком.
"""

from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "practice_6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:,.2f}".format

order_files = [
    DATA_DIR / "orders_2024_01.csv",
    DATA_DIR / "orders_2024_02.csv",
    DATA_DIR / "orders_2024_03.csv",
]
df_orders = pd.concat(
    [pd.read_csv(f, parse_dates=["order_datetime"]) for f in order_files],
    ignore_index=True,
)
sheets = pd.read_excel(DATA_DIR / "products.xlsx", sheet_name=["products", "categories"])
df_products = sheets["products"]
df_categories = sheets["categories"]


# category_name есть и на листе products, и на листе categories — без суффиксов pandas
# переименует оба столбца в category_name_x / category_name_y (из-за этого падал скрипт)
df_prod_full = df_products.merge(df_categories, on="category_id", how="left",
                                 suffixes=("", "_cat"))
mismatch = (df_prod_full["category_name"] != df_prod_full["category_name_cat"]).sum()
print(f"Расхождений category_name между листами products и categories: {mismatch}")
df_prod_full = df_prod_full.drop(columns="category_name_cat")

df_orders_prod = df_orders.merge(df_prod_full, on="product_id", how="left")
df_orders_prod["revenue"] = df_orders_prod["price"] * df_orders_prod["quantity"]
print("Заказы + товары + категории:", df_orders_prod.shape)
print(df_orders_prod[["order_id", "user_id", "product_name", "category_name",
                      "price", "quantity", "status", "revenue"]].head())
print(f"Товаров без категории после объединения: {df_orders_prod['category_name'].isna().sum()}")

print("\nПроверка качества:")
print(f"  дубликатов строк: {df_orders_prod.duplicated().sum()}")
print(f"  повторяющихся order_id: {df_orders_prod['order_id'].duplicated().sum()}")
print(f"  цен <= 0: {(df_orders_prod['price'] <= 0).sum()}")
print("  статусы:", df_orders_prod["status"].value_counts().to_dict())

# Чек — это сумма оплаченного заказа: отменённые и возвращённые заказы денег не приносят,
# поэтому в расчёт берём только status == "paid"
df_paid = df_orders_prod[df_orders_prod["status"] == "paid"]
print(f"\nОплаченных строк: {len(df_paid)} из {len(df_orders_prod)}")

order_checks = (
    df_paid
    .groupby(["user_id", "order_id"], as_index=False)
    .agg(check_sum=("revenue", "sum"), items=("quantity", "sum"))
)
print(f"Позиций в одном заказе: {df_paid.groupby('order_id').size().max()} — "
      "каждый заказ состоит из одной позиции, чек = price × quantity")
print("\nСуммы чеков (первые строки):")
print(order_checks.head())

print("\n=== Задание 1. Средний и медианный чек пользователя ===")
user_checks = (
    order_checks
    .groupby("user_id")
    .agg(
        orders_count=("order_id", "nunique"),
        revenue_sum=("check_sum", "sum"),
        avg_check=("check_sum", "mean"),
        median_check=("check_sum", "median"),
    )
)
print(user_checks.head(10).round(2))
print("\nОписательная статистика:")
print(user_checks[["orders_count", "avg_check", "median_check"]].describe().round(2))
avg_gt_median = (user_checks["avg_check"] > user_checks["median_check"]).mean()
print(f"\nСредний чек по всем оплаченным заказам: {order_checks['check_sum'].mean():,.2f}, "
      f"медианный: {order_checks['check_sum'].median():,.2f}")
print(f"У {avg_gt_median:.0%} пользователей средний чек выше медианного: распределение чека "
      "внутри пользователя скошено вправо (крупные заказы с quantity = 3–4 тянут среднее вверх).")

print("\n=== Задание 2. Z-нормализация среднего и медианного чека ===")
features = user_checks[["avg_check", "median_check"]]
features_z = (features - features.mean()) / features.std(ddof=0)
user_checks["avg_check_z"] = features_z["avg_check"]
user_checks["median_check_z"] = features_z["median_check"]
print(user_checks[["avg_check", "avg_check_z", "median_check", "median_check_z"]].head(10).round(3))
print("\nПроверка: после Z-нормализации среднее ≈ 0, стандартное отклонение = 1")
print(user_checks[["avg_check_z", "median_check_z"]].agg(["mean", "std"]).round(3))
print(f"Корреляция Z среднего и Z медианного чека: "
      f"{user_checks['avg_check_z'].corr(user_checks['median_check_z']):.2f}")

print("\n=== Задание 3. Самый высокий и самый низкий Z средний чек ===")
cols = ["orders_count", "revenue_sum", "avg_check", "avg_check_z", "median_check", "median_check_z"]
user_max = user_checks["avg_check_z"].idxmax()
user_min = user_checks["avg_check_z"].idxmin()
print(f"Максимальный Z среднего чека — пользователь {user_max}:")
print(user_checks.loc[[user_max], cols].round(3))
print(f"\nМинимальный Z среднего чека — пользователь {user_min}:")
print(user_checks.loc[[user_min], cols].round(3))

print("\nТоп-5 пользователей с самым высоким Z средний чек:")
print(user_checks.nlargest(5, "avg_check_z")[cols].round(3))
print("\nТоп-5 пользователей с самым низким Z средний чек:")
print(user_checks.nsmallest(5, "avg_check_z")[cols].round(3))

for label, uid in [("максимальный", user_max), ("минимальный", user_min)]:
    user_rows = df_paid[df_paid["user_id"] == uid]
    print(f"\nПользователь {uid} ({label} Z): средняя цена {user_rows['price'].mean():,.2f}, "
          f"среднее quantity {user_rows['quantity'].mean():.2f}, "
          f"категории: {user_rows['category_name'].value_counts().to_dict()}")

z = user_checks["avg_check_z"]
n_outliers = (z.abs() > 3).sum()
cv = user_checks["avg_check"].std() / user_checks["avg_check"].mean()
print(f"\nПользователей с |Z| > 3 по среднему чеку: {n_outliers}")
print("\nВывод:")
print(f"- Средний чек пользователей лежит в диапазоне {user_checks['avg_check'].min():,.0f} — "
      f"{user_checks['avg_check'].max():,.0f} (коэффициент вариации {cv:.1%}); Z-значения от "
      f"{z.min():.2f} до {z.max():.2f}.")
print(f"- Самый высокий нормализованный средний чек у пользователя {user_max} "
      f"(Z = {z.max():.2f}), самый низкий — у {user_min} (Z = {z.min():.2f}).")
if n_outliers == 0:
    print("- Ни один пользователь не выходит за |Z| > 3: «премиальных» или аномальных по чеку "
          "клиентов нет, различия между пользователями умеренные. У каждого пользователя "
          f"около {user_checks['orders_count'].median():.0f} оплаченных заказов, поэтому его "
          "средний чек — устойчивая оценка, и крайние значения объясняются в основном "
          "случайным набором цен и количества в заказах, а не особым поведением.")
else:
    print(f"- {n_outliers} пользователей выходят за |Z| > 3 — это кандидаты в «премиальные» "
          "клиенты, их стоит рассмотреть отдельно.")
print("- Медианный чек менее чувствителен к отдельным крупным заказам, поэтому для сегментации "
      "клиентов полезно использовать оба признака.")

out_path = OUT_DIR / "user_checks.csv"
user_checks.reset_index().to_csv(out_path, index=False)
print("\nПризнаки сохранены:", out_path)
