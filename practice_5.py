"""
Практическая работа №5: Оптимизация и очистка данных
Вариант 6

1. Найти пользователей с аномально высокой суммарной выручкой (выше 95-го перцентиля).
2. Проверить, есть ли у них дубликаты заказов или странные статусы.
3. Сформировать список таких пользователей для дальнейшего расследования.
"""

from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "practice_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:,.2f}".format

orders_files = [
    DATA_DIR / "orders_2024_01.csv",
    DATA_DIR / "orders_2024_02.csv",
    DATA_DIR / "orders_2024_03.csv",
]
orders_dtypes = {
    "order_id": "int32",
    "user_id": "int32",
    "product_id": "int16",
    "price": "float32",
    "quantity": "int16",
    "status": "category",
}
df_orders = pd.concat(
    [pd.read_csv(f, dtype=orders_dtypes, parse_dates=["order_datetime"]) for f in orders_files],
    ignore_index=True,
)
df_orders["revenue"] = df_orders["price"].astype("float64") * df_orders["quantity"]
print("Заказы:", df_orders.shape)
print(df_orders.head())

# --- Флаги качества данных на уровне строки ---
order_cols = ["order_id", "user_id", "order_datetime", "product_id", "price", "quantity", "status"]
# 1) полные дубликаты строк
df_orders["is_full_dup"] = df_orders.duplicated(subset=order_cols, keep="first")
# 2) повтор идентификатора заказа
df_orders["is_id_dup"] = df_orders.duplicated(subset="order_id", keep="first")
# 3) «повторные» заказы: тот же пользователь и тот же товар с разницей меньше REPEAT_WINDOW —
#    возможный двойной клик / повторная отправка формы с новым order_id
REPEAT_WINDOW = pd.Timedelta(minutes=10)
REPEAT_LABEL = "10 мин"
df_orders = df_orders.sort_values(["user_id", "product_id", "order_datetime"])
gap = df_orders.groupby(["user_id", "product_id"])["order_datetime"].diff()
df_orders["is_repeat"] = gap < REPEAT_WINDOW
df_orders = df_orders.sort_index()

bad_values = ["", " ", "none", "null", "unknown", "-"]
valid_statuses = ["paid", "cancelled", "refunded"]
df_orders["status_clean"] = (
    df_orders["status"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace(bad_values, pd.NA)
)
df_orders["is_strange_status"] = ~df_orders["status_clean"].isin(valid_statuses)
df_orders["is_bad_value"] = (df_orders["price"] <= 0) | (df_orders["quantity"] <= 0)
df_orders["is_cancelled"] = df_orders["status_clean"] == "cancelled"
df_orders["is_refunded"] = df_orders["status_clean"] == "refunded"
df_orders["revenue_paid"] = df_orders["revenue"].where(df_orders["status_clean"] == "paid", 0)

print("\nСтатусы в исходных данных:")
print(df_orders["status"].value_counts(dropna=False))
print("\nПроверка качества по всей таблице:")
print(f"  полных дубликатов строк: {df_orders['is_full_dup'].sum()}")
print(f"  повторов order_id: {df_orders['is_id_dup'].sum()}")
print(f"  повторных заказов (тот же user + товар < {REPEAT_LABEL}): {df_orders['is_repeat'].sum()}")
print(f"  нестандартных статусов: {df_orders['is_strange_status'].sum()}")
print(f"  цен/количеств <= 0: {df_orders['is_bad_value'].sum()}")

print("\n=== Задание 1. Пользователи с аномально высокой выручкой ===")
user_revenue = (
    df_orders
    .groupby("user_id")
    .agg(
        revenue_sum=("revenue", "sum"),
        revenue_paid=("revenue_paid", "sum"),
        orders_count=("order_id", "nunique"),
        rows_count=("order_id", "size"),
    )
)
p95 = user_revenue["revenue_sum"].quantile(0.95)
median = user_revenue["revenue_sum"].median()
print(f"Всего пользователей: {len(user_revenue)}")
print(f"95-й перцентиль суммарной выручки: {p95:,.2f}")
print(f"Медиана суммарной выручки: {median:,.2f}")

anomal_users = user_revenue[user_revenue["revenue_sum"] > p95].sort_values(
    "revenue_sum", ascending=False)
print(f"Пользователей выше 95-го перцентиля: {len(anomal_users)}")
print(f"Их доля в общей выручке: "
      f"{anomal_users['revenue_sum'].sum() / user_revenue['revenue_sum'].sum():.1%}")
print(anomal_users)

# p95 по определению всегда отсекает ~5% пользователей; проверим, выбросы ли это статистически
q1, q3 = user_revenue["revenue_sum"].quantile([0.25, 0.75])
iqr_upper = q3 + 1.5 * (q3 - q1)
z_revenue = ((user_revenue["revenue_sum"] - user_revenue["revenue_sum"].mean())
             / user_revenue["revenue_sum"].std(ddof=0))
print(f"\nВерхняя граница IQR: {iqr_upper:,.2f}; пользователей выше неё: "
      f"{(user_revenue['revenue_sum'] > iqr_upper).sum()}")
print(f"Максимальный Z суммарной выручки: {z_revenue.max():.2f}")
print(f"Максимум выше медианы на {anomal_users['revenue_sum'].max() / median - 1:.1%}")

print("\n=== Задание 2. Проверка дубликатов и странных статусов ===")
quality = (
    df_orders
    .groupby("user_id")
    .agg(
        full_dup_count=("is_full_dup", "sum"),
        id_dup_count=("is_id_dup", "sum"),
        repeat_count=("is_repeat", "sum"),
        strange_status_count=("is_strange_status", "sum"),
        bad_value_count=("is_bad_value", "sum"),
        cancelled_count=("is_cancelled", "sum"),
        refunded_count=("is_refunded", "sum"),
    )
)
user_stats = user_revenue.join(quality)
user_stats["cancelled_share"] = user_stats["cancelled_count"] / user_stats["rows_count"]
user_stats["refunded_share"] = user_stats["refunded_count"] / user_stats["rows_count"]
user_stats["problem_share"] = user_stats["cancelled_share"] + user_stats["refunded_share"]
user_stats["unpaid_revenue_share"] = 1 - user_stats["revenue_paid"] / user_stats["revenue_sum"]
user_stats["is_anomal"] = user_stats["revenue_sum"] > p95

# «Странный» профиль статусов — доля отмен+возвратов выше 95-го перцентиля по всем пользователям
problem_p95 = user_stats["problem_share"].quantile(0.95)
user_stats["high_problem_share"] = user_stats["problem_share"] > problem_p95

# Если убрать отменённые и возвращённые заказы — останется ли пользователь «аномальным»?
p95_paid = user_stats["revenue_paid"].quantile(0.95)
user_stats["is_anomal_paid"] = user_stats["revenue_paid"] > p95_paid

group_compare = user_stats.groupby("is_anomal").agg(
    users=("revenue_sum", "size"),
    revenue_mean=("revenue_sum", "mean"),
    revenue_paid_mean=("revenue_paid", "mean"),
    rows_mean=("rows_count", "mean"),
    cancelled_share=("cancelled_share", "mean"),
    refunded_share=("refunded_share", "mean"),
    unpaid_revenue_share=("unpaid_revenue_share", "mean"),
    repeats_mean=("repeat_count", "mean"),
)
group_compare.index = group_compare.index.map({False: "обычные", True: "выше p95"})
print("Сравнение групп (средние по пользователям):")
print(group_compare.to_string(formatters={
    c: "{:.1%}".format for c in ["cancelled_share", "refunded_share", "unpaid_revenue_share"]}))

anomal_stats = user_stats[user_stats["is_anomal"]].sort_values("revenue_sum", ascending=False)
n_anom = len(anomal_stats)
print(f"\nУ {(anomal_stats['full_dup_count'] > 0).sum()} из {n_anom} есть полные дубликаты строк")
print(f"У {(anomal_stats['id_dup_count'] > 0).sum()} из {n_anom} есть повторы order_id")
print(f"У {(anomal_stats['repeat_count'] > 0).sum()} из {n_anom} есть повторные заказы "
      f"того же товара в течение {REPEAT_LABEL}")
print(f"У {(anomal_stats['strange_status_count'] > 0).sum()} из {n_anom} есть "
      "пропущенные/нестандартные статусы")
print(f"У {(anomal_stats['bad_value_count'] > 0).sum()} из {n_anom} есть цены/количества <= 0")
print(f"Порог высокой доли отмен+возвратов (p95 по пользователям): {problem_p95:.1%}; "
      f"превышают его {anomal_stats['high_problem_share'].sum()} из {n_anom}")
print(f"95-й перцентиль оплаченной выручки: {p95_paid:,.2f}; остаются выше него "
      f"{anomal_stats['is_anomal_paid'].sum()} из {n_anom}")

print("\nДетали по аномальным пользователям:")
print(anomal_stats[["revenue_sum", "revenue_paid", "rows_count", "full_dup_count", "id_dup_count",
                    "repeat_count", "strange_status_count", "cancelled_share", "refunded_share",
                    "is_anomal_paid"]]
      .to_string(formatters={"cancelled_share": "{:.1%}".format,
                             "refunded_share": "{:.1%}".format}))

repeat_rows = df_orders[df_orders["is_repeat"] & df_orders["user_id"].isin(anomal_stats.index)]
if len(repeat_rows):
    first = repeat_rows.iloc[0]
    uid, pid, t = first["user_id"], first["product_id"], first["order_datetime"]
    example = df_orders[(df_orders["user_id"] == uid) & (df_orders["product_id"] == pid)
                        & df_orders["order_datetime"].between(t - REPEAT_WINDOW, t)]
    print(f"\nПример повторного заказа: пользователь {uid}, товар {pid}")
    print(example.sort_values("order_datetime")[order_cols].to_string(index=False))

print("\n=== Задание 3. Список для расследования ===")


def investigation_reason(row):
    reasons = []
    if row["full_dup_count"] > 0:
        reasons.append(f"полные дубликаты: {int(row['full_dup_count'])}")
    if row["id_dup_count"] > 0:
        reasons.append(f"повторы order_id: {int(row['id_dup_count'])}")
    if row["strange_status_count"] > 0:
        reasons.append(f"нестандартные статусы: {int(row['strange_status_count'])}")
    if row["bad_value_count"] > 0:
        reasons.append(f"цена/количество <= 0: {int(row['bad_value_count'])}")
    if not row["is_anomal_paid"]:
        reasons.append("без отмен и возвратов выручка не аномальна")
    if row["repeat_count"] > 0:
        reasons.append(f"повторные заказы того же товара < {REPEAT_LABEL}: "
                       f"{int(row['repeat_count'])}")
    if row["high_problem_share"]:
        reasons.append(f"высокая доля отмен/возвратов ({row['problem_share']:.0%})")
    if not reasons:
        reasons.append("высокая выручка без технических ошибок — проверить как крупного клиента")
    return "; ".join(reasons)


def priority(row):
    # высокий — технические ошибки данных; средний — подозрительное поведение
    if (row["full_dup_count"] or row["id_dup_count"] or row["strange_status_count"]
            or row["bad_value_count"]):
        return "высокий"
    if row["repeat_count"] or row["high_problem_share"] or not row["is_anomal_paid"]:
        return "средний"
    return "низкий"


investigation = anomal_stats.copy()
investigation["reason"] = investigation.apply(investigation_reason, axis=1)
investigation["priority"] = investigation.apply(priority, axis=1)
investigation = investigation.reset_index()[[
    "user_id", "priority", "revenue_sum", "revenue_paid", "orders_count", "rows_count",
    "full_dup_count", "id_dup_count", "repeat_count", "strange_status_count", "bad_value_count",
    "cancelled_count", "refunded_count", "problem_share", "reason",
]]
priority_order = {"высокий": 0, "средний": 1, "низкий": 2}
investigation["priority_rank"] = investigation["priority"].map(priority_order)
investigation = (
    investigation
    .sort_values(["priority_rank", "revenue_sum"], ascending=[True, False])
    .drop(columns="priority_rank")
)
print(investigation["priority"].value_counts())
print(investigation.round(3).to_string(index=False))

out_path = OUT_DIR / "users_for_investigation.csv"
investigation.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\nСписок сохранён:", out_path)

print("\nВывод:")
n_iqr = (user_revenue["revenue_sum"] > iqr_upper).sum()
print(f"- Порог аномальной выручки (p95) = {p95:,.0f}; выше него {n_anom} пользователей, "
      f"которые дают {anomal_stats['revenue_sum'].sum() / user_stats['revenue_sum'].sum():.1%} "
      "выручки.")
if n_iqr == 0:
    print(f"- Статистическими выбросами они не являются: по правилу IQR выбросов нет, "
          f"максимальный Z = {z_revenue.max():.2f}, лидер всего на "
          f"{anomal_stats['revenue_sum'].max() / median - 1:.0%} выше медианы. Перцентильный "
          "порог всегда отсекает верхние 5%, даже когда распределение ровное.")
else:
    print(f"- По правилу IQR выбросов {n_iqr} — у них выручка действительно выделяется.")
tech_issues = int(anomal_stats[["full_dup_count", "id_dup_count", "strange_status_count",
                                "bad_value_count"]].sum().sum())
if tech_issues == 0:
    print("- Дубликатов строк, повторов order_id, нестандартных статусов и некорректных цен "
          "у них нет: высокая выручка не является артефактом загрузки данных.")
else:
    print(f"- Найдено {tech_issues} технических проблем (дубликаты, статусы, цены) — "
          "часть выручки может быть артефактом загрузки данных.")
print(f"- Доля отмен у аномальных пользователей "
      f"{group_compare.loc['выше p95', 'cancelled_share']:.1%} против "
      f"{group_compare.loc['обычные', 'cancelled_share']:.1%} у остальных; выручка от "
      f"неоплаченных заказов — {group_compare.loc['выше p95', 'unpaid_revenue_share']:.1%} против "
      f"{group_compare.loc['обычные', 'unpaid_revenue_share']:.1%}. По оплаченной выручке выше "
      f"p95 остаются {anomal_stats['is_anomal_paid'].sum()} из {n_anom}.")
print(f"- Повторные заказы того же товара в течение {REPEAT_LABEL} есть у "
      f"{(anomal_stats['repeat_count'] > 0).sum()} из {n_anom} — их стоит проверить на двойное "
      "оформление.")
print("- Пользователи без технических ошибок и с обычной долей отмен — реальные крупные "
      "клиенты; расследование начинают с высокого и среднего приоритета.")
