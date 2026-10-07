"""
Практическая работа №4: Временные ряды и пользовательские сессии
Вариант 6

1. Построить недельную агрегированную таблицу выручки.
2. Для каждой недели посчитать накопительную (expanding) выручку.
3. Определить неделю, после которой общая выручка превысила половину итоговой.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "practice_4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False 
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:,.2f}".format

orders_files = [
    DATA_DIR / "orders_2024_01.csv",
    DATA_DIR / "orders_2024_02.csv",
    DATA_DIR / "orders_2024_03.csv",
]
df_orders = pd.concat([pd.read_csv(f) for f in orders_files], ignore_index=True)
df_orders["order_datetime"] = pd.to_datetime(df_orders["order_datetime"])

df_orders["order_datetime_utc"] = df_orders["order_datetime"].dt.tz_localize("UTC")
df_orders["order_datetime_msk"] = df_orders["order_datetime_utc"].dt.tz_convert("Europe/Moscow")

orders_ts = df_orders.set_index("order_datetime_msk").sort_index()
orders_ts["revenue"] = orders_ts["price"] * orders_ts["quantity"]
print("Заказы с временным индексом:")
print(orders_ts[["order_id", "user_id", "price", "quantity", "revenue"]].head())

print("\n=== Задание 1. Недельная агрегированная таблица выручки ===")
weekly_orders = orders_ts.resample("W").agg(
    orders_count=("order_id", "nunique"),
    items_count=("quantity", "sum"),
    revenue_sum=("revenue", "sum"),
)
weekly_orders["week_start"] = (weekly_orders.index - pd.Timedelta(days=6)).date
weekly_orders["week_end"] = weekly_orders.index.date
daily_counts = orders_ts["order_id"].resample("D").count()
weekly_orders["days_with_orders"] = (daily_counts > 0).resample("W").sum()
weekly_orders["is_full_week"] = weekly_orders["days_with_orders"] == 7
print(weekly_orders[["week_start", "week_end", "days_with_orders", "orders_count",
                     "items_count", "revenue_sum"]].to_string())
print(f"\nПериод данных (MSK): {orders_ts.index.min()} — {orders_ts.index.max()}")
partial = weekly_orders[~weekly_orders["is_full_week"]]
for _, w in partial.iterrows():
    print(f"Неделя {w['week_start']} — {w['week_end']} неполная ({w['days_with_orders']} дн. "
          f"с заказами, выручка {w['revenue_sum']:,.2f}) — её нельзя сравнивать с остальными.")
last_utc, last_msk = df_orders["order_datetime"].max(), orders_ts.index.max()
if last_msk.date() != last_utc.date():
    print(f"Причина: после перевода UTC -> MSK (+3 ч) последние заказы {last_utc.date()} "
          f"сдвинулись на {last_msk.date()} и попали в следующую неделю.")
full = weekly_orders[weekly_orders["is_full_week"]]["revenue_sum"]
print(f"Полные недели: средняя выручка {full.mean():,.2f}, разброс "
      f"{full.min():,.2f} — {full.max():,.2f} (коэффициент вариации {full.std() / full.mean():.1%})")

print("\n=== Задание 2. Накопительная выручка по неделям ===")
weekly_orders["revenue_expanding"] = weekly_orders["revenue_sum"].expanding().sum()
total_revenue = weekly_orders["revenue_sum"].sum()
weekly_orders["expanding_share"] = weekly_orders["revenue_expanding"] / total_revenue
print(weekly_orders[["week_end", "revenue_sum", "revenue_expanding", "expanding_share"]]
      .to_string(formatters={"expanding_share": "{:.1%}".format}))

print("\n=== Задание 3. Неделя преодоления 50% итоговой выручки ===")
half_revenue = total_revenue / 2
over_half = weekly_orders[weekly_orders["revenue_expanding"] > half_revenue]
half_week = over_half.index[0]
half_week_pos = weekly_orders.index.get_loc(half_week)
row = weekly_orders.loc[half_week]
prev_cum = weekly_orders["revenue_expanding"].iloc[half_week_pos - 1] if half_week_pos > 0 else 0

print(f"Итоговая выручка за период: {total_revenue:,.2f}")
print(f"Половина итоговой выручки: {half_revenue:,.2f}")
print(f"Накоплено к концу предыдущей недели: {prev_cum:,.2f} "
      f"({prev_cum / total_revenue:.1%})")
print(f"Половина выручки превышена по итогам недели {row['week_start']} — {row['week_end']} "
      f"(неделя №{half_week_pos + 1} из {len(weekly_orders)}): "
      f"накоплено {row['revenue_expanding']:,.2f} ({row['expanding_share']:.1%})")

daily_cum = orders_ts["revenue"].resample("D").sum().cumsum()
half_day = daily_cum[daily_cum > half_revenue].index[0]
print(f"С точностью до дня: 50% выручки набрано {half_day.date()}")
period_mid = orders_ts.index.min() + (orders_ts.index.max() - orders_ts.index.min()) / 2
print(f"Середина периода по календарю: {period_mid.date()}")
gap_days = (half_day - period_mid.normalize()).days
if full.std() / full.mean() < 0.1 and abs(gap_days) <= 7:
    print("Вывод: недельная выручка почти постоянна, поэтому накопительная кривая растёт "
          "практически линейно и половина выручки набирается около середины периода — "
          "ярко выраженных пиков продаж или сезонности в данных нет.")
else:
    side = "раньше" if gap_days < 0 else "позже"
    print(f"Вывод: половина выручки набрана на {abs(gap_days)} дн. {side} середины периода — "
          "выручка распределена по неделям неравномерно.")

fig, ax1 = plt.subplots(figsize=(11, 5))
x = weekly_orders.index.tz_localize(None)
ax1.bar(x, weekly_orders["revenue_sum"], width=5, color="#4C72B0", label="Выручка за неделю")
ax1.set_ylabel("Выручка за неделю")
ax2 = ax1.twinx()
ax2.plot(x, weekly_orders["revenue_expanding"], color="#DD8452", marker="o",
         label="Накопительная выручка")
ax2.axhline(half_revenue, color="gray", linestyle="--", label="50% итоговой выручки")
ax2.axvline(half_week.tz_localize(None), color="red", linestyle=":", label="Неделя превышения 50%")
ax2.set_ylabel("Накопительная выручка")
ax2.set_ylim(bottom=0)
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
ax1.set_title("Недельная и накопительная выручка")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT_DIR / "weekly_revenue.png", dpi=120)
if SHOW_PLOTS:
    plt.show()
plt.close(fig)

weekly_orders.to_csv(OUT_DIR / "weekly_revenue.csv")
print("\nТаблица и график сохранены в", OUT_DIR)
