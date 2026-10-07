"""
Практическая работа №7: Исследовательский анализ данных и базовая визуализация
Вариант 6

1. Построить bar chart по status.
2. Найти долю каждого статуса.
3. Построить pivot_table: weekday × status.
4. Визуализировать pivot через heatmap.
5. Объяснить, в какие дни больше проблемных заказов.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "practice_7"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:,.2f}".format


def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=120)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print("График сохранён:", OUT_DIR / name)


orders_files = [
    DATA_DIR / "orders_2024_01.csv",
    DATA_DIR / "orders_2024_02.csv",
    DATA_DIR / "orders_2024_03.csv",
]
df_orders = pd.concat([pd.read_csv(f) for f in orders_files], ignore_index=True)
df_orders["order_datetime"] = pd.to_datetime(df_orders["order_datetime"])
df_orders["revenue"] = df_orders["price"] * df_orders["quantity"]
df_orders["order_date"] = df_orders["order_datetime"].dt.date
df_orders["order_hour"] = df_orders["order_datetime"].dt.hour
df_orders["weekday"] = df_orders["order_datetime"].dt.day_name()

print(df_orders.head())
print(df_orders.shape)
print(df_orders.isnull().sum())

df_status = df_orders.drop_duplicates(subset="order_id").copy()

bad_values = ["", " ", "none", "null", "unknown", "-"]
print("\nСтатусы до очистки:")
print(df_status["status"].value_counts(dropna=False))
df_status["status"] = (
    df_status["status"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace(bad_values, pd.NA)
    .fillna("unknown")
)

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df_status["weekday"] = pd.Categorical(df_status["weekday"], categories=weekday_order, ordered=True)
print(f"\nУникальных заказов: {len(df_status)} (строк-позиций: {len(df_orders)})")

print("\n=== Задание 1. Bar chart по status ===")
status_counts = df_status["status"].value_counts()
print(status_counts)
ax = status_counts.plot(kind="bar", color=sns.color_palette("deep", len(status_counts)),
                        figsize=(8, 5))
for i, v in enumerate(status_counts):
    ax.text(i, v, f"{v}", ha="center", va="bottom")
plt.title("Количество заказов по статусам")
plt.xlabel("status")
plt.ylabel("Количество заказов")
plt.xticks(rotation=0)
save_plot("status_bar.png")

print("\n=== Задание 2. Доля каждого статуса ===")
status_share = df_status["status"].value_counts(normalize=True)
status_table = pd.DataFrame({"count": status_counts, "share": status_share})
print(status_table.to_string(formatters={"share": "{:.2%}".format}))

print("\n=== Задание 3. pivot_table weekday × status ===")
pivot = pd.pivot_table(
    df_status,
    index="weekday",
    columns="status",
    values="order_id",
    aggfunc="count",
    fill_value=0,
    observed=False,
)
pivot = pivot[status_counts.index]
print(pivot)

pivot_share = pivot.div(pivot.sum(axis=1), axis=0)
print("\nДоли статусов внутри каждого дня недели:")
print(pivot_share.round(3))

print("\n=== Задание 4. Heatmap pivot-таблицы ===")
plt.figure(figsize=(9, 5))
sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues")
plt.title("Количество заказов по дням недели и статусам")
save_plot("weekday_status_heatmap.png")

plt.figure(figsize=(9, 5))
sns.heatmap(pivot_share, annot=True, fmt=".1%", cmap="Reds")
plt.title("Доля статусов внутри дня недели")
save_plot("weekday_status_share_heatmap.png")

print("\n=== Задание 5. Проблемные заказы по дням недели ===")
problem_statuses = [s for s in ["cancelled", "refunded", "unknown"] if s in pivot.columns]
problem = pd.DataFrame({
    "orders_total": pivot.sum(axis=1),
    "problem_orders": pivot[problem_statuses].sum(axis=1),
})
for s in problem_statuses:
    problem[f"{s}_share"] = pivot_share[s]
problem["problem_share"] = problem["problem_orders"] / problem["orders_total"]
print(problem.to_string(formatters={c: "{:.2%}".format for c in problem.columns if c.endswith("share")}))

problem["problem_share"].plot(kind="bar", color="#C44E52", figsize=(8, 5))
plt.axhline(problem["problem_orders"].sum() / problem["orders_total"].sum(),
            color="gray", linestyle="--", label="Среднее за неделю")
plt.title("Доля проблемных заказов (cancelled + refunded + unknown) по дням недели")
plt.ylabel("Доля")
plt.xticks(rotation=45)
plt.legend()
save_plot("problem_share_by_weekday.png")

overall = problem["problem_orders"].sum() / problem["orders_total"].sum()
worst = problem["problem_share"].idxmax()
best = problem["problem_share"].idxmin()
weekend = problem.loc[["Saturday", "Sunday"]]
weekdays = problem.loc[weekday_order[:5]]
weekend_share = weekend["problem_orders"].sum() / weekend["orders_total"].sum()
weekdays_share = weekdays["problem_orders"].sum() / weekdays["orders_total"].sum()
max_abs = problem["problem_orders"].idxmax()

# Значимы ли различия между днями? Критерий хи-квадрат для таблицы
# «день недели × (проблемный / нормальный заказ)» — считаем вручную, без scipy
observed = pd.DataFrame({
    "problem": problem["problem_orders"],
    "ok": problem["orders_total"] - problem["problem_orders"],
})
expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.values.sum()
chi2 = ((observed.values - expected) ** 2 / expected).sum()
dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
CHI2_CRIT_005 = {6: 12.592}  # критическое значение χ² при α = 0.05 для 6 степеней свободы
chi2_crit = CHI2_CRIT_005[dof]
significant = chi2 > chi2_crit
margin = 1.96 * np.sqrt(overall * (1 - overall) / problem["orders_total"].mean())
print(f"\nХи-квадрат: {chi2:.2f} при {dof} степенях свободы, критическое значение "
      f"(α = 0.05): {chi2_crit}. Различия между днями "
      f"{'статистически значимы' if significant else 'статистически НЕ значимы'}.")
print(f"95% доверительный интервал доли проблемных заказов за один день: ±{margin:.1%}")

print("\nВывод:")
print(f"- Доля проблемных заказов в среднем {overall:.1%}.")
print(f"- Больше всего проблемных заказов в относительном выражении: {worst} "
      f"({problem.loc[worst, 'problem_share']:.1%}), меньше всего: {best} "
      f"({problem.loc[best, 'problem_share']:.1%}).")
print(f"- Выходные: {weekend_share:.1%} проблемных заказов, будни: {weekdays_share:.1%}.")
if "cancelled" in problem_statuses:
    print(f"- Основной вклад даёт статус cancelled: максимальная доля отмен — "
          f"{problem['cancelled_share'].idxmax()} ({problem['cancelled_share'].max():.1%}).")
print(f"- По абсолютному числу проблемных заказов лидирует {max_abs} "
      f"({problem.loc[max_abs, 'problem_orders']} шт.), но абсолютные значения зависят от "
      "общего числа заказов в этот день, поэтому сравнивать дни корректнее по долям.")
spread = problem["problem_share"].max() - problem["problem_share"].min()
if significant:
    print(f"- Разница между днями ({spread * 100:.1f} п.п.) статистически значима: в {worst} проблемных "
          "заказов действительно больше — стоит проверить загрузку склада, доставки и "
          "поддержки в этот день.")
else:
    print(f"- Однако разница между лучшим и худшим днём всего {spread * 100:.1f} п.п. при погрешности "
          f"±{margin * 100:.1f} п.п. для одного дня, а критерий хи-квадрат ({chi2:.1f} < {chi2_crit}) не "
          "находит значимых различий. Доля проблемных заказов от дня недели практически не "
          "зависит: «худший» день — скорее случайное колебание, чем закономерность, и искать "
          "причину отмен стоит в других признаках (товар, цена, пользователь).")
