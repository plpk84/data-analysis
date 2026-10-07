"""
Практическая работа №8: Аналитические выводы, сравнение групп и интерпретация визуализаций
Вариант 6

1. Построить heatmap корреляций.
2. Найти самую сильную связь.
3. Построить scatter plot для этой пары признаков.
4. Проверить наличие выбросов.
5. Написать интерпретацию.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "practice_8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False  # True — показывать графики в окне (в Jupyter графики видны всегда)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:,.2f}".format  # без экспоненциальной записи


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
df_orders["weekday_num"] = df_orders["order_datetime"].dt.dayofweek  # 0 = понедельник

numeric_cols = ["price", "quantity", "revenue", "order_hour", "weekday_num"]
print(df_orders[numeric_cols].describe().round(2))

print("\n=== Задание 1. Матрица корреляций ===")
corr = df_orders[numeric_cols].corr()
print(corr.round(3))

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Матрица корреляций (Пирсон)")
save_plot("corr_heatmap.png")

print("\n=== Задание 2. Самая сильная связь ===")
mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
pairs = (
    corr.where(mask_upper)
    .stack()
    .dropna()
    .rename("r")
    .to_frame()
)
pairs["abs_r"] = pairs["r"].abs()
pairs = pairs.sort_values("abs_r", ascending=False)
print("Все пары признаков по силе связи:")
print(pairs.round(3))

(x_col, y_col), strongest = pairs.index[0], pairs.iloc[0]
print(f"\nСамая сильная связь: {x_col} — {y_col}, r = {strongest['r']:.3f}")
print(f"Вторая по силе: {pairs.index[1][0]} — {pairs.index[1][1]}, r = {pairs.iloc[1]['r']:.3f}")

print(f"\n=== Задание 3. Scatter plot {x_col} vs {y_col} ===")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_orders, x=x_col, y=y_col, hue="quantity", palette="viridis",
                alpha=0.5, s=15)
plt.title(f"Зависимость {y_col} от {x_col} (r = {strongest['r']:.2f})")
save_plot(f"scatter_{x_col}_{y_col}.png")

print("\n=== Задание 4. Проверка выбросов (IQR) ===")


def spearman(a, b):
    """Корреляция Спирмена = корреляция Пирсона между рангами (без scipy)."""
    return a.rank().corr(b.rank())


def iqr_bounds(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


outlier_mask = pd.Series(False, index=df_orders.index)
for col in [x_col, y_col]:
    lower, upper = iqr_bounds(df_orders[col])
    col_mask = (df_orders[col] < lower) | (df_orders[col] > upper)
    outlier_mask |= col_mask
    print(f"{col}: границы [{lower:,.2f}; {upper:,.2f}], выбросов {col_mask.sum()} "
          f"({col_mask.mean():.1%})")

non_positive = (df_orders["price"] <= 0) | (df_orders["quantity"] <= 0)
print(f"Некорректных цен или количеств (<= 0): {non_positive.sum()}")
print(f"Строк-выбросов хотя бы по одному признаку пары: {outlier_mask.sum()} "
      f"({outlier_mask.mean():.1%})")
print("\nПримеры выбросов (максимальные значения):")
print(df_orders[outlier_mask].nlargest(5, y_col)[["order_id", "price", "quantity", "revenue"]])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x=df_orders[x_col], ax=axes[0])
axes[0].set_title(f"Boxplot {x_col}")
sns.boxplot(x=df_orders[y_col], ax=axes[1])
axes[1].set_title(f"Boxplot {y_col}")
save_plot("boxplots_pair.png")

# Насколько выбросы влияют на коэффициент корреляции
df_clean = df_orders[~outlier_mask & ~non_positive]
r_all = df_orders[x_col].corr(df_orders[y_col])
r_clean = df_clean[x_col].corr(df_clean[y_col])
rho_all = spearman(df_orders[x_col], df_orders[y_col])
robust = pd.DataFrame({
    "rows": [len(df_orders), len(df_clean)],
    "pearson_r": [r_all, r_clean],
    "spearman_rho": [rho_all, spearman(df_clean[x_col], df_clean[y_col])],
}, index=["все данные", "без выбросов"])
print("\nУстойчивость связи к выбросам:")
print(robust.round(3))

outliers = df_orders[outlier_mask]
if len(outliers):
    print(f"\nСостав выбросов: quantity = {sorted(outliers['quantity'].unique().tolist())}, "
          f"price от {outliers['price'].min():,.2f} до {outliers['price'].max():,.2f}")
print("\nАсимметрия (skew) признаков:")
print(df_orders[["price", "quantity", "revenue"]].skew().round(2).to_string())

plt.figure(figsize=(8, 6))
plt.scatter(df_clean[x_col], df_clean[y_col], s=8, alpha=0.4, label="обычные")
plt.scatter(outliers[x_col], outliers[y_col], s=12, alpha=0.8, color="red",
            label="выбросы (IQR)")
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.legend()
plt.title(f"{y_col} vs {x_col}: выбросы по IQR выделены")
save_plot(f"scatter_{x_col}_{y_col}_outliers.png")

print("\n=== Задание 5. Интерпретация ===")
r_q = corr.loc["quantity", "revenue"]
weak = pairs.iloc[-1]
cv_price = df_orders["price"].std() / df_orders["price"].mean()
cv_qty = df_orders["quantity"].std() / df_orders["quantity"].mean()
# Какую часть разброса цены объясняет сам товар (межгрупповая дисперсия / общая)
product_mean_price = df_orders.groupby("product_id")["price"].transform("mean")
price_by_product = product_mean_price.var() / df_orders["price"].var()

print(f"1. Наблюдение: самая сильная связь — {x_col} и {y_col} (r = {r_all:.2f}), "
      f"на втором месте quantity–revenue (r = {r_q:.2f}). Время заказа (час, день недели) "
      f"с суммами не связано: у остальных пар |r| <= {pairs['abs_r'].iloc[2]:.3f}, самая слабая "
      f"{weak.name[0]}–{weak.name[1]} (r = {weak['r']:.3f}).")
if {x_col, y_col} == {"price", "revenue"}:
    print(f"2. Интерпретация: revenue = price × quantity, поэтому связь «техническая» — один "
          "признак вычислен из другого. Цена меняется от "
          f"{df_orders['price'].min():,.0f} до {df_orders['price'].max():,.0f} "
          f"(коэф. вариации {cv_price:.0%}), количество — от {df_orders['quantity'].min()} до "
          f"{df_orders['quantity'].max()} (коэф. вариации {cv_qty:.0%}); вклад цены в разброс "
          "выручки больше, поэтому r(price, revenue) выше. На scatter plot видны "
          f"{df_orders['quantity'].nunique()} «луча» — по одному на каждое значение quantity: "
          "разброс вокруг общей линии создаёт именно количество.")
else:
    print(f"2. Интерпретация: {x_col} и {y_col} растут вместе; если один признак вычислен из "
          "другого (revenue = price × quantity), связь техническая и не описывает поведение "
          "покупателей.")
if len(outliers) and outliers["quantity"].min() == df_orders["quantity"].max():
    nature = (f"все они — заказы с максимальным quantity = {df_orders['quantity'].max()} и "
              f"ценой от {outliers['price'].min():,.0f}, т.е. это естественный верхний хвост "
              "произведения price × quantity (revenue скошена вправо, skew = "
              f"{df_orders['revenue'].skew():.2f}), а не ошибки")
else:
    nature = "их стоит проверить отдельно"
change = "почти не меняется" if abs(r_clean - r_all) < 0.02 else "заметно меняется"
out_counts = {col: ((df_orders[col] < lo) | (df_orders[col] > hi)).sum()
              for col, (lo, hi) in ((c, iqr_bounds(df_orders[c])) for c in [x_col, y_col])}
out_text = "; ".join(f"{col}: {n} выбросов (skew = {df_orders[col].skew():.2f})"
                     for col, n in out_counts.items())
print(f"3. Выбросы по IQR — {out_text}. Всего {outlier_mask.sum()} строк "
      f"({outlier_mask.mean():.1%}): {nature}. После их удаления r {change} "
      f"({r_all:.2f} -> {r_clean:.2f}), Спирмен = {rho_all:.2f} близок к Пирсону — связь "
      "устойчивая и не держится на крайних значениях.")
n_missing = df_orders[["price", "quantity", "revenue"]].isna().sum().sum()
if non_positive.sum() == 0 and n_missing == 0:
    print("4. Некорректных значений (цена или количество <= 0) и пропусков нет — "
          "очистка перед расчётом корреляций не требуется.")
else:
    print(f"4. Найдено {non_positive.sum()} строк с ценой или количеством <= 0 — это ошибки данных, "
          "их нужно исключить перед расчётом выручки.")
print(f"5. Следствие: выручку позиции определяют цена и количество, а не время заказа. При этом "
      f"товар объясняет лишь {price_by_product:.1%} разброса цены — один и тот же товар "
      "продаётся по очень разным ценам, поэтому «ценовой сегмент товара» в этих данных "
      "почти не работает как признак. Ограничения: корреляция не означает причинность, "
      "Пирсон измеряет только линейную связь, а анализ сделан на одном наборе признаков "
      "за 3 месяца.")
