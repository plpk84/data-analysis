"""
Практическая работа №2: Группировка, обработка пропусков и объединение данных
Вариант 6

1. Найти общую выручку по регионам (создать Revenue, затем .groupby('Region').sum()).
2. Посчитать количество заказов для каждого продукта.
3. Удалить столбцы, содержащие хотя бы один NaN (.dropna(axis=1)).
4. Переименовать столбцы: OrderID -> ID, Quantity -> Qty.
5. Добавить к sales новую строку с данными: [111, 'Monitor', 1, 15000, 'Moscow'].
"""

import pandas as pd
import numpy as np

sales = pd.DataFrame({
    'OrderID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Product': ['Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone',
                'Laptop', 'Tablet', 'Phone', 'Laptop', 'Tablet'],
    'Quantity': [1, 2, 1, 3, 1, 2, 1, 4, 1, 2],
    'Price': [50000, 30000, 20000, 50000, np.nan,
              50000, 20000, 30000, 50000, np.nan],
    'Region': ['Moscow', 'SPb', 'Kazan', 'Moscow', 'SPb',
               'Kazan', np.nan, 'SPb', 'Moscow', 'Kazan']
})
products = pd.DataFrame({
    'Product': ['Laptop', 'Phone', 'Tablet', 'Monitor'],
    'Category': ['Electronics', 'Electronics', 'Electronics', 'Accessories'],
    'Weight_kg': [2.5, 0.2, 0.5, 3.0]
})
print("Таблица продаж:")
print(sales)
print("\nТаблица товаров:")
print(products)

print("\n=== Задание 1. Общая выручка по регионам ===")
sales['Revenue'] = sales['Quantity'] * sales['Price']
revenue_by_region = sales.groupby('Region')[['Revenue']].sum()
print(sales[['OrderID', 'Region', 'Quantity', 'Price', 'Revenue']])
print("\nВыручка по регионам:")
print(revenue_by_region.sort_values('Revenue', ascending=False))

print("\n=== Задание 2. Количество заказов по продуктам ===")
orders_by_product = sales.groupby('Product')['OrderID'].count()
print(orders_by_product)
# То же самое через value_counts()
print(sales['Product'].value_counts())

print("\n=== Задание 3. Удаление столбцов с NaN (.dropna(axis=1)) ===")
print("Пропуски по столбцам:")
print(sales.isnull().sum())
sales_no_nan_cols = sales.dropna(axis=1)
print("\nСтолбцы до:", list(sales.columns))
print("Столбцы после:", list(sales_no_nan_cols.columns))
print(sales_no_nan_cols)

print("\n=== Задание 4. Переименование OrderID -> ID, Quantity -> Qty ===")
sales_renamed = sales.rename(columns={'OrderID': 'ID', 'Quantity': 'Qty'})
print(sales_renamed.head())

print("\n=== Задание 5. Добавление новой строки ===")
new_row = pd.DataFrame(
    [[111, 'Monitor', 1, 15000, 'Moscow']],
    columns=['OrderID', 'Product', 'Quantity', 'Price', 'Region']
)
new_row['Revenue'] = new_row['Quantity'] * new_row['Price']
sales = pd.concat([sales, new_row], ignore_index=True)
print(sales)
