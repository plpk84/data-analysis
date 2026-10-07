"""
Практическая работа №1: Основы работы с данными в Pandas
Вариант 6

1. Вывести статистическую сводку (.describe()).
2. Выбрать строки 3-7 (используя .iloc[]).
3. Выбрать строки, где Name начинается с буквы 'A' или 'E' (.str.startswith()).
4. Найти стандартное отклонение Age.
5. Нормализовать Salary: вычесть минимум, разделить на (макс - мин).
"""

import pandas as pd
import numpy as np

data = {
    'ID': range(1, 11),
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva',
             'Frank', 'Grace', 'Helen', 'Ivan', 'Julia'],
    'Age': [23, 35, 29, 40, 28, 45, 32, 27, 38, 31],
    'City': ['Moscow', 'SPb', 'Moscow', 'Kazan', 'SPb',
             'Moscow', 'SPb', 'Kazan', 'Moscow', 'SPb'],
    'Salary': [50000, 70000, 60000, 80000, 55000,
               90000, 65000, 58000, 75000, 62000]
}
df = pd.DataFrame(data)
print("Исходный DataFrame:")
print(df)

print("\n=== Задание 1. Статистическая сводка (.describe()) ===")
print(df.describe())

print("\n=== Задание 2. Строки 3-7 (.iloc[3:8]) ===")
print(df.iloc[3:8])

print("\n=== Задание 3. Name начинается с 'A' или 'E' ===")
mask_ae = df['Name'].str.startswith(('A', 'E'))
print(df[mask_ae])

print("\n=== Задание 4. Стандартное отклонение Age ===")
age_std = df['Age'].std()
print(f"Стандартное отклонение Age: {age_std:.2f}")

print("\n=== Задание 5. Нормализация Salary ===")
salary_min = df['Salary'].min()
salary_max = df['Salary'].max()
df['Salary_Norm'] = (df['Salary'] - salary_min) / (salary_max - salary_min)
print(df[['Name', 'Salary', 'Salary_Norm']].round(3))
print(f"Минимум: {salary_min}, максимум: {salary_max}")
