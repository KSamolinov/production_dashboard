import pandas as pd
import matplotlib.dates as mdates
import matplotlib
from matplotlib.figure import Figure
from datetime import datetime as dt
import io

matplotlib.use('Agg')

def _empty_plot(message: str) -> bytes:
    fig = Figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    return buf.getvalue()

def prepare_data(data, nomen=None, place=None):
    '''
    Фнукция выбора данных для работы
    '''
    if nomen:
        data = data[data['Номенклатура'] == nomen]
    if place:
        data = data[data['Участок'] == place]
    return data

def defects_data(data: pd.DataFrame) -> pd.DataFrame:
    defect_columns = [col for col in data.columns if col.startswith('Виды брака_')]

    df = pd.DataFrame()
    for col in defect_columns:
        df[col] = data[col] * data['Цена']

    # Сумма по каждому виду брака
    price_table = df.sum().reset_index()
    price_table.columns = ["defect", "sum"]

    # убираем префикс "Виды брака_"
    price_table["defect"] = price_table["defect"].str.replace("^Виды брака_", "", regex=True)

    # оставляем только строки с суммой > 0
    price_table = price_table[price_table["sum"] > 0].sort_values('sum', ascending=False).reset_index(drop=True)

    # добавляем строку "Отсутствует синхронизацию..."
    nan_count = len(data.loc[data['Номенклатура'] == "Отсутствует информация"])
    if nan_count > 0:
        price_table.loc[len(price_table)] = {
            "defect": f"Отсутствует синхронизация по {nan_count} позициям",
            "sum": None
        }

    # возвращаем таблицу с None, чтобы фронт мог отрисовать colspan
    return price_table.where(pd.notnull(price_table), None)

def _apply_styles(ax, title_size=16, label_size=14, tick_size=12):
    ax.title.set_fontsize(title_size)
    ax.xaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontsize(label_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_size)

def plot_line_total(data, period):
    start_date = pd.to_datetime(period[0], dayfirst=True)
    end_date = pd.to_datetime(period[1], dayfirst=True)

    filtered = data[(data['Дата'] >= start_date) & (data['Дата'] <= end_date)]
    if filtered.empty:
        return _empty_plot("Нет данных за указанный период")

    grouped = filtered.groupby('Дата')['Количество'].sum().sort_index()

    fig = Figure(figsize=(20, 7))
    ax = fig.subplots()
    ax.plot(grouped.index, grouped.values, linestyle='-', marker='o')  # цвет не задаём жёстко
    ax.set_title(f'Вырубка карт: {start_date.strftime("%d.%m.%y")} — {end_date.strftime("%d.%m.%y")}')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Количество')
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    fig.autofmt_xdate()
    _apply_styles(ax)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf.getvalue()

def plot_line_defects(data, period):
    start_date = pd.to_datetime(period[0], dayfirst=True)
    end_date = pd.to_datetime(period[1], dayfirst=True)

    filtered = data[(data['Дата'] >= start_date) & (data['Дата'] <= end_date)]
    if filtered.empty:
        return _empty_plot("Нет данных за указанный период")

    grouped = filtered.groupby('Дата')[['Количество', 'кол-во брака']].sum().sort_index()
    grouped['% брака'] = grouped.apply(
        lambda row: (row['кол-во брака'] / row['Количество']) * 100 if row['Количество'] else 0,
        axis=1
    )

    total_qty = grouped['Количество'].sum()
    mean_value = (grouped['кол-во брака'].sum() / total_qty * 100) if total_qty else 0

    fig = Figure(figsize=(20, 7))
    ax = fig.subplots()
    ax.plot(grouped.index, grouped['% брака'], linestyle='-', marker='o')
    ax.axhline(mean_value, linestyle='--', linewidth=2, label=f'Среднее: {mean_value:.2f}%')
    ax.set_title(f'% брака: {start_date.strftime("%d.%m.%y")} — {end_date.strftime("%d.%m.%y")}')
    ax.set_xlabel('Дата')
    ax.set_ylabel('% брака')
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    fig.autofmt_xdate()
    _apply_styles(ax)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf.getvalue()

def bar_plot_defects(data, period):
    # исходный список
    defect_cols = [col for col in data.columns if col.startswith("Виды брака")] + [
        "Брак без видимых причин", "Излишки"
    ]
    # берём только реально существующие в data.columns
    defect_cols = [c for c in defect_cols if c in data.columns]
    if not defect_cols:
        return _empty_plot("Нет дефектных колонок")

    start_date = pd.to_datetime(period[0], dayfirst=True)
    end_date = pd.to_datetime(period[1], dayfirst=True)

    filtered = data[(data['Дата'] >= start_date) & (data['Дата'] <= end_date)]
    if filtered.empty:
        return _empty_plot("Нет данных за указанный период")

    grouped = filtered[defect_cols].sum(numeric_only=True)
    grouped = pd.to_numeric(grouped, errors="coerce").fillna(0).astype(int)
    grouped = grouped[grouped > 0]

    if grouped.empty:
        return _empty_plot("Нет данных о браке за период")

    grouped = grouped.sort_values(ascending=False)
    labels = [col.replace('Виды брака_', '') for col in grouped.index]

    fig = Figure(figsize=(20, 8))
    ax = fig.subplots()
    ax.grid(True)
    ax.bar(labels, grouped.values)
    ax.set_title(f'Типы дефектов: {start_date.strftime("%d.%m.%y")} — {end_date.strftime("%d.%m.%y")}')
    ax.set_xlabel('Тип дефекта')
    ax.set_ylabel('Количество')
    ax.tick_params(axis='x', rotation=60)
    fig.tight_layout()
    _apply_styles(ax)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf.getvalue()

def pie_plot_defects(data, period):
    defect_cols = [col for col in data.columns if col.startswith("Виды брака")] + [
        "Брак без видимых причин", "Излишки"
    ]
    defect_cols = [c for c in defect_cols if c in data.columns]
    if not defect_cols:
        return _empty_plot("Нет дефектных колонок")

    start_date = pd.to_datetime(period[0], dayfirst=True)
    end_date = pd.to_datetime(period[1], dayfirst=True)

    filtered = data[(data['Дата'] >= start_date) & (data['Дата'] <= end_date)]
    if filtered.empty:
        return _empty_plot("Нет данных за указанный период")

    grouped = filtered[defect_cols].sum(numeric_only=True)
    grouped = pd.to_numeric(grouped, errors="coerce").fillna(0).astype(int)
    grouped_nonzero = grouped[grouped > 0]

    if grouped_nonzero.empty:
        return _empty_plot("Нет данных о браке за период")

    labels = [col.replace('Виды брака_', '') for col in grouped_nonzero.index]

    fig = Figure(figsize=(12, 12))
    ax = fig.subplots()
    ax.pie(grouped_nonzero.values, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title(f'Структура дефектов: {start_date.strftime("%d.%m.%y")} — {end_date.strftime("%d.%m.%y")}', fontsize=16)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf.getvalue()

def get_kpis(data, period):
    start = pd.to_datetime(period[0])
    end = pd.to_datetime(period[1])

    # 'Дата' к datetime только если ещё не datetime64
    if not pd.api.types.is_datetime64_any_dtype(data['Дата']):
        data = data.copy()
        data['Дата'] = pd.to_datetime(data['Дата'], errors='coerce')

    filtered = data[(data['Дата'] >= start) & (data['Дата'] <= end)].copy()
    for col in ['Количество', 'кол-во брака', 'Стоимость брака']:
        filtered[col] = pd.to_numeric(filtered[col], errors='coerce').fillna(0)

    filtered = filtered[filtered['Количество'] > 0]

    total = filtered['Количество'].sum()
    defects = filtered['кол-во брака'].sum()
    percent = (defects / total * 100) if total > 0 else 0
    total_money = filtered['Стоимость брака'].sum()

    return int(total), int(defects), round(percent, 2), round(total_money, 2)
