import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

pd.set_option('future.no_silent_downcasting', True)

BASE_DIR = Path(__file__).resolve().parent

CARDS_DATA_PATH = Path(os.getenv("CARDS_DATA_PATH", BASE_DIR / "data/cards prod/"))
BRELOKI_DATA_PATH = Path(os.getenv("BRELOKI_DATA_PATH", BASE_DIR / "data/table prod/"))
LOCAL_OUTPUT_PATH = Path(os.getenv("LOCAL_OUTPUT_PATH", BASE_DIR / "data"))
LOCAL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def num_z(x, y):
    # Преобразуем дату из строки в datetime
    date_obj = pd.to_datetime(y, dayfirst=True)
    cutoff_date = datetime(2025, 3, 1)  # 01.03.2025

    x_str = str(x).strip().replace('Y', 'У').replace('.0', '')

    if date_obj < cutoff_date:
        # Если строка уже начинается с '25УП' или '24УП', возвращаем её
        if x_str.startswith('25УП') or x_str.startswith('24УП'):
            return x_str
        else:
            # Обработка для строк длиной 5 символов
            if len(x_str) == 5:
                return f'24УП-{x_str.zfill(6)}'
            # Обработка для строк длиной 4 символа
            elif len(x_str) < 5:
                return f'25УП-{x_str.zfill(6)}'
            else:
                # Можно добавить обработку для других случаев или вернуть исходное значение
                return x_str
    else:
        # Для дат после cutoff_date можно определить другую логику или вернуть исходное значение
        if '25УП' not in x_str:
            return f'25УП-{x_str.zfill(6)}'

        return x_str

def reorder_cols(df):
    cols = df.columns.tolist()
    base_cols = ['Дата', '№ заказа', 'Наименование', 'Количество', 'Доп. Расход чипов', 'Излишки']
    new_order = [col for col in base_cols if col in cols]
    brak_columns = [col for col in cols if col.startswith('Виды брака_')]
    new_order.extend(brak_columns)
    remaining_cols = [col for col in cols if col not in new_order]
    new_order.extend(remaining_cols)
    return df[new_order]

def add_nomenklature(df, col):
    order_file = LOCAL_OUTPUT_PATH / 'order_money.xlsx'
    if not order_file.exists():
        print('Файл с данными о деньгах отсутствует или недоступен')
        return df

    order_num = pd.read_excel(order_file, header=3).rename(columns={'Номенклатура, Вид номенклатуры' : 'Номенклатура'})

    if col == 'Заказ на производство':
        order_num['Заказ на производство'] = order_num['Заказ на производство'].apply(lambda x: str(x)[22:33])
        df = df.merge(order_num[['Заказ на производство', 'Номенклатура', 'Цена', 'Количество']],
                      left_on='№ заказа', right_on='Заказ на производство', how='left', suffixes=('_i', '_a_n'))

    elif col == 'Заказ клиента':
        order_num['Заказ клиента'] = order_num['Заказ клиента'].apply(lambda x: str(x)[14:25])
        df = df.merge(order_num[['Заказ клиента', 'Номенклатура', 'Цена', 'Количество']],
                      left_on='№ заказа БСМ', right_on='Заказ клиента', how='left', suffixes=('_i', '_a_n'))

    if "Номенклатура_a_n" in df.columns:
        df["Номенклатура"] = df["Номенклатура_a_n"].fillna("Отсутствует информация")
        df.drop(columns=["Номенклатура_a_n"], inplace=True)
    elif "Номенклатура" in df.columns:
        df["Номенклатура"] = df["Номенклатура"].fillna("Отсутствует информация")
    else:
        df["Номенклатура"] = "Отсутствует информация"

    return df

def cards_data_collect():
    full_data = pd.DataFrame()

    for file_path in CARDS_DATA_PATH.rglob("*.xls*"):
        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = [s for s in xls.sheet_names if s[-1].isdigit()]
            for sheet in tqdm(sheet_names, desc=f"Обработка карточек {file_path.name}"):
                tmp_data = pd.read_excel(file_path, sheet_name=sheet)
                tmp_data = tmp_data.iloc[:, :tmp_data.columns.get_loc('Брак без видимых причин') + 1]

                for i in range(len(tmp_data.columns)):
                    if 'Unnamed' in tmp_data.columns[i]:
                        tmp_data.rename(columns={tmp_data.columns[i]: f'{tmp_data.columns[i-1]}'}, inplace=True)
                for i in range(len(tmp_data.columns)):
                    if pd.notna(tmp_data.iloc[0, i]):
                        tmp_data.columns.values[i] = f'{tmp_data.columns[i]}_{tmp_data.iloc[0, i]}'

                tmp_data = tmp_data[1:]
                # tmp_data = tmp_data.loc[~tmp_data['Дата'].isin(['Количество брака за день', 'Цех rfid'])]
                tmp_data['Дата'] = pd.to_datetime(tmp_data['Дата'], dayfirst=True, errors='coerce').ffill()
                tmp_data = tmp_data.loc[~tmp_data['№ заказа'].isna()]
                tmp_data = tmp_data.loc[pd.notna(tmp_data['№ заказа'])]
                tmp_data = tmp_data.fillna(0)
                tmp_data['Количество'] = pd.to_numeric(tmp_data['Количество'], downcast='integer')

                full_data = pd.concat([full_data, tmp_data], ignore_index=True)
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

    full_data['Участок'] = 'Цех 1'


    full_data['№ заказа'] = full_data.apply(
        lambda row: num_z(row['№ заказа'], row['Дата']),
        axis=1
    )

    return full_data

def breloki_data_collect():
    full_data = pd.DataFrame()

    for file_path in BRELOKI_DATA_PATH.rglob("% брака цех 2.xls*"):
        try:
            df = pd.read_excel(file_path, sheet_name='Брак 2025 год')
            df = df.rename(columns={'Причина брака': 'Виды брака', 'Наименование тиража': 'Наименование'})

            for i in range(len(df.columns)):
                if 'Unnamed' in df.columns[i]:
                    df.rename(columns={df.columns[i]: f'{df.columns[i-1]}'}, inplace=True)
            for i in range(len(df.columns)):
                if pd.notna(df.iloc[0, i]):
                    df.columns.values[i] = f'{df.columns[i]}_{df.iloc[0, i]}'

            df = df[1:]
            df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')

            defect_cols = [c for c in df.columns if c.startswith('Виды брака_')] + ['Излишки']

            df['№ заказа'] = df['№ заказа'].astype(str).apply(lambda x: x.strip().replace('.0', ''))

            for col in defect_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            df['кол-во брака'] = df[defect_cols].sum(axis=1)
            df['Участок'] = 'Цех 2'
            df = df.loc[~df['Дата'].isna()].fillna(0)

            full_data = pd.concat([full_data, df], ignore_index=True)
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

    full_data['№ заказа'] = full_data.apply(
        lambda row: num_z(row['№ заказа'], row['Дата']),
        axis=1
    )

    return full_data

def payment_table_data_collect():
    full_data = pd.DataFrame()

    for file_path in BRELOKI_DATA_PATH.rglob("Брак цех 3.xls*"):
        try:
            df = pd.read_excel(file_path)
            df = df.iloc[:, :17]
            df = df.rename(columns={'Причина брака': 'Виды брака', 'Наименование тиража': 'Наименование'})
            df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')

            for i in range(len(df.columns)):
                if 'Unnamed' in df.columns[i]:
                    df.rename(columns={df.columns[i]: f'{df.columns[i - 1]}'}, inplace=True)
            for i in range(len(df.columns)):
                if pd.notna(df.iloc[0, i]):
                    df.columns.values[i] = f'{df.columns[i]}_{df.iloc[0, i]}'
            df = df[2:]

            df = df.loc[df['Дата'] > '2025-01-01']
            full_data = pd.concat([full_data, df], ignore_index=True)
            full_data['Участок'] = 'Цех 3'

        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

    full_data['№ заказа БСМ'] = full_data.apply(
        lambda row: num_z(row['№ заказа БСМ'], row['Дата']),
        axis=1
    )

    full_data = full_data.drop(columns=[
        '№ заказа 1С', 'FCT Eperso', 'Ед/изм', 'Кол-во по заказу'])

    return full_data

def save_full_data(df: pd.DataFrame):
    out_xlsx = LOCAL_OUTPUT_PATH / 'card_full_data.xlsx'
    out_csv = LOCAL_OUTPUT_PATH / 'card_full_data.csv'

    df.to_excel(out_xlsx, index=False)
    df.to_csv(out_csv, index=False)
    print(f'''Сохранено: 
{out_xlsx}
{out_csv}''')

def main():
    print("=== Поиск файлов ===")

    card_files = list(CARDS_DATA_PATH.rglob("*.xls*"))
    breloki_files = list(BRELOKI_DATA_PATH.rglob("*.xls*"))

    if card_files:
        print("Файлы карточек:")
        for f in card_files:
            print("  -", f.relative_to(BASE_DIR))
    else:
        print("❌ Файлы карточек не найдены")

    if breloki_files:
        print("Файлы брелоков:")
        for f in breloki_files:
            print("  -", f.relative_to(BASE_DIR))
    else:
        print("❌ Файлы брелоков не найдены")

    print("\n=== Сборка данных ===")

    cards_df = cards_data_collect() if card_files else pd.DataFrame()
    cards_df = add_nomenklature(cards_df, 'Заказ на производство')
    breloki_df = breloki_data_collect() if breloki_files else pd.DataFrame()
    breloki_df = add_nomenklature(breloki_df, 'Заказ на производство')
    p_table = payment_table_data_collect()
    p_table = add_nomenklature(p_table, 'Заказ клиента').rename(columns={'№ заказа БСМ':'№ заказа'})

    if cards_df.empty and breloki_df.empty:
        print("⚠️ Нет данных для обработки — выходим.")
        return

    full_df = pd.concat([cards_df, breloki_df], ignore_index=True).fillna(0)

    # объединяем количество
    if {'Количество_i', 'Количество_a_n'}.issubset(full_df.columns):
        full_df['Количество_i'] = full_df[['Количество_i', 'Количество_a_n']].max(axis=1)
        full_df = full_df.drop('Количество_a_n', axis=1).rename(columns={'Количество_i': 'Количество'})

    full_df = pd.concat([full_df, p_table], ignore_index=True).fillna(0)

    # считаем браки
    defect_columns = [col for col in full_df.columns if col.startswith('Виды брака_')]

    for col in defect_columns:
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce").fillna(0)

    full_df['кол-во брака'] = full_df[defect_columns].sum(axis=1)

    # % брака
    full_df["кол-во брака"] = pd.to_numeric(full_df["кол-во брака"], errors="coerce").fillna(0)
    full_df["Количество"] = pd.to_numeric(full_df["Количество"], errors="coerce").fillna(0)

    full_df["% брака"] = (
            (full_df["кол-во брака"] / full_df["Количество"].replace(0, pd.NA)) * 100
    ).fillna(0).astype(float).round(2)

    # стоимость брака
    if 'Цена' in full_df.columns:
        full_df['Стоимость брака'] = full_df['кол-во брака'] * full_df['Цена']

    # финальная перестановка колонок
    full_df = reorder_cols(full_df)
    #
    try:
        full_df = full_df.drop(columns=['Заказ на производство'], axis=1)
    except:
        pass

    try:
        full_df = full_df.drop(columns=['СКМ'], axis=1)
    except:
        pass

    full_df = full_df.fillna(0)

    save_full_data(full_df)


if __name__ == '__main__':
    main()
