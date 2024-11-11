from datetime import date, datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

base_dir = Path(__file__).resolve().parent


def get_holidays_api(fiscal_year: int) -> Dict[str, str]:
    """
    指定された年と翌年の祝日データをAPIで取得する

    :param fiscal_year: 取得したい年
    :return: 祝日データの辞書 {日付: 祝日名}
    """
    
    holiday_list = []
    for year in range(fiscal_year, fiscal_year + 2):
        url = f"https://holidays-jp.shogo82148.com/{year}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch holidays for {year}. Status code: {response.status_code}")

        response_data = response.json()
        holidays = response_data.get('holidays', [])
        holiday_list.extend(holidays)

    holidays_dict = {holiday['date']: holiday['name'] for holiday in holiday_list}
    return holidays_dict


def get_date_range(fiscal_year) -> tuple[date, date]:
    """
    指定された年度の年度開始日と終了日を返す関数

    :param fiscal_year: 年度
    :return: 年度の開始日と終了日
    """

    target_date = datetime.strptime(f'{fiscal_year}/04/01', '%Y/%m/%d')
    start_date = date(target_date.year, 4, 1)
    end_date = date(target_date.year + 1, 3, 31)

    return start_date, end_date


def generate_calendar_dates(fiscal_year: int) -> pd.DataFrame:
    """
    与えられた年度の休日を生成する関数。

    以下のフィールドを持つDataFrameを返す。
    - date: 日付
    - weekday: 曜日（0: 月曜日, 1: 火曜日, ..., 6: 日曜日）
    - weekday_name: 曜日名
    - is_weekend: 週末かどうか
    - is_holiday: 祝日かどうか
    - holiday_name: 祝日名

    :params fiscal_year: 年度
    :returns: DataFrame
    """

    # APIで休日リストを取得
    holidays_dict = get_holidays_api(fiscal_year)
    holidays = pd.DataFrame(holidays_dict.items(), columns=['date', 'holiday_name'])
    holidays['date'] = pd.to_datetime(holidays['date'])

    # 日付の範囲を生成
    start, end = get_date_range(fiscal_year)
    date_range = pd.date_range(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    calendar = pd.DataFrame(date_range, columns=['date'])

    # 曜日番号と曜日名を追加
    calendar['weekday'] = calendar['date'].dt.weekday
    calendar['weekday_name'] = calendar['date'].dt.day_name(locale='ja_JP')

    # 週末かどうかを追加
    calendar['is_weekend'] = calendar['weekday'].isin([5, 6])

    # 休日かどうかを追加
    calendar['is_holiday'] = calendar['date'].isin(holidays['date']) | calendar['weekday'].isin([5, 6])

    # 休日名を追加
    calendar = pd.merge(calendar, holidays, on='date', how='left')
    calendar.loc[calendar['is_weekend'], 'holiday_name'] = '週末'
    return calendar


def get_hours(row: pd, schedule_df: pd.DataFrame) -> float:
    """
    与えられた日付に対して、スケジュールデータから対応する目標時間を取得する関数。

    :param pandas row: 日付の行データ
    :param pandas DataFrame schedule_df: スケジュールデータ
    :return: float hours : 平日・休日に応じた目標時間
    """

    row_date = pd.Timestamp(row['date'])

    for i in range(len(schedule_df)):
        start_date = schedule_df.loc[i, 'apply_start_dt']
        end_date = schedule_df.loc[i + 1, 'apply_start_dt'] if i + 1 < len(schedule_df) else pd.Timestamp('9999-12-31')

        # 条件に該当する期間かチェック
        if start_date <= row_date < end_date:
            return schedule_df.loc[i, 'holiday_time'] if row['is_holiday'] else schedule_df.loc[i, 'weekday_time']

    raise ValueError('No schedule found')


def create_calendar():
    """
    メインの処理を実行する関数。

    1. カレンダーの日付を生成する。
    2. スケジュールデータを読み込む。
    3. 各日付に対して目標時間を設定する。
    4. 結果をCSVおよびExcelファイルとして保存する。
    """

    df = generate_calendar_dates(2024)

    # CSVのデータを読み込む
    schedule_df = pd.read_csv('training_schedule.csv')
    schedule_df['apply_start_dt'] = pd.to_datetime(schedule_df['apply_start_dt'])

    # 各日付に対して目標時間を設定
    df['hours'] = df.apply(get_hours, axis=1, schedule_df=schedule_df)

    # 結果を保存
    result_df = df[['date', 'weekday_name', 'is_holiday', 'holiday_name', 'hours']]
    output_dir = base_dir / 'results'
    output_dir.mkdir(exist_ok=True)
    result_df.to_csv('results/calendar.csv', index=False)
    result_df.to_excel('results/calendar.xlsx', index=False)


if __name__ == '__main__':
    create_calendar()
