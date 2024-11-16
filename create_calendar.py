from dataclasses import dataclass
from datetime import date
from enum import IntEnum
from pathlib import Path
from typing import Dict, Tuple, Optional

import polars as pl
import requests


class Weekday(IntEnum):
    """曜日を表す列挙型"""
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7

    @classmethod
    def is_weekend(cls, day: int) -> bool:
        """週末かどうかを判定"""
        return day in (cls.SATURDAY, cls.SUNDAY)


@dataclass
class FiscalYear:
    """年度を表すデータクラス"""
    year: int

    def get_date_range(self) -> Tuple[date, date]:
        """年度の開始日と終了日を取得"""
        start_date = date(self.year, 4, 1)
        end_date = date(self.year + 1, 3, 31)
        return start_date, end_date


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


class CalendarGenerator:
    """カレンダー生成クラス"""

    def __init__(self, fiscal_year: FiscalYear, schedule_path: Path) -> None:
        """
        :param fiscal_year: 年度
        :param schedule_path: スケジュールファイルのパス
        """
        self.fiscal_year = fiscal_year
        self.schedule_path = schedule_path
        self.output_dir = Path(__file__).resolve().parent / 'results'
        self.output_dir.mkdir(exist_ok=True)
        self.schedule_df: Optional[pl.DataFrame] = None

    def generate(self) -> None:
        """
        カレンダーの生成と保存

        処理の流れ:
        1. _load_schedule() -> スケジュールを読み込み
        2. _generate_calendar_dates() -> カレンダーの基本データを生成
        3. _apply_schedule() -> スケジュールを適用
        4. _select_output_columns() -> 出力用に列を選択
        5. write_csv() -> CSVファイルとして保存
        """
        self.schedule_df = self._load_schedule()
        calendar_df = self._generate_calendar_dates()
        calendar_df = self._apply_schedule(calendar_df, self.schedule_df)
        result_df = self._select_output_columns(calendar_df)
        result_df.write_csv(self.output_dir / 'calendar.csv')

    def _generate_calendar_dates(self) -> pl.DataFrame:
        """
        カレンダーデータの生成

        処理の流れ:
        1. _create_base_calendar() -> 基本カレンダーを作成
        2. _create_holiday_frame() -> 祝日データを取得
        3. _add_holiday_info() -> 休日情報を追加
        """
        calendar_df = self._create_base_calendar()
        holidays_df = self._create_holiday_frame()
        return self._add_holiday_info(calendar_df, holidays_df)

    def _create_base_calendar(self) -> pl.DataFrame:
        """
        基本カレンダーの作成

        処理の流れ:
        1. 日付範囲の生成
        2. _add_calendar_columns() -> 基本列の追加
        """
        start_date, end_date = self.fiscal_year.get_date_range()
        schedule_start_date = self.schedule_df.filter(pl.col('apply_start_dt') <= start_date).max()
        schedule_start_date = schedule_start_date.to_series().to_list()[0]
        date_range = pl.date_range(schedule_start_date, end_date, interval='1d', eager=True)
        df = pl.DataFrame({'date': date_range})
        return self._add_calendar_columns(df)

    @staticmethod
    def _add_calendar_columns(df: pl.DataFrame) -> pl.DataFrame:
        """カレンダーの基本列を追加"""
        # 曜日関連の列を追加
        df = df.with_columns([
            pl.col('date').dt.weekday().alias('weekday'),
            pl.col('date').dt.strftime('%A').alias('weekday_name'),
        ])

        # 週末判定の列を追加
        df = df.with_columns([
            pl.col('weekday')
            .map_elements(Weekday.is_weekend, return_dtype=pl.Boolean)
            .alias('is_weekend')
        ])

        return df

    def _create_holiday_frame(self) -> pl.DataFrame:
        """祝日データフレームの作成"""
        holidays_dict = get_holidays_api(self.fiscal_year.year)

        return pl.DataFrame({
            'date': pl.Series([k for k in holidays_dict.keys()]).cast(pl.Date),
            'holiday_name': pl.Series([v for v in holidays_dict.values()])
        })

    @staticmethod
    def _add_holiday_info(calendar_df: pl.DataFrame, holidays_df: pl.DataFrame) -> pl.DataFrame:
        """休日情報の追加"""
        # 祝日データを結合
        df = calendar_df.join(holidays_df, on='date', how='left')

        # 休日フラグを追加
        df = df.with_columns([
            (pl.col('is_weekend') | pl.col('holiday_name').is_not_null()).alias('is_holiday')
        ])

        # 週末の場合は休日名を設定
        df = df.with_columns([
            pl.when(pl.col('is_weekend'))
            .then(pl.lit('週末'))
            .otherwise(pl.col('holiday_name'))
            .alias('holiday_name')
        ])

        return df

    def _load_schedule(self) -> pl.DataFrame:
        """スケジュールの読み込みと前処理"""
        df = pl.read_csv(self.schedule_path)
        return df.with_columns([
            pl.col('apply_start_dt').str.strptime(pl.Date, format='%Y-%m-%d')
        ])

    def _apply_schedule(self, calendar_df: pl.DataFrame, schedule_df: pl.DataFrame) -> pl.DataFrame:
        """スケジュールの適用"""
        # スケジュールを結合
        df = calendar_df.join(
            schedule_df,
            left_on='date',
            right_on='apply_start_dt',
            how='left'
        )

        # 直前の有効なスケジュールを適用
        df = df.with_columns([
            pl.col('weekday_time').forward_fill(),
            pl.col('holiday_time').forward_fill(),
        ])

        # 休日/平日に応じた時間を設定
        df = df.with_columns([
            pl.when(pl.col('is_holiday'))
            .then(pl.col('holiday_time'))
            .otherwise(pl.col('weekday_time'))
            .alias('hours')
        ])

        # 対象年度のみのデータに絞る
        start_date = self.fiscal_year.get_date_range()[0]
        df = df.filter(pl.col('date') >= start_date)
        return df

    @staticmethod
    def _select_output_columns(df: pl.DataFrame) -> pl.DataFrame:
        """出力用の列を選択"""
        return df.select([
            'date',
            'weekday_name',
            'is_holiday',
            'holiday_name',
            'hours'
        ])


def main(fiscal_year) -> None:
    """メイン処理"""
    fiscal_year = FiscalYear(year=fiscal_year)
    generator = CalendarGenerator(
        fiscal_year=fiscal_year,
        schedule_path=Path('training_schedule.csv')
    )
    generator.generate()


if __name__ == '__main__':
    main(2024)
