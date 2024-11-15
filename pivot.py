from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List

import plotly.express as px
import polars as pl


@dataclass
class DateRange:
    """日付範囲を管理するデータクラス"""
    start_date: str
    end_date: str
    _start_dt: Optional[date] = None
    _end_dt: Optional[date] = None

    def __post_init__(self):
        """文字列の日付をdateオブジェクトに変換"""
        self._start_dt = datetime.strptime(self.start_date, "%Y/%m/%d").date()
        self._end_dt = datetime.strptime(self.end_date, "%Y/%m/%d").date()

    @property
    def start_dt(self) -> date:
        return self._start_dt

    @property
    def end_dt(self) -> date:
        return self._end_dt


class StudyDataProcessor:
    """学習データの処理を担当するクラス"""

    def __init__(self, file_path: str, date_range: DateRange):
        self.file_path = file_path
        self.date_range = date_range
        self.df: Optional[pl.DataFrame] = None
        self.weekly_summary: Optional[pl.DataFrame] = None
        self.weekly_total: Optional[pl.DataFrame] = None
        self.period_total: Optional[pl.DataFrame] = None
        self._subjects: Optional[List[str]] = None

    def load_and_preprocess_data(self) -> None:
        """データの読み込みと前処理"""
        self.df = pl.read_csv(self.file_path, encoding="utf-8")
        self.df = self.df.with_columns([
            pl.col("日付").str.slice(0, 10).str.to_datetime("%Y/%m/%d").alias("date"),
            pl.col("日付").str.slice(0, 10).str.to_datetime("%Y/%m/%d").dt.strftime("%Y-W%W").alias("週")
        ])
        self.df = self.df.filter(
            (pl.col("date") >= pl.lit(self.date_range.start_dt)) &
            (pl.col("date") <= pl.lit(self.date_range.end_dt))
        )

        # データ読み込み後に科目リストを取得
        self._subjects = self.df["科目"].unique().to_list()

    @property
    def subjects(self) -> List[str]:
        if self._subjects is None:
            raise ValueError("データが読み込まれていません")
        return self._subjects

    def calculate_summaries(self) -> None:
        """各種サマリーの計算"""
        if self.df is None:
            raise ValueError("データが読み込まれていません。load_and_preprocess_data()を先に実行してください。")

        self.weekly_summary = self.df.group_by(["週", "科目"]).agg([
            pl.col("実績時間").sum().alias("週間実績時間"),
            pl.col("目標時間").sum().alias("週間目標時間"),
            (pl.col("実績時間").sum() / pl.col("目標時間").sum() * 100).round(1).alias("達成率(%)")
        ]).sort(["週", "科目"])

        self.weekly_total = self.df.group_by("週").agg([
            pl.col("実績時間").sum().alias("週間実績時間"),
            pl.col("目標時間").sum().alias("週間目標時間"),
            (pl.col("実績時間").sum() / pl.col("目標時間").sum() * 100).round(1).alias("達成率(%)")
        ]).with_columns(pl.lit("合計").alias("科目")).sort("週")

        self.period_total = self.df.group_by("科目").agg([
            pl.col("実績時間").sum().alias("合計実績時間"),
            pl.col("目標時間").sum().alias("合計目標時間"),
            (pl.col("実績時間").sum() / pl.col("目標時間").sum() * 100).round(1).alias("達成率(%)")
        ]).sort("科目")

    def display_summaries(self) -> None:
        """サマリーの表示"""
        if any(x is None for x in [self.weekly_summary, self.weekly_total, self.period_total]):
            raise ValueError("サマリーが計算されていません。calculate_summaries()を先に実行してください。")

        print(f"\n集計期間: {self.date_range.start_date} ～ {self.date_range.end_date}")
        print("\n週別サマリー:")
        print(self.weekly_summary)
        print("\n週間合計:")
        print(self.weekly_total)
        print("\n期間全体:")
        print(self.period_total)


class StudyDataVisualizer:
    """学習データの可視化を担当するクラス"""

    def __init__(self, processor: StudyDataProcessor):
        self.processor = processor

    def create_weekly_stacked_bar(self) -> px.bar:
        """週ごとのスタックバーグラフの作成"""
        if self.processor.df is None:
            raise ValueError("データが読み込まれていません。")

        weekly_data = self.processor.df.group_by(["週", "科目"]).agg([
            pl.col("実績時間").sum().alias("実績時間"),
            pl.col("目標時間").sum().alias("目標時間")
        ]).sort(["週", "科目"])

        weekly_pivoted = weekly_data.pivot(
            index="週",
            on="科目",
            values=["実績時間", "目標時間"]
        )

        result_weeks = []
        for week in weekly_pivoted['週']:
            result_weeks.append(f"{week}/実績時間")
            result_weeks.append(f"{week}/目標時間")

        # 初期データ構造を作成
        result_data = {
            '期間': result_weeks,
            **{subject: [] for subject in self.processor.subjects}
        }

        # データを埋める
        for week in weekly_pivoted['週']:
            for subject in self.processor.subjects:
                result_data[subject].append(weekly_pivoted.filter(pl.col('週') == week)[f'実績時間_{subject}'][0])
                result_data[subject].append(weekly_pivoted.filter(pl.col('週') == week)[f'目標時間_{subject}'][0])

        return px.bar(
            result_data,
            x='期間',
            y=self.processor.subjects,
            title=f'{self.processor.date_range.start_dt} ~ {self.processor.date_range.end_dt} 週別の学習実績と目標時間',
            labels={'value': '時間', 'variable': '科目'},
            barmode='stack',
        )

    def create_total_weekly_bar(self) -> px.bar:
        """週ごとの総計バーグラフの作成"""
        if self.processor.weekly_total is None:
            raise ValueError("週間合計が計算されていません。")

        return px.bar(
            self.processor.weekly_total,
            x="週",
            y=["週間実績時間", "週間目標時間"],
            title="週ごとの総計実績時間と目標時間",
            labels={"value": "時間", "variable": "カテゴリー"},
            barmode="group"
        )

    def create_period_total_bar(self) -> px.bar:
        """期間全体の総計バーグラフの作成"""
        if self.processor.period_total is None:
            raise ValueError("期間合計が計算されていません。")

        return px.bar(
            self.processor.period_total,
            x="科目",
            y=["合計実績時間", "合計目標時間"],
            title="集計期間全体の科目別実績時間と目標時間",
            labels={"value": "時間", "variable": "カテゴリー"},
            barmode="group"
        )


def main():
    """メイン処理"""
    # 日付範囲の設定
    date_range = DateRange('2024/04/01', '2024/04/30')

    # データ処理インスタンスの作成と処理実行
    processor = StudyDataProcessor("study.csv", date_range)
    processor.load_and_preprocess_data()
    processor.calculate_summaries()
    processor.display_summaries()

    # 可視化インスタンスの作成とグラフの表示
    visualizer = StudyDataVisualizer(processor)

    fig1 = visualizer.create_weekly_stacked_bar()
    fig1.show()

    fig2 = visualizer.create_total_weekly_bar()
    fig2.show()

    fig3 = visualizer.create_period_total_bar()
    fig3.show()


if __name__ == "__main__":
    main()
