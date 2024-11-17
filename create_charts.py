from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Tuple

import plotly.express as px
import polars as pl
from plotly.graph_objs import Bar, Figure


@dataclass
class LabelData:
    """データフレーム・グラフで共通して使うラベルのデータクラス"""
    weekly_actual_time: str = "週間実績計"
    weekly_target_time: str = "週間目標計"
    period_total_actual_time: str = "実績計"
    period_total_target_time: str = "目標計"
    achievement_rate: str = "達成率(%)"
    category: str = "科目"
    week: str = "週"


@dataclass
class DateRange:
    """グラフを作成する日付範囲を管理するデータクラス"""
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

    @property
    def range(self) -> Tuple[str, str]:
        return self._start_dt.strftime("%y/%m/%d(%a)"), self._end_dt.strftime("%y/%m/%d(%a)")


class StudyDataProcessor:
    """学習データの集計を担当するクラス"""

    def __init__(self, file_path: str, date_range: DateRange):
        self.file_path = file_path
        self.date_range = date_range
        self.source_df: Optional[pl.DataFrame] = None
        self.weekly_detail: Optional[pl.DataFrame] = None
        self.weekly_detail_pivot: Optional[pl.DataFrame] = None
        self.weekly_total: Optional[pl.DataFrame] = None
        self.period_total: Optional[pl.DataFrame] = None
        self._subjects: Optional[List[str]] = None
        self.labels = LabelData()

    def _load_and_preprocess_data(self) -> None:
        """
        データの読み込みと前処理

        - 集計用の週の列を追加
        - 集計対象期間のデータのみ抽出
        - ソース内の科目をリスト化
        """
        self.source_df = pl.read_csv(self.file_path, encoding="utf-8")
        date_column_conversion = pl.col("日付").str.slice(0, 10).str.to_datetime("%Y/%m/%d")
        self.source_df = self.source_df.with_columns([
            date_column_conversion.alias("date"),
            date_column_conversion.dt.strftime("%y年%W週").alias("週")
        ])
        self.source_df = self.source_df.filter(
            (pl.col("date") >= pl.lit(self.date_range.start_dt)) &
            (pl.col("date") <= pl.lit(self.date_range.end_dt))
        )

        # データ読み込み後に科目リストを取得
        self._subjects = self.source_df["科目"].unique().to_list()

    @property
    def subjects(self) -> List[str]:
        if self._subjects is None:
            raise ValueError("データが読み込まれていません")
        return self._subjects

    def _aggregate_data(self, group_cols, actual_col: str, target_col: str) -> pl.DataFrame:
        """
        集計の共通処理を行うヘルパーメソッド

        :param group_cols: グループ化する列名
        :param actual_col: 実績時間の列名
        :param target_col: 目標時間の列名
        :return: 集計結果のDataFrame
        """
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        return self.source_df.group_by(group_cols).agg([
            pl.col("実績時間").sum().alias(actual_col),
            pl.col("目標時間").sum().alias(target_col),
            (pl.col("実績時間").sum() / pl.col("目標時間").sum() * 100).round(1).alias(self.labels.achievement_rate)
        ]).sort(group_cols)

    def calculate_summaries(self) -> None:
        """各種サマリーの計算"""

        # データの読み込みと前処理
        self._load_and_preprocess_data()

        # 週別・科目別の詳細
        self.weekly_detail = self._aggregate_data(
            group_cols=[self.labels.week, self.labels.category],
            actual_col=self.labels.weekly_actual_time,
            target_col=self.labels.weekly_target_time,
        )
        # 週別・科目別のピボットデータを作成
        self.weekly_detail_pivot = self._create_weekly_pivot()

        # 週間合計
        self.weekly_total = self._aggregate_data(
            group_cols=self.labels.week,
            actual_col=self.labels.weekly_actual_time,
            target_col=self.labels.weekly_target_time,
        ).with_columns(pl.lit("合計").alias(self.labels.category))

        # 期間全体
        self.period_total = self._aggregate_data(
            group_cols=self.labels.category,
            actual_col=self.labels.period_total_actual_time,
            target_col=self.labels.period_total_target_time,
        )

    def _create_weekly_pivot(self) -> pl.DataFrame:
        """週別の実績・目標時間用のピボットデータを作成"""
        group = [self.labels.week, self.labels.category]
        weekly_data = self.source_df.group_by(group).agg([
            pl.col("実績時間").sum().alias("実績時間"),
            pl.col("目標時間").sum().alias("目標時間"),
            (pl.col("実績時間").sum() / pl.col("目標時間").sum() * 100).round(1).alias(self.labels.achievement_rate)
        ]).sort(group)

        weekly_pivot = weekly_data.pivot(
            index=self.labels.week,
            on=self.labels.category,
            values=["実績時間", "目標時間", self.labels.achievement_rate]
        )

        result_weeks = []
        for week in weekly_pivot[self.labels.week]:
            result_weeks.append(f"{week} 実績")
            result_weeks.append(f"{week} 目標")

        # 初期データ構造を作成
        result_data = {
            '期間': result_weeks,
            **{subject: [] for subject in self.subjects},  # 時間
            **{f"{subject} {self.labels.achievement_rate}": [] for subject in self.subjects}  # 達成率
        }

        # データを埋める
        for week in weekly_pivot[self.labels.week]:
            for subject in self.subjects:
                # 週ごとの目標・実績データを取得
                subject_weekly_time = weekly_pivot.filter(pl.col('週') == week)
                result_data[subject].append(subject_weekly_time[f'実績時間_{subject}'][0])
                result_data[subject].append(subject_weekly_time[f'目標時間_{subject}'][0])

                achievement_rate = subject_weekly_time[f'{self.labels.achievement_rate}_{subject}'][0]
                achievement_rate_key = f"{subject} {self.labels.achievement_rate}"
                result_data[achievement_rate_key].append(achievement_rate)
                result_data[achievement_rate_key].append(achievement_rate)

        return pl.DataFrame(result_data, strict=False)

    def display_summaries(self) -> None:
        """サマリーの表示"""
        if any(x is None for x in [self.weekly_detail, self.weekly_total, self.period_total]):
            raise ValueError("サマリーが計算されていません。calculate_summaries()を先に実行してください。")

        print(f"\n集計期間: {self.date_range.start_date} ～ {self.date_range.end_date}")
        print("\n週別サマリー:")
        print(self.weekly_detail)
        print("\n週間合計:")
        print(self.weekly_total)
        print("\n期間全体:")
        print(self.period_total)


class StudyDataVisualizer:
    """学習データの可視化を担当するクラス"""

    def __init__(self, processor: StudyDataProcessor):
        self.processor = processor

    def create_weekly_stacked_bar(self) -> Figure:
        """週ごと・科目ごとの積み上げ棒グラフ作成"""
        if self.processor.weekly_detail_pivot is None:
            raise ValueError("週別ピボットデータが計算されていません。")

        achievement_rate_cols = [
            f"{subject} {self.processor.labels.achievement_rate}" for subject in self.processor.subjects
        ]

        return px.bar(
            self.processor.weekly_detail_pivot,
            x='期間',
            y=self.processor.subjects,
            title='{} ~ {} 週別の実績時間と目標時間'.format(*self.processor.date_range.range),
            labels={'value': '時間', 'variable': '科目'},
            barmode='stack',
            hover_data=achievement_rate_cols
        )

    def _create_common_bar(self, data: pl.DataFrame, x: str, y: List[str], title: str) -> Figure:
        """棒グラフの共通生成処理"""
        if data is None:
            raise ValueError("データが計算されていません。")

        return px.bar(
            data,
            x=x,
            y=y,
            title=title,
            labels={"value": "時間", "variable": "種別"},
            barmode="group",
            hover_data=[self.processor.labels.achievement_rate],
        )

    def create_total_weekly_bar(self) -> Figure:
        """週ごとの総計棒グラフの作成"""

        return px.bar(
            self.processor.weekly_total,
            x=self.processor.labels.week,
            y=[self.processor.labels.weekly_actual_time, self.processor.labels.weekly_target_time],
            title='{} ~ {} 週ごとの総計実績時間と目標時間'.format(*self.processor.date_range.range),
            labels={"value": "時間", "variable": "種別"},
            barmode="group",
            hover_data=[self.processor.labels.achievement_rate], )

    def create_period_total_bar(self) -> Figure:
        """期間全体の総計棒グラフの作成"""
        if self.processor.period_total is None:
            raise ValueError("データが計算されていません。")

        # 科目ごとの色の定義
        color_map = {
            "国語": {"target": "#F7BA89", "actual": "#F49D57"},
            "数学": {"target": "#7DD4AF", "actual": "#46C28E"},
            "英語": {"target": "#E78974", "actual": "#DE5739"},
            "社会": {"target": "#B78AF6", "actual": "#9959F3"},
            "理科": {"target": "#8A8FF6", "actual": "#585FF3"},
            "化学": {"target": "#8A8FF6", "actual": "#585FF3"}
        }

        df = self.processor.period_total
        fig = Figure()

        bar_settings = {
            "target": {
                "name": "目標",
                "color": [color_map[subject]["target"] for subject in df[self.processor.labels.category]],
                "y": df[self.processor.labels.period_total_target_time],
            },
            "actual": {
                "name": "実績",
                "color": [color_map[subject]["actual"] for subject in df[self.processor.labels.category]],
                "y": df[self.processor.labels.period_total_actual_time],
            }
        }

        for bar_type, settings in bar_settings.items():
            fig.add_trace(Bar(
                name=settings["name"],
                x=df[self.processor.labels.category],
                y=settings["y"],
                marker_color=settings["color"],
                customdata=list(zip(df[self.processor.labels.category], df[self.processor.labels.achievement_rate])),
                hovertemplate=f"科目: %{{customdata[0]}}<br>{settings['name']}: %{{y:.1f}}時間<br>達成率: %{{customdata[1]:.1f}}%<extra></extra>",
            ))

        # レイアウトの設定
        font_size = 20
        fig.update_layout(
            title=dict(text="{} ~ {} の実績時間と目標時間".format(*self.processor.date_range.range),
                       font=dict(size=font_size), ),
            xaxis_title="科目",
            xaxis=dict(title_font=dict(size=font_size), tickfont=dict(size=font_size)),
            yaxis_title="時間",
            yaxis=dict(title_font=dict(size=font_size), tickfont=dict(size=font_size)),
            barmode='group',
            hoverlabel=dict(font_size=font_size),
            showlegend=False,
        )

        return fig


def main():
    """メイン処理"""
    # 日付範囲の設定
    date_range = DateRange('2024/04/01', '2024/04/07')

    # データ処理インスタンスの作成と処理実行
    processor = StudyDataProcessor("study.csv", date_range)
    processor.calculate_summaries()
    # processor.display_summaries()

    # 可視化インスタンスの作成とグラフの表示
    visualizer = StudyDataVisualizer(processor)
    # fig1 = visualizer.create_weekly_stacked_bar()
    # fig2 = visualizer.create_total_weekly_bar()
    fig3 = visualizer.create_period_total_bar()

    # グラフの保存
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)
    # fig1.write_html(output_dir / "weekly_stacked_bar.html")
    # fig2.write_html(output_dir / "total_weekly_bar.html")
    fig3.write_html(output_dir / "period_total_bar.html")


if __name__ == "__main__":
    main()
