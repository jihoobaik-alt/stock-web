from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from io import BytesIO
import platform

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 0. 설정
# ============================================================
START = "2023-01-01"
END = "2027-01-01"  # 종료일 제외. 계속 최신 자료를 쓰려면 None

KST = timezone(timedelta(hours=9))

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = COLUMNS[:4]

KOREAN_NAMES = {
    "Open": "시가",
    "High": "고가",
    "Low": "저가",
    "Close": "종가",
    "Volume": "거래량",
}

N_TREES = 32
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 8
MAX_THRESHOLDS = 16

MIN_DATA_ROWS = 250
TEST_RATIO = 0.20
RANDOM_SEED = 42

CACHE_VERSION = "python315-yfinance-v1"

DISCLAIMER = (
    "예측 결과는 교육·연구 및 단순 참고용이며, "
    "실제 가격이나 수익을 보장하지 않습니다. "
    "투자의 책임은 전적으로 투자자 본인에게 있습니다."
)


def normalize(text):
    return "".join(str(text).split()).casefold()


def today_kst():
    return pd.Timestamp(datetime.now(KST).date())


def request_period():
    today = today_kst()
    start = pd.Timestamp(START)

    end = (
        today
        if END is None
        else min(pd.Timestamp(END), today)
    )

    if start >= end:
        raise ValueError("시작일은 종료일보다 이전이어야 합니다.")

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def environment_info():
    names = [
        "streamlit",
        "numpy",
        "pandas",
        "pyarrow",
        "yfinance",
        "curl_cffi",
        "exchange-calendars",
    ]

    return {
        "Python": platform.python_version(),
        "구현": platform.python_implementation(),
        **{name: version(name) for name in names},
    }


def normalize_dates(values, convert_to_kst=False):
    """
    시간대를 제거한 날짜 인덱스로 정리합니다.

    시세의 시간대가 있으면 한국시간으로 변환합니다.
    달력의 세션 라벨은 시각이 아니라 거래일 날짜이므로
    시간대만 제거합니다.
    """
    index = pd.DatetimeIndex(values)

    if index.hasnans:
        raise RuntimeError("날짜 인덱스에 빈 날짜가 있습니다.")

    if index.tz is not None:
        if convert_to_kst:
            index = index.tz_convert(KST)

        index = index.tz_localize(None)

    return index.normalize().as_unit("ns")


# ============================================================
# 1. 기업 목록 수집 및 검색
# ============================================================
def download_bytes(url):
    request = Request(
        url,
        headers={"User-Agent": "StockStudy/1.0"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            return response.read()

    except HTTPError as error:
        raise RuntimeError(
            f"종목 목록 서버 오류: HTTP {error.code}. "
            "접근 제한이면 반복 요청하거나 우회하지 마세요."
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"종목 목록 다운로드 실패: {error.reason}"
        ) from error


def download_stock_list():
    stocks = {}

    for market, suffix in [
        ("kospi", "KS"),
        ("kosdaq", "KQ"),
    ]:
        url = (
            "https://new.real.download.dws.co.kr/common/master/"
            f"{market}_code.mst.zip"
        )

        content = download_bytes(url)

        with ZipFile(BytesIO(content)) as archive:
            data = archive.read(f"{market}_code.mst")

        for row in data.splitlines():
            if len(row) < 61:
                continue

            try:
                code = row[:9].decode("ascii").strip()[-6:]
                name = row[21:61].decode("cp949").strip()
            except UnicodeDecodeError:
                continue

            if len(code) == 6 and code.isalnum() and name:
                stocks[f"{code}.{suffix}"] = name

    if not stocks:
        raise RuntimeError("종목 목록을 읽지 못했습니다.")

    return sorted(
        [(name, symbol) for symbol, name in stocks.items()],
        key=lambda item: normalize(item[0]),
    )


def search_stocks(stocks, query):
    query = normalize(query)

    query = {
        "네이버": "naver",
        "현대자동차": "현대차",
    }.get(query, query)

    if not query:
        raise ValueError("기업 이름이나 종목코드를 입력하세요.")

    exact = [
        (name, symbol)
        for name, symbol in stocks
        if query in {
            normalize(name),
            normalize(symbol),
            normalize(symbol.split(".")[0]),
        }
    ]

    matches = exact or [
        (name, symbol)
        for name, symbol in stocks
        if query in normalize(name) or query in normalize(symbol)
    ]

    if not matches:
        raise ValueError(
            "검색 결과가 없습니다. "
            "국내 코스피·코스닥 기업명이나 종목코드를 확인하세요."
        )

    return matches


# ============================================================
# 2. 시세 수집 및 거래일 정리
# ============================================================
def load_prices(symbol, start, end):
    try:
        raw = yf.Ticker(symbol).history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=False,
            keepna=True,
            timeout=30,
        )
    except Exception as error:
        raise RuntimeError(
            f"시세 요청 실패: {type(error).__name__}: {error}. "
            "접근·요청 제한이 있다면 반복 요청하거나 우회하지 마세요."
        ) from error

    if raw is None or raw.empty:
        raise RuntimeError(
            "주가 자료를 받지 못했습니다. "
            "종목 지원 여부, 인터넷 연결, 제공처의 요청 제한을 "
            "확인해 주세요."
        )

    if not set(COLUMNS).issubset(raw.columns):
        raise RuntimeError("필요한 OHLCV 열이 없습니다.")

    prices = raw[COLUMNS].copy()

    prices.index = normalize_dates(
        prices.index,
        convert_to_kst=True,
    )
    prices.index.name = "Date"

    prices = prices.loc[
        ~prices.index.duplicated(keep="last")
    ].sort_index()

    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)

    prices = prices.loc[
        (prices.index >= start_date)
        & (prices.index < end_date)
    ].copy()

    if prices.empty:
        raise RuntimeError("조회 기간에 자료가 없습니다.")

    prices = prices.apply(
        pd.to_numeric,
        errors="coerce",
    ).astype(float)

    prices = prices.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # 다음 거래일까지 계산할 수 있도록 달력을 넉넉히 만듭니다.
    try:
        calendar = xcals.get_calendar(
            "XKRX",
            start=prices.index[0],
            end=end_date + pd.Timedelta(days=40),
        )

        sessions = normalize_dates(calendar.sessions)

    except Exception as error:
        raise RuntimeError(
            "한국거래소 거래일 달력 생성에 실패했습니다. "
            f"pandas={pd.__version__}, "
            f"exchange-calendars={version('exchange-calendars')}. "
            f"원인: {error}"
        ) from error

    # 달력에 없는 날짜는 무조건 삭제하지 않습니다.
    unexpected = prices.index.difference(sessions)
    non_session_rows = prices.loc[unexpected].copy()

    if not non_session_rows.empty:
        no_trade_data = (
            non_session_rows["Volume"].eq(0)
            | non_session_rows[COLUMNS].isna().all(axis=1)
        )

        review = non_session_rows.loc[~no_trade_data]

        if not review.empty:
            details = review.head(10).to_string()

            raise RuntimeError(
                "시세와 거래일 달력이 일치하지 않습니다. "
                "달력에 없는 날짜에 거래 자료가 있어 "
                "임의로 삭제하지 않았습니다.\n\n"
                f"{details}\n\n"
                f"pandas={pd.__version__}, "
                f"exchange-calendars={version('exchange-calendars')}"
            )

        non_session_rows["Issue"] = (
            "달력상 비거래일이며 거래량 0 또는 OHLCV 전체 결측"
        )

    # 실제 거래일의 누락 행은 삭제하지 않고 결측으로 유지합니다.
    past_sessions = sessions[
        (sessions >= prices.index[0])
        & (sessions < end_date)
    ]

    if len(past_sessions) == 0:
        raise RuntimeError("사용할 과거 거래일이 없습니다.")

    grid = prices.reindex(past_sessions)
    grid.index.name = "Date"

    invalid = (
        grid.isna().any(axis=1)
        | grid[PRICE_COLUMNS].le(0).any(axis=1)
        | grid["Volume"].le(0)
        | grid["High"].lt(grid["Low"])
        | grid["High"].lt(
            grid[["Open", "Close"]].max(axis=1)
        )
        | grid["Low"].gt(
            grid[["Open", "Close"]].min(axis=1)
        )
    )

    issues = grid.loc[invalid].copy()
    grid.loc[invalid, COLUMNS] = np.nan

    if grid.dropna(subset=COLUMNS).empty:
        raise RuntimeError("사용 가능한 시세 자료가 없습니다.")

    if grid.iloc[-1].isna().any():
        raise RuntimeError(
            f"최근 거래일 {grid.index[-1]:%Y-%m-%d}의 "
            "유효한 시세가 없습니다. "
            "제공 지연·거래정지·결측 여부를 확인하세요. "
            "오래된 자료를 최신 자료 대신 사용하지 않습니다."
        )

    following = sessions[
        sessions > grid.index[-1]
    ]

    if len(following) == 0:
        raise RuntimeError("다음 거래일을 찾지 못했습니다.")

    return {
        "raw": raw,
        "grid": grid,
        "issues": issues,
        "non_session_rows": non_session_rows,
        "target_date": following[0],
        "fetched_at": datetime.now(KST).strftime(
            "%Y-%m-%d %H:%M:%S KST"
        ),
    }


# ============================================================
# 3. 학습 배열 검사
# ============================================================
def checked_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if (
        X.ndim != 2
        or y.ndim != 2
        or X.size == 0
        or y.size == 0
        or len(X) != len(y)
    ):
        raise ValueError("학습 배열의 크기가 올바르지 않습니다.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("학습 배열에 결측값이나 무한대가 있습니다.")

    return X, y


# ============================================================
# 4. 회귀용 의사결정트리
# ============================================================
@dataclass(slots=True)
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    value: np.ndarray | None = None


class DecisionTreeRegressor:
    def __init__(
        self,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_features=None,
        max_thresholds=MAX_THRESHOLDS,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        X, y = checked_xy(X, y)

        self.rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]

        self.n_features_ = (
            X.shape[1]
            if self.n_features is None
            else max(1, min(int(self.n_features), X.shape[1]))
        )

        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        leaf = Node(value=y.mean(axis=0))

        if (
            depth >= self.max_depth
            or n_samples < 2 * self.min_samples_leaf
            or np.var(y, axis=0).mean() <= 1e-12
        ):
            return leaf

        candidates = self.rng.choice(
            n_features,
            self.n_features_,
            replace=False,
        )

        total_sum = y.sum(axis=0)
        parent_score = np.sum(total_sum ** 2) / n_samples

        best_gain = 1e-10
        best_split = None

        for feature in candidates:
            order = np.argsort(
                X[:, feature],
                kind="mergesort",
            )
            sorted_x = X[order, feature]

            # 서로 다른 값 사이에서만 분할합니다.
            positions = np.flatnonzero(
                sorted_x[:-1] < sorted_x[1:]
            ) + 1

            positions = positions[
                (positions >= self.min_samples_leaf)
                & (
                    n_samples - positions
                    >= self.min_samples_leaf
                )
            ]

            if len(positions) == 0:
                continue

            # 실행 시간을 줄이기 위한 근사 탐색입니다.
            if len(positions) > self.max_thresholds:
                selected = np.linspace(
                    0,
                    len(positions) - 1,
                    self.max_thresholds,
                    dtype=int,
                )
                positions = positions[selected]

            cumulative = np.cumsum(y[order], axis=0)

            left_sums = cumulative[positions - 1]
            right_sums = total_sum - left_sums

            # 제곱오차 감소량과 동등한 분할 점수
            scores = np.sum(
                left_sums ** 2 / positions[:, None]
                + right_sums ** 2
                / (n_samples - positions)[:, None],
                axis=1,
            )

            gains = scores - parent_score
            best_index = int(np.argmax(gains))

            if gains[best_index] > best_gain:
                best_gain = float(gains[best_index])
                position = int(positions[best_index])

                best_split = (
                    int(feature),
                    float(sorted_x[position - 1]),
                )

        if best_split is None:
            return leaf

        feature, threshold = best_split
        left_mask = X[:, feature] <= threshold

        return Node(
            feature=feature,
            threshold=threshold,
            left=self._grow_tree(
                X[left_mask],
                y[left_mask],
                depth + 1,
            ),
            right=self._grow_tree(
                X[~left_mask],
                y[~left_mask],
                depth + 1,
            ),
        )

    def predict(self, X):
        if self.root is None:
            raise RuntimeError("먼저 트리를 학습해야 합니다.")

        X = np.asarray(X, dtype=float)

        if (
            X.ndim != 2
            or X.shape[1] != self.n_features_in_
            or not np.isfinite(X).all()
        ):
            raise ValueError("예측 입력을 확인하세요.")

        result = []

        for x in X:
            node = self.root

            while node.value is None:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            result.append(node.value)

        return np.asarray(result)


# ============================================================
# 5. 회귀용 랜덤 포레스트
# ============================================================
class RandomForestRegressor:
    def __init__(
        self,
        n_trees=N_TREES,
        random_state=RANDOM_SEED,
    ):
        self.n_trees = n_trees
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y, on_tree=None):
        X, y = checked_xy(X, y)

        if self.n_trees < 1:
            raise ValueError("트리는 1개 이상이어야 합니다.")

        self.trees = []
        rng = np.random.default_rng(self.random_state)

        # 가격 변화율과 거래량의 크기 차이를 줄입니다.
        # 이번 학습 구간의 통계만 사용합니다.
        self.y_mean_ = y.mean(axis=0)
        scale = y.std(axis=0)
        self.y_scale_ = np.where(scale > 1e-12, scale, 1.0)

        standardized_y = (
            y - self.y_mean_
        ) / self.y_scale_

        n_samples, n_features = X.shape
        selected_features = max(1, int(np.sqrt(n_features)))

        for number in range(self.n_trees):
            indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
            )

            tree = DecisionTreeRegressor(
                n_features=selected_features,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )

            tree.fit(
                X[indices],
                standardized_y[indices],
            )
            self.trees.append(tree)

            if on_tree is not None:
                on_tree(number + 1, self.n_trees)

        return self

    def predict(self, X):
        if not self.trees:
            raise RuntimeError("먼저 랜덤 포레스트를 학습해야 합니다.")

        prediction = np.mean(
            [tree.predict(X) for tree in self.trees],
            axis=0,
        )

        return prediction * self.y_scale_ + self.y_mean_


# ============================================================
# 6. 입력 특성 및 가격 복원
# ============================================================
def make_features(df):
    close = df["Close"]
    volume = df["Volume"]
    daily_return = close.pct_change(fill_method=None)

    X = pd.DataFrame(index=df.index)

    for column in ["Open", "High", "Low"]:
        X[f"{column}_relative_close"] = df[column] / close - 1

    X["daily_range"] = (df["High"] - df["Low"]) / close

    for period in [1, 2, 3, 5, 10]:
        X[f"return_{period}"] = close.pct_change(
            periods=period,
            fill_method=None,
        )

    for lag in range(1, 6):
        X[f"return_lag_{lag}"] = daily_return.shift(lag)

    for window in [5, 10, 20, 60]:
        X[f"close_to_ma_{window}"] = (
            close / close.rolling(window).mean() - 1
        )

    for window in [5, 20]:
        X[f"volatility_{window}"] = (
            daily_return.rolling(window).std()
        )

    X["log_volume"] = np.log1p(volume)
    X["log_volume_change"] = np.log1p(volume).diff()

    for window in [5, 20]:
        X[f"volume_relative_{window}"] = (
            volume / volume.rolling(window).mean() - 1
        )

    X["weekday"] = X.index.dayofweek

    return X.replace([np.inf, -np.inf], np.nan)


def restore_predictions(predictions, base_close):
    # pandas/NumPy에서 얻은 배열을 직접 수정하지 않도록 복사
    result = np.array(predictions, dtype=float, copy=True)

    base_close = np.asarray(
        base_close,
        dtype=float,
    ).reshape(-1, 1)

    result[:, :4] = base_close * (1 + result[:, :4])
    result[:, :4] = np.maximum(result[:, :4], 1e-8)

    # 형식적 OHLC 관계 보정이며 정확도를 보장하지 않습니다.
    result[:, 1] = np.maximum(
        result[:, 1],
        np.maximum(result[:, 0], result[:, 3]),
    )
    result[:, 2] = np.minimum(
        result[:, 2],
        np.minimum(result[:, 0], result[:, 3]),
    )

    result[:, 4] = np.maximum(0, np.rint(result[:, 4]))

    return result


# ============================================================
# 7. 시간순 검증 및 최종 예측
# ============================================================
def train_and_forecast(bundle, name, symbol, report=None):
    def notify(percent, text):
        if report is not None:
            report(int(percent), text)

    if not 0 < TEST_RATIO < 1:
        raise ValueError("TEST_RATIO는 0과 1 사이여야 합니다.")

    df = bundle["grid"]

    notify(5, "입력 특성과 다음 거래일 정답을 만들고 있어요.")

    features = make_features(df)
    next_values = df[COLUMNS].shift(-1)

    targets = next_values.copy()
    targets[PRICE_COLUMNS] = (
        next_values[PRICE_COLUMNS].div(df["Close"], axis=0) - 1
    )
    targets.columns = [
        "Open_Return",
        "High_Return",
        "Low_Return",
        "Close_Return",
        "Volume",
    ]
    targets = targets.replace([np.inf, -np.inf], np.nan)

    target_dates = pd.Series(
        df.index,
        index=df.index,
        name="TargetDate",
    ).shift(-1)

    valid = (
        features.notna().all(axis=1)
        & targets.notna().all(axis=1)
    )

    if int(valid.sum()) < MIN_DATA_ROWS:
        raise RuntimeError(
            f"유효한 학습 사례가 {MIN_DATA_ROWS}개 미만입니다. "
            "자료가 충분한 다른 종목을 선택하거나 "
            "START 날짜를 앞당겨 주세요."
        )

    if features.iloc[-1].isna().any():
        raise RuntimeError(
            "최신 날짜의 입력 특성을 만들 수 없습니다. "
            "최근 60거래일 안의 결측 자료를 확인하세요."
        )

    X = features.loc[valid].to_numpy(dtype=float)
    y = targets.loc[valid].to_numpy(dtype=float)

    sample_dates = features.index[valid]
    base_close = df.loc[valid, "Close"].to_numpy(dtype=float)
    actual_next = next_values.loc[valid].to_numpy(dtype=float)
    baseline = df.loc[valid, COLUMNS].to_numpy(dtype=float)

    learning_data = features.loc[valid].join(
        targets.loc[valid].add_prefix("Target_")
    )
    learning_data.insert(
        0,
        "TargetDate",
        target_dates.loc[valid],
    )
    learning_data.index.name = "InputDate"

    split = int(len(X) * (1 - TEST_RATIO))
    first_test_date = sample_dates[split]

    # 학습 정답 날짜도 첫 검증 입력일보다 이전이어야 합니다.
    train_mask = (
        target_dates.loc[valid] < first_test_date
    ).to_numpy()

    if int(train_mask.sum()) < 2:
        raise RuntimeError("학습 구간이 너무 짧습니다.")

    forest = RandomForestRegressor()

    forest.fit(
        X[train_mask],
        y[train_mask],
        on_tree=lambda done, total: notify(
            10 + 35 * done // total,
            f"검증용 모델 학습 · 트리 {done}/{total}",
        ),
    )

    test_prediction = restore_predictions(
        forest.predict(X[split:]),
        base_close[split:],
    )

    actual_test = actual_next[split:]
    baseline_test = baseline[split:]

    model_mae = np.abs(
        test_prediction - actual_test
    ).mean(axis=0)

    baseline_mae = np.abs(
        baseline_test - actual_test
    ).mean(axis=0)

    denominator = np.where(
        baseline_mae > 0,
        baseline_mae,
        np.nan,
    )

    metrics = pd.DataFrame(
        {
            "RF_MAE": model_mae,
            "전일값_MAE": baseline_mae,
            "개선율(%)": 100 * (1 - model_mae / denominator),
        },
        index=[KOREAN_NAMES[c] for c in COLUMNS],
    ).rename_axis("항목")

    test_details = pd.DataFrame(
        index=pd.DatetimeIndex(
            target_dates.loc[valid].iloc[split:].to_numpy(),
            name="Date",
        )
    )
    test_details["InputDate"] = sample_dates[split:].to_numpy()

    for j, column in enumerate(COLUMNS):
        test_details[f"Actual_{column}"] = actual_test[:, j]
        test_details[f"Predicted_{column}"] = test_prediction[:, j]
        test_details[f"Baseline_{column}"] = baseline_test[:, j]

    # 검증 후, 정답을 아는 전체 사례로 다시 학습
    forest.fit(
        X,
        y,
        on_tree=lambda done, total: notify(
            50 + 45 * done // total,
            f"최종 모델 학습 · 트리 {done}/{total}",
        ),
    )

    last_close = float(df["Close"].iloc[-1])

    prediction = restore_predictions(
        forest.predict(
            features.iloc[[-1]].to_numpy(dtype=float)
        ),
        [last_close],
    )[0]

    change_pct = float(
        (prediction[3] / last_close - 1) * 100
    )

    forecast = pd.DataFrame(
        [prediction],
        columns=COLUMNS,
        index=pd.DatetimeIndex(
            [bundle["target_date"]],
            name="Date",
        ),
    )
    forecast.insert(0, "InputDate", df.index[-1])
    forecast.insert(0, "Symbol", symbol)
    forecast.insert(0, "Name", name)
    forecast["Close_Change_Pct"] = change_pct

    notify(100, "예측을 완료했어요.")

    return {
        "name": name,
        "symbol": symbol,
        "input_date": df.index[-1],
        "target_date": bundle["target_date"],
        "last_close": last_close,
        "prediction": prediction,
        "change_pct": change_pct,
        "forecast": forecast,
        "metrics": metrics,
        "raw": bundle["raw"],
        "grid": df,
        "prices": df.dropna(subset=COLUMNS),
        "issues": bundle["issues"],
        "non_session_rows": bundle["non_session_rows"],
        "learning_data": learning_data,
        "test_predictions": test_details,
        "n_samples": len(X),
        "n_train": int(train_mask.sum()),
        "n_test": len(X) - split,
        "fetched_at": bundle["fetched_at"],
    }
