import logging
from io import BytesIO
from math import isfinite
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st

import stock_forecast as core


st.set_page_config(
    page_title="개미의 주가 예측",
    page_icon="🐜",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent

RESULT_ITEMS = [
    ("Open", "예측 시가"),
    ("High", "예측 고가"),
    ("Low", "예측 저가"),
    ("Close", "예상 종가"),
    ("Volume", "예상 거래량"),
]


# ============================================================
# 1. 디자인
# ============================================================
st.markdown(
    """
    <style>
    .result-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 20px;
        padding: 13px 0;
    }
    .result-label {
        font-size: 1.4rem;
        color: #222;
    }
    .result-value {
        font-size: 1.65rem;
        font-weight: 700;
        text-align: right;
        color: #111;
    }
    .rate-title {
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 26px;
        color: #222;
    }
    .rate-value {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    @media (max-width: 640px) {
        .result-label { font-size: 1.05rem; }
        .result-value { font-size: 1.25rem; }
        .rate-title { font-size: 1.3rem; }
        .rate-value { font-size: 2.4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. 기본 개미 일러스트
# ============================================================
ANT_SVG = """
<svg xmlns="http://www.w3.org/2000/svg"
     width="360" height="360" viewBox="0 0 360 360">
  <title>귀여운 개미</title>
  <ellipse cx="180" cy="335" rx="95" ry="12" fill="#eee"/>
  <g fill="none" stroke="#51382e" stroke-width="7"
     stroke-linecap="round" stroke-linejoin="round">
    <path d="M139 190 L95 170 L72 140"/>
    <path d="M138 215 L87 225 L62 253"/>
    <path d="M145 250 L104 285 L89 324"/>
    <path d="M221 190 L265 170 L288 140"/>
    <path d="M222 215 L273 225 L298 253"/>
    <path d="M215 250 L256 285 L271 324"/>
    <path d="M150 90 L128 48 L106 33"/>
    <path d="M210 90 L232 48 L254 33"/>
  </g>
  <circle cx="106" cy="33" r="8" fill="#51382e"/>
  <circle cx="254" cy="33" r="8" fill="#51382e"/>
  <g stroke="#51382e" stroke-width="4">
    <ellipse cx="180" cy="281" rx="66" ry="48" fill="#b9855d"/>
    <ellipse cx="180" cy="211" rx="49" ry="53" fill="#c48e65"/>
    <ellipse cx="180" cy="129" rx="81" ry="67" fill="#d5a77e"/>
  </g>
  <ellipse cx="151" cy="122" rx="14" ry="20" fill="#2e221d"/>
  <ellipse cx="209" cy="122" rx="14" ry="20" fill="#2e221d"/>
  <ellipse cx="147" cy="116" rx="5" ry="7" fill="white"/>
  <ellipse cx="205" cy="116" rx="5" ry="7" fill="white"/>
  <ellipse cx="125" cy="150" rx="13" ry="8" fill="#e8a5a0"/>
  <ellipse cx="235" cy="150" rx="13" ry="8" fill="#e8a5a0"/>
  <path d="M161 153 Q180 174 199 153"
        fill="none" stroke="#51382e"
        stroke-width="4" stroke-linecap="round"/>
</svg>
"""


# ============================================================
# 3. 자료 캐시
# ============================================================
@st.cache_data(
    ttl=86400,
    max_entries=1,
    show_spinner=False,
)
def cached_stock_list():
    return core.download_stock_list()


@st.cache_data(
    ttl=1800,
    max_entries=16,
    show_spinner=False,
)
def cached_prices(symbol, start, end, code_version):
    return core.load_prices(symbol, start, end)


# ============================================================
# 4. 결과 및 다운로드 도우미
# ============================================================
def clear_result():
    st.session_state["result"] = None
    st.session_state["error"] = None


def csv_bytes(frame):
    return frame.to_csv(
        date_format="%Y-%m-%d",
    ).encode("utf-8-sig")


def result_zip(result):
    buffer = BytesIO()

    files = {
        "market_prices_raw.csv": result["raw"],
        "market_prices.csv": result["prices"],
        "market_prices_grid.csv": result["grid"],
        "market_prices_issues.csv": result["issues"],
        "non_session_rows.csv": result["non_session_rows"],
        "learning_data.csv": result["learning_data"],
        "metrics.csv": result["metrics"],
        "test_predictions.csv": result["test_predictions"],
        "next_day_prediction.csv": result["forecast"],
    }

    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for filename, frame in files.items():
            archive.writestr(filename, csv_bytes(frame))

        archive.writestr(
            "README.txt",
            core.DISCLAIMER
            + "\n\n가격 오차는 원, 거래량 오차는 주 단위입니다."
            + "\n개선율은 예측 성공 확률이나 수익률이 아닙니다.",
        )

    return buffer.getvalue()


def show_ant():
    st.write("")
    st.write("")

    image_path = APP_DIR / "ant.png"

    if image_path.exists():
        st.image(str(image_path), width=290)
    else:
        st.image(ANT_SVG, width=290)

    st.caption("개미의 한 걸음, 데이터로 살펴보는 다음 거래일")


def show_result(result):
    if result is None:
        st.subheader("예측 결과")
        st.caption("기업을 검색하면 기준일과 예측 대상일이 표시돼요.")
    else:
        st.subheader(f"{result['name']} · {result['symbol']}")
        st.caption(
            f"자료 기준: {result['input_date']:%Y-%m-%d} 마감"
            f"  |  예측 대상: {result['target_date']:%Y-%m-%d}"
        )
        st.caption(
            f"기준일 종가: {result['last_close']:,.2f}원"
            f"  |  조회: {result['fetched_at']}"
        )

    rows = []

    # HTML에는 고정 문구와 계산된 숫자만 사용합니다.
    for index, (column, title) in enumerate(RESULT_ITEMS):
        if result is None:
            text = "—"
        elif column == "Volume":
            text = f"{result['prediction'][index]:,.0f} 주"
        else:
            text = f"{result['prediction'][index]:,.2f} 원"

        rows.append(
            '<div class="result-row">'
            f'<span class="result-label">{title}</span>'
            f'<span class="result-value">{text}</span>'
            "</div>"
        )

    st.markdown("".join(rows), unsafe_allow_html=True)

    rate = result["change_pct"] if result is not None else None
    rate_text = "—" if rate is None else f"{rate:+.2f}%"

    color = "#222222"
    if rate is not None:
        if rate > 0:
            color = "#c74343"
        elif rate < 0:
            color = "#326cc0"

    st.markdown(
        '<div class="rate-title">최종 예측 종가 등락률</div>'
        f'<div class="rate-value" style="color:{color}">'
        f"{rate_text}</div>",
        unsafe_allow_html=True,
    )

    if result is None:
        st.info("왼쪽에서 기업명을 검색해 주세요.")
        return

    st.caption("등락률은 자료 기준일 종가 대비 예상 변화율이에요.")

    today = core.today_kst()

    if result["target_date"] < today:
        st.warning(
            "예측 대상일이 현재 날짜보다 과거입니다. "
            "다시 검색하거나 stock_forecast.py의 END 설정을 확인하세요."
        )
    elif result["target_date"] == today:
        st.info(
            "실행 당일 자료는 제외했어요. "
            "이 결과는 오늘 전체 일봉에 대한 추정이며 "
            "오늘 장중 시세는 반영하지 않았어요."
        )
    else:
        st.caption(
            "마지막 수집일의 다음 거래일에 대한 예측이에요. "
            "실시간 시세 기반 예측은 아니에요."
        )

    score = result["metrics"].loc["종가"]
    improvement = float(score["개선율(%)"])

    st.caption(
        f"검증 종가 MAE: {score['RF_MAE']:,.2f}원"
        f"  |  전일 종가 반복 기준: {score['전일값_MAE']:,.2f}원"
    )

    if isfinite(improvement) and improvement <= 0:
        st.warning(
            "이번 검증에서는 종가 예측이 "
            "전일 종가를 그대로 쓰는 단순 기준보다 좋지 않았어요."
        )

    if len(result["non_session_rows"]) > 0:
        st.caption(
            f"달력상 비거래일 중 거래량 0 또는 전체 결측인 "
            f"{len(result['non_session_rows']):,}개 행은 "
            "원자료에 보존하고 모델에서 제외했어요."
        )

    with st.expander("검증 결과 · 과거 자료 · CSV 다운로드"):
        st.write(
            f"학습 가능 사례 {result['n_samples']:,}개 / "
            f"검증 시 학습 {result['n_train']:,}개 / "
            f"검증 {result['n_test']:,}개"
        )

        st.dataframe(result["metrics"].round(3))

        st.caption(
            "MAE는 작을수록 좋아요. 가격 오차는 원, "
            "거래량 오차는 주 단위예요. "
            "개선율은 단순 기준 대비 오차 감소율이며 "
            "예측 성공 확률이나 수익률이 아니에요."
        )

        st.caption(
            "앞쪽 자료로 학습하고 뒤쪽 자료로 검증했어요. "
            "검증 구간에서는 각 날짜까지 관측된 정보로 "
            "다음 거래일을 예측했어요."
        )

        st.line_chart(result["grid"][["Close"]].tail(120))

        st.caption(
            f"거래일 중 결측·비정상 값·거래량 0으로 "
            f"사용하지 않은 행: {len(result['issues']):,}개"
        )

        prefix = result["symbol"].replace(".", "_")

        st.download_button(
            "최종 예측 CSV 내려받기",
            data=csv_bytes(result["forecast"]),
            file_name=f"{prefix}_prediction.csv",
            mime="text/csv",
            on_click="ignore",
        )

        st.download_button(
            "과거 자료·학습 자료·검증 결과 ZIP 내려받기",
            data=result_zip(result),
            file_name=f"{prefix}_results.zip",
            mime="application/zip",
            on_click="ignore",
        )


# ============================================================
# 5. 세션 초기화
# ============================================================
for key, default in {
    "matches": [],
    "result": None,
    "error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


requested_stock = None
left, right = st.columns([1, 1.65], gap="large")


# ============================================================
# 6. 왼쪽: 기업 검색 및 개미
# ============================================================
with left:
    st.header("기업 검색창")

    with st.form("company_search"):
        query = st.text_input(
            "기업명 또는 종목코드",
            value="삼성전자",
            placeholder="예: 삼성전자, 네이버, 005930",
        )
        submitted = st.form_submit_button(
            "검색 · 예측",
            type="primary",
        )

    if submitted:
        clear_result()
        st.session_state["matches"] = []
        st.session_state.pop("stock_choice", None)

        try:
            with st.spinner("종목 목록을 검색하고 있어요."):
                matches = core.search_stocks(
                    cached_stock_list(),
                    query,
                )

            st.session_state["matches"] = matches

            if len(matches) == 1:
                requested_stock = matches[0]

        except Exception as error:
            logging.exception("기업 검색 실패")
            st.session_state["error"] = str(error)

    matches = st.session_state["matches"]

    if matches:
        selected = st.selectbox(
            "예측할 종목 확인",
            options=matches,
            format_func=lambda item: f"{item[0]} ({item[1]})",
            key="stock_choice",
            on_change=clear_result,
        )

        if len(matches) > 1:
            st.caption(
                f"{len(matches)}개 종목이 검색됐어요. "
                "선택한 뒤 아래 버튼을 눌러 주세요."
            )

        if st.button("선택한 기업 예측하기", type="primary"):
            requested_stock = selected

    st.caption(
        "국내 코스피·코스닥 종목 대상 · 실행 당일 자료 제외"
    )

    show_ant()


# ============================================================
# 7. 오른쪽: 수집·학습·예측
# ============================================================
with right:
    if requested_stock is not None:
        clear_result()
        name, symbol = requested_stock

        progress = st.progress(
            0,
            text=f"{name}의 과거 자료를 수집하고 있어요.",
        )

        def report(percent, text):
            progress.progress(percent, text=text)

        try:
            start, end = core.request_period()

            bundle = cached_prices(
                symbol,
                start,
                end,
                core.CACHE_VERSION,
            )

            st.session_state["result"] = core.train_and_forecast(
                bundle=bundle,
                name=name,
                symbol=symbol,
                report=report,
            )

        except Exception as error:
            logging.exception("주가 예측 실패")
            st.session_state["error"] = str(error)

        finally:
            progress.empty()

    if st.session_state["error"]:
        st.error(st.session_state["error"])

    show_result(st.session_state["result"])


# ============================================================
# 8. 경고 문구와 실행 환경
# ============================================================
st.write("")
st.warning(core.DISCLAIMER)
st.caption("이 프로그램은 매수·매도 주문을 실행하지 않습니다.")

with st.expander("Python · 패키지 버전 확인"):
    st.json(core.environment_info())
