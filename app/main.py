from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import graphs
import data_collection
import io
import os
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# -----------------------------------------------------------------------------
# App & logging
# -----------------------------------------------------------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

scheduler = BackgroundScheduler(timezone="Europe/Moscow")

# -----------------------------------------------------------------------------
# Paths & templates
# -----------------------------------------------------------------------------
# --- Paths & templates ---
BASE_DIR = Path(__file__).resolve().parent

print(BASE_DIR)

CARDS_DIR = BASE_DIR / "data" / "cards prod"
STATS_DIR = BASE_DIR / "data" / "table prod"

print(CARDS_DIR)
print(STATS_DIR)

LOCAL_OUTPUT_PATH = Path(os.getenv("LOCAL_OUTPUT_PATH", BASE_DIR / "data"))
LOCAL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(BASE_DIR, "data", "card_full_data.csv")

print(DATA_PATH)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def _check_dir(p: Path, must_write: bool = False) -> dict:
    info = {"path": str(p), "exists": p.exists(), "is_dir": p.is_dir(), "readable": False, "writable": False, "sample": []}
    try:
        if p.exists() and p.is_dir():
            # показать несколько элементов каталога
            info["sample"] = sorted([f.name for f in p.iterdir()])[:5]
            # проверка чтения
            for _ in p.iterdir():
                info["readable"] = True
                break
            # проверка записи при необходимости
            if must_write:
                testfile = p / ".write_test"
                try:
                    testfile.write_text("ok", encoding="utf-8")
                    testfile.unlink(missing_ok=True)
                    info["writable"] = True
                except Exception:
                    info["writable"] = False
        return info
    except Exception:
        return info

def verify_paths(strict: bool = False) -> dict:
    cards = _check_dir(CARDS_DIR, must_write=False)
    table = _check_dir(STATS_DIR, must_write=False)
    outd  = _check_dir(LOCAL_OUTPUT_PATH, must_write=True)
    status = {
        "cards": cards,
        "table": table,
        "out": outd,
        "ok": (
            cards["exists"] and cards["is_dir"] and cards["readable"] and
            table["exists"] and table["is_dir"] and table["readable"] and
            outd["exists"] and outd["is_dir"] and (outd["writable"] or not strict)
        )
    }
    return status


# Стартовая проверка: логируем и (опционально) блокируем старт при полном отсутствии путей
@app.on_event("startup")
async def _startup_check():
    status = verify_paths(strict=False)
    # Если сетевые пути недоступны — можно либо упасть, либо просто логнуть и работать в деградации
    if not status["ok"]:
        # выбери поведение:
        # 1) Жёсткая ошибка: не стартуем
        #   raise RuntimeError(f"Paths not available: {status}")
        # 2) Мягко: просто лог (оставлю мягкий вариант по умолчанию)
        print("[WARN] Some paths are not available:", status)

# Healthcheck для Docker
@app.get("/healthz")
def healthz():
    status = verify_paths(strict=True)
    code = 200 if status["ok"] else 503
    return JSONResponse(content=status, status_code=code)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def update_data_job():
    """Фоновое обновление данных (запускается планировщиком)."""
    try:
        logging.info("🔄 Запуск обновления данных...")
        data_collection.main()
        logging.info("✅ Обновление данных завершено")
    except Exception as e:
        logging.exception(f"Ошибка при обновлении данных: {e}")

def load_data() -> pd.DataFrame:
    print(f"[DEBUG] DATA_PATH = {DATA_PATH.resolve()}")
    print(f"[DEBUG] Exists? {DATA_PATH.exists()}")
    # если файла нет — соберём
    if not DATA_PATH.exists():
        data_collection.main()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Файл данных не найден: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
    return df


    # Приводим даты
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
    return df

def get_default_period() -> list[str]:
    """Последние 14 дней, в ISO (YYYY-MM-DD)."""
    today = datetime.today().date()
    start = today - timedelta(days=13)
    return [start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")]

DEFAULT_PERIOD = get_default_period()

def to_display(d: str) -> str:
    return pd.to_datetime(d).strftime("%d.%m.%y")

def parse_period(start: str, end: str) -> list[pd.Timestamp]:
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(end, errors="coerce")

    if pd.isna(s) or pd.isna(e):
        s, e = pd.to_datetime(DEFAULT_PERIOD[0]), pd.to_datetime(DEFAULT_PERIOD[1])
    if s > e:
        s, e = e, s
    return [s, e]

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class KPIResponse(BaseModel):
    total_cards: int
    total_defects: int
    defect_percent: float
    total_money: float

# -----------------------------------------------------------------------------
# Endpoints: справочники
# -----------------------------------------------------------------------------
@app.get("/nomenclature")
async def get_nomenclature():
    df = load_data()
    nomen_list = sorted(df["Номенклатура"].dropna().astype(str).unique().tolist())
    return JSONResponse(nomen_list)

@app.get("/places")
async def get_places():
    df = load_data()
    places = sorted(df["Участок"].dropna().astype(str).unique().tolist())
    return JSONResponse(places)

# -----------------------------------------------------------------------------
# Index (шаблон)
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    df = load_data()
    start_iso, end_iso = DEFAULT_PERIOD
    total_cards, total_defects, defect_percent, total_money = graphs.get_kpis(
        df, parse_period(start_iso, end_iso)
    )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "start_display": to_display(start_iso),
            "end_display": to_display(end_iso),
            "total_cards": f'{total_cards:,.2f}'.replace(",", " ").replace(".00", "") + " шт",
            "total_defects": f'{total_defects:,.2f}'.replace(",", " ").replace(".00", "") + " шт",
            "defect_percent": defect_percent,
            "total_money": f"{total_money:,.2f}".replace(",", " ").replace(".00", "") + " ₽",
        },
    )

# -----------------------------------------------------------------------------
# Endpoints: графики
# -----------------------------------------------------------------------------
@app.get("/plots/line_total")
async def plot_line_total(
    start: str = DEFAULT_PERIOD[0],
    end: str = DEFAULT_PERIOD[1],
    nomen: str | None = None,
    place: str | None = None,
):
    df = load_data()
    filtered = graphs.prepare_data(df, nomen=nomen, place=place)
    img_bytes = graphs.plot_line_total(filtered, parse_period(start, end))
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

@app.get("/plots/line_defects")
async def plot_line_defects(
    start: str = DEFAULT_PERIOD[0],
    end: str = DEFAULT_PERIOD[1],
    nomen: str | None = None,
    place: str | None = None,
):
    df = load_data()
    filtered = graphs.prepare_data(df, nomen=nomen, place=place)
    img_bytes = graphs.plot_line_defects(filtered, parse_period(start, end))
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

@app.get("/plots/bar_defects")
async def bar_defects(
    start: str = DEFAULT_PERIOD[0],
    end: str = DEFAULT_PERIOD[1],
    nomen: str | None = None,
    place: str | None = None,
):
    df = load_data()
    filtered = graphs.prepare_data(df, nomen=nomen, place=place)
    img_bytes = graphs.bar_plot_defects(filtered, parse_period(start, end))
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

@app.get("/plots/pie_defects")
async def pie_defects(
    start: str = DEFAULT_PERIOD[0],
    end: str = DEFAULT_PERIOD[1],
    nomen: str | None = None,
    place: str | None = None,
):
    df = load_data()
    filtered = graphs.prepare_data(df, nomen=nomen, place=place)
    img_bytes = graphs.pie_plot_defects(filtered, parse_period(start, end))
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

# -----------------------------------------------------------------------------
# Endpoints: таблицы и KPI
# -----------------------------------------------------------------------------
@app.get("/tables/defects")
async def defects_table(
    start: str,
    end: str,
    nomen: str | None = None,
    place: str | None = None,
):
    df = load_data()
    filtered = graphs.prepare_data(df, nomen=nomen, place=place)

    start_dt, end_dt = parse_period(start, end)
    period_df = filtered[(filtered["Дата"] >= start_dt) & (filtered["Дата"] <= end_dt)]

    defects_df = graphs.defects_data(period_df)

    if defects_df.empty:
        return JSONResponse([{"defect": "Нет данных", "sum": None}])

    safe_df = defects_df.copy()
    # чистим только числовые значения, None оставляем
    safe_df["sum"] = safe_df["sum"].apply(
        lambda x: 0
        if isinstance(x, (int, float)) and (pd.isna(x) or x in [float("inf"), float("-inf")])
        else x
    )
    return JSONResponse(safe_df.to_dict(orient="records"))

@app.get("/kpi", response_model=KPIResponse)
async def get_kpi_data(
    start: str,
    end: str,
    nomen: str | None = None,
    place: str | None = None,
):
    df = load_data()
    filtered = graphs.prepare_data(df, nomen=nomen, place=place)
    total_cards, total_defects, defect_percent, total_money = graphs.get_kpis(
        filtered, parse_period(start, end)
    )
    return KPIResponse(
        total_cards=total_cards,
        total_defects=total_defects,
        defect_percent=defect_percent,
        total_money=total_money,
    )

# -----------------------------------------------------------------------------
# Manual trigger (по желанию: вручную обновить данные)
# -----------------------------------------------------------------------------
@app.post("/update")
async def manual_update():
    update_data_job()
    return {"status": "ok", "message": "Данные обновлены вручную"}

# -----------------------------------------------------------------------------
# Scheduler lifecycle (startup/shutdown)
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Старт приложения: включаем планировщик обновления данных раз в час."""
    minutes = int(os.getenv("REFRESH_INTERVAL_MINUTES", "60"))
    try:
        if not scheduler.running:
            scheduler.add_job(
                update_data_job,
                IntervalTrigger(minutes=minutes),
                id="update_data_job",
                replace_existing=True,
            )
            scheduler.start()
            logging.info(f"⏰ Планировщик запущен: каждые {minutes} мин.")
    except Exception as e:
        logging.exception(f"Ошибка при запуске планировщика: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Остановка приложения: корректно выключаем планировщик."""
    try:
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logging.info("🛑 Планировщик остановлен")
    except Exception as e:
        logging.exception(f"Ошибка при остановке планировщика: {e}")

