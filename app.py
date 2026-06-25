"""
AI Interaction Diagnostic
=========================
Однофайловое FastAPI-приложение для анализа качества взаимодействия
с генеративным ИИ. Интерфейс встроен в этот файл в переменной INDEX_HTML.

Запуск:
    uvicorn app:app --reload

Переменные окружения:
    GIGACHAT_API_KEY=<ключ GigaChat>   # необязательно для анализа диалогов
    DATABASE_URL=sqlite:///./ai_analyzer.db
    MAX_UPLOAD_MB=5
"""

from __future__ import annotations

# =============================================================================
# 1. Импорты
# =============================================================================

import html
import io
import ipaddress
import os
import re
import socket
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Generator, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text, create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

try:
    import pdfplumber
except ImportError:  # PDF-анализ остаётся необязательной возможностью
    pdfplumber = None

try:
    import fitz  # PyMuPDF: рендеринг страниц PDF для OCR
except ImportError:
    fitz = None

try:
    from PIL import Image
    import pytesseract
    from pytesseract import Output as TesseractOutput
except ImportError:
    Image = None
    pytesseract = None
    TesseractOutput = None

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except ImportError:
    colors = None
    TA_CENTER = None
    A4 = None
    canvas = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    cm = None
    pdfmetrics = None
    TTFont = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None


# =============================================================================
# 2. Конфигурация
# =============================================================================

load_dotenv()

APP_TITLE = "AI Interaction Diagnostic"
BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'ai_analyzer.db'}")
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY", "").strip()
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat:latest").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_TEXT_LENGTH = 120_000
MIN_TEAM_REPORTS = 10
PERSONAL_TEAM = "Личная диагностика"
MAX_URL_BYTES = int(os.getenv("MAX_URL_BYTES", "2000000"))
URL_TIMEOUT_SECONDS = int(os.getenv("URL_TIMEOUT_SECONDS", "12"))
MAX_URL_REDIRECTS = int(os.getenv("MAX_URL_REDIRECTS", "3"))
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "rus+eng").strip() or "rus+eng"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
PDF_FONT_PATH = os.getenv("PDF_FONT_PATH", "").strip()
PDF_BOLD_FONT_PATH = os.getenv("PDF_BOLD_FONT_PATH", "").strip()

if pytesseract is not None and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Веса из расчётного примера приложения В. В production-версии должны
# заменяться результатами фактического экспертного оценивания.
COMPONENT_WEIGHTS = {
    "P": 0.271,
    "V": 0.418,
    "A": 0.191,
    "R": 0.120,
}

ALLOWED_FILE_EXTENSIONS = {".txt", ".md", ".pdf"}


# =============================================================================
# 3. База данных и модели
# =============================================================================

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SessionDB(Base):
    """Диалоговая сессия встроенного чата."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class MessageDB(Base):
    """Отдельное сообщение в диалоговой сессии."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class ReportDB(Base):
    """Сохранённый результат анализа без хранения исходного текста файла."""

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=True, index=True)
    source_type = Column(String(32), nullable=False)  # chat | text | file
    team_name = Column(String(120), nullable=False, default=PERSONAL_TEAM)
    metrics = Column(JSON, nullable=False)
    recommendations = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


Base.metadata.create_all(bind=engine)


def ensure_legacy_schema() -> None:
    """Добавляет поле team_name в базу, созданную старой версией приложения."""

    inspector = inspect(engine)
    if "reports" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("reports")}
    if "team_name" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE reports "
                    "ADD COLUMN team_name VARCHAR(120) DEFAULT 'Личная диагностика'"
                )
            )


ensure_legacy_schema()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# 4. Схемы запросов и общие вспомогательные функции
# =============================================================================


class ChatRequest(BaseModel):
    session_id: int = Field(gt=0)
    message: str = Field(min_length=1, max_length=12_000)


class TextAnalysisRequest(BaseModel):
    dialogue: str = Field(min_length=3, max_length=MAX_TEXT_LENGTH)
    team_name: str = Field(default=PERSONAL_TEAM, max_length=120)


class UrlAnalysisRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    team_name: str = Field(default=PERSONAL_TEAM, max_length=120)


def clean_team_name(value: Optional[str]) -> str:
    name = (value or "").strip()
    return name[:120] if name else PERSONAL_TEAM


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def mean(values: Iterable[float]) -> float:
    prepared = list(values)
    return sum(prepared) / len(prepared) if prepared else 0.0


def round_score(value: float) -> float:
    return round(clamp(value), 3)


def score_percent(value: float) -> int:
    return int(round(clamp(value) * 100))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def contains_any(text_value: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text_value, flags=re.IGNORECASE) for pattern in patterns)


def get_or_404(db: Session, model: Any, record_id: int, message: str) -> Any:
    record = db.query(model).filter(model.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=message)
    return record


# =============================================================================
# 5. Парсинг диалогов и защита данных
# =============================================================================

ROLE_PATTERN = re.compile(
    r"^\s*(?P<role>user|human|you|пользователь|сотрудник|assistant|ai|bot|model|"
    r"ассистент|модель|бот)\s*[:—-]\s*(?P<content>.*)$",
    flags=re.IGNORECASE,
)

USER_ROLE_ALIASES = {"user", "human", "you", "пользователь", "сотрудник"}
ASSISTANT_ROLE_ALIASES = {"assistant", "ai", "bot", "model", "ассистент", "модель", "бот"}

SENSITIVE_PATTERNS = {
    "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "phone": r"(?:(?:\+7|8)\s?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})",
    "bank_card": r"\b(?:\d[ -]?){13,19}\b",
    "passport": r"\b\d{4}\s?\d{6}\b",
}


def normalize_role(raw_role: str) -> Optional[str]:
    role = normalize_text(raw_role).strip(":—- ")
    if role in USER_ROLE_ALIASES:
        return "user"
    if role in ASSISTANT_ROLE_ALIASES:
        return "assistant"
    return None


def validate_user_side(value: Optional[str]) -> str:
    side = normalize_text(value or "right")
    if side not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="Положение сообщений пользователя должно быть left или right")
    return side


def merge_adjacent_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for item in messages:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] = f'{merged[-1]["content"]}\n{content}'.strip()
        else:
            merged.append({"role": role, "content": content})
    return merged


def parse_dialogue(text_value: str) -> list[dict[str, str]]:
    """Разбирает текст диалога с явными ролями User/AI и русскими аналогами."""

    messages: list[dict[str, str]] = []
    current_role: Optional[str] = None
    current_parts: list[str] = []

    def flush_current() -> None:
        nonlocal current_role, current_parts
        content = "\n".join(current_parts).strip()
        if current_role and content:
            messages.append({"role": current_role, "content": content})
        current_role = None
        current_parts = []

    for raw_line in text_value.splitlines():
        line = raw_line.strip()
        if not line:
            if current_parts:
                current_parts.append("")
            continue

        match = ROLE_PATTERN.match(line)
        if match:
            flush_current()
            current_role = normalize_role(match.group("role"))
            current_parts = [match.group("content").strip()]
        elif current_role:
            current_parts.append(line)

    flush_current()

    if not messages and text_value.strip():
        messages = [{"role": "user", "content": text_value.strip()}]

    return merge_adjacent_messages(messages)


def user_prompts_from_messages(messages: list[dict[str, str]]) -> list[str]:
    return [message["content"].strip() for message in messages if message["role"] == "user" and message["content"].strip()]


def detect_sensitive_data(text_value: str) -> list[str]:
    return [label for label, pattern in SENSITIVE_PATTERNS.items() if re.search(pattern, text_value, re.IGNORECASE)]


def positioned_words_to_messages(
    words: list[dict[str, Any]],
    page_width: float,
    user_side: str,
) -> list[dict[str, str]]:
    """Группирует слова PDF/OCR в строки и определяет роль по левой/правой стороне."""

    if not words or page_width <= 0:
        return []

    prepared = []
    for word in words:
        token = str(word.get("text", "")).strip()
        if not token:
            continue
        try:
            prepared.append({
                "text": token,
                "x0": float(word.get("x0", word.get("left", 0))),
                "x1": float(word.get("x1", float(word.get("left", 0)) + float(word.get("width", 0)))),
                "top": float(word.get("top", word.get("y", 0))),
            })
        except (TypeError, ValueError):
            continue

    prepared.sort(key=lambda item: (item["top"], item["x0"]))
    lines: list[dict[str, Any]] = []

    # Для OCR используется более крупный допуск по высоте строки.
    for word in prepared:
        if not lines or abs(word["top"] - lines[-1]["top"]) > 8:
            lines.append({"top": word["top"], "words": [word]})
        else:
            lines[-1]["words"].append(word)

    messages: list[dict[str, str]] = []
    for line in lines:
        line_words = sorted(line["words"], key=lambda item: item["x0"])
        content = " ".join(item["text"] for item in line_words).strip()
        if len(content) < 2:
            continue

        explicit = ROLE_PATTERN.match(content)
        if explicit:
            role = normalize_role(explicit.group("role"))
            content = explicit.group("content").strip()
        else:
            x0 = min(item["x0"] for item in line_words)
            x1 = max(item["x1"] for item in line_words)
            center = (x0 + x1) / 2
            is_right = center >= page_width / 2
            role = "user" if is_right == (user_side == "right") else "assistant"

        if role and content:
            messages.append({"role": role, "content": content})

    return merge_adjacent_messages(messages)


def extract_messages_from_text_pdf(raw_bytes: bytes, user_side: str) -> list[dict[str, str]]:
    """Извлекает текстовый слой PDF с координатами.

    Сначала используется PyMuPDF: он устойчивее к PDF с неполным FontBBox и не
    выводит предупреждение pdfminer при чтении таких файлов. pdfplumber остаётся
    резервным вариантом для окружений без PyMuPDF.
    """

    messages: list[dict[str, str]] = []

    if fitz is not None:
        try:
            document = fitz.open(stream=raw_bytes, filetype="pdf")
            for page in document:
                raw_words = page.get_text("words", sort=True)
                words = [
                    {
                        "text": str(word[4]).strip(),
                        "x0": float(word[0]),
                        "x1": float(word[2]),
                        "top": float(word[1]),
                    }
                    for word in raw_words
                    if len(word) >= 5 and str(word[4]).strip()
                ]
                if len(words) >= 3:
                    messages.extend(
                        positioned_words_to_messages(words, float(page.rect.width), user_side)
                    )
            document.close()
            if messages:
                return merge_adjacent_messages(messages)
        except Exception:
            # Ниже используется резервное извлечение через pdfplumber.
            messages = []

    if pdfplumber is None:
        return []

    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
                if len(words) >= 3:
                    messages.extend(
                        positioned_words_to_messages(words, float(page.width), user_side)
                    )
    except Exception:
        return []

    return merge_adjacent_messages(messages)


def extract_messages_from_scanned_pdf(raw_bytes: bytes, user_side: str) -> list[dict[str, str]]:
    """Распознаёт PDF-скан: OCR + классификация роли по положению реплики."""

    if fitz is None or pytesseract is None or Image is None or TesseractOutput is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Для PDF-сканов установите PyMuPDF, Pillow и pytesseract, "
                "а также Tesseract OCR с языками rus и eng."
            ),
        )

    try:
        document = fitz.open(stream=raw_bytes, filetype="pdf")
        messages: list[dict[str, str]] = []
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            data = pytesseract.image_to_data(
                image,
                lang=OCR_LANGUAGE,
                output_type=TesseractOutput.DICT,
                config="--psm 6",
            )
            words: list[dict[str, Any]] = []
            for index, token in enumerate(data.get("text", [])):
                token = (token or "").strip()
                if not token:
                    continue
                try:
                    confidence = float(data["conf"][index])
                except (TypeError, ValueError, KeyError):
                    confidence = -1
                if confidence < 20:
                    continue
                left = float(data["left"][index])
                top = float(data["top"][index])
                width = float(data["width"][index])
                words.append({"text": token, "x0": left, "x1": left + width, "top": top})
            messages.extend(positioned_words_to_messages(words, float(image.width), user_side))
        document.close()
        return merge_adjacent_messages(messages)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Не удалось распознать PDF-скан: {error}") from error


def extract_messages_from_pdf(raw_bytes: bytes, user_side: str) -> list[dict[str, str]]:
    messages = extract_messages_from_text_pdf(raw_bytes, user_side)
    if len(messages) >= 2 and user_prompts_from_messages(messages):
        return messages

    scanned_messages = extract_messages_from_scanned_pdf(raw_bytes, user_side)
    if scanned_messages:
        return scanned_messages

    return messages


def extract_messages_from_upload(upload: UploadFile, user_side: str = "right") -> list[dict[str, str]]:
    filename = upload.filename or ""
    extension = Path(filename).suffix.lower()
    user_side = validate_user_side(user_side)

    if extension not in ALLOWED_FILE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_FILE_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Поддерживаются только файлы: {allowed}")

    raw_bytes = upload.file.read()
    if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Размер файла не должен превышать {MAX_UPLOAD_MB} МБ")

    if extension in {".txt", ".md"}:
        return parse_dialogue(raw_bytes.decode("utf-8", errors="ignore"))

    messages = extract_messages_from_pdf(raw_bytes, user_side)
    if not messages:
        raise HTTPException(status_code=400, detail="В PDF не удалось найти или распознать текст")

    # Если в PDF не выявлено сообщений пользователя, не выдаём ложную уверенность.
    if not user_prompts_from_messages(messages):
        raise HTTPException(
            status_code=400,
            detail="Не удалось определить сообщения пользователя. Измените сторону своих сообщений или загрузите экспорт с метками ролей.",
        )

    return messages


class PublicChatHTMLParser(HTMLParser):
    """Извлекает сообщения из открытой HTML-страницы без выполнения JavaScript."""

    BLOCK_TAGS = {"p", "div", "article", "section", "li", "br", "h1", "h2", "h3", "h4"}
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.messages: list[dict[str, str]] = []
        self.generic_parts: list[str] = []
        self.tag_stack: list[str] = []
        self.skip_depth = 0
        self.current_role: Optional[str] = None
        self.current_role_depth: Optional[int] = None
        self.current_parts: list[str] = []

    def flush_message(self) -> None:
        content = " ".join(part.strip() for part in self.current_parts if part.strip()).strip()
        if self.current_role and content:
            self.messages.append({"role": self.current_role, "content": content})
        self.current_role = None
        self.current_role_depth = None
        self.current_parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self.tag_stack.append(tag)
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        attrs_dict = {key.lower(): (value or "") for key, value in attrs}
        raw_role = attrs_dict.get("data-message-author-role") or attrs_dict.get("data-role")
        role = normalize_role(raw_role) if raw_role else None
        if role:
            self.flush_message()
            self.current_role = role
            self.current_role_depth = len(self.tag_stack)
        elif tag in self.BLOCK_TAGS:
            self.generic_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth and tag in self.SKIP_TAGS:
            self.skip_depth -= 1
        if self.current_role_depth == len(self.tag_stack):
            self.flush_message()
        if self.tag_stack:
            self.tag_stack.pop()
        if not self.skip_depth and tag in self.BLOCK_TAGS:
            self.generic_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned:
            return
        self.generic_parts.append(cleaned)
        if self.current_role:
            self.current_parts.append(cleaned)

    def finish(self) -> tuple[list[dict[str, str]], str]:
        self.flush_message()
        generic_text = "\n".join(self.generic_parts)
        return merge_adjacent_messages(self.messages), generic_text


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def validate_public_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Укажите корректную публичную ссылку с http:// или https://")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Ссылки с учётными данными не поддерживаются")
    if parsed.port not in {None, 80, 443}:
        raise HTTPException(status_code=400, detail="Нестандартные порты в публичных ссылках не поддерживаются")

    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise HTTPException(status_code=400, detail="Локальные адреса недоступны")

    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise HTTPException(status_code=400, detail="Не удалось определить адрес сайта") from error

    for item in addresses:
        ip_value = ipaddress.ip_address(item[4][0])
        if (
            ip_value.is_private
            or ip_value.is_loopback
            or ip_value.is_link_local
            or ip_value.is_multicast
            or ip_value.is_reserved
            or ip_value.is_unspecified
        ):
            raise HTTPException(status_code=400, detail="Ссылка ведёт на недоступный внутренний адрес")

    return value.strip()


def fetch_public_page(url: str) -> str:
    """Загружает только страницы, доступные обычному HTTP-клиенту без входа.

    Важно: публичность ссылки в браузере не гарантирует доступ для сервера.
    Некоторые сервисы, включая DeepSeek, используют Cloudflare/Turnstile и
    возвращают 403 автоматическим клиентам. Такой доступ не обходим.
    """

    current_url = validate_public_url(url)
    opener = build_opener(NoRedirectHandler())
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
    }

    for _ in range(MAX_URL_REDIRECTS + 1):
        try:
            response = opener.open(Request(current_url, headers=headers), timeout=URL_TIMEOUT_SECONDS)
        except HTTPError as error:
            if error.code in {301, 302, 303, 307, 308}:
                location = error.headers.get("Location")
                if not location:
                    raise HTTPException(status_code=400, detail="Сайт вернул редирект без адреса") from error
                current_url = validate_public_url(urljoin(current_url, location))
                continue

            host = (urlparse(current_url).hostname or "").lower()
            if error.code in {401, 403} and host.endswith("deepseek.com"):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "DeepSeek открывает эту ссылку в браузере, но блокирует "
                        "автоматическое чтение сервером. Это не означает, что ссылка "
                        "закрыта. Сохраните чат через «Печать → Сохранить как PDF» и "
                        "загрузите PDF: сервис определит реплики по расположению слева/справа."
                    ),
                ) from error

            if error.code in {401, 403}:
                raise HTTPException(
                    status_code=422,
                    detail="Сайт запретил автоматическое чтение страницы. Откройте чат в браузере и загрузите экспорт TXT или PDF.",
                ) from error
            raise HTTPException(status_code=400, detail=f"Не удалось получить страницу: HTTP {error.code}") from error
        except URLError as error:
            raise HTTPException(status_code=400, detail=f"Не удалось открыть ссылку: {error.reason}") from error

        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "text/plain", "application/json"}:
            raise HTTPException(status_code=400, detail="Ссылка должна вести на публичную текстовую или HTML-страницу")

        raw = response.read(MAX_URL_BYTES + 1)
        if len(raw) > MAX_URL_BYTES:
            raise HTTPException(status_code=400, detail="Страница слишком большая для анализа")
        charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="ignore")

    raise HTTPException(status_code=400, detail="Слишком много перенаправлений")


def extract_messages_from_public_url(url: str) -> list[dict[str, str]]:
    page = fetch_public_page(url)
    parser = PublicChatHTMLParser()
    try:
        parser.feed(page)
        parser.close()
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать структуру страницы: {error}") from error

    messages, generic_text = parser.finish()
    if not messages:
        messages = parse_dialogue(generic_text)

    if not messages or not user_prompts_from_messages(messages):
        raise HTTPException(
            status_code=400,
            detail=(
                "На странице не удалось надёжно выделить реплики. "
                "Ссылка должна быть публичной и содержать текст диалога без авторизации."
            ),
        )
    return messages


# =============================================================================
# 6. Диагностическая методика P, V, A, R
# =============================================================================

GOAL_PATTERNS = [
    r"\b(подготов|состав|напиш|создай|сформир|разработ|проанализ|сравни|структурир|"
    r"предлож|проверь|объясни|перевед|резюмир|сделай|построй|выяви)\w*\b",
    r"\b(write|create|prepare|analyse|analyze|compare|summari[sz]e|draft|build|review)\b",
]

CONTEXT_PATTERNS = [
    r"\b(проект|команда|клиент|продукт|компани|пользовател|сегмент|приложен|"
    r"период|квартал|бюджет|метрик|данн|отч[её]т|сервис|рынок|релиз|задач)\w*\b",
    r"\b(we|our|customer|product|team|project|metric|data|quarter|budget)\b",
    r"\d+[%₽$]|\b\d{1,4}\b",
]

CONSTRAINT_PATTERNS = [
    r"\b(не |без |только |исключи|огранич|срок|дедлайн|до \d|не используй|"
    r"не добавляй|не делай|обязательн|критери|требован)\w*",
    r"\b(without|only|do not|must|deadline|limit|constraint)\b",
]

FORMAT_PATTERNS = [
    r"\b(формат|таблиц|список|план|структур|json|markdown|письмо|отч[её]т|"
    r"презентац|реестр|чек-?лист|шаблон|sql|код)\w*\b",
    r"\b(format|table|list|plan|outline|json|markdown|email|report|template|code)\b",
]

REFINEMENT_PATTERNS = [
    r"\b(добав|уточн|перепиш|измени|сократ|расшир|раздел|исправ|адаптир|"
    r"сравни|проверь|покажи|обнови)\w*\b",
    r"\b(add|clarify|rewrite|change|shorten|expand|split|fix|update|compare|check)\b",
]

VERIFICATION_PATTERNS = [
    r"\b(проверь|верифицир|сверь|источник|ссылк|ошибк|противореч|"
    r"допущен|ограничен|риск|альтернатив|обосн|подтверд)\w*\b",
    r"\b(check|verify|source|citation|error|contradiction|assumption|limitation|risk|alternative|evidence)\b",
]

ANALYTICAL_PATTERNS = [
    r"\b(моя гипотез|по моим данным|вот данные|исходн(ые|ая) данные|сегмент|"
    r"критери|сравн|декомпоз|этап|сценар|вариант|выборк|метрик|вывод)\w*\b",
    r"\b(my hypothesis|our data|input data|segment|criterion|compare|decompose|stage|scenario|sample|metric)\b",
    r"\d+[%₽$]|\b\d{1,4}\b",
]

DELEGATION_PATTERNS = [
    r"\b(сам реши|прими решение за меня|выбери лучший|сделай полностью|"
    r"реши за меня|готовое решение|всё за меня|самостоятельно выбери)\b",
    r"\b(decide for me|choose the best|do everything|make the final decision)\b",
]

HIGH_RISK_PATTERNS = [
    r"\b(персональн|паспорт|зарплат|увольнен|найм|кандидат|финанс|бюджет|"
    r"договор|юридическ|правов|клиентск|плат[её]ж|налог|медицин|безопасност)\w*\b",
]

MEDIUM_RISK_PATTERNS = [
    r"\b(аналитик|отч[её]т|стратеги|проект|план|риск|презентац|исследован|"
    r"маркетинг|коммуникац|метрик|гипотез)\w*\b",
]

TASK_LABELS = {
    "analysis": [r"\b(анализ|проанализ|гипотез|метрик|данн|отч[её]т|исследован)\w*\b"],
    "project": [r"\b(проект|релиз|команда|план|риск|задач|спринт)\w*\b"],
    "writing": [r"\b(письмо|текст|стать|пост|ваканси|описан|редактир)\w*\b"],
    "coding": [r"\b(код|python|sql|api|скрипт|программ)\w*\b"],
    "ideation": [r"\b(иде[яи]|вариант|брейншторм|придум)\w*\b"],
}


def detect_task_risk(all_prompts: list[str]) -> tuple[str, str]:
    merged = normalize_text(" ".join(all_prompts))
    if contains_any(merged, HIGH_RISK_PATTERNS):
        return "high", "Высокий"
    if contains_any(merged, MEDIUM_RISK_PATTERNS):
        return "medium", "Средний"
    return "low", "Низкий"


def detect_task_types(all_prompts: list[str]) -> list[str]:
    merged = normalize_text(" ".join(all_prompts))
    found = [label for label, patterns in TASK_LABELS.items() if contains_any(merged, patterns)]
    return found or ["general"]


def evidence_label(condition: bool, present: str, missing: str) -> dict[str, Any]:
    return {"present": condition, "label": present if condition else missing}


def analyze_interaction(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Рассчитывает показатели P, V, A, R по наблюдаемым текстовым признакам."""

    prompts = user_prompts_from_messages(messages)
    if not prompts:
        raise HTTPException(status_code=400, detail="Не удалось найти сообщения пользователя для анализа")

    first_prompt = normalize_text(prompts[0])
    merged = normalize_text(" ".join(prompts))
    followups = prompts[1:]
    followups_text = normalize_text(" ".join(followups))
    risk_code, risk_label = detect_task_risk(prompts)
    task_types = detect_task_types(prompts)

    # P — качество постановки задачи
    has_goal = contains_any(first_prompt, GOAL_PATTERNS)
    has_context = contains_any(first_prompt, CONTEXT_PATTERNS) or len(first_prompt.split()) >= 24
    has_constraints = contains_any(first_prompt, CONSTRAINT_PATTERNS)
    has_format = contains_any(first_prompt, FORMAT_PATTERNS)
    has_refinement = bool(followups) and contains_any(followups_text, REFINEMENT_PATTERNS)

    p_features = [float(has_goal), float(has_context), float(has_constraints), float(has_format)]
    if followups:
        p_features.append(float(has_refinement))
    p_score = round_score(mean(p_features))

    # V — наблюдаемая проверка результата. Для низкорисковой задачи отсутствие
    # явной проверки интерпретируется нейтрально, а не как отрицательный признак.
    has_fact_check = contains_any(merged, VERIFICATION_PATTERNS)
    has_compare = contains_any(merged, [r"\b(сравни|альтернатив|вариант|сопостав)\w*\b", r"\b(compare|alternative)\b"])
    has_assumptions = contains_any(merged, [r"\b(допущен|ограничен|обосн|данных не хватает)\w*\b", r"\b(assumption|limitation)\b"])
    has_source_request = contains_any(merged, [r"\b(источник|ссылк|подтверд)\w*\b", r"\b(source|citation|evidence)\b"])
    verification_signals = [float(has_fact_check), float(has_compare), float(has_assumptions), float(has_source_request)]
    raw_v = mean(verification_signals)
    if risk_code == "low" and raw_v == 0:
        v_score = 0.5
        verification_note = "Для низкорисковой задачи отсутствие проверки учтено нейтрально."
    else:
        v_score = round_score(raw_v)
        verification_note = "Оценка основана на наблюдаемых запросах на проверку и критическую оценку."

    # A — аналитическое участие сотрудника
    has_own_data = contains_any(merged, ANALYTICAL_PATTERNS)
    has_decomposition = contains_any(merged, [r"\b(этап|шаг|сначала|затем|раздел|декомпоз)\w*\b", r"\b(step|stage|first|then|split|decompose)\b"])
    has_criteria = contains_any(merged, [r"\b(критери|приоритет|оцен(и|ка)|выбор|услови)\w*\b", r"\b(criteria|priority|evaluate|condition)\b"])
    asks_to_critique_own_material = contains_any(merged, [r"\b(мой|наше|черновик|материал).{0,60}\b(проверь|улучши|раскритикуй|доработай)\w*", r"\b(my draft|our draft).{0,60}\b(review|improve|critic)\b"])
    a_score = round_score(mean([float(has_own_data), float(has_decomposition), float(has_criteria), float(asks_to_critique_own_material)]))

    # R — риск чрезмерного делегирования
    delegation = contains_any(merged, DELEGATION_PATTERNS)
    complex_task = risk_code in {"medium", "high"} or any(task in {"analysis", "project"} for task in task_types)
    no_context_for_complex_task = complex_task and p_score < 0.5
    high_risk_without_check = risk_code == "high" and v_score < 0.5
    no_own_contribution = a_score < 0.25
    r_score = round_score(
        0.45 * float(delegation)
        + 0.25 * float(no_context_for_complex_task)
        + 0.20 * float(high_risk_without_check)
        + 0.10 * float(no_own_contribution)
    )

    overall = round_score(
        COMPONENT_WEIGHTS["P"] * p_score
        + COMPONENT_WEIGHTS["V"] * v_score
        + COMPONENT_WEIGHTS["A"] * a_score
        + COMPONENT_WEIGHTS["R"] * (1 - r_score)
    )

    if overall < 0.40:
        level = "Требуется развитие практик"
        level_code = "needs_development"
    elif overall < 0.70:
        level = "Базовый уровень взаимодействия"
        level_code = "basic"
    else:
        level = "Устойчивое и контролируемое взаимодействие"
        level_code = "mature"

    evidence = {
        "P": [
            evidence_label(has_goal, "Указана цель или ожидаемое действие.", "Не обнаружена явная цель запроса."),
            evidence_label(has_context, "Передан предметный контекст или исходные данные.", "Недостаточно предметного контекста."),
            evidence_label(has_constraints, "Указаны ограничения или критерии.", "Не обнаружены ограничения или критерии."),
            evidence_label(has_format, "Указан ожидаемый формат результата.", "Не указан ожидаемый формат результата."),
            evidence_label(has_refinement, "Есть уточнение после промежуточного ответа.", "Не обнаружены содержательные уточнения."),
        ],
        "V": [
            evidence_label(has_fact_check, "Есть запрос на проверку фактов, ошибок или рисков.", "Нет явного запроса на проверку фактов или ошибок."),
            evidence_label(has_compare, "Есть сравнение вариантов или альтернатив.", "Не обнаружено сравнение вариантов."),
            evidence_label(has_assumptions, "Есть запрос на раскрытие допущений или ограничений.", "Не обнаружен запрос на допущения и ограничения."),
            evidence_label(has_source_request, "Есть запрос на источники или подтверждение вывода.", "Не обнаружен запрос на источники или подтверждение."),
        ],
        "A": [
            evidence_label(has_own_data, "Пользователь вносит собственные данные, гипотезы или критерии.", "Не обнаружен собственный содержательный вклад."),
            evidence_label(has_decomposition, "Задача декомпозирована на этапы или блоки.", "Не обнаружена декомпозиция задачи."),
            evidence_label(has_criteria, "Пользователь задаёт критерии выбора или оценки.", "Не обнаружены собственные критерии выбора."),
            evidence_label(asks_to_critique_own_material, "Есть запрос на критику или доработку собственного материала.", "Не обнаружен запрос на критику собственного материала."),
        ],
        "R": [
            evidence_label(delegation, "Обнаружена формулировка, передающая модели итоговое решение.", "Нет явной передачи итогового решения модели."),
            evidence_label(no_context_for_complex_task, "Комплексная задача поставлена без достаточного контекста.", "Контекст задачи достаточен для её сложности."),
            evidence_label(high_risk_without_check, "Высокорисковая задача не сопровождается наблюдаемой проверкой.", "Для задачи не выявлен риск отсутствия проверки."),
            evidence_label(no_own_contribution, "Недостаточно наблюдаемого содержательного вклада пользователя.", "Содержательный вклад пользователя обнаружен."),
        ],
    }

    recommendations = build_recommendations(
        p_score=p_score,
        v_score=v_score,
        a_score=a_score,
        r_score=r_score,
        risk_code=risk_code,
    )

    sensitive_data = detect_sensitive_data("\n".join(prompt for prompt in prompts))

    return {
        "schema_version": "2.0",
        "overall_score": score_percent(overall),
        "overall_value": overall,
        "level": level,
        "level_code": level_code,
        "prompt_count": len(prompts),
        "task_types": task_types,
        "task_risk": {"code": risk_code, "label": risk_label},
        "components": {
            "P": {"score": score_percent(p_score), "value": p_score, "title": "Постановка задачи"},
            "V": {"score": score_percent(v_score), "value": v_score, "title": "Проверка результата"},
            "A": {"score": score_percent(a_score), "value": a_score, "title": "Аналитическое участие"},
            "R": {"score": score_percent(r_score), "value": r_score, "title": "Риск чрезмерного делегирования"},
        },
        "evidence": evidence,
        "verification_note": verification_note,
        "weights": COMPONENT_WEIGHTS,
        "sensitive_data_warning": sensitive_data,
        "recommendations": recommendations,
    }


def build_recommendations(p_score: float, v_score: float, a_score: float, r_score: float, risk_code: str) -> list[str]:
    recommendations: list[str] = []

    if p_score < 0.60:
        recommendations.append("Добавьте цель, исходные данные, ограничения и ожидаемый формат результата.")
    if v_score < 0.60 and risk_code != "low":
        recommendations.append("Для значимой задачи запросите проверку фактов, допущений, ограничений или альтернатив.")
    if a_score < 0.50:
        recommendations.append("Добавьте собственные гипотезы, критерии выбора или промежуточные выводы перед запросом к модели.")
    if r_score >= 0.40:
        recommendations.append("Разделите задачу на этапы и оставьте за собой выбор итогового решения.")
    if risk_code == "high":
        recommendations.append("Для высокорисковой задачи сверяйте ответ с внутренними правилами, источниками и профильным экспертом.")
    if not recommendations:
        recommendations.append("Диалог содержит признаки зрелого взаимодействия: сохраните практику уточнений и проверки результата.")

    return recommendations[:4]


# =============================================================================
# 7. LLM-клиент (GigaChat подключается только при наличии зависимости и ключа)
# =============================================================================


def get_llm_response(history: list[dict[str, str]]) -> str:
    """Возвращает ответ GigaChat или понятное сообщение о конфигурации."""

    if not GIGACHAT_API_KEY:
        return (
            "Чат не настроен: добавьте GIGACHAT_API_KEY в файл .env. "
            "Анализ загруженных диалогов доступен без ключа."
        )

    try:
        from gigachat import GigaChat  # импортируется только при реальном использовании

        with GigaChat(
            credentials=GIGACHAT_API_KEY,
            verify_ssl_certs=False,
            model=GIGACHAT_MODEL,
            timeout=60,
        ) as client:
            response = client.chat({"messages": history})
            return response.choices[0].message.content
    except ImportError:
        return "Не установлена библиотека gigachat. Выполните: pip install gigachat"
    except Exception as error:
        return f"Не удалось получить ответ модели: {error}"


# =============================================================================
# 8. FastAPI-приложение и API
# =============================================================================

app = FastAPI(
    title=APP_TITLE,
    version="2.1.0",
    description="Диагностика качества взаимодействия сотрудника с генеративным ИИ.",
)


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_TITLE,
        "gigachat_configured": bool(GIGACHAT_API_KEY),
        "version": "2.1.0",
    }


# ---------- Сессии и чат ----------

@app.get("/api/sessions")
def list_sessions(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    sessions = db.query(SessionDB).order_by(SessionDB.created_at.desc()).all()
    return [{"id": item.id, "created_at": item.created_at.isoformat()} for item in sessions]


@app.post("/api/sessions")
def create_session(db: Session = Depends(get_db)) -> dict[str, Any]:
    session = SessionDB()
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"id": session.id, "created_at": session.created_at.isoformat()}


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: int, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    get_or_404(db, SessionDB, session_id, "Сессия не найдена")
    messages = (
        db.query(MessageDB)
        .filter(MessageDB.session_id == session_id)
        .order_by(MessageDB.timestamp.asc())
        .all()
    )
    return [
        {"id": item.id, "role": item.role, "content": item.content, "timestamp": item.timestamp.isoformat()}
        for item in messages
    ]


@app.post("/api/chat")
def send_chat_message(request: ChatRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    get_or_404(db, SessionDB, request.session_id, "Сессия не найдена")

    user_message = MessageDB(session_id=request.session_id, role="user", content=request.message.strip())
    db.add(user_message)
    db.commit()

    history_records = (
        db.query(MessageDB)
        .filter(MessageDB.session_id == request.session_id)
        .order_by(MessageDB.timestamp.desc())
        .limit(20)
        .all()
    )
    history = [
        {"role": item.role, "content": item.content}
        for item in reversed(history_records)
    ]

    answer = get_llm_response(history)
    assistant_message = MessageDB(session_id=request.session_id, role="assistant", content=answer)
    db.add(assistant_message)
    db.commit()

    return {"answer": answer}


# ---------- Анализ ----------

def save_report(
    db: Session,
    *,
    source_type: str,
    metrics: dict[str, Any],
    team_name: str,
    session_id: Optional[int] = None,
) -> ReportDB:
    report = ReportDB(
        session_id=session_id,
        source_type=source_type,
        team_name=clean_team_name(team_name),
        metrics=metrics,
        recommendations=metrics["recommendations"],
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@app.post("/api/analyze/text")
def analyze_text(payload: TextAnalysisRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    messages = parse_dialogue(payload.dialogue)
    metrics = analyze_interaction(messages)
    report = save_report(db, source_type="text", metrics=metrics, team_name=payload.team_name)
    return serialize_report(report)


@app.post("/api/analyze/file")
def analyze_file(
    file: UploadFile = File(...),
    team_name: str = Form(PERSONAL_TEAM),
    user_side: str = Form("right"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    messages = extract_messages_from_upload(file, user_side=user_side)
    extracted_length = sum(len(item["content"]) for item in messages)
    if extracted_length > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="Текст после извлечения превышает допустимый объём")

    metrics = analyze_interaction(messages)
    report = save_report(db, source_type="file", metrics=metrics, team_name=team_name)
    return serialize_report(report)


@app.post("/api/analyze/url")
def analyze_url(payload: UrlAnalysisRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    messages = extract_messages_from_public_url(payload.url)
    extracted_length = sum(len(item["content"]) for item in messages)
    if extracted_length > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="Текст по ссылке превышает допустимый объём")

    metrics = analyze_interaction(messages)
    report = save_report(db, source_type="url", metrics=metrics, team_name=payload.team_name)
    return serialize_report(report)


@app.post("/api/analyze/session/{session_id}")
def analyze_session(
    session_id: int,
    team_name: str = Form(PERSONAL_TEAM),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    get_or_404(db, SessionDB, session_id, "Сессия не найдена")
    records = (
        db.query(MessageDB)
        .filter(MessageDB.session_id == session_id)
        .order_by(MessageDB.timestamp.asc())
        .all()
    )
    messages = [{"role": item.role, "content": item.content} for item in records]
    metrics = analyze_interaction(messages)
    report = save_report(
        db,
        source_type="chat",
        metrics=metrics,
        team_name=team_name,
        session_id=session_id,
    )
    return serialize_report(report)


# ---------- История и агрегированная аналитика ----------

def serialize_report(report: ReportDB) -> dict[str, Any]:
    return {
        "id": report.id,
        "session_id": report.session_id,
        "source_type": report.source_type,
        "team_name": report.team_name or PERSONAL_TEAM,
        "created_at": report.created_at.isoformat(),
        "metrics": report.metrics,
        "recommendations": report.recommendations,
    }


@app.get("/api/reports")
def list_reports(limit: int = 30, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    safe_limit = min(max(limit, 1), 100)
    reports = db.query(ReportDB).order_by(ReportDB.created_at.desc()).limit(safe_limit).all()
    return [serialize_report(report) for report in reports]


@app.get("/api/reports/{report_id}")
def get_report(report_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    report = get_or_404(db, ReportDB, report_id, "Отчёт не найден")
    return serialize_report(report)


def register_report_fonts() -> tuple[str, str]:
    """Подключает системные шрифты с кириллицей для визуального PDF-отчёта."""

    if pdfmetrics is None or TTFont is None or canvas is None:
        raise HTTPException(status_code=500, detail="Для выгрузки PDF установите библиотеку reportlab.")

    regular_name = "AIReportRegular"
    bold_name = "AIReportBold"
    registered = set(pdfmetrics.getRegisteredFontNames())
    if regular_name in registered and bold_name in registered:
        return regular_name, bold_name

    candidates = [
        (PDF_FONT_PATH, PDF_BOLD_FONT_PATH),
        (r"C:\\Windows\\Fonts\\arial.ttf", r"C:\\Windows\\Fonts\\arialbd.ttf"),
        (r"C:\\Windows\\Fonts\\calibri.ttf", r"C:\\Windows\\Fonts\\calibrib.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ]

    for regular_path, bold_path in candidates:
        if regular_path and Path(regular_path).exists():
            pdfmetrics.registerFont(TTFont(regular_name, regular_path))
            pdfmetrics.registerFont(TTFont(bold_name, bold_path if bold_path and Path(bold_path).exists() else regular_path))
            return regular_name, bold_name

    raise HTTPException(
        status_code=500,
        detail="Не найден системный шрифт с кириллицей. Укажите PDF_FONT_PATH в файле .env.",
    )


def _pdf_wrap(text_value: Any, font_name: str, font_size: float, max_width: float) -> list[str]:
    """Переносит строку по ширине с сохранением слов и кириллицы."""

    text_value = re.sub(r"\s+", " ", str(text_value or "").strip())
    if not text_value:
        return [""]

    words = text_value.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _pdf_draw_wrapped(
    pdf: Any,
    text_value: Any,
    x: float,
    y: float,
    max_width: float,
    font_name: str,
    font_size: float,
    color: Any,
    leading: Optional[float] = None,
    max_lines: Optional[int] = None,
) -> float:
    leading = leading or font_size * 1.25
    lines = _pdf_wrap(text_value, font_name, font_size, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;: ") + "…"
    pdf.setFont(font_name, font_size)
    pdf.setFillColor(color)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _pdf_draw_panel(pdf: Any, x: float, y: float, width: float, height: float, fill: Any, stroke: Any, radius: float = 12) -> None:
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(0.7)
    pdf.roundRect(x, y, width, height, radius, stroke=1, fill=1)


def _pdf_draw_score_ring(pdf: Any, center_x: float, center_y: float, radius: float, score: int, regular: str, bold: str, palette: dict[str, Any]) -> None:
    pdf.setLineWidth(9)
    pdf.setStrokeColor(palette["ring_base"])
    pdf.circle(center_x, center_y, radius, stroke=1, fill=0)
    pdf.setStrokeColor(palette["accent"])
    pdf.arc(
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
        90,
        -360 * max(0, min(100, score)) / 100,
    )
    pdf.setFillColor(palette["panel"])
    pdf.circle(center_x, center_y, radius - 7, stroke=0, fill=1)
    pdf.setFillColor(palette["text"])
    pdf.setFont(bold, 22)
    number = str(score)
    pdf.drawCentredString(center_x, center_y + 3, number)
    pdf.setFont(regular, 7.5)
    pdf.setFillColor(palette["muted"])
    pdf.drawCentredString(center_x, center_y - 13, "из 100")


def _pdf_draw_metric_card(
    pdf: Any,
    key: str,
    title: str,
    score: int,
    x: float,
    y: float,
    width: float,
    height: float,
    regular: str,
    bold: str,
    palette: dict[str, Any],
) -> None:
    _pdf_draw_panel(pdf, x, y, width, height, palette["metric"], palette["line"], 11)
    _pdf_draw_wrapped(pdf, title, x + 12, y + height - 18, width - 24, regular, 8.2, palette["muted"], max_lines=2)
    pdf.setFont(bold, 19)
    pdf.setFillColor(palette["text"])
    pdf.drawString(x + 12, y + 39, str(score))
    bar_x, bar_y, bar_w = x + 12, y + 24, width - 24
    pdf.setFillColor(palette["bar_bg"])
    pdf.roundRect(bar_x, bar_y, bar_w, 5, 2.5, stroke=0, fill=1)
    pdf.setFillColor(palette["accent_2"])
    pdf.roundRect(bar_x, bar_y, bar_w * max(0, min(100, score)) / 100, 5, 2.5, stroke=0, fill=1)
    pdf.setFont(regular, 7.2)
    pdf.setFillColor(palette["muted"])
    pdf.drawString(x + 12, y + 10, "Чем ниже, тем лучше" if key == "R" else "Чем выше, тем лучше")


def _pdf_draw_detail_card(
    pdf: Any,
    title: str,
    items: list[dict[str, Any]],
    x: float,
    y: float,
    width: float,
    height: float,
    regular: str,
    bold: str,
    palette: dict[str, Any],
) -> None:
    _pdf_draw_panel(pdf, x, y, width, height, palette["metric"], palette["line"], 11)
    pdf.setFont(bold, 9.5)
    pdf.setFillColor(palette["text"])
    pdf.drawString(x + 12, y + height - 17, title)
    current_y = y + height - 31
    for item in items:
        present = bool(item.get("present"))
        bullet_color = palette["accent_2"] if present else palette["muted"]
        pdf.setFillColor(bullet_color)
        pdf.setFont(bold, 7.7)
        pdf.drawString(x + 12, current_y, "✓" if present else "—")
        next_y = _pdf_draw_wrapped(
            pdf,
            item.get("label", ""),
            x + 22,
            current_y,
            width - 34,
            regular,
            7.5,
            palette["text"] if present else palette["muted"],
            leading=9.1,
            max_lines=2,
        )
        current_y = next_y - 1.4
        if current_y < y + 8:
            break


def _pdf_palette() -> dict[str, Any]:
    """Цветовая схема PDF совпадает с тёмной темой веб-дашборда."""

    return {
        "bg": colors.HexColor("#0B1020"),
        "panel": colors.HexColor("#131B31"),
        "metric": colors.HexColor("#0E1529"),
        "text": colors.HexColor("#EDF2FF"),
        "muted": colors.HexColor("#9CA9C7"),
        "line": colors.HexColor("#2A3654"),
        "accent": colors.HexColor("#7C8CFF"),
        "accent_2": colors.HexColor("#58D7C4"),
        "ring_base": colors.HexColor("#27314C"),
        "bar_bg": colors.HexColor("#222C45"),
        "recommendations": colors.HexColor("#182E43"),
        "recommendations_line": colors.HexColor("#2D6372"),
        "warning": colors.HexColor("#F8DDA8"),
    }


def _draw_report_header(
    pdf: Any,
    metrics: dict[str, Any],
    score: int,
    regular: str,
    bold: str,
    palette: dict[str, Any],
    page_width: float,
    page_height: float,
    margin: float,
) -> tuple[float, float, float]:
    """Рисует верхнюю часть индивидуального дашборда и возвращает геометрию."""

    usable_width = page_width - margin * 2
    header_top = page_height - margin
    header_bottom = page_height - 125
    _pdf_draw_panel(
        pdf,
        margin,
        header_bottom,
        usable_width,
        header_top - header_bottom,
        palette["panel"],
        palette["line"],
        16,
    )
    _pdf_draw_score_ring(pdf, margin + 58, header_bottom + 52, 41, score, regular, bold, palette)

    task_types = ", ".join(str(value) for value in metrics.get("task_types", [])) or "—"
    risk_label = str((metrics.get("task_risk") or {}).get("label") or "—")
    prompt_count = metrics.get("prompt_count", "—")
    text_x = margin + 122

    pdf.setFont(bold, 17)
    pdf.setFillColor(palette["text"])
    pdf.drawString(text_x, header_top - 29, "Индивидуальный отчёт")
    _pdf_draw_wrapped(
        pdf,
        metrics.get("level", "Нет интерпретации"),
        text_x,
        header_top - 48,
        usable_width - 170,
        bold,
        10.3,
        palette["accent_2"],
        max_lines=1,
    )
    pdf.setFont(regular, 8.6)
    pdf.setFillColor(palette["muted"])
    pdf.drawString(text_x, header_top - 69, f"Запросов: {prompt_count} · Риск задачи: {risk_label}")
    _pdf_draw_wrapped(
        pdf,
        f"Типы задач: {task_types}",
        text_x,
        header_top - 87,
        usable_width - 170,
        regular,
        8.1,
        palette["muted"],
        max_lines=1,
    )
    return usable_width, header_bottom, header_top


def _draw_full_evidence_page(
    pdf: Any,
    evidence: dict[str, Any],
    regular: str,
    bold: str,
    palette: dict[str, Any],
    page_width: float,
    page_height: float,
    margin: float,
) -> None:
    """Создаёт отдельную страницу без обрезания оснований оценки."""

    titles = {
        "P": "Постановка задачи",
        "V": "Проверка результата",
        "A": "Аналитическое участие",
        "R": "Риск делегирования",
    }
    usable_width = page_width - margin * 2

    pdf.setFillColor(palette["bg"])
    pdf.rect(0, 0, page_width, page_height, stroke=0, fill=1)
    header_top = page_height - margin
    header_bottom = page_height - 86
    _pdf_draw_panel(
        pdf,
        margin,
        header_bottom,
        usable_width,
        header_top - header_bottom,
        palette["panel"],
        palette["line"],
        16,
    )
    pdf.setFont(bold, 16)
    pdf.setFillColor(palette["text"])
    pdf.drawString(margin + 16, header_top - 31, "Основания оценки")
    _pdf_draw_wrapped(
        pdf,
        "Полный перечень обнаруженных и не обнаруженных признаков по каждому компоненту.",
        margin + 16,
        header_top - 51,
        usable_width - 32,
        regular,
        8.5,
        palette["muted"],
        max_lines=1,
    )

    gap = 12
    card_width = (usable_width - gap) / 2
    card_height = 205
    top_y = header_bottom - 18 - card_height
    bottom_y = top_y - gap - card_height
    positions = (
        ("P", margin, top_y),
        ("V", margin + card_width + gap, top_y),
        ("A", margin, bottom_y),
        ("R", margin + card_width + gap, bottom_y),
    )
    for key, x, y in positions:
        _pdf_draw_detail_card(
            pdf,
            titles[key],
            list(evidence.get(key, [])),
            x,
            y,
            card_width,
            card_height,
            regular,
            bold,
            palette,
        )

    pdf.setFont(regular, 7.0)
    pdf.setFillColor(palette["muted"])
    pdf.drawString(margin + 2, margin - 2, f"Сформировано: {datetime.now().strftime('%d.%m.%Y %H:%M')}")


def report_to_pdf_bytes(report: ReportDB) -> bytes:
    """Формирует одностраничный PDF-дашборд со всеми основаниями оценки."""

    if colors is None or A4 is None or canvas is None:
        raise HTTPException(status_code=500, detail="Для выгрузки PDF установите библиотеку reportlab.")

    regular, bold = register_report_fonts()
    buffer = io.BytesIO()
    page_width, page_height = A4[1], A4[0]  # альбомная A4
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    pdf.setTitle(f"AI Interaction Diagnostic — отчёт {report.id}")
    pdf.setAuthor("AI Interaction Diagnostic")

    palette = _pdf_palette()
    metrics = report.metrics or {}
    components = metrics.get("components", {})
    evidence = metrics.get("evidence", {})
    score = int(metrics.get("overall_score") or 0)
    recommendations = list(report.recommendations or metrics.get("recommendations", []))[:4]
    titles = {
        "P": "Постановка задачи",
        "V": "Проверка результата",
        "A": "Аналитическое участие",
        "R": "Риск делегирования",
    }

    margin = 18
    usable_width = page_width - margin * 2
    pdf.setFillColor(palette["bg"])
    pdf.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    # Компактная верхняя панель.
    header_top = page_height - margin
    header_bottom = page_height - 100
    _pdf_draw_panel(
        pdf, margin, header_bottom, usable_width, header_top - header_bottom,
        palette["panel"], palette["line"], 16,
    )
    _pdf_draw_score_ring(pdf, margin + 53, header_bottom + 41, 33, score, regular, bold, palette)

    task_types = ", ".join(str(value) for value in metrics.get("task_types", [])) or "—"
    risk_label = str((metrics.get("task_risk") or {}).get("label") or "—")
    prompt_count = metrics.get("prompt_count", "—")
    text_x = margin + 105

    pdf.setFont(bold, 15.5)
    pdf.setFillColor(palette["text"])
    pdf.drawString(text_x, header_top - 28, "Индивидуальный отчёт")
    _pdf_draw_wrapped(
        pdf, metrics.get("level", "Нет интерпретации"), text_x, header_top - 47,
        usable_width - 130, bold, 9.4, palette["accent_2"], max_lines=1,
    )
    pdf.setFont(regular, 7.9)
    pdf.setFillColor(palette["muted"])
    pdf.drawString(text_x, header_top - 65, f"Запросов: {prompt_count} · Риск задачи: {risk_label}")
    _pdf_draw_wrapped(
        pdf, f"Типы задач: {task_types}", text_x, header_top - 80,
        usable_width - 130, regular, 7.1, palette["muted"], max_lines=1,
    )

    # Четыре карточки компонентов.
    metrics_y = header_bottom - 88
    metric_height = 78
    metric_gap = 10
    metric_width = (usable_width - metric_gap * 3) / 4
    for index, key in enumerate(("P", "V", "A", "R")):
        component = components.get(key, {})
        _pdf_draw_metric_card(
            pdf, key, component.get("title") or titles[key],
            int(component.get("score") or 0),
            margin + index * (metric_width + metric_gap), metrics_y,
            metric_width, metric_height, regular, bold, palette,
        )

    # Все основания оценки помещаются на той же странице в 4 карточках.
    detail_gap = 10
    detail_width = (usable_width - detail_gap) / 2
    detail_height = 140
    detail_top_y = metrics_y - 11 - detail_height
    detail_bottom_y = detail_top_y - 9 - detail_height
    positions = (
        ("P", margin, detail_top_y),
        ("V", margin + detail_width + detail_gap, detail_top_y),
        ("A", margin, detail_bottom_y),
        ("R", margin + detail_width + detail_gap, detail_bottom_y),
    )
    for key, x, y in positions:
        _pdf_draw_detail_card(
            pdf, titles[key], list(evidence.get(key, [])), x, y,
            detail_width, detail_height, regular, bold, palette,
        )

    # Рекомендации и пояснение внизу страницы.
    recommendations_y = 19
    recommendations_h = detail_bottom_y - recommendations_y - 10
    _pdf_draw_panel(
        pdf, margin, recommendations_y, usable_width, recommendations_h,
        palette["recommendations"], palette["recommendations_line"], 12,
    )
    pdf.setFont(bold, 9.2)
    pdf.setFillColor(palette["text"])
    pdf.drawString(margin + 12, recommendations_y + recommendations_h - 16, "Рекомендации")

    if recommendations:
        content_top = recommendations_y + recommendations_h - 29
        column_gap = 12
        column_width = (usable_width - 38 - column_gap) / 2
        for index, item in enumerate(recommendations):
            column = index % 2
            row = index // 2
            x = margin + 14 + column * (column_width + column_gap)
            y = content_top - row * 23
            pdf.setFillColor(palette["text"])
            pdf.setFont(bold, 6.3)
            pdf.drawString(x, y, "•")
            _pdf_draw_wrapped(
                pdf, item, x + 9, y, column_width - 9, regular, 6.25,
                palette["text"], leading=7.2, max_lines=2,
            )
    else:
        _pdf_draw_wrapped(
            pdf, "Рекомендации отсутствуют.", margin + 14,
            recommendations_y + recommendations_h - 30, usable_width - 28,
            regular, 6.5, palette["text"], max_lines=1,
        )

    warning = metrics.get("sensitive_data_warning") or []
    footer = metrics.get("verification_note") or "Оценка основана на наблюдаемых текстовых признаках конкретного диалога."
    if warning:
        footer = f"Возможные чувствительные данные: {', '.join(str(item) for item in warning)}. {footer}"
    _pdf_draw_wrapped(
        pdf, footer, margin + 12, recommendations_y + 9, usable_width - 24,
        regular, 5.8, palette["warning"] if warning else palette["muted"],
        leading=6.8, max_lines=1,
    )
    pdf.setFont(regular, 5.7)
    pdf.setFillColor(palette["muted"])
    pdf.drawRightString(page_width - margin, 9, f"Сформировано: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
@app.get("/api/reports/{report_id}/pdf")
def download_report_pdf(report_id: int, db: Session = Depends(get_db)) -> Response:
    report = get_or_404(db, ReportDB, report_id, "Отчёт не найден")
    pdf_bytes = report_to_pdf_bytes(report)
    filename = f"AI_Interaction_Report_{report_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/teams")
def list_teams(db: Session = Depends(get_db)) -> list[str]:
    rows = (
        db.query(ReportDB.team_name)
        .filter(ReportDB.team_name.isnot(None))
        .distinct()
        .order_by(ReportDB.team_name.asc())
        .all()
    )
    return [row[0] for row in rows if row[0] and row[0] != PERSONAL_TEAM]


def get_team_analytics_payload(team_name: str, db: Session) -> dict[str, Any]:
    """Собирает данные командного дашборда для веб-интерфейса и PDF."""

    team = clean_team_name(team_name)
    reports = (
        db.query(ReportDB)
        .filter(ReportDB.team_name == team)
        .order_by(ReportDB.created_at.asc())
        .all()
    )

    if len(reports) < MIN_TEAM_REPORTS:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Для агрегированной аналитики требуется не менее {MIN_TEAM_REPORTS} отчётов. "
                f"Сейчас доступно: {len(reports)}."
            ),
        )

    component_keys = ("P", "V", "A", "R")
    components: dict[str, float] = {}
    for key in component_keys:
        values = [
            report.metrics.get("components", {}).get(key, {}).get("score")
            for report in reports
            if report.metrics.get("components", {}).get(key, {}).get("score") is not None
        ]
        components[key] = round(mean([float(value) for value in values]), 1) if values else 0.0

    levels = Counter(report.metrics.get("level", "Нет данных") for report in reports)
    risk_levels = Counter(report.metrics.get("task_risk", {}).get("label", "Нет данных") for report in reports)

    return {
        "team_name": team,
        "report_count": len(reports),
        "average_overall_score": round(
            mean([float(report.metrics.get("overall_score", 0)) for report in reports]), 1
        ),
        "components": components,
        "levels": dict(levels),
        "task_risks": dict(risk_levels),
        "privacy_note": "Показатели агрегированы. Тексты диалогов и индивидуальные отчёты не отображаются.",
    }


def team_analytics_to_pdf_bytes(data: dict[str, Any]) -> bytes:
    """Формирует скачиваемый PDF командного дашборда без ручной печати."""

    if colors is None or A4 is None or canvas is None:
        raise HTTPException(status_code=500, detail="Для выгрузки PDF установите библиотеку reportlab.")

    regular, bold = register_report_fonts()
    buffer = io.BytesIO()
    page_width, page_height = A4[1], A4[0]
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    pdf.setTitle(f"AI Interaction Diagnostic — командный отчёт: {data['team_name']}")
    pdf.setAuthor("AI Interaction Diagnostic")
    palette = _pdf_palette()

    pdf.setFillColor(palette["bg"])
    pdf.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    margin = 20
    usable_width = page_width - margin * 2
    header_top = page_height - margin
    header_bottom = page_height - 148
    _pdf_draw_panel(pdf, margin, header_bottom, usable_width, header_top - header_bottom, palette["panel"], palette["line"], 16)
    _pdf_draw_score_ring(
        pdf,
        margin + 58,
        header_bottom + 61,
        41,
        int(round(float(data.get("average_overall_score", 0)))),
        regular,
        bold,
        palette,
    )

    text_x = margin + 122
    pdf.setFont(bold, 17)
    pdf.setFillColor(palette["text"])
    pdf.drawString(text_x, header_top - 30, "Командный отчёт")
    pdf.setFont(bold, 11)
    pdf.setFillColor(palette["accent_2"])
    _pdf_draw_wrapped(pdf, data.get("team_name", "Команда"), text_x, header_top - 50, usable_width - 170, bold, 11, palette["accent_2"], max_lines=1)
    pdf.setFont(regular, 8.8)
    pdf.setFillColor(palette["muted"])
    pdf.drawString(text_x, header_top - 73, f"Обезличенных отчётов: {data.get('report_count', 0)}")
    _pdf_draw_wrapped(
        pdf,
        data.get("privacy_note", ""),
        text_x,
        header_top - 92,
        usable_width - 170,
        regular,
        8.0,
        palette["muted"],
        max_lines=2,
    )

    titles = {
        "P": "Постановка задачи",
        "V": "Проверка результата",
        "A": "Аналитическое участие",
        "R": "Риск чрезмерного делегирования",
    }
    metrics_y = header_bottom - 95
    gap = 10
    card_width = (usable_width - gap * 3) / 4
    components = data.get("components", {})
    for index, key in enumerate(("P", "V", "A", "R")):
        _pdf_draw_metric_card(
            pdf,
            key,
            titles[key],
            int(round(float(components.get(key, 0)))),
            margin + index * (card_width + gap),
            metrics_y,
            card_width,
            78,
            regular,
            bold,
            palette,
        )

    detail_gap = 10
    detail_width = (usable_width - detail_gap) / 2
    detail_height = 147
    detail_y = metrics_y - detail_height - 14
    level_items = [
        {"present": True, "label": f"{label}: {count}"}
        for label, count in (data.get("levels") or {}).items()
    ] or [{"present": False, "label": "Нет данных о распределении уровней."}]
    risk_items = [
        {"present": True, "label": f"{label}: {count}"}
        for label, count in (data.get("task_risks") or {}).items()
    ] or [{"present": False, "label": "Нет данных о распределении рисков."}]
    _pdf_draw_detail_card(pdf, "Распределение уровней", level_items, margin, detail_y, detail_width, detail_height, regular, bold, palette)
    _pdf_draw_detail_card(pdf, "Распределение рисков задач", risk_items, margin + detail_width + detail_gap, detail_y, detail_width, detail_height, regular, bold, palette)

    note_y = 45
    note_h = max(75, detail_y - note_y - 14)
    _pdf_draw_panel(pdf, margin, note_y, usable_width, note_h, palette["recommendations"], palette["recommendations_line"], 12)
    pdf.setFont(bold, 10)
    pdf.setFillColor(palette["text"])
    pdf.drawString(margin + 13, note_y + note_h - 19, "Интерпретация")
    note_text = (
        "Отчёт отражает агрегированные показатели практик взаимодействия с генеративным ИИ. "
        "Он предназначен для обучения и улучшения процессов, а не для оценки отдельных сотрудников или кадровых решений."
    )
    _pdf_draw_wrapped(pdf, note_text, margin + 13, note_y + note_h - 37, usable_width - 26, regular, 8.4, palette["text"], leading=10.5, max_lines=5)
    pdf.setFont(regular, 7.0)
    pdf.setFillColor(palette["muted"])
    pdf.drawString(margin + 13, note_y + 14, f"Сформировано: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@app.get("/api/team-analytics")
def team_analytics(team_name: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    return get_team_analytics_payload(team_name, db)


@app.get("/api/team-analytics/pdf")
def download_team_analytics_pdf(team_name: str, db: Session = Depends(get_db)) -> Response:
    data = get_team_analytics_payload(team_name, db)
    pdf_bytes = team_analytics_to_pdf_bytes(data)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", data["team_name"]).strip("_") or "team"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="AI_Team_Report_{safe_name}.pdf"'},
    )



# =============================================================================
# 9. Встроенный фронтенд
# =============================================================================

INDEX_HTML = r"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Interaction Diagnostic</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #131b31;
      --panel-2: #192440;
      --text: #edf2ff;
      --muted: #9ca9c7;
      --line: rgba(255,255,255,.11);
      --accent: #7c8cff;
      --accent-2: #58d7c4;
      --warn: #ffbf69;
      --danger: #ff7d88;
      --shadow: 0 20px 55px rgba(0,0,0,.24);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      background: radial-gradient(circle at 20% 0%, #22305b 0%, transparent 34%), var(--bg);
    }
    .app { max-width: 1280px; margin: 0 auto; padding: 28px 20px 54px; }
    .topbar { display:flex; justify-content:space-between; gap:20px; align-items:flex-start; margin-bottom:26px; }
    .brand { display:flex; gap:14px; align-items:center; }
    .logo { width:48px; height:48px; display:grid; place-items:center; border-radius:15px; font-weight:800; color:#10162b; background:linear-gradient(135deg,var(--accent),var(--accent-2)); }
    h1 { font-size:22px; margin:0 0 5px; letter-spacing:-.02em; }
    .subtitle { color:var(--muted); font-size:14px; margin:0; max-width:760px; }
    .status { border:1px solid var(--line); color:var(--muted); padding:8px 11px; border-radius:12px; font-size:12px; white-space:nowrap; }
    .layout { display:grid; grid-template-columns:220px 1fr; gap:18px; }
    .sidebar, .card { background:rgba(19,27,49,.90); border:1px solid var(--line); box-shadow:var(--shadow); border-radius:18px; }
    .sidebar { padding:12px; height:fit-content; position:sticky; top:18px; }
    .nav-btn { width:100%; border:0; background:transparent; color:var(--muted); text-align:left; cursor:pointer; padding:12px 13px; border-radius:11px; font:inherit; margin:2px 0; transition:.18s; }
    .nav-btn:hover, .nav-btn.active { background:rgba(124,140,255,.16); color:var(--text); }
    .view { display:none; } .view.active { display:block; }
    .card { padding:24px; margin-bottom:18px; }
    .card h2 { margin:0 0 7px; font-size:19px; } .card h3 { margin:0 0 10px; font-size:15px; }
    .help { color:var(--muted); line-height:1.55; font-size:14px; margin:0 0 18px; }
    label { display:block; font-size:13px; color:#cbd5f0; margin:14px 0 7px; }
    textarea, input, select { width:100%; border:1px solid var(--line); background:#0e1529; color:var(--text); border-radius:12px; padding:12px; font:inherit; outline:none; }
    textarea { min-height:200px; resize:vertical; line-height:1.5; }
    textarea:focus, input:focus, select:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(124,140,255,.12); }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .actions { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:16px; }
    button.primary, button.secondary, button.ghost { border:0; border-radius:11px; padding:11px 15px; cursor:pointer; font:600 14px inherit; transition:.18s; }
    button.primary { color:#10162b; background:linear-gradient(135deg,var(--accent),var(--accent-2)); }
    button.primary:hover { transform:translateY(-1px); filter:brightness(1.05); }
    button.secondary, a.secondary { background:#263557; color:var(--text); } button.secondary:hover, a.secondary:hover { background:#30436c; }
    button.ghost { background:transparent; color:var(--muted); border:1px solid var(--line); }
    a.download-link { display:inline-flex; align-items:center; text-decoration:none; }
    .print-dashboard-btn { display:inline-flex; align-items:center; }
    .check { display:flex; align-items:flex-start; gap:9px; color:var(--muted); font-size:12px; line-height:1.45; margin-top:12px; }
    .check input { width:auto; margin-top:2px; }
    .notice { border-left:3px solid var(--warn); padding:10px 12px; background:rgba(255,191,105,.08); color:#f8dda8; border-radius:7px; font-size:13px; margin-top:14px; }
    .error { border-left-color:var(--danger); background:rgba(255,125,136,.08); color:#ffd0d4; }
    .report { display:none; }
    .score-hero { display:flex; gap:24px; align-items:center; padding-bottom:20px; border-bottom:1px solid var(--line); margin-bottom:20px; }
    .score-ring { width:112px; height:112px; border-radius:50%; display:grid; place-items:center; background:conic-gradient(var(--accent) calc(var(--score) * 1%), rgba(255,255,255,.09) 0); position:relative; }
    .score-ring:after { content:""; position:absolute; inset:9px; border-radius:50%; background:var(--panel); }
    .score-value { z-index:1; text-align:center; font-size:27px; font-weight:800; } .score-value span { font-size:12px; color:var(--muted); font-weight:500; display:block; }
    .level { color:var(--accent-2); font-weight:700; margin:3px 0 7px; } .meta { color:var(--muted); font-size:13px; line-height:1.5; }
    .metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0; }
    .metric { background:#0e1529; padding:14px; border-radius:13px; border:1px solid var(--line); }
    .metric .name { color:var(--muted); font-size:12px; line-height:1.35; min-height:32px; }
    .metric .num { font-size:24px; font-weight:750; margin:7px 0; }
    .bar { height:7px; background:rgba(255,255,255,.08); border-radius:99px; overflow:hidden; } .bar > i { display:block; height:100%; background:linear-gradient(90deg,var(--accent),var(--accent-2)); border-radius:99px; }
    .detail-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; }
    .detail { background:#0e1529; border:1px solid var(--line); border-radius:13px; padding:14px; }
    .detail ul, .recommendations ul { margin:9px 0 0; padding-left:18px; color:#dbe4ff; font-size:13px; line-height:1.55; }
    .detail li.no { color:#a9b6d3; } .recommendations { margin-top:14px; padding:15px; border-radius:13px; background:rgba(88,215,196,.08); border:1px solid rgba(88,215,196,.20); }
    .chat-window { min-height:370px; max-height:500px; overflow:auto; background:#0e1529; border:1px solid var(--line); border-radius:14px; padding:16px; }
    .message { margin:0 0 14px; max-width:82%; } .message.user { margin-left:auto; text-align:right; } .message .role { font-size:11px; color:var(--muted); margin:0 0 4px; }
    .bubble { display:inline-block; text-align:left; padding:11px 13px; border-radius:13px; background:#202d4c; white-space:pre-wrap; line-height:1.45; font-size:14px; }
    .message.user .bubble { background:rgba(124,140,255,.23); }
    .history-item { padding:14px; background:#0e1529; border:1px solid var(--line); border-radius:13px; margin:10px 0; display:flex; justify-content:space-between; gap:12px; align-items:center; }
    .history-item p { margin:4px 0 0; color:var(--muted); font-size:12px; }
    .empty { color:var(--muted); padding:18px 0; }
    .loader { display:none; margin:14px 0; color:var(--muted); font-size:13px; } .loader.show { display:block; }
    .mini-loader { display:inline-block; width:14px; height:14px; border:2px solid rgba(255,255,255,.2); border-top-color:var(--accent-2); border-radius:50%; animation:spin .75s linear infinite; vertical-align:-2px; margin-right:7px; }
    @keyframes spin { to { transform:rotate(360deg); } }
    .privacy { color:var(--muted); font-size:12px; margin:10px 0 0; }
    .pill { display:inline-block; padding:3px 7px; border-radius:999px; background:rgba(124,140,255,.16); color:#cbd4ff; font-size:11px; margin:3px 3px 0 0; }
    @media (max-width: 900px) { .layout { grid-template-columns:1fr; } .sidebar { position:static; display:flex; overflow:auto; gap:4px; } .nav-btn { white-space:nowrap; width:auto; } .metrics { grid-template-columns:repeat(2,1fr); } }
    @media (max-width: 600px) { .app { padding:16px 12px 34px; } .topbar { display:block; } .status { display:inline-block; margin-top:12px; } .card { padding:17px; } .grid2, .detail-grid { grid-template-columns:1fr; } .score-hero { align-items:flex-start; } .metrics { grid-template-columns:1fr 1fr; } }
  
  .topbar-actions {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .theme-toggle {
      border: 1px solid var(--line);
      background: transparent;
      color: var(--muted);
      border-radius: 12px;
      padding: 8px 11px;
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      transition: .18s;
    }

    .theme-toggle:hover {
      color: var(--text);
      border-color: var(--accent);
      background: rgba(124, 140, 255, .12);
    }

    .app-footer {
      margin-top: 28px;
      padding: 18px 4px 0;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
      text-align: center;
    }

    body.light-theme {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --panel-2: #eef3fb;
      --text: #172033;
      --muted: #60708a;
      --line: rgba(29, 48, 82, .16);
      --accent: #526cf6;
      --accent-2: #20b9aa;
      --warn: #b76e00;
      --danger: #d84a57;
      --shadow: 0 18px 45px rgba(44, 62, 100, .10);
      background: radial-gradient(circle at 20% 0%, #dce7ff 0%, transparent 34%), var(--bg);
    }

    body.light-theme .sidebar,
    body.light-theme .card {
      background: rgba(255, 255, 255, .94);
    }

    body.light-theme textarea,
    body.light-theme input,
    body.light-theme select,
    body.light-theme .chat-window,
    body.light-theme .metric,
    body.light-theme .detail,
    body.light-theme .history-item {
      background: #f8faff;
      color: var(--text);
    }

    body.light-theme .bubble {
      background: #e8eefb;
      color: var(--text);
    }

    body.light-theme .message.user .bubble {
      background: rgba(82, 108, 246, .16);
    }

    body.light-theme .detail ul,
    body.light-theme .recommendations ul {
      color: #2a3852;
    }

    body.light-theme .notice {
      background: #fff6e7;
      color: #7a4d00;
    }

    body.light-theme .notice.error {
      background: #fff0f1;
      color: #a52c37;
    }

    body.light-theme .recommendations {
      background: #ecfbf8;
      border-color: rgba(32, 185, 170, .25);
    }

    body.light-theme .nav-btn:hover,
    body.light-theme .nav-btn.active {
      background: rgba(82, 108, 246, .12);
    }

    @media (max-width: 600px) {
      .topbar-actions {
        margin-top: 12px;
        justify-content: flex-start;
      }
    }
    
  </style>
</head>
<body>
  <main class="app">
    <header class="topbar">
      <div class="brand"><div class="logo">AI</div><div><h1>AI Interaction Diagnostic</h1><p class="subtitle">Объяснимая диагностика качества взаимодействия с генеративным ИИ: задача, проверка, участие и риск делегирования.</p></div></div>
      <div class="topbar-actions">
    <button id="themeToggle" class="theme-toggle" type="button">
        ☀ Светлая тема
    </button>
    <div id="healthStatus" class="status">Проверка сервиса...</div>
    </div>
    </header>

    <div class="layout">
      <aside class="sidebar">
        <button class="nav-btn active" data-view="analysis">Диагностика</button>
        <button class="nav-btn" data-view="chat">Встроенный чат</button>
        <button class="nav-btn" data-view="history">История отчётов</button>
        <button class="nav-btn" data-view="team">Командная аналитика</button>
      </aside>

      <section>
        <div id="analysis" class="view active">
          <article class="card">
            <h2>Анализ диалога</h2>
            <p class="help">Вставьте текст, загрузите TXT/MD/PDF или укажите публичную ссылку на чат. В PDF без меток сервис определяет роль по положению реплики слева или справа.</p>
            <div class="grid2"><div><label for="teamName">Команда (необязательно)</label><input id="teamName" maxlength="120" placeholder="Например, Product Team"></div><div><label for="fileInput">Файл диалога</label><input id="fileInput" type="file" accept=".txt,.md,.pdf"></div></div>
            <div class="grid2"><div><label for="pdfUserSide">В PDF мои сообщения расположены</label><select id="pdfUserSide"><option value="right" selected>Справа</option><option value="left">Слева</option></select></div><div><label for="chatUrl">Публичная ссылка на чат</label><input id="chatUrl" type="url" placeholder="https://..."></div></div>
            <p class="privacy">Ссылки анализируются только тогда, когда сайт разрешает автоматическое чтение без входа. DeepSeek может открывать публичный чат в браузере, но блокировать серверный импорт; в этом случае сохраните страницу в PDF и загрузите файл.</p>
            <label for="dialogueText">Текст диалога</label>
            <textarea id="dialogueText" placeholder="User: Подготовь структуру аналитической записки...\nAI: ...\nUser: Проверь допущения и ограничения..."></textarea>
            <label class="check"><input type="checkbox" id="safeData"><span>Подтверждаю, что текст не содержит паролей, персональных данных, коммерческой тайны и иной запрещённой для передачи информации.</span></label>
            <div class="actions"><button class="primary" id="analyzeBtn">Проанализировать</button><button class="ghost" id="fillExampleBtn">Вставить пример</button></div>
            <div class="notice">Сервис оценивает только наблюдаемые признаки диалога. Результат не является оценкой общей профессиональной эффективности сотрудника.</div>
            <div id="analysisLoader" class="loader"><span class="mini-loader"></span>Выполняется анализ структуры диалога...</div>
            <div id="analysisError"></div>
          </article>
          <div id="analysisReport" class="report"></div>
        </div>

        <div id="chat" class="view">
          <article class="card">
            <h2>Встроенный чат</h2><p class="help">Чат сохраняет историю текущей сессии. Для генерации ответов задайте <code>GIGACHAT_API_KEY</code> в файле <code>.env</code>.</p>
            <div id="chatWindow" class="chat-window"></div>
            <label for="chatInput">Сообщение</label><textarea id="chatInput" style="min-height:86px" placeholder="Опишите рабочую задачу, контекст и ожидаемый формат результата..."></textarea>
            <div class="actions"><button class="primary" id="sendBtn">Отправить</button><button class="secondary" id="analyzeChatBtn">Проанализировать сессию</button><button class="ghost" id="newSessionBtn">Новая сессия</button></div>
            <div id="chatLoader" class="loader"><span class="mini-loader"></span>Получаем ответ...</div>
          </article>
          <div id="chatReport" class="report"></div>
        </div>

        <div id="history" class="view"><article class="card"><h2>История отчётов</h2><p class="help">Сохраняются метрики и рекомендации. Текст загруженного файла не сохраняется.</p><div id="reportsList" class="empty">Загрузка...</div></article><div id="historyReport" class="report"></div></div>

        <div id="team" class="view"><article class="card"><h2>Командная аналитика</h2><p class="help">Доступна только при наличии не менее 10 отчётов в одной команде. Индивидуальные диалоги и оценки не отображаются.</p><div class="grid2"><div><label for="analyticsTeam">Команда</label><input id="analyticsTeam" placeholder="Название команды"></div><div style="display:flex;align-items:end"><button class="primary" id="teamBtn" style="width:100%">Показать аналитику</button></div></div><div id="teamError"></div></article><div id="teamReport" class="report"></div></div>
      </section>
    </div>

    <footer class="app-footer">
      Сделано в рамках курсовой работы НИУ ВШЭ.<br>
      Студент программы «Продуктовый подход и аналитика данных в HR», 2025 — Усков Никита.<br>
      © 2025–2026. Все права защищены.
    </footer>
  </main>

  <script>
    const state = { sessionId: null };
    const $ = (id) => document.getElementById(id);
    function applyTheme(theme) {
      const isLight = theme === 'light';

      document.body.classList.toggle('light-theme', isLight);

      const button = $('themeToggle');
      if (button) {
        button.textContent = isLight
          ? '◐ Тёмная тема'
          : '☀ Светлая тема';
      }

      localStorage.setItem('ai-diagnostic-theme', theme);
    }

    function toggleTheme() {
      const nextTheme = document.body.classList.contains('light-theme')
        ? 'dark'
        : 'light';

      applyTheme(nextTheme);
    }

    function escapeHtml(value) { const box = document.createElement('div'); box.textContent = String(value ?? ''); return box.innerHTML; }
    function setError(targetId, message='') { const target = $(targetId); target.innerHTML = message ? `<div class="notice error">${escapeHtml(message)}</div>` : ''; }
    function setLoading(id, isLoading) { $(id).classList.toggle('show', isLoading); }
    async function request(url, options={}) {
      const response = await fetch(url, options);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || 'Не удалось выполнить запрос');
      return data;
    }

    async function checkHealth() {
      try {
        const health = await request('/api/health');
        $('healthStatus').textContent = health.gigachat_configured ? 'GigaChat подключён' : 'Режим анализа: чат не настроен';
      } catch { $('healthStatus').textContent = 'Сервис недоступен'; }
    }

    function showView(viewId) {
      document.querySelectorAll('.nav-btn').forEach(button => button.classList.toggle('active', button.dataset.view === viewId));
      document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === viewId));
      if (viewId === 'history') loadReports();
    }

    function componentCard(key, component) {
      const isRisk = key === 'R';
      const label = isRisk ? 'Чем ниже, тем лучше' : 'Чем выше, тем лучше';
      const width = Math.max(0, Math.min(100, component.score || 0));
      return `<div class="metric"><div class="name">${escapeHtml(component.title)}</div><div class="num">${width}</div><div class="bar"><i style="width:${width}%"></i></div><div class="meta">${label}</div></div>`;
    }

    function reportHtml(report) {
      const m = report.metrics || report;
      const c = m.components || {};
      const evidence = m.evidence || {};
      const score = Number(m.overall_score || 0);
      const recs = report.recommendations || m.recommendations || [];
      const componentOrder = ['P', 'V', 'A', 'R'];
      const titles = {P:'Постановка задачи', V:'Проверка результата', A:'Аналитическое участие', R:'Риск делегирования'};
      const cards = componentOrder.map(key => componentCard(key, c[key] || {title: titles[key], score:0})).join('');
      const details = componentOrder.map(key => {
        const rows = (evidence[key] || []).map(item => `<li class="${item.present ? '' : 'no'}">${item.present ? '✓' : '—'} ${escapeHtml(item.label)}</li>`).join('') || '<li class="no">Недостаточно данных для детализации.</li>';
        return `<div class="detail"><h3>${titles[key]}</h3><ul>${rows}</ul></div>`;
      }).join('');
      const tasks = (m.task_types || []).map(item => `<span class="pill">${escapeHtml(item)}</span>`).join('');
      const sensitive = (m.sensitive_data_warning || []).length ? `<div class="notice error">Обнаружены возможные чувствительные данные: ${escapeHtml(m.sensitive_data_warning.join(', '))}. Не передавайте такие данные в production-среду.</div>` : '';
      const pdfButton = report.id ? `<div class="actions"><a class="secondary download-link" href="/api/reports/${report.id}/pdf" download>Скачать отчёт в PDF</a></div>` : '';
      return `<article class="card dashboard-report"><div class="score-hero"><div class="score-ring" style="--score:${score}"><div class="score-value">${score}<span>из 100</span></div></div><div><h2>Индивидуальный отчёт</h2><div class="level">${escapeHtml(m.level || 'Нет интерпретации')}</div><div class="meta">Запросов: ${m.prompt_count ?? '—'} · Риск задачи: ${escapeHtml(m.task_risk?.label || '—')}<br>Типы задач: ${tasks || '—'}</div></div></div><div class="metrics">${cards}</div><div class="detail-grid">${details}</div><div class="recommendations"><h3>Рекомендации</h3><ul>${recs.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul><div class="privacy">${escapeHtml(m.verification_note || '')}</div></div>${sensitive}${pdfButton}</article>`;
    }

    function renderReport(targetId, report) {
      const target = $(targetId);
      target.classList.add('report');
      target.style.display = 'block';
      target.innerHTML = reportHtml(report);
    }

    async function analyzeTextOrFile() {
      setError('analysisError');
      if (!$('safeData').checked) { setError('analysisError', 'Подтвердите отсутствие чувствительных данных перед анализом.'); return; }
      const file = $('fileInput').files[0];
      const dialogue = $('dialogueText').value.trim();
      const chatUrl = $('chatUrl').value.trim();
      const teamName = $('teamName').value.trim() || 'Личная диагностика';
      const userSide = $('pdfUserSide').value;
      if (!file && !dialogue && !chatUrl) { setError('analysisError', 'Вставьте текст, выберите файл или укажите публичную ссылку.'); return; }
      setLoading('analysisLoader', true);
      try {
        let report;
        if (file) {
          const formData = new FormData();
          formData.append('file', file);
          formData.append('team_name', teamName);
          formData.append('user_side', userSide);
          report = await request('/api/analyze/file', {method:'POST', body:formData});
        } else if (chatUrl) {
          report = await request('/api/analyze/url', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({url: chatUrl, team_name: teamName})
          });
        } else {
          report = await request('/api/analyze/text', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({dialogue, team_name:teamName})
          });
        }
        renderReport('analysisReport', report);
      } catch (error) { setError('analysisError', error.message); }
      finally { setLoading('analysisLoader', false); }
    }


    function fillExample() {
      $('dialogueText').value = `User: Нужно подготовить план анализа снижения конверсии в оплату в мобильном приложении за второй квартал. Снижение составило 8 % относительно первого квартала. Предложи структуру исследования в формате: гипотеза, необходимые данные, метод проверки, возможные ограничения. Не делай выводов без данных.\nAI: Предлагаю начать с проверки технических, поведенческих и маркетинговых факторов.\nUser: Добавь альтернативные объяснения и укажи, какие сегменты пользователей нужно сравнить.\nAI: Дополняю структуру исследования.\nUser: Проверь, какие гипотезы нельзя проверить только по данным веб-аналитики, и укажи нужные источники данных.`;
    }

    async function initSession() {
      const sessions = await request('/api/sessions');
      if (sessions.length) state.sessionId = sessions[0].id;
      else state.sessionId = (await request('/api/sessions', {method:'POST'})).id;
      await loadMessages();
    }

    function addMessage(role, content) {
      const windowEl = $('chatWindow');
      const item = document.createElement('div'); item.className = `message ${role === 'user' ? 'user' : ''}`;
      item.innerHTML = `<div class="role">${role === 'user' ? 'Вы' : 'AI'}</div><div class="bubble">${escapeHtml(content)}</div>`;
      windowEl.appendChild(item); windowEl.scrollTop = windowEl.scrollHeight;
    }

    async function loadMessages() {
      if (!state.sessionId) return;
      const messages = await request(`/api/sessions/${state.sessionId}/messages`);
      $('chatWindow').innerHTML = '';
      if (!messages.length) addMessage('assistant', 'Опишите рабочую задачу. Для качественного результата укажите цель, контекст, ограничения и формат ответа.');
      else messages.forEach(message => addMessage(message.role, message.content));
    }

    async function sendMessage() {
      const message = $('chatInput').value.trim(); if (!message || !state.sessionId) return;
      $('chatInput').value = ''; addMessage('user', message); setLoading('chatLoader', true);
      try { const result = await request('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({session_id:state.sessionId, message})}); addMessage('assistant', result.answer); }
      catch (error) { addMessage('assistant', `Ошибка: ${error.message}`); }
      finally { setLoading('chatLoader', false); }
    }

    async function analyzeChat() {
      if (!state.sessionId) return;
      setLoading('chatLoader', true);
      try { const data = new FormData(); data.append('team_name', 'Личная диагностика'); const report = await request(`/api/analyze/session/${state.sessionId}`, {method:'POST', body:data}); renderReport('chatReport', report); }
      catch (error) { alert(error.message); }
      finally { setLoading('chatLoader', false); }
    }

    async function newSession() { state.sessionId = (await request('/api/sessions', {method:'POST'})).id; await loadMessages(); $('chatReport').style.display='none'; }

    async function loadReports() {
      try {
        const reports = await request('/api/reports');
        const container = $('reportsList');
        if (!reports.length) { container.innerHTML = '<div class="empty">Сохранённых отчётов пока нет.</div>'; return; }
        container.innerHTML = reports.map(item => { const m = item.metrics || {}; return `<div class="history-item"><div><strong>Отчёт #${item.id} · ${escapeHtml(item.team_name || 'Личная диагностика')}</strong><p>${new Date(item.created_at).toLocaleString('ru-RU')} · ${escapeHtml(item.source_type)} · ${m.overall_score ?? '—'} / 100</p></div><div class="actions" style="margin-top:0"><a class="secondary download-link" href="/api/reports/${item.id}/pdf" download>PDF</a><button class="secondary" data-report-id="${item.id}">Открыть</button></div></div>`; }).join('');
        container.querySelectorAll('[data-report-id]').forEach(button => button.addEventListener('click', async () => { const report = await request(`/api/reports/${button.dataset.reportId}`); renderReport('historyReport', report); }));
      } catch (error) { $('reportsList').innerHTML = `<div class="notice error">${escapeHtml(error.message)}</div>`; }
    }

    async function loadTeamAnalytics() {
      setError('teamError'); const team = $('analyticsTeam').value.trim(); if (!team) { setError('teamError', 'Введите название команды.'); return; }
      try {
        const data = await request(`/api/team-analytics?team_name=${encodeURIComponent(team)}`);
        const components = data.components || {}; const cards = Object.entries(components).map(([key,value]) => componentCard(key, {title:{P:'Постановка задачи',V:'Проверка результата',A:'Аналитическое участие',R:'Риск делегирования'}[key], score:value})).join('');
        const levels = Object.entries(data.levels || {}).map(([k,v]) => `<li>✓ ${escapeHtml(k)} — ${v}</li>`).join('') || '<li class="no">— Нет данных.</li>';
        const risks = Object.entries(data.task_risks || {}).map(([k,v]) => `<li>✓ ${escapeHtml(k)} — ${v}</li>`).join('') || '<li class="no">— Нет данных.</li>';
        const pdfUrl = `/api/team-analytics/pdf?team_name=${encodeURIComponent(data.team_name)}`;
        $('teamReport').style.display='block';
        $('teamReport').innerHTML = `<article class="card dashboard-report"><h2>${escapeHtml(data.team_name)}</h2><p class="help">${escapeHtml(data.privacy_note)}</p><div class="score-hero"><div class="score-ring" style="--score:${data.average_overall_score}"><div class="score-value">${data.average_overall_score}<span>средний балл</span></div></div><div><div class="level">${data.report_count} обезличенных отчётов</div><div class="meta">Командная диагностика по агрегированным данным</div></div></div><div class="metrics">${cards}</div><div class="detail-grid"><div class="detail"><h3>Распределение уровней</h3><ul>${levels}</ul></div><div class="detail"><h3>Распределение рисков задач</h3><ul>${risks}</ul></div></div><div class="recommendations"><h3>Интерпретация</h3><div class="privacy">Отчёт предназначен для обучения и улучшения процессов. Он не используется для оценки отдельных сотрудников или кадровых решений.</div></div><div class="actions"><a class="secondary download-link" href="${pdfUrl}" download>Скачать командный отчёт в PDF</a></div></article>`;
      } catch (error) { setError('teamError', error.message); }
    }

    document.querySelectorAll('.nav-btn').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));
    $('analyzeBtn').addEventListener('click', analyzeTextOrFile); $('fillExampleBtn').addEventListener('click', fillExample);
    $('sendBtn').addEventListener('click', sendMessage); $('analyzeChatBtn').addEventListener('click', analyzeChat); $('newSessionBtn').addEventListener('click', newSession); $('teamBtn').addEventListener('click', loadTeamAnalytics);
    $('chatInput').addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') sendMessage(); });
    $('themeToggle').addEventListener('click', toggleTheme);

    const savedTheme = localStorage.getItem('ai-diagnostic-theme') || 'dark';
    applyTheme(savedTheme);

    checkHealth();
    initSession().catch(error => console.error(error));  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML


# =============================================================================
# 10. Локальный запуск
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # При прямом запуске используется именно текущий файл, а не случайный app.py.
    uvicorn.run(app, host="0.0.0.0", port=8000)
