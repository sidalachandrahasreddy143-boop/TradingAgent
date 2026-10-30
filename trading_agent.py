"""Local, deterministic trading research and paper-trading assistant.

The application does not send user requests or provider data to an LLM. Request
routing, ticker validation, order checks, and confirmation are enforced locally.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

QLOO_BASE_URL = "https://hackathon.api.qloo.com"
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
REQUEST_TIMEOUT = (5, 15)
HTTP_ATTEMPTS = 3
MAX_ORDER_SHARES = 5
MAX_QUERY_LENGTH = 2_000

TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")
SEED_TYPES = {
    "artist": "urn:entity:artist",
    "movie": "urn:entity:movie",
    "tv_show": "urn:entity:tv_show",
    "book": "urn:entity:book",
    "brand": "urn:entity:brand",
}
CURATED_BRAND_TICKERS = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "amazon": "AMZN",
    "amazon com": "AMZN",
    "starbucks": "SBUX",
    "nike": "NKE",
    "coca cola": "KO",
    "the coca cola company": "KO",
    "mcdonalds": "MCD",
    "mcdonald s": "MCD",
    "netflix": "NFLX",
    "disney": "DIS",
    "the walt disney company": "DIS",
    "walmart": "WMT",
    "target": "TGT",
}


class Intent(str, Enum):
    TASTE_DISCOVERY = "taste_discovery"
    STOCK_ANALYSIS = "stock_analysis"
    PORTFOLIO_STATUS = "portfolio_status"
    PAPER_TRADE_REQUEST = "paper_trade_request"
    GENERAL_HELP = "general_help"
    UNSUPPORTED = "unsupported"


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class SeedType(str, Enum):
    ARTIST = "artist"
    MOVIE = "movie"
    TV_SHOW = "tv_show"
    BOOK = "book"
    BRAND = "brand"


class UserRequest(BaseModel):
    """Validated request data extracted by deterministic local rules."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    ticker: str | None = None
    taste_seed: str | None = None
    seed_type: SeedType | None = None
    trade_side: TradeSide | None = None
    quantity: int | None = Field(default=None, ge=1)
    explicitly_confirmed: bool = False

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper().lstrip("$")
        if not TICKER_PATTERN.fullmatch(normalized):
            raise ValueError("Ticker must be a valid 1-5 character symbol.")
        return normalized

    @field_validator("taste_seed")
    @classmethod
    def validate_taste_seed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or len(normalized) > 160:
            raise ValueError("Taste seed must contain 1-160 characters.")
        return normalized


class AgentResponse(BaseModel):
    request_type: Intent
    result: str
    evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    next_step: str | None = None


class BrandCandidate(BaseModel):
    name: str
    entity_id: str | None = None
    affinity: float | str | None = None
    verified_ticker: str | None = None
    explainability: Any | None = None


class TasteDiscoveryResult(BaseModel):
    matched_seed: str
    qloo_entity_id: str
    brands: list[BrandCandidate]
    affinity_note: str = (
        "Qloo affinity is aggregate cultural relatedness only; it is not "
        "financial sentiment, a price prediction, expected return, or a "
        "buy/sell signal."
    )


@dataclass(frozen=True)
class Settings:
    qloo_api_key: str | None = None
    twelvedata_api_key: str | None = None
    finnhub_api_key: str | None = None
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    max_order_shares: int = MAX_ORDER_SHARES

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_DIR / ".env", override=False)
        return cls(
            qloo_api_key=os.getenv("QLOO_API_KEY") or None,
            twelvedata_api_key=os.getenv("TWELVEDATA_API_KEY") or None,
            finnhub_api_key=os.getenv("FINNHUB_API_KEY") or None,
            alpaca_api_key=os.getenv("ALPACA_API_KEY") or None,
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY") or None,
        )

    def missing_for(self, intent: Intent) -> list[str]:
        required: dict[Intent, tuple[str, ...]] = {
            Intent.TASTE_DISCOVERY: ("QLOO_API_KEY",),
            Intent.STOCK_ANALYSIS: ("TWELVEDATA_API_KEY", "FINNHUB_API_KEY"),
            Intent.PORTFOLIO_STATUS: ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
            Intent.PAPER_TRADE_REQUEST: (
                "TWELVEDATA_API_KEY",
                "ALPACA_API_KEY",
                "ALPACA_SECRET_KEY",
            ),
        }
        values = {
            "QLOO_API_KEY": self.qloo_api_key,
            "TWELVEDATA_API_KEY": self.twelvedata_api_key,
            "FINNHUB_API_KEY": self.finnhub_api_key,
            "ALPACA_API_KEY": self.alpaca_api_key,
            "ALPACA_SECRET_KEY": self.alpaca_secret_key,
        }
        return [name for name in required.get(intent, ()) if not values[name]]


class ServiceError(RuntimeError):
    """Safe provider error with no request URL, response body, or credentials."""


@dataclass(frozen=True)
class PendingOrder:
    ticker: str
    side: TradeSide
    quantity: int


def _http_error_message(status_code: int) -> str:
    if status_code in (401, 403):
        return "The provider rejected the request. Check the configured API key and account access."
    if status_code == 429:
        return "The provider rate limit was reached. Wait briefly and try again."
    if status_code >= 500:
        return "The provider is temporarily unavailable. Try again later."
    return f"The provider returned HTTP {status_code}."


def request_json(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str | int | bool] | None = None,
    method: Literal["GET", "POST"] = "GET",
    json_body: dict[str, str] | None = None,
    retries: int = HTTP_ATTEMPTS,
) -> dict[str, Any] | list[Any]:
    """Make bounded HTTP requests; never include response text in exceptions."""

    if retries < 1 or retries > HTTP_ATTEMPTS:
        raise ValueError("Retry count is outside the configured limit.")
    if method not in ("GET", "POST"):
        raise ValueError("Unsupported HTTP method.")
    for attempt in range(retries):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.Timeout as exc:
            if attempt + 1 < retries and method == "GET":
                time.sleep(0.25 * (attempt + 1))
                continue
            raise ServiceError("The provider request timed out.") from exc
        except requests.RequestException as exc:
            if attempt + 1 < retries and method == "GET":
                time.sleep(0.25 * (attempt + 1))
                continue
            raise ServiceError("The provider request failed due to a network error.") from exc

        status = response.status_code
        retryable = status == 429 or status >= 500
        if retryable and attempt + 1 < retries and method == "GET":
            time.sleep(0.25 * (attempt + 1))
            continue
        if status >= 400:
            raise ServiceError(_http_error_message(status))
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ServiceError("The provider returned an invalid JSON response.") from exc
        if not isinstance(payload, (dict, list)):
            raise ServiceError("The provider returned an unexpected response format.")
        return payload
    raise ServiceError("The provider request failed after bounded retries.")


def _qloo_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if isinstance(results, dict):
        entities = results.get("entities", [])
    elif isinstance(results, list):
        entities = results
    else:
        entities = payload.get("entities", [])
    if isinstance(entities, dict):
        entities = [entities]
    if not isinstance(entities, list):
        return []
    return [entity for entity in entities if isinstance(entity, dict)]


def _map_brand_ticker(name: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    return CURATED_BRAND_TICKERS.get(normalized)


def _parse_affinity(entity: dict[str, Any]) -> float | str | None:
    query = entity.get("query")
    if isinstance(query, dict):
        value = query.get("affinity")
    else:
        value = entity.get("query.affinity")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    if isinstance(value, str) and len(value) <= 80:
        return value
    return None


def _parse_seed_type(text: str) -> SeedType:
    lowered = text.casefold()
    if re.search(r"\b(tv show|television|series)\b", lowered):
        return SeedType.TV_SHOW
    if re.search(r"\b(movie|film)\b", lowered):
        return SeedType.MOVIE
    if re.search(r"\b(book|novel)\b", lowered):
        return SeedType.BOOK
    if re.search(
        r"\b(?:fans? of|interest(?:ed)? in)\s+(?:the\s+)?brand\b",
        lowered,
    ):
        return SeedType.BRAND
    return SeedType.ARTIST


def _extract_seed(text: str) -> str | None:
    patterns = (
        r"\bfans?\s+of\s+(.+?)(?:[?.!,]|$)",
        r"\binterest(?:ed)?\s+in\s+(.+?)(?:[?.!,]|$)",
        r"\b(?:like|love|enjoy)\s+(.+?)(?:[?.!,]|$)",
        r"\bbased\s+on\s+(?:my\s+)?(?:taste|interest)\s+in\s+(.+?)(?:[?.!,]|$)",
        r"\b(?:about|for)\s+(?:the\s+)?(?:artist|movie|film|tv show|book|brand)\s+(.+?)(?:[?.!,]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            seed = re.sub(
                r"^(?:the\s+)?(?:artist|movie|film|tv\s+show|book|brand)\s+",
                "",
                match.group(1).strip(),
                flags=re.IGNORECASE,
            )
            seed = re.sub(r"\s+(?:please|thanks)$", "", seed, flags=re.IGNORECASE).strip()
            return seed or None
    return None


def _extract_ticker(text: str) -> str | None:
    candidates = re.findall(r"(?<![A-Za-z0-9])\$?([A-Z][A-Z0-9]*(?:\.[A-Z])?)(?![A-Za-z0-9])", text)
    stop_words = {"I", "A", "AN", "THE", "BUY", "SELL", "SHARE", "SHARES", "STOCK", "STOCKS", "USD", "AI"}
    for candidate in candidates:
        if candidate not in stop_words and TICKER_PATTERN.fullmatch(candidate):
            return candidate
    return None


def parse_user_request(query: str) -> UserRequest:
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
        return UserRequest(intent=Intent.UNSUPPORTED)

    text = query.strip()
    lowered = text.casefold()
    if re.search(r"\b(ignore|bypass|override)\b.*\b(rules|safeguards|safety|confirmation)\b", lowered) or re.search(
        r"\b(sell|liquidate)\s+(everything|all|all\s+positions)\b", lowered
    ):
        return UserRequest(intent=Intent.UNSUPPORTED)

    trade_match = re.search(
        r"\b(buy|purchase|sell)\s+(?P<quantity>one|a|an|\d+)\s+"
        r"(?:(?:share|shares|stock|stocks)\s+(?:of\s+)?)?"
        r"(?P<ticker>\$?[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z])?)\b",
        text,
        flags=re.IGNORECASE,
    )
    has_trade_word = bool(re.search(r"\b(buy|purchase|sell)\b", lowered))
    if trade_match or has_trade_word:
        side: TradeSide | None = None
        ticker: str | None = None
        quantity: int | None = None
        if trade_match:
            side = TradeSide.BUY if trade_match.group(1).casefold() in {"buy", "purchase"} else TradeSide.SELL
            raw_quantity = trade_match.group("quantity").casefold()
            quantity = {"one": 1, "a": 1, "an": 1}.get(raw_quantity)
            if quantity is None:
                try:
                    quantity = int(raw_quantity)
                except ValueError:
                    quantity = None
            ticker = trade_match.group("ticker").lstrip("$").upper()
        else:
            if re.search(r"\b(buy|purchase)\b", lowered):
                side = TradeSide.BUY
            elif re.search(r"\bsell\b", lowered):
                side = TradeSide.SELL
            ticker = _extract_ticker(text)

        confirmed = bool(
            re.search(r"\b(yes|confirm|confirmed|i confirm|approved)\b", lowered)
            and re.search(r"\bpaper\s+order\b|\bconfirm(?:ed)?\s+(?:the\s+)?order\b", lowered)
        )
        try:
            return UserRequest(
                intent=Intent.PAPER_TRADE_REQUEST,
                ticker=ticker,
                trade_side=side,
                quantity=quantity,
                explicitly_confirmed=confirmed,
            )
        except ValidationError:
            return UserRequest(
                intent=Intent.PAPER_TRADE_REQUEST,
                ticker=ticker if ticker and TICKER_PATTERN.fullmatch(ticker) else None,
                trade_side=side,
                explicitly_confirmed=confirmed,
            )

    if re.search(r"\b(portfolio|positions?|balance|buying power|stocks? do i own)\b", lowered):
        return UserRequest(intent=Intent.PORTFOLIO_STATUS)

    if re.search(
        r"\b(associated with fans|cultural|culture|taste|brands? based on|fans of|interest(?:ed)? in)\b",
        lowered,
    ):
        seed = _extract_seed(text)
        seed_type = _parse_seed_type(text)
        return UserRequest(
            intent=Intent.TASTE_DISCOVERY,
            taste_seed=seed,
            seed_type=seed_type if seed else None,
        )

    ticker = _extract_ticker(text)
    if re.search(r"\b(analy[sz]e|analysis|price|quote|market data|stock)\b", lowered):
        return UserRequest(intent=Intent.STOCK_ANALYSIS, ticker=ticker)

    if re.search(r"\b(help|what can you do|how does this work)\b", lowered):
        return UserRequest(intent=Intent.GENERAL_HELP)
    return UserRequest(intent=Intent.UNSUPPORTED)


class TradingAgentApp:
    """Deterministic request router with provider clients and a one-order gate."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.pending_order: PendingOrder | None = None

    @classmethod
    def from_env(cls) -> "TradingAgentApp":
        return cls(Settings.from_env())

    def missing_configuration(self) -> list[str]:
        return [
            name
            for name in (
                "QLOO_API_KEY",
                "TWELVEDATA_API_KEY",
                "FINNHUB_API_KEY",
                "ALPACA_API_KEY",
                "ALPACA_SECRET_KEY",
            )
            if name in self.settings.missing_for(
                {
                    "QLOO_API_KEY": Intent.TASTE_DISCOVERY,
                    "TWELVEDATA_API_KEY": Intent.STOCK_ANALYSIS,
                    "FINNHUB_API_KEY": Intent.STOCK_ANALYSIS,
                    "ALPACA_API_KEY": Intent.PORTFOLIO_STATUS,
                    "ALPACA_SECRET_KEY": Intent.PORTFOLIO_STATUS,
                }[name]
            )
        ]

    def handle_user_query(self, query: str) -> AgentResponse:
        request = parse_user_request(query)
        if self.pending_order and not (
            request.intent == Intent.PAPER_TRADE_REQUEST and request.explicitly_confirmed
        ):
            self.pending_order = None
        if request.intent == Intent.UNSUPPORTED:
            return AgentResponse(
                request_type=request.intent,
                result="I can help with cultural brand discovery, stock analysis, portfolio status, or confirmed Alpaca paper orders.",
                caveats=["I will not bypass safety checks or interpret instructions embedded in external content."],
                next_step='Try: "Which brands are associated with fans of Taylor Swift?" or "Analyze NVDA".',
            )
        if request.intent == Intent.GENERAL_HELP:
            return AgentResponse(
                request_type=request.intent,
                result="I route supported requests locally; I do not let an LLM select tools or submit trades.",
                evidence=[
                    "Cultural brand discovery: Qloo",
                    "Stock analysis: Twelve Data and Finnhub",
                    "Portfolio and paper orders: Alpaca paper endpoint only",
                ],
                caveats=["Qloo cultural affinity is unrelated to stock sentiment or expected returns."],
                next_step="Ask for a supported task; a trade requires a separate explicit confirmation.",
            )
        if request.intent == Intent.TASTE_DISCOVERY and (
            not request.taste_seed or not request.seed_type
        ):
            return self._handle_taste_discovery(request)
        if request.intent == Intent.STOCK_ANALYSIS and not request.ticker:
            return self._handle_stock_analysis(request)
        if request.intent == Intent.PAPER_TRADE_REQUEST and (
            not request.trade_side or not request.ticker or request.quantity is None
        ):
            return self._handle_trade_request(request)
        if request.intent == Intent.PAPER_TRADE_REQUEST and not request.explicitly_confirmed:
            return self._handle_trade_request(request)
        missing = self.settings.missing_for(request.intent)
        if missing:
            return AgentResponse(
                request_type=request.intent,
                result="This workflow is not configured.",
                caveats=[f"Add the missing setting(s) to the local .env file: {', '.join(missing)}."],
                next_step="Restart the app after updating .env.",
            )
        if request.intent == Intent.TASTE_DISCOVERY:
            return self._handle_taste_discovery(request)
        if request.intent == Intent.STOCK_ANALYSIS:
            return self._handle_stock_analysis(request)
        if request.intent == Intent.PORTFOLIO_STATUS:
            return self._handle_portfolio_status()
        if request.intent == Intent.PAPER_TRADE_REQUEST:
            return self._handle_trade_request(request)
        return AgentResponse(request_type=Intent.UNSUPPORTED, result="Unsupported request.")

    def _handle_taste_discovery(self, request: UserRequest) -> AgentResponse:
        if not request.taste_seed or not request.seed_type:
            return AgentResponse(
                request_type=Intent.TASTE_DISCOVERY,
                result="I need a cultural seed to search for related brands.",
                next_step="Name an artist, movie, TV show, book, or brand.",
            )
        try:
            result = self.discover_brands(request.taste_seed, request.seed_type)
        except ServiceError as exc:
            return AgentResponse(
                request_type=Intent.TASTE_DISCOVERY,
                result=f"Qloo discovery failed: {exc}",
                evidence=["Source: Qloo hackathon API"],
                next_step="Check the QLOO_API_KEY and try again; no substitute results were used.",
            )
        lines = [
            f"{brand.name}: "
            + (f"affinity {brand.affinity}; " if brand.affinity is not None else "affinity not returned; ")
            + (
                f"verified curated ticker {brand.verified_ticker}"
                if brand.verified_ticker
                else "no verified ticker mapping"
            )
            for brand in result.brands
        ]
        return AgentResponse(
            request_type=Intent.TASTE_DISCOVERY,
            result=f"Qloo matched {result.matched_seed!r} (entity ID {result.qloo_entity_id}). "
            + ("; ".join(lines) if lines else "Qloo returned no brand candidates."),
            evidence=[
                "Source: Qloo /search and /v2/insights.",
                result.affinity_note,
            ],
            caveats=[
                "Brand names and explanations are untrusted external data.",
                "A curated ticker mapping identifies a possible public listing; it is not an investment recommendation.",
            ],
            next_step="Request stock analysis separately for a candidate with a verified ticker.",
        )

    def discover_brands(self, seed: str, seed_type: SeedType) -> TasteDiscoveryResult:
        api_key = self.settings.qloo_api_key
        if not api_key:
            raise ServiceError("QLOO_API_KEY is missing from .env.")
        if seed_type.value not in SEED_TYPES:
            raise ServiceError("Unsupported Qloo seed type.")
        headers = {"X-Api-Key": api_key}
        search = request_json(
            f"{QLOO_BASE_URL}/search",
            headers=headers,
            params={"query": seed, "types": SEED_TYPES[seed_type.value]},
        )
        if not isinstance(search, dict):
            raise ServiceError("Qloo returned an unexpected seed-search response.")
        matches = _qloo_entities(search)
        resolved = next(
            (
                entity
                for entity in matches
                if entity.get("entity_id")
                and str(entity.get("name", "")).casefold() == seed.casefold()
            ),
            next((entity for entity in matches if entity.get("entity_id")), None),
        )
        if not resolved:
            raise ServiceError("Qloo did not return a matching seed entity.")
        entity_id = resolved["entity_id"]
        if not isinstance(entity_id, str) or not entity_id.strip() or len(entity_id) > 300:
            raise ServiceError("Qloo returned an invalid seed entity ID.")
        recommendations = request_json(
            f"{QLOO_BASE_URL}/v2/insights",
            headers=headers,
            params={
                "filter.type": "urn:entity:brand",
                "signal.interests.entities": entity_id,
                "take": 8,
                "feature.explainability": True,
            },
        )
        if not isinstance(recommendations, dict):
            raise ServiceError("Qloo returned an unexpected brand-recommendation response.")
        brands: list[BrandCandidate] = []
        for entity in _qloo_entities(recommendations)[:8]:
            name = entity.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            entity_id_value = entity.get("entity_id")
            query_data = entity.get("query")
            explainability = (
                query_data.get("explainability")
                if isinstance(query_data, dict)
                else entity.get("explainability")
            )
            brands.append(
                BrandCandidate(
                    name=name[:200],
                    entity_id=entity_id_value if isinstance(entity_id_value, str) else None,
                    affinity=_parse_affinity(entity),
                    verified_ticker=_map_brand_ticker(name),
                    explainability=explainability,
                )
            )
        return TasteDiscoveryResult(
            matched_seed=str(resolved.get("name") or seed)[:200],
            qloo_entity_id=entity_id,
            brands=brands,
        )

    def _twelvedata(self, endpoint: str, params: dict[str, str | int]) -> dict[str, Any]:
        key = self.settings.twelvedata_api_key
        if not key:
            raise ServiceError("TWELVEDATA_API_KEY is missing from .env.")
        return request_json(
            f"{TWELVEDATA_BASE_URL}/{endpoint}",
            headers={"Authorization": f"apikey {key}"},
            params=params,
        )

    def _finnhub_news(self, ticker: str) -> list[dict[str, Any]]:
        key = self.settings.finnhub_api_key
        if not key:
            raise ServiceError("FINNHUB_API_KEY is missing from .env.")
        end = date.today()
        payload = request_json(
            f"{FINNHUB_BASE_URL}/company-news",
            headers={"X-Finnhub-Token": key},
            params={
                "symbol": ticker,
                "from": (end - timedelta(days=7)).isoformat(),
                "to": end.isoformat(),
            },
        )
        # Finnhub's company-news endpoint returns a list; reject unexpected data.
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and payload.get("error"):
            raise ServiceError("Finnhub reported an error while fetching company news.")
        return []

    def _handle_stock_analysis(self, request: UserRequest) -> AgentResponse:
        if not request.ticker:
            return AgentResponse(
                request_type=Intent.STOCK_ANALYSIS,
                result="I need a ticker symbol to analyze.",
                next_step='Provide a 1-5 character ticker, for example "Analyze NVDA".',
            )
        ticker = request.ticker
        evidence: list[str] = []
        caveats = [
            "This is descriptive market data, not personalized financial advice or a forecast.",
            "Market-data availability and freshness depend on the provider and account plan.",
        ]
        parts: list[str] = []
        try:
            price = self._twelvedata("price", {"symbol": ticker})
            value = Decimal(str(price["price"]))
            if not value.is_finite() or value <= 0:
                raise ValueError("invalid price")
            parts.append(f"Latest price: ${value:.2f}")
            evidence.append("Price source: Twelve Data.")
        except (ServiceError, KeyError, ValueError, InvalidOperation):
            caveats.append("Current price could not be verified from Twelve Data.")

        try:
            history = self._twelvedata(
                "time_series", {"symbol": ticker, "interval": "1day", "outputsize": 10}
            )
            rows = history.get("values")
            closes = [Decimal(str(row["close"])) for row in rows if isinstance(row, dict)]
            if len(closes) >= 2:
                change = closes[0] - closes[-1]
                trend = "higher" if change > 0 else "lower" if change < 0 else "unchanged"
                parts.append(f"Recent 10-session trend: {trend} (${change:+.2f} across returned observations)")
            else:
                caveats.append("Recent price history was incomplete.")
        except (ServiceError, KeyError, TypeError, ValueError, InvalidOperation):
            caveats.append("Price history could not be verified from Twelve Data.")

        try:
            rsi_data = self._twelvedata(
                "rsi", {"symbol": ticker, "interval": "1day", "time_period": 14}
            )
            rsi = Decimal(str(rsi_data["values"][0]["rsi"]))
            parts.append(f"RSI(14): {rsi:.1f}")
        except (ServiceError, KeyError, IndexError, TypeError, ValueError, InvalidOperation):
            caveats.append("RSI could not be retrieved.")

        try:
            sma_50 = self._twelvedata(
                "sma", {"symbol": ticker, "interval": "1day", "time_period": 50}
            )
            sma_200 = self._twelvedata(
                "sma", {"symbol": ticker, "interval": "1day", "time_period": 200}
            )
            short = Decimal(str(sma_50["values"][0]["sma"]))
            long = Decimal(str(sma_200["values"][0]["sma"]))
            parts.append(f"50-day SMA: ${short:.2f}; 200-day SMA: ${long:.2f}")
        except (ServiceError, KeyError, IndexError, TypeError, ValueError, InvalidOperation):
            caveats.append("One or both simple moving averages could not be retrieved.")

        try:
            headlines = self._finnhub_news(ticker)
            safe_headlines = [
                str(item.get("headline", "")).strip()[:240]
                for item in headlines[:5]
                if isinstance(item.get("headline"), str) and item.get("headline", "").strip()
            ]
            parts.append("Recent news: " + (" | ".join(safe_headlines) if safe_headlines else "none returned"))
            evidence.append("News source: Finnhub company-news feed.")
        except ServiceError:
            caveats.append("Recent news could not be retrieved from Finnhub.")

        result = f"{ticker} market-data summary. " + ("; ".join(parts) if parts else "No market observations were available.")
        return AgentResponse(
            request_type=Intent.STOCK_ANALYSIS,
            result=result,
            evidence=evidence,
            caveats=caveats,
            next_step="Ask separately if you want to prepare a paper-order preview.",
        )

    def _alpaca_headers(self) -> dict[str, str]:
        if not self.settings.alpaca_api_key or not self.settings.alpaca_secret_key:
            raise ServiceError("Alpaca paper API credentials are missing from .env.")
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_secret_key,
        }

    def _handle_portfolio_status(self) -> AgentResponse:
        headers = self._alpaca_headers()
        try:
            account = request_json(
                f"{ALPACA_PAPER_BASE_URL}/v2/account", headers=headers
            )
            if not isinstance(account, dict):
                raise ServiceError("Alpaca returned an unexpected account format.")
            positions = request_json(
                f"{ALPACA_PAPER_BASE_URL}/v2/positions",
                headers=headers,
            )
            if not isinstance(positions, list):
                raise ServiceError("Alpaca returned an unexpected positions format.")
        except ServiceError as exc:
            return AgentResponse(
                request_type=Intent.PORTFOLIO_STATUS,
                result=f"Could not read the Alpaca paper portfolio: {exc}",
                evidence=["Source: Alpaca paper API."],
            )
        except (requests.RequestException, ValueError):
            return AgentResponse(
                request_type=Intent.PORTFOLIO_STATUS,
                result="Could not read the Alpaca paper portfolio due to a network or response error.",
                evidence=["Source: Alpaca paper API."],
            )
        safe_positions = []
        for position in positions:
            if isinstance(position, dict) and isinstance(position.get("symbol"), str):
                safe_positions.append(
                    f"{position['symbol']} {position.get('qty', '?')} share(s)"
                )
        equity = account.get("equity", "unavailable")
        buying_power = account.get("buying_power", "unavailable")
        return AgentResponse(
            request_type=Intent.PORTFOLIO_STATUS,
            result=f"Paper account equity: {equity}; buying power: {buying_power}. "
            + ("Positions: " + ", ".join(safe_positions) if safe_positions else "No open positions."),
            evidence=["Source: Alpaca paper account and positions endpoints."],
            caveats=["Portfolio figures are account-sensitive and were not logged."],
        )

    def _handle_trade_request(self, request: UserRequest) -> AgentResponse:
        if not request.trade_side or not request.ticker or request.quantity is None:
            self.pending_order = None
            return AgentResponse(
                request_type=Intent.PAPER_TRADE_REQUEST,
                result="I cannot prepare an order from an incomplete or invalid request.",
                next_step='Specify side, ticker, and quantity, for example "Buy 1 share of AAPL".',
            )
        if request.quantity > min(self.settings.max_order_shares, MAX_ORDER_SHARES):
            self.pending_order = None
            return AgentResponse(
                request_type=Intent.PAPER_TRADE_REQUEST,
                result=f"Order refused: the per-order limit is {min(self.settings.max_order_shares, MAX_ORDER_SHARES)} shares.",
            )
        proposed = PendingOrder(request.ticker, request.trade_side, request.quantity)
        if not request.explicitly_confirmed:
            self.pending_order = proposed
            return AgentResponse(
                request_type=Intent.PAPER_TRADE_REQUEST,
                result=f"Paper-order preview only: {proposed.side.value.upper()} {proposed.quantity} share(s) of {proposed.ticker}. No order was submitted.",
                caveats=["Submission requires a separate confirmation matching this exact paper order."],
                next_step=f'To proceed, reply: "Yes, confirm paper order: {proposed.side.value} {proposed.quantity} share(s) of {proposed.ticker}".',
            )
        if self.pending_order != proposed:
            self.pending_order = None
            return AgentResponse(
                request_type=Intent.PAPER_TRADE_REQUEST,
                result="Order refused: there is no matching pending preview. Prepare a new preview first.",
                next_step="Submit the complete order as a new request to receive a preview.",
            )
        self.pending_order = None
        try:
            result = self._submit_paper_order(proposed)
        except ServiceError as exc:
            return AgentResponse(
                request_type=Intent.PAPER_TRADE_REQUEST,
                result=f"Order not submitted: {exc}",
                evidence=["No live brokerage endpoint is configured."],
            )
        return AgentResponse(
            request_type=Intent.PAPER_TRADE_REQUEST,
            result=result,
            evidence=["Order endpoint: Alpaca paper trading only."],
            caveats=["Paper trading is simulated and is not an investment recommendation."],
        )

    def _submit_paper_order(self, order: PendingOrder) -> str:
        if not TICKER_PATTERN.fullmatch(order.ticker):
            raise ServiceError("Ticker validation failed.")
        if order.side not in (TradeSide.BUY, TradeSide.SELL):
            raise ServiceError("Order side validation failed.")
        if not isinstance(order.quantity, int) or not 1 <= order.quantity <= min(
            self.settings.max_order_shares, MAX_ORDER_SHARES
        ):
            raise ServiceError("Quantity exceeds the configured per-order limit.")
        headers = self._alpaca_headers()
        try:
            account = request_json(
                f"{ALPACA_PAPER_BASE_URL}/v2/account", headers=headers
            )
            if not isinstance(account, dict):
                raise ServiceError("Alpaca returned an unexpected account format.")
            if (
                account.get("status") != "ACTIVE"
                or account.get("trading_blocked")
                or account.get("account_blocked")
                or account.get("trade_suspended_by_user")
            ):
                raise ServiceError("The paper account is inactive or trading is blocked.")
            positions = request_json(
                f"{ALPACA_PAPER_BASE_URL}/v2/positions",
                headers=headers,
            )
            if not isinstance(positions, list):
                raise ServiceError("Alpaca returned an unexpected positions format.")
        except ServiceError:
            raise
        except (requests.RequestException, ValueError) as exc:
            raise ServiceError("Could not validate current paper-account state.") from exc

        if order.side == TradeSide.SELL:
            held = Decimal("0")
            for position in positions:
                if isinstance(position, dict) and position.get("symbol") == order.ticker:
                    try:
                        held = Decimal(str(position.get("qty", "0")))
                    except InvalidOperation:
                        raise ServiceError("Alpaca returned an invalid position quantity.")
                    break
            if held < order.quantity:
                raise ServiceError("The paper account does not hold enough shares to sell that quantity.")
        else:
            try:
                price_data = self._twelvedata("price", {"symbol": order.ticker})
                price = Decimal(str(price_data["price"]))
                buying_power = Decimal(str(account["buying_power"]))
            except (ServiceError, KeyError, ValueError, InvalidOperation) as exc:
                raise ServiceError("Could not verify price and buying power; the order was blocked.") from exc
            if not price.is_finite() or price <= 0 or not buying_power.is_finite():
                raise ServiceError("Could not validate price or buying power; the order was blocked.")
            if price * order.quantity > buying_power:
                raise ServiceError("The order exceeds current paper-account buying power.")

        try:
            result = request_json(
                f"{ALPACA_PAPER_BASE_URL}/v2/orders",
                headers={**headers, "Content-Type": "application/json"},
                method="POST",
                json_body={
                    "symbol": order.ticker,
                    "qty": str(order.quantity),
                    "side": order.side.value,
                    "type": "market",
                    "time_in_force": "day",
                },
                retries=1,
            )
        except ServiceError:
            raise
        order_id = result.get("id")
        if not isinstance(order_id, str) or not order_id:
            raise ServiceError("Alpaca did not confirm an order ID.")
        return f"Paper order submitted: {order.side.value.upper()} {order.quantity} {order.ticker}; order ID {order_id}."

    def langchain_tools(self) -> list[Any]:
        """Create typed LangChain tool adapters without giving an LLM order authority."""
        try:
            from langchain_core.tools import tool
        except ImportError as exc:
            raise RuntimeError(
                "Install dependencies from requirements.txt to expose LangChain tools."
            ) from exc

        @tool
        def qloo_discover_brands(
            cultural_seed: str,
            seed_type: Literal["artist", "movie", "tv_show", "book", "brand"],
        ) -> str:
            """Resolve a cultural seed with Qloo and return related brands; affinity is not financial evidence."""
            result = self.discover_brands(cultural_seed, SeedType(seed_type))
            return result.model_dump_json()

        return [qloo_discover_brands]


def handle_user_query(
    query: str,
    app: TradingAgentApp | None = None,
) -> AgentResponse:
    """Classify, validate, and route one request using deterministic code."""

    instance = app or TradingAgentApp.from_env()
    return instance.handle_user_query(query)


def main() -> int:
    app = TradingAgentApp.from_env()
    missing = app.missing_configuration()
    if missing:
        print(
            "Setup note: some workflows need additional .env settings: "
            + ", ".join(missing)
        )
    print("Local trading research assistant. Type 'quit' to exit. No LLM selects or executes tools.")
    while True:
        try:
            query = input("\nRequest> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0
        if query.casefold() in {"quit", "exit"}:
            return 0
        response = handle_user_query(query, app)
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    sys.exit(main())
