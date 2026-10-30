<<<<<<< HEAD
# Local culture-informed trading research assistant

This project replaces the notebook's Colab-specific secret lookup and
top-level API examples with a local Python application. Request classification
and order authorization are deterministic Python code; no LLM chooses tools or
submits an order.

## VS Code setup

From this folder, open a PowerShell terminal and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` in VS Code and add provider keys locally. Never paste key values
into Python files, notebook output, chat messages, or source control. The local
`.gitignore` excludes `.env`, virtual environments, and generated Python files.
Restart the terminal after changing `.env`.

Run the interactive application:

```powershell
.\.venv\Scripts\python.exe trading_agent.py
```

Run the offline test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Only the key for the selected workflow is required:

| Workflow | Settings |
| --- | --- |
| Cultural brand discovery | `QLOO_API_KEY` |
| Stock analysis | `TWELVEDATA_API_KEY`, `FINNHUB_API_KEY` |
| Portfolio status | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` |
| Paper-order preview/confirmation | `TWELVEDATA_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` |

No Gemini key is needed. The app does not send prompts, headlines, API keys, or
account data to a language model.

## Supported requests

- `Which public brands are associated with fans of Taylor Swift?`
- `Analyze NVDA`
- `What is my paper portfolio balance?`
- `Buy one share of AAPL`
- `Yes, confirm paper order: buy 1 share of AAPL`

The first order request only creates an in-memory preview. A separate
confirmation must repeat the same side, ticker, and quantity. Confirmation
expires if the app exits. The application checks the current Alpaca paper
account, available buying power for buys, and shares held for sells. It caps
orders at five shares. Requests that try to bypass safeguards or trade all
positions are refused.

## Data and limitations

- Qloo uses the hackathon `/search` endpoint and then `/v2/insights` with
  `filter.type=urn:entity:brand`. Affinity is aggregate cultural relatedness,
  not sentiment, price prediction, expected return, or a trading signal.
- Brand-to-ticker mapping is deliberately limited to exact, curated names.
  Unmapped brands are shown without a ticker; the app never guesses.
- Stock analysis uses Twelve Data price/history/indicator endpoints and Finnhub
  company news. Provider availability, plan entitlements, delays, and returned
  observations may vary. Results are descriptive, not investment advice.
- Alpaca calls are fixed to `https://paper-api.alpaca.markets`. The application
  contains no live-trading endpoint. A confirmation does not guarantee an
  order will be accepted or filled.
- Provider HTTP requests use bounded timeouts and retries for safe GET
  requests. Order POSTs are not retried, to avoid duplicate orders.

The tests mock all HTTP responses. The test suite does not establish that
provider credentials, live provider schemas, data entitlements, or actual
brokerage behavior work in a particular account.

## Original notebook

`TradingAgent2Code.ipynb` is retained as a lightweight VS Code entry point. The
maintained application and safety checks live in `trading_agent.py`; use the
terminal command above for the full interactive workflow. The old notebook's
Colab-install cells, direct example API calls, and saved API outputs are
removed.
=======
# TradingAgent
>>>>>>> 1a3301b6b4018cac47ddb0136d468aed59e66016
