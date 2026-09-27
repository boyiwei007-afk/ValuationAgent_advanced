# Third-party software, services, and data

This file records direct third-party dependencies and external services used by the submitted build. Versions below match the tested `economic_agent` environment and `web/package-lock.json` as of 2026-09-25. Transitive Python versions are pinned in `requirements.lock`; transitive JavaScript versions are pinned in `web/package-lock.json`. Each installed distribution/package retains its own license text and notices.

## Python runtime

| Component | Tested version | License | Use in this project | Source |
| --- | ---: | --- | --- | --- |
| FastAPI | 0.141.1 | MIT | HTTP API | https://github.com/fastapi/fastapi |
| HTTPX | 0.28.1 | BSD-3-Clause | Model, search, filing, and market HTTP clients | https://github.com/encode/httpx |
| LangGraph | 1.2.11 | MIT | Deterministic workflow orchestration | https://github.com/langchain-ai/langgraph |
| Pydantic | 2.13.5 | MIT | Typed input/output contracts and validation | https://github.com/pydantic/pydantic |
| python-multipart | 0.0.32 | Apache-2.0 | Uploaded file handling | https://github.com/Kludex/python-multipart |
| Questionary | 2.1.1 | MIT | Interactive CLI prompts | https://github.com/tmbo/questionary |
| Rich | 14.3.4 | MIT | CLI rendering | https://github.com/Textualize/rich |
| Typer | 0.27.2 | MIT | CLI commands | https://github.com/fastapi/typer |
| Uvicorn | 0.52.4 | BSD-3-Clause | ASGI server | https://github.com/encode/uvicorn |
| openpyxl | 3.1.5 | MIT | XLSX report generation and workbook parsing | https://foss.heptapod.net/openpyxl/openpyxl |
| pypdf | 6.18.1 | BSD-3-Clause | PDF text extraction and report tests | https://github.com/py-pdf/pypdf |
| python-docx | 1.2.0 | MIT | DOCX parsing | https://github.com/python-openxml/python-docx |
| ReportLab | 5.0.1 | BSD-3-Clause | PDF report generation | https://www.reportlab.com/opensource/ |
| pytest | 9.1.1 | MIT | Automated regression tests | https://github.com/pytest-dev/pytest |

## Web application

| Component | Tested version | License | Use in this project | Source |
| --- | ---: | --- | --- | --- |
| React / React DOM | 19.3.0 | MIT | Web user interface | https://github.com/facebook/react |
| react-markdown | 10.1.0 | MIT | Safe Markdown rendering | https://github.com/remarkjs/react-markdown |
| remark-gfm | 4.0.1 | MIT | GitHub-flavored Markdown support | https://github.com/remarkjs/remark-gfm |
| Vite | 8.3.0 | MIT | Frontend build and development server | https://github.com/vitejs/vite |
| @vitejs/plugin-react | 6.1.1 | MIT | React build integration | https://github.com/vitejs/vite-plugin-react |
| happy-dom | 20.14.5 | MIT | Component test DOM | https://github.com/capricorn86/happy-dom |
| Oxlint | 1.82.0 | MIT | Frontend linting | https://github.com/oxc-project/oxc |
| @types/react / @types/react-dom | 19.3.0 | MIT | Type declarations used during development | https://github.com/DefinitelyTyped/DefinitelyTyped |

## External models and data services

These services are called over HTTPS and are not redistributed. Their commercial terms, privacy policies, rate limits, and data licenses apply separately.

| Service/data | Version or interface | Use in this project | Source / terms |
| --- | --- | --- | --- |
| DeepSeek API | OpenAI-compatible Chat Completions; tested with `deepseek-flash` and `deepseek-v4-pro` | Core LLM reasoning, tool selection, document extraction, and interaction | https://api-docs.deepseek.com/ |
| Tavily Search API | HTTP API, provider adapter `tavily` | Public-web research and supplementary source discovery | https://docs.tavily.com/ |
| 巨潮资讯网 (CNINFO) | Public announcement catalogue and issuer PDF filings | Official A-share annual-report discovery and source documents | https://www.cninfo.com.cn/ |
| Tushare Pro | HTTP API, optional runtime connection | A-share structured financial/market data and peer selection | https://tushare.pro/ |

Credentials are supplied at runtime, kept in process memory, and are not included in source, logs, exported reports, or this notice.

## Real-disclosure acceptance fixtures (2026-09-27)

Official issuer 2024 annual reports were downloaded from CNINFO for local software validation: Yili (600887), Midea (000333), CATL (300750), with the earlier Kweichow Moutai (600519) audit retained as regression evidence. These are third-party disclosures, not project-authored data and not licensed as this project's source code. Annual-report PDFs are not redistributed in release archives. Only source URLs, publication dates, hashes, acceptance summaries and minimal numerical golden rows are included; downloading and reuse remain subject to issuer/CNINFO terms. Exact sources and hashes are recorded in `docs/真实公司数据验收_2026-09-27.md` and the opt-in acceptance script's source manifests.

## Referenced projects and protocols

- TradingAgents: CLI workflow/layout inspiration only; no upstream source is vendored. Source: https://github.com/TauricResearch/TradingAgents
- OpenAI function calling: protocol/design reference for compatible tool calls; no OpenAI SDK source is vendored. Source: https://developers.openai.com/api/docs/guides/function-calling

## Domain parameter data

`src/valuationagent/finance/data/industry_parameters_v2.json` is a normalized snapshot attributed in the file to `行业参数数据库_v2_全行业(1).xlsx` and records the supplied source SHA-256. The original workbook is not present in this repository. The snapshot therefore reports partial metadata and does not claim a complete observation period, sample size, owner, update frequency, or confidence interval. The food/beverage and liquor route is an explicit quality-C fallback to an existing generic row, not a claimed industry statistic.

