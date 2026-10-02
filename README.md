# ValuationAgent Advance

一个以自由对话为入口的估值 Agent。模型负责理解、计划、检索、解释与迭代；有来源约束的事实核验和确定性金融工具负责正式计算。

## 当前重构状态

本分支采用**单一工作区 Agent**，不保留旧研究 API、旧 CLI 向导或计算后专用对话路由。计算前后的提问都使用同一工具循环、同一模型连接和同一消息历史。

- 自动模式默认生成估值草案；审阅模式先冻结方案，等待用户批准。
- 每次计算及重算均在执行前保存 ModelSpec 和来源快照。
- 计算历史用于审计与对比，不提供旧工作流切换、激活或分支入口。
- 正式估值缺少行业时明确阻断，不再静默使用参考模型；仅显式 demo 使用合成演示引擎。
- 网页支持原文下钻、可见任务计划、输入复核、风险检查和报告下载。
- 数据不足、模型未连接和计费失败均保留已有进度，不伪造数值结果。
- 重大选择以选项呈现，同时保留自由输入；选择只形成普通消息，不替代财务审批。
- 按模型规划多年取证：DCF 目标近十年、自动预测最低四年；相对估值目标近三年核对趋势，不套用 DCF 门槛。
- 原文、检索线索、已核验事实、待修复候选分别展示；单个检索目标耗尽不锁死其他年份和任务。
- LLM 批量读取带原文行号与表头的证据包，解释科目语义；程序逐项校验，不要求逐字段复制长引文。
- 支持官方披露、公开财务网页及已连接的 Tushare 多年三表快照；第三方数据通过交叉核对后保留 C 级草案限制，不把一致性当成官方核验。

**这是破坏性重构，不提供旧协议兼容。自动化回归通过不等于真实公司金融准确性或生产就绪验收。** 清理范围、验证结果及未完成项见 [一致性记录](docs/REFACTOR_CONFORMANCE.md)。

## 启动

在本目录使用正确的 Python 环境重新进行 editable 安装，避免命令仍指向相邻旧仓库：

```powershell
python -m pip install -e ".[documents,reports,dev]"
npm.cmd --prefix web ci
npm.cmd --prefix web run build
python -m valuationagent.cli.main serve
```

打开 http://127.0.0.1:8000，在页面连接模型和数据服务。密钥仅用于当前进程，不写入工作区数据库。安装只需首次或依赖变化时执行；日常激活环境后使用 `valuationagent` 或 `valuationagent serve`。改动前端源码后才需重新 build。

未配置访问凭证时仅允许本机连接。受保护部署须设置 `VALUATION_ACCESS_TOKEN`、显式主机白名单和 HTTPS 反向代理，详见[部署与验收边界](docs/DEPLOYMENT_READINESS.md)。访问凭证不是模型密钥；当前不是多租户系统。

CLI 使用同一工作区。激活环境后，在本目录直接启动：

```powershell
conda activate economic_agent
valuationagent

valuationagent --workspace workspace_实际ID
valuationagent --review
valuationagent serve
valuationagent replay 冻结复算包.json
```

首次启动在终端输入 Base URL、模型 ID 和隐藏的 API Key，并验证工具调用能力。对话内 `/model` 可重新配置，`/search` 配置 Tavily，`/market` 可选配置 Tushare，`/upload 文件路径` 添加附件，`/approve` 审阅并批准方案，`/exit` 退出。密钥不保存到磁盘；已有 `VALUATION_LLM_*` 环境变量也可使用。

CLI 保留品牌欢迎页、Markdown 对话卡片与窄屏适配，执行时展示当前工具、耗时、证据数量和任务计划。`/status` 显示流程工作台及年度覆盖，`/status --json` 才输出完整数据。方案可输入 `A`、`1` 或 `A 补充要求`，也可直接写自己的方案；Web 点击选项会填入输入框，修改后发送。

升级代码后请重启已有 Web/CLI 进程。旧格式工作区不会自动迁移；若提示 `WORKSPACE_STATE_INCOMPATIBLE`，原始记录仍在，请新建工作区验证当前 Agent，不要继续使用未重启的旧服务。

当前执行引擎标记为 `workspace-agent-2026-10-02.15`，可在网页、CLI `/status` 或 `/health` 核对。代码更新不会替换正在运行的旧进程。

## 文件工作区与交付

上传与检索下载文件统一使用 `list_files → inspect_file → read_file`：模型自行选择文本检索、原始文本行、PDF单页布局/阅读顺序、Excel矩形单元格，原文不会被覆盖。格式转换工具不判断财务含义；新增视图保留来源哈希和引用位置。

- CLI 保留欢迎页和工作台；新增 `/files`、`/read 文件ID`、`/artifacts`、`/export md|pdf|html|json`、`/save artifact_id`。
- 文件导出到启动目录的 `artifacts`，附带哈希清单，不覆盖不同内容的本地文件。Web 在对话中展示可下载的文件交付卡。
- Agent 可生成系统结果/缺口报告，也可写明确标注“未审阅”的研究笔记；笔记不能成为原始财务证据。没有计算结果就不生成虚构价格。
- 可选页图阅读：一次安装 `python -m pip install -e ".[vision]"`；CLI `/vision` 或 Web 模型连接里的图片选项明确开启后，才向**当前模型**发送选定页图。接口须支持图片，不自动切换模型、不额外启动第二模型。
- 默认仍是文本模式。视觉读数不等于机器核验；纯扫描件尚不能仅凭模型自评自动进入计算。当前没有任意 Python/shell 执行工具。

日常启动仍只需激活环境后运行 `valuationagent` 或 `valuationagent serve`，不需每次安装。部署安全边界、限额与验收见 [文件工作区说明](docs/FILE_WORKSPACE.md)。

文档理解由主 LLM 负责，不再注册旧表格契约或调用旧候选提取工具。当前唯一主链是 `extract_observations → prepare_observation_review → review_observations`：模型解释原文片段、期间、币种、口径和会计含义，再对照原文逐维复核；程序核验定位、完整数值和换算、来源及模型约束，不使用模型自评置信分放行。复核属于同一模型的第二阶段检查，不是独立审计。`inspect_extraction_progress` 提供已尝试视图及替代策略，避免把解析失败当成缺资料反复下载。Web 与 CLI 同步展示原文定位、语义复核、模型准入和方法准备度；日常启动仍为 `valuationagent` 或 `valuationagent serve`。

`valuationagent` 和 `valuationagent chat` 调用同一入口；直接启动不是恢复旧向导。旧 interactive / wizard / research / run --chat、确认卡协议和计算 runner 的第二套 LLM 审核路径已删除。

`.8` 新增通用 `pdf_geometry` 坐标解码视图，处理普通 PDF 文本层把相邻数值列拼接的问题，不包含公司专用模板。LLM 可直接引用工具返回的行号，不必重新抄写空格；整行引用中的金额由模型选择、程序验证完整数字并保存精确位置。同值多次出现仍须明确选择。模型一次发出多项工具调用时按顺序执行，错误反馈给出原文位置和修复方向，不靠增加无限重试掩盖失败。

本次新增依赖后需在已激活环境中执行一次 `python -m pip install -e ".[documents,reports,dev]"`，随后重启 Web/CLI。以后日常启动不需要重复安装。

远程自建模型可使用同一OpenAI兼容连接；本次千问的服务器检查结果、SSH隧道、`json_content` 工具格式及测试边界见 [远程模型接入](docs/REMOTE_MODEL.md)。不需要改变日常 CLI 启动命令。

自建SGLang/vLLM若支持请求级思考控制，可显式选择`chat_template`推理协议；思考模式和采样温度均可在Web模型配置或CLI `/model`内调整。JSON网关每次一个动作，原生工具协议仍支持有界批量；两种连接走同一文件、证据和估值后端，不放宽财务准入。

## 删除历史对话

网页右侧每条历史对话提供删除按钮，确认后永久删除该工作区的消息、证据记录、任务内报告及独占计算记录。删除当前对话会回到新对话状态，删除其他对话不打断当前选择。排队或运行中的任务、其他工作区共享的计算记录会阻止删除；状态变更时需刷新后重新确认。上传/下载的原始文件可能被共享，因此保留文件库字节；已导出到电脑的文件也不删除。删除不等于全盘安全擦除。

## 数据与接口边界

新数据库是 `var/workspace-agent.sqlite3`。原 `valuationagent.sqlite3` 不删除、不自动迁移，也不混入新工作区。上传原文件仍在 runtime 目录中，不通过静态目录公开。

- `POST /api/workspaces`：创建任务，默认 automatic。
- `POST /api/workspaces/{id}/messages`：幂等排队执行一轮 Agent。
- `GET /api/workspaces/{id}`：只读快照，不触发计算或生成版本。
- `DELETE /api/workspaces/{id}?revision=<当前修订>`：用户确认后删除历史；返回204，运行中/修订冲突返回409，不存在返回404。
- `POST /api/workspaces/{id}/prevaluation-review`：生成有哈希的输入复核包。
- `POST /api/workspaces/{id}/approvals`：批准当前方案。
- `POST /api/workspaces/{id}/versions`：用户明确修改参数并创建新计算。
- `GET /api/workspaces/{id}/sources/{file_id}`：范围内来源分页或指定 block_id 下钻。
- `/api/files`、`/api/model-sessions`：通用文件和模型连接。
- `GET /api/runs/...`：确定性计算、事件与报告的只读查询。
- 旧 `/api/research-sessions`、run 对话和历史版本激活/分支接口不再注册。

## 验证

```powershell
python -m pytest -q
npm.cmd --prefix web test
npm.cmd --prefix web run test:components
npm.cmd --prefix web run lint
npm.cmd --prefix web run build
```

已删除退役入口和确认卡专属测试，并将仍有效的证据、财务、语言、异常恢复和接口约束迁移到当前契约；不通过恢复旧代码满足旧界面断言。

预算停止会保存续做检查点，但不会在后台自动无限运行。下一轮重新检查未覆盖年度与已有候选，重复查询使用缓存；某一目标限制仍保留，以避免无进展付费检索。

估值执行失败或检查点停止、且尚无数值结果时，保存可下载的“执行中断与输入缺口报告”。它明确标记估值未完成，不冒充数值估值报告，不计入生产验收成功。状态工具在证据未改变且重复读取后暂时隐藏，改写计划文字不能重新开放；实际证据或任务变化后恢复。

步骤预算按最多两个 40 次调用窗口执行：只有第一窗口取得新的原文快照或无警告事实、且仍有时间预算时才续行；总时间与检索配额不重置。批量工具用于减少无效往返，不是承诺任意公司一次完成。第三方数据缺日期、单位、口径或存在冲突仍不能入模；扫描 PDF 的 OCR 和金融子公司专项权益桥接尚未实现。

`.4` 修复批量工具将未带行号的有效引文错误替换为第一行的问题：每项使用连续 `quote` 或原始行号，二者同时提供时必须一致。模型在预算将尽时用结构化 `checkpoint` 保存未完成步骤，也能触发有实质进展的同轮续做；整批失败现在参与无进展保护，不再伪装成成功。

显式联网冒烟测试为 `scripts/live_workspace_smoke.py`，从环境读取 `DEEPSEEK_API_KEY` 与 `TAVILY_API_KEY`，最多调用模型 12 次。它只验证模型协议、对话和检索，不代表真实公司端到端估值通过。

更多设计和边界见 [当前架构](docs/ADVANCE_ARCHITECTURE.md)。

跨公司真实取证测试使用 `scripts/live_document_acceptance.py --company "公司名 证券代码" --directory tmp/新的测试目录`，使用 `VALUATION_LLM_BASE_URL`、`VALUATION_LLM_MODEL`、`VALUATION_LLM_API_KEY` 及 `TAVILY_API_KEY` 环境配置，也可指定 `--base-url`、`--model`、`--tool-call-format`；默认最多80次模型调用、2轮任务，每轮最多1800秒，会消耗接口额度或服务器算力。`--upload-only --file 路径` 支持仅附件测试。结果保存实际事实状态、计算是否发生及检查点，不把候选数当成估值成功。`scripts/replay_observation_calls.py <数据目录> --session <会话ID>` 将日志复制到临时数据库离线回放原始提取调用，不修改用户数据库、不调用付费接口；它不替代 LLM 和金融准确性验收。本轮结果与未完成项见 [真实取证验收记录](docs/ACCEPTANCE_2026-10-01.md)。

真实估值脚本现在以非零退出码明确表示未完成：必须有非演示的成功数值、同一计算版本的数值报告、报告完整性及冻结输入离线复算通过。缺口报告、研究笔记、旧版本报告、仅有 result 对象均不能通过。即使这些门槛通过，也不代表原始数据已获独立语义审计。

`scripts/audit_observations.py --database <测试数据库> --case <独立核对案例JSON>` 只读比较原文哈希、页码、字段、期间、数值、单位及口径；多会话数据库须明确 `--session-id`。初始案例在 `tests/fixtures/source_audits/`，只覆盖已单独查看页图的少量字段，不作为 Agent 取数模板或输入答案。最新失败与修正记录见 [2026-10-02验收](docs/ACCEPTANCE_2026-10-02.md)。
