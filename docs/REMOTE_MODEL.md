# 远程千问接入与检查记录

## 最新连接状态

2026-10-07 只读复核：主进程 PID `4149509`，路径 `/gongjifs/suzhenfeng/xry/synthetic_data/models/Qwen3.6-35B-A3B`，监听 `127.0.0.1:8000`，真实模型 ID `qwen36-teacher`，上下文上限 **32768**。启动参数有 `--reasoning-parser qwen3`，无 `--tool-call-parser`。仅恢复本地 SSH 隧道，没有重启、修改或另起远端模型。当天开始时本地 Web `8000/health` 无响应，不能沿用 10 月 4 日的 `.37` 状态。

当前 `.75` 隔离实测沿用 `.63` 配置：`qwen3_coder`、`chat_template`、`thinking=enabled`、temperature=1、top_p=0.95、presence_penalty=1.5、top_k=20，输出预算仍为8192/16384，请求超时600秒。Web选择「Qwen 原生标签」并点击「应用 Qwen3.6 通用采样建议」；CLI `/model` 同样可显式应用。新配置不覆盖已有连接，需在新后端重新连接。

最新协议修复不依赖取消预算：文件读取联合 schema 已兼容；主循环允许正常最终正文经过完成条件校验，控制/复核阶段仍须工具调用。`finish_reason=stop` 但只有私有思考的响应记为 `TOOL_NO_DECISION`，不是 `LLM_REASONING_LIMIT`。诊断事件保留用量及解析原因，不记录私有思考正文。`.67`单步输出扩展不再传播到所有后续工具步骤。重启本地后端后检查 `/health` 的 `agent_version` 是否为 `.75`；仅重建前端不会更新后台进程。

采样参考 [Qwen3.6 官方说明](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md)。同输入、4096上限的小对照中，coding配置耗尽无工具，通用配置889 tokens产生工具；仅一次组合对照，不能证明单个参数因果关系或任意公司成功率。另已复现完整 `update_task` 的布尔参数 `True` 被旧客户端拒绝；当前参考 [SGLang解析器](https://github.com/sgl-project/sglang/blob/v0.5.10/python/sglang/srt/function_call/qwen3_coder_detector.py)做类型限定兼容，不用 `eval`、不猜布尔值、不执行截断JSON或私有思考。

以下 `.49` 及更早内容均为历史诊断，不代表当前运行状态。最新验收与未通过项目见 `DEPLOYMENT_READINESS.md`。

`.49`最新版的真实简单PE基线再次通过（开启思考，145.49秒，数值报告和复算通过），但完整算例未通过。关闭思考的原生请求对照仍反复漏填单位依据，且观察到截断输出绝大多数为空白；不推荐把关闭思考或`native_json`当作通用成功配置。详见`DEPLOYMENT_READINESS.md`最新收据与边界。保持原用户模型设置，没有重启本地Web或远端模型。

2026-10-04只读复核：实际启动PID4149509，模型路径`/gongjifs/suzhenfeng/xry/synthetic_data/models/Qwen3.6-35B-A3B`，SGLang安装版本`0.5.10.post1`，监听`127.0.0.1:8000`，`--context-length 32768`、`--reasoning-parser qwen3`。没有重启、修改服务或另起模型实例。GPU共享推理并发不是多模型实例的证据；性能和占用应以实际时刻为准。

本机Web的`http://127.0.0.1:8000/health`仍是`.37`（PID98168），源码已到`.49`，远端隧道是本机18000；不要混淆这两个8000端口。新代码仅在隔离验收进程加载，现有Web需结束任务后手动重启，并重新建立模型连接才能采用新预算。无需每次重装依赖。

扩大预算的`.46`复测实际失败：11118输入tokens加20480输出tokens（31598总量）仍未产生工具决策，记录`user-scenario-reasoning-20261004-06`。不要无限增大或反复点继续。原生工具请求探针在开启思考时返回文本`[{"name":"connection_check","parameters":{}}]`而非标准tool_calls，新增显式`native_json`适配可通过连接检查，但完整任务尚未验收。该模式未改变默认连接，也未修改远端启动参数。

`.45`真实开启思考的完整算例再次耗尽：最后输入15581、输出16384，合计31965，已逼近服务公开的32768容量。取消`max_tokens`不会取消服务器上下文限制，还可能采用更小的服务默认值；本项目保留可配置上限、取消检查、时间限制和截断JSON拒绝。`.46`另测12288初始／20480上限／600秒请求超时，仍保持原模型与开启思考；测试条件不同于默认Web，不把隔离条件下的结果冒充默认配置验收。

`.44`开启思考的简单用户PE基线真实通过，含工具录入、正式计算、可下载数值报告和离线复算，记录`output/acceptance/user-pe-reasoning-20261004-01/result.json`，耗时84.72秒。该次只执行一个基线，不代表所有方法或自主研究可用。完整DCF多方法用例尚未通过；关闭思考的对照仍会出现字段引用错误，不推荐把关闭思考当作免验收方案。

`.44`完整开启思考的复测`user-scenario-reasoning-20261004-04`结束于21次调用、872秒，已保存23项输入，仍没有数值估值。扩展预算和上下文压缩恢复过一次工具调用，但最后输入14771 tokens加输出16384 tokens后仍未产生决策；最后一次合计31155已接近实际32768容量，不能靠取消客户端上限获得无限空间。中途有单位引用和预测参数／依据错误，需要修复工具交互而不是继续无界重试。该次没有修改模型、思考模式或远端服务。

2026-10-04 服务恢复后，本地隧道的 `/v1/models` 已再次返回 `qwen36-teacher`，上下文容量为32768。实际推理可用：旧版入口在启用思考时以1800输出tokens耗尽，未生成工具；关闭思考后的公牛集团测试能取回35项供应商输入，但反复提交港股可比导致 `AGENT_NO_PROGRESS`，仍无估值报告。**连接恢复不等于自主估值已通过。**

当前修改将共享客户端的初始输出预算设为8192，扩展上限16384，Web与CLI均可配置。思考耗尽且本次未执行工具时，工具循环按倍增预算有限重试，最多两次，保留模型、思考模式及全部用户约束；达到上限、取消或时间预算耗尽则停止。截断JSON绝不执行。输入和输出仍共享远端总上下文，调大输出不能超过模型可用容量；不是无限预算或无限重试。默认请求超时180秒，可配置至600秒。

使用用户完整合成测试原话、开启思考的真实回归已通过 `set_turn_plan` 与 `update_task`，禁止联网约束保持；完整计算、报告和回放以 `output/acceptance/user-scenario-reasoning-20261004-01/acceptance.json` 的最终检查为准，不能把入口成功当作全部通过。

该次最终因扩展后的上下文不足而失败；`-02`缩小文件工具目录后，真实输入从约19736降到14968 tokens，8192耗尽后的同输入16384重试成功执行了`update_task`，但下一调用16384仍只思考，最终失败。`.40`进一步精简离线提示，`-03`首个主循环输入10498 tokens，最终仍因字段修复后再次思考耗尽而失败。没有自动关闭用户连接的思考模式或修改远端服务；disabled仅用于明确记录的隔离对照。

部分关闭思考的完整工具输出也达到8192上限，说明预算错误不能一概当作“思考太长”。现在记录输出字符和空白字符计数（不保存私有思考），收窄无关工具和重复的引文参数。参考[SGLang官方结构化输出说明](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/structured_outputs.mdx)；尚未证明某个上游issue就是当前部署的根因，不盲目添加未验证的服务端参数。

参考[Qwen官方思考预算示例](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)，受控思考与最终输出是不同预算问题。其示例依赖tokenizer和续写接口，不应将未验证的供应商参数直接当成通用修复。SGLang仓库也有[Qwen3.6预算未生效的报告](https://github.com/sgl-project/sglang/issues/25536)，涉及特定版本与模型；本项目尚未验证当前部署属于同一缺陷，不以该issue代替本地诊断。

## 恢复前故障记录（不是当前状态）

2026-10-04，Agent版本`.37`通过既有SSH认证进行只读复查，未输入或保存密码，结果如下：

- `ps -ef`中未找到SGLang、vLLM或launch_server模型进程；没有可报告的当前模型PID，也没有观察到多个冲突模型服务。
- 服务器未提供`ss`，使用`netstat -lntp`核对，8000／8001均无监听。远端自身访问`http://127.0.0.1:8000/v1/models`返回连接拒绝；本地18000此前连接重置不能再简单解释为隧道故障或模型加载。
- 四张NVIDIA GeForce RTX 5090各使用4 MiB／32607 MiB显存，没有观察到模型权重占用。
- 最新日志`/gongjifs/suzhenfeng/xry/synthetic_data/logs/qwen36_sglang_shared_20261001_153615.log`末尾显示Gloo广播通信`Connection reset by peer`，随后在日志时间`2026-10-03 13:59:46`记录`SIGQUIT received`。日志时区未核对，且这些末尾信息不足以确定最初的子进程失败原因，不能推断为显存不足。

当前服务不健康，无法从当前`/v1/models`确认运行模型路径或真实模型ID。下面的PID、模型名称、监听地址和成功推理是历史观察，不代表当前状态。没有停止、重启、启动第二个模型或修改远端文件；恢复服务需要操作者处理或另行授权。恢复后须重新核对监听、模型ID并完成最小推理，再继续真实自主估值验收，不能只看本地端口监听。`.37`尚未完成该验收。

## 历史只读检查记录

通过用户指定 SSH 地址只读检查，未停止、重启或修改远端进程和文件，也未启动第二个模型实例。没有收集或保存登录密码。

- 主进程 PID：`1831174`，`python -m sglang.launch_server`。
- 实际模型路径：`/gongjifs/suzhenfeng/xry/synthetic_data/models/Qwen3.6-35B-A3B`。
- 实际监听：`127.0.0.1:8000`。服务器没有 `ss`，改用只读 `netstat -lntp`；未发现8001监听。
- `/v1/models` 返回真实 ID：`qwen36-teacher`，`max_model_len=32768`。
- 加载完成后，`/v1/models` 和 `/health` 均返回200；首次检查时尚在加载，并非服务已失败。
- GPU：4张 RTX 5090，每卡总显存32607MiB；检查时使用量依次为25591、24979、24979、24980MiB。
- 2026-10-01 05:19 UTC只读复查：主进程仍为1831174，仍只监听127.0.0.1:8000，`/health`返回200。稍前GPU使用量为25913、25301、25301、25302MiB；这些是共享服务的瞬时值，不是本次测试的独占占用。
- TP工作进程：1831428—1831431。它们是同一服务的4卡张量并行工作进程，不是4个重复模型服务。在已检查的进程和端口中未发现冲突实例。
- 当前日志：`qwen36_sglang_shared_20260930_184107.log`。历史日志中的旧故障不能当作此次部署故障。

**用户手动开启持久隧道后已验证建立。** 本地`127.0.0.1:18000/v1/models`可访问真实模型，普通推理与工具探针均已通过。隧道由用户窗口维护，本次没有停止该隧道或修改远端服务。

## 建立本地隧道

2026-10-03 本地诊断发现：同一 `/v1/models` 请求使用系统代理时返回 502，直连回环地址时返回 200。当前客户端对 `localhost`、IPv4/IPv6 回环地址禁用环境和系统代理，外部模型地址仍保留原代理设置；不修改系统代理、不重启远端服务。模型列表可访问仍不等于推理和工具协议可用，需分别测试。

在一个新的 PowerShell 窗口运行并保持窗口打开；密码仅在 SSH 提示时手动输入，不放入命令、文件或脚本。若出现主机指纹确认，请向服务器管理方核实，不要关闭主机密钥检查。

```powershell
ssh -N -L 18000:127.0.0.1:8000 -p 40013 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 root@xsj1.550w.link
```

在运行 Agent 的另一个 PowerShell 窗口设置本机请求绕过代理并验证：

```powershell
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = "127.0.0.1,localhost"
curl.exe --noproxy "*" http://127.0.0.1:18000/v1/models
```

应得到模型列表而非网页或代理错误。以这次返回的模型 ID 为准；如果本地18000已被占用，先辨明占用者，不停止未知进程或再启动冲突隧道。关闭隧道窗口会断开访问；`ExitOnForwardFailure` 只证明转发端口建立，不证明模型可推理。

## Agent 配置

Web“连接核心推理模型→自定义接口”，或 CLI `/model` 中填写：

| 配置 | 本次部署的值 |
| --- | --- |
| Base URL | `http://127.0.0.1:18000/v1` |
| 模型 ID | `qwen36-teacher`（再次核对本地 `/v1/models`） |
| API Key | `EMPTY`，仅限当前未启用API鉴权且经SSH访问的服务 |
| 工具调用格式 | 本次 Qwen 部署显式选 `qwen3_coder`，不是通用默认值 |
| 推理参数协议 | `chat_template`（SGLang / vLLM） |
| 思考模式 | `enabled`；`disabled`亦可请求，但单独关闭思考未解决完整任务的循环问题 |
| 采样 | temperature=1、top_p=0.95、presence_penalty=1.5、top_k=20；仍需任务验收 |
| 初始输出预算 / 扩展上限 | 8192 / 16384 tokens，含思考与工具输出 |
| 单次请求超时 | 默认180秒；慢速远端可显式配置300秒 |

历史原生工具探针曾返回 `content='[{"name":"connection_check","parameters":{}}]'`、`tool_calls=null`，因此仅替换URL不够。当前`json_content`显式使用JSON动作协议：请求中提供完整工具目录，结构化解码将工具名绑定到对应参数schema，不要求服务端原生工具解析器；历史调用/结果转换成一致的JSON文本协议，来源仍是不可信数据。每次仅一个动作，收到真实结果后再决策，避免批量猜参数和多份正文淹没上下文。完整回复转回应用标准工具消息，参数再次由原始Pydantic工具契约严格校验，才允许执行。自然语言、代码块、未知工具、重复键、非有限数值、截断或超额调用全部拒绝。

此前完整生成schema遇到SGLang不支持的正则前瞻／后顾，临时只约束外壳后又出现必填参数遗漏。现行协议按工具隔离定义引用，仅从生成schema移除不兼容的正则前瞻／后顾；**工具目录中的完整约束和本地原始校验保留**，没有放宽金额、字段或财务准入。标准`native`模式不变。`chat_template`协议显式把思考开关转换为请求级`chat_template_kwargs.enable_thinking`；不按模型名字猜能力，不修改服务启动参数。协议有效不代表模型能够完成自主研究。

环境配置也支持：

```powershell
$env:VALUATION_LLM_BASE_URL = "http://127.0.0.1:18000/v1"
$env:VALUATION_LLM_MODEL = "qwen36-teacher"
$env:VALUATION_LLM_API_KEY = "EMPTY"
$env:VALUATION_LLM_TOOL_CALL_FORMAT = "qwen3_coder"
$env:VALUATION_LLM_REASONING_PROTOCOL = "chat_template"
$env:VALUATION_LLM_THINKING = "enabled"
$env:VALUATION_LLM_TEMPERATURE = "1"
$env:VALUATION_LLM_TOP_P = "0.95"
$env:VALUATION_LLM_PRESENCE_PENALTY = "1.5"
$env:VALUATION_LLM_TOP_K = "20"
$env:VALUATION_LLM_OUTPUT_TOKEN_BUDGET = "8192"
$env:VALUATION_LLM_MAX_OUTPUT_TOKENS = "16384"
$env:VALUATION_LLM_TIMEOUT_SECONDS = "300"
valuationagent
```

环境变量仅对当前窗口及其新进程生效，不会改变已运行 Web 服务里的旧连接。新选项需要运行更新后的本地代码；本次没有替用户重启已有服务。

## Python 普通远程推理

安装 OpenAI Python 客户端后，隧道健康时可使用以下例子。这里是普通聊天，不是估值验收；Agent 本身使用现有 HTTP 客户端，无需新增 OpenAI SDK 依赖。

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18000/v1", api_key="EMPTY")
models = client.models.list().data
model_id = "qwen36-teacher"
if model_id not in {item.id for item in models}:
    raise RuntimeError("模型 ID 已变化，请检查 /v1/models")
response = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "请简短解释企业价值与股权价值的区别。"}],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

## 实测边界与后续

2026-10-03至10-04最新连接检查：本地`127.0.0.1:18000/v1/models`连接重置，不能依据端口监听宣称远程服务健康。备用DeepSeek凭证读取模型目录返回200，但最小推理请求返回402；目录成功不等于推理可用。未重启隧道、远端服务或启动第二个模型。`.36`的输入／推导专项测试和真实API机械探针不调用LLM，不能替代连接恢复后的新会话自主估值验收。后文为较早模型可访问时的真实测试历史。

本地回归828项Python测试通过（314.68秒），最终日志脱敏调整另有30项相关测试通过；Web API 6项、组件检查、lint、build及diff空白检查通过。测试数量不代表金融准确率。

2026-10-01持久隧道实测：`tmp/qwen-tunnel-reading-v6`在10次模型调用、约312秒内完成瑞芯微2023年年报的2021—2023收入/归母净利润提取。6项通过原文绑定、同一LLM复核和字段准入，零待修复项；六个数值逐项对照原文一致。没有为该公司编写提取规则，也没有向模型预填这些数值。它是附件读取专项，不是公司估值验收。

`tmp/qwen-free-chat-v1`通过统一工作区真实调用解释PE/DCF，未检索、未建立估值运行。此前多轮失败也保留在隔离测试目录，包括输出截断、上下文超限及重复更新；没有改用户已有工作区。

在附件测试工作区又真实生成了研究笔记和Markdown状态报告，并完成回读；文件已导出到`tmp/qwen-tunnel-reading-v6/exported/`。状态报告标记`insufficient_data`、`numeric_result_available=false`，不是正式数值估值。生成/回读能力已验证，估值完整性不能由“有报告文件”代替。

完整公司检索→数值估值→报告与单项读取是不同验收层次。苏泊尔早期完整测试尚未产出数值；不能把连接成功或6个数值正确说成任意公司自动估值成功。最新评测记录以各隔离目录的`acceptance.json`为准。

最终单步JSON协议测试`tmp/qwen-wuliangye-pe-v2`：80次模型调用、约341秒，成功检索/下载官方年报，7条记录通过字段准入（其中一个重复字段，合计6个不同年度/科目），涉及2021—2024收入/归母净利润；7条仍待修复。尚未取得匹配最新基期、有效股数及可比倍数的完整输入，预算用尽时仍在翻页，**没有执行数值估值，也没有生成该公司的数值报告**。苏泊尔多轮测试也未闭环；对无进展的隔离测试曾通过本地工作区取消接口停止，没有停止远端模型。结果不能解释为这些公司的资料不存在。

剩余主要问题是长任务的目标保持、已读资料到规范字段的映射及补缺策略，不能只靠扩大模型调用预算解决。字段约束拒绝错误币种/维度/股数口径是必要保护，不会为了演示“出数”而关闭。当前适合继续研发和受监督试用，不具备任意范围内公司无人值守自动估值的验收结论。

开启持久隧道后，用通用 `scripts/live_document_acceptance.py` 验证，不再写死DeepSeek端点：

```powershell
python scripts/live_document_acceptance.py --company "待测公司及证券代码" --directory tmp/新的测试目录 --calls 80 --turns 2
```

搜索凭证通过 `TAVILY_API_KEY` 环境变量传入，不写进脚本。`--upload-only --file 文件路径 --objective "读取并提取指定字段"` 可先独立测试用户附件，不调用搜索。目录必须全新。模型上下文上限为32768，长文、多轮工具及输出预算还需在真实连续任务中验证；当前不能承诺任意公司自动成功。

`--resume`只允许继续带有本脚本`acceptance.json`、公司名称一致且仅含记录中一个工作区的评测目录，不能改变上传/联网权限。续测另存`acceptance-resume-*.json`，不覆盖初测记录，不重置搜索历史。已取消的工作区不可继续；另建评测目录。调用日志只记录token计数、结束原因和内容长度，不输出思维链或凭证；`valuation_and_report_complete`要求真实计算结果和带数值状态的报告同时存在。

服务端工具解析机制参考 [SGLang 官方文档](https://docs.sglang.io/docs/advanced_features/tool_parser)，请求级思考控制参考 [Qwen3.6 官方部署说明](https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx)。本轮没有按文档重启远程服务，仅在本地适配实测响应。

## 后续迭代 `.11`

该历史阶段曾在服务端约束嵌套参数；`.14`已删除这部分解码约束，原因和现行策略见下一节。新增PDF全文定位、动态工作状态、隔离原文复核和冻结基准敏感性报告。实际进展与未通过的完整估值测试见[本轮验收记录](ACCEPTANCE_QWEN_2026-10-01.md)。不能将连通、提取或合成基准交互通过等同于任意公司的真实估值通过。

## `.14` 参数生成与执行校验分离

真实对照测试使用同一模型、提示、温度及思考设置：完整嵌套Schema时，模型从valuation_date开始输出，漏掉此前定义的company/ticker；只约束动作外壳时，这些参数均完整返回。XGrammar默认按Schema属性声明顺序生成，与该现象吻合，但此测试不是所有后端的行为证明。[XGrammar官方接口说明](https://xgrammar.mlc.ai/docs/latest/api/python/grammar.html)

现行JSON协议只约束单项数组、已注册工具名称和参数对象，不向远端解码器施加参数键顺序或财务值枚举。完整工具契约仍提供给LLM；本地Pydantic校验、工具白名单、当前文件/主体范围、原文绑定、复核及计算准入全部保留。未知工具、额外参数、非法单位、重复键、不完整JSON和越界引用不会因此执行。不是“取消金融校验”，也不要求更改或重启远程服务。
