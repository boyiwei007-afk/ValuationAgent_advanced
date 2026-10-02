# 远程千问接入与检查记录

## 本次实测状态

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
| 工具调用格式 | `json_content` / JSON内容工具调用 |
| 推理参数协议 | `chat_template`（SGLang / vLLM） |
| 思考模式 | `enabled`；`disabled`亦可请求，但单独关闭思考未解决完整任务的循环问题 |
| 采样温度 | 默认0；本轮也实测0.6，不能只改温度就当作质量验收通过 |

当前服务普通推理正常，但原生工具探针返回 `content='[{"name":"connection_check","parameters":{}}]'`、`tool_calls=null`。仅替换URL不够。`json_content`显式使用JSON动作协议：请求中提供完整工具目录，结构化解码只约束动作名称/参数对象的外壳，不要求服务端工具解析器；历史调用/结果转换成一致的JSON文本协议，来源仍是不可信数据。每次仅一个动作，收到真实结果后再决策，避免批量猜参数和多份正文淹没上下文。完整回复转回应用标准工具消息，参数再次由原始Pydantic工具契约严格校验，才允许执行。自然语言、代码块、未知工具、重复键、非有限数值、截断或超额调用全部拒绝。

此前直接把完整工具schema交给SGLang强制工具解码，遇到正则不支持、截断及重复动作。当前JSON协议不再让服务端解码器编译全部财务参数schema；**工具目录中的完整约束和本地原始校验保留**，没有放宽金额、字段或财务准入。标准`native`模式不变。`chat_template`协议显式把思考开关转换为请求级`chat_template_kwargs.enable_thinking`；不按模型名字猜能力，不修改服务启动参数。

环境配置也支持：

```powershell
$env:VALUATION_LLM_BASE_URL = "http://127.0.0.1:18000/v1"
$env:VALUATION_LLM_MODEL = "qwen36-teacher"
$env:VALUATION_LLM_API_KEY = "EMPTY"
$env:VALUATION_LLM_TOOL_CALL_FORMAT = "json_content"
$env:VALUATION_LLM_REASONING_PROTOCOL = "chat_template"
$env:VALUATION_LLM_THINKING = "enabled"
$env:VALUATION_LLM_TEMPERATURE = "0.6"
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
