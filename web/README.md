# ValuationAgent Web

React / Vite 统一工作区，与 CLI 共用 WorkspaceAgentRuntime、消息、证据、计算与审计记录。不包含旧向导、确认卡或结果专用聊天入口。

## 启动

在父目录构建 `npm.cmd --prefix web ci`、`npm.cmd --prefix web run build`，随后在 `economic_agent` 环境运行 `valuationagent serve`，访问 `http://127.0.0.1:8000/`。此工作台依赖本地 Python 服务，不是独立静态演示页。

开发：后端运行在 8000 时，在此目录执行 `npm.cmd run dev`。默认 5173；开发代理目标由 `VALUATION_BACKEND_URL` 指定。独立前端部署的 API 地址由构建变量 `VITE_API_BASE_URL` 指定，并需配置后端 CORS。

## 模块

- `WorkspaceApp.jsx`：统一对话、模型/搜索连接、附件、任务计划、证据下钻、方案审批和只读计算历史。
- `MessageBody.jsx`：显示模型回复中的列表、表格与代码，禁用原始 HTML、危险链接和远程图片加载。
- `api.js`：统一请求、校验错误处理、文件上传；不保留旧 API 适配。
- `ui.jsx`、`Icons.jsx`：当前工作区共用的界面组件。
- `workspace.css` 及共用样式：工作区布局与响应式展示。

金额和假设在请求、存储及导出中沿用十进制字符串。修改假设通过后端新计算完成，不在浏览器里模拟估值结果。历史计算仅用于审计与对比，不能激活旧工作流。

API Key 只在连接表单中暂存并提交给本地后端临时会话；不进入浏览器存储或工作区数据库。所有 Agent 对话都需要连接模型，不能靠未连接时的规则回复冒充 Agent。服务默认本地单用户使用，不具备公网多用户鉴权。

## 验证

`npm.cmd test`、`npm.cmd run test:components`、`npm.cmd run lint`、`npm.cmd run build` 分别检查 API 单元测试、虚拟 DOM 交互、静态规则与生产构建；不调用真实模型。

`npm.cmd run test:integration` 使用 `VALUATION_TEST_URL`（默认 8001）验证真实服务的静态资源、工作区消息持久化、缺模型错误、导出与退役路由拒绝，会创建测试工作区；请指向独立测试数据目录。

虚拟 DOM 验证不等于浏览器视觉验收。当前验证范围与未完成项见父目录 `docs/REFACTOR_CONFORMANCE.md`。
