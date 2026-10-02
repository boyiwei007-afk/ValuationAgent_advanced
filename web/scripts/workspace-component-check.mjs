import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'
import { Window } from 'happy-dom'

const window = new Window({ url: 'http://valuation.test/#workspace_aaaaaaaa', settings: {
  disableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true,
  disableCSSFileLoading: true, disableIframePageLoading: true,
} })
for (const key of ['window', 'document', 'navigator', 'HTMLElement', 'HTMLInputElement', 'HTMLTextAreaElement', 'Node', 'Event', 'KeyboardEvent', 'MouseEvent', 'File', 'Blob', 'FormData']) {
  Object.defineProperty(globalThis, key, { configurable: true, value: key === 'window' ? window : window[key] })
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true
const { createElement, act } = await import('react')
const { createRoot } = await import('react-dom/client')
const vite = await createServer({ root: fileURLToPath(new URL('../', import.meta.url)), server: { middlewareMode: true }, appType: 'custom', logLevel: 'error' })
const workspace = { workspace_id: 'workspace_aaaaaaaa', revision: 3, title: '统一工作区测试', status: 'awaiting_approval', current_phase: 'approval', run_policy: 'review' }
const otherWorkspace = { ...workspace, workspace_id: 'workspace_bbbbbbbb', title: '另一段历史对话' }
let history = [workspace, otherWorkspace]
let deletionBlocked = false
let delaySnapshot = false
let releaseSnapshot
const source = { file_id: 'file_example', name: '测试原文.txt', warnings: [], block_count: 1, sha256: 'a'.repeat(64) }
const snapshot = {
  workspace, execution: { active: false }, messages: [{ message_id: 'msg_one', role: 'assistant', content: '统一消息历史 **包括估值后追问**' }],
  research: { session: { draft: {}, documents: [source], facts: [{ fact_id: 'fact_one', block_id: 'file_example:1', metric: '营业收入', period: '2025', raw_value: '100', unit: '元', status: 'confirmed', verification: { source_assessment: { source_tier: 'C', binding: 'verified', admission: 'corroborated_draft', publication_date_known: true, limitations: ['跨来源一致不证明上游独立'] } } }], model_name: 'fixture', pending_decision: { question: '选择估值口径', options: [{ label: '合并口径', description: '披露限制' }, { label: '分部口径', description: '需要独立输入' }] } }, messages: [{ message_id: 'legacy', role: 'assistant', content: '旧分流不应显示' }] },
  runtime: { agent_version: 'workspace-agent-2026-09-30.7' },
  artifacts: [{ artifact_id: 'artifact_fixture', kind: 'research_note', filename: 'research-note.md', number: 1, source_revision: 4, status: 'unreviewed', size_bytes: 200, sha256: 'b'.repeat(64) },
    { artifact_id: 'artifact_interrupted', kind: 'interruption_report', filename: 'valuation-interruption.md', number: 2, source_revision: 4, status: 'execution_incomplete', size_bytes: 200, sha256: 'c'.repeat(64) }],
  research_plan: { history_policy: { target_years: 10, instruction: '目标十年，最低门槛不是目标。' }, evidence_counts: { observations_verified: 2, semantic_review_pending: 1, semantic_review_supported: 1 }, extraction_recovery: { files: [{ file_id: 'file_example', failure_count: 1, latest_issues: ['AMOUNT_BOUNDARY'], next_choices: [{ tool: 'read_file', arguments: { view: 'pdf_layout' } }] }] }, annual_coverage: [{ year: 2025, priority: 'baseline', candidate_metrics: ['营业收入', '利息费用'], bound_metrics: ['营业收入', '利息费用'], model_review_count: 1, confirmed_metrics: ['revenue'], not_directly_verified: ['ebit_margin'], binding_issues: [] }], table_repairs: [{ file_id: 'file_example', issue: '年度列待绑定', affected_count: 3 }], method_readiness: [{ method: 'pe', status: 'inputs_ready', reason: '仅准备通过' }, { method: 'dcf', status: 'blocked', reason: '专项桥接待审查' }] },
  connections: { model: { available: true } }, findings: [], decisions: [], requirements: [],
  versions: [], checkpoints: [], plan: [{ title: '读取原文', status: 'completed' }],
}
const review = { executable_methods: ['dcf'], unresolved_items: [], assumptions: { revenue_growth_scenarios: { base: ['0.05', '0.04'] } }, inputs: {}, state_hash: 'a'.repeat(64), research_revision: 1, fact_ids: [], evidence_ids: [] }
const requests = []
globalThis.fetch = async (url, options = {}) => {
  requests.push({ url, method: options.method || 'GET', body: options.body ? JSON.parse(options.body) : null })
  if (options.method === 'DELETE') {
    if (deletionBlocked) return { ok: false, status: 409, json: async () => ({ detail: '任务仍在执行，请先停止' }) }
    const id = url.split('/').at(-1).split('?')[0]
    history = history.filter(item => item.workspace_id !== id)
    return { ok: true, status: 204 }
  }
  if (delaySnapshot && url === `/api/workspaces/${workspace.workspace_id}`) {
    delaySnapshot = false
    return new Promise(resolve => { releaseSnapshot = () => resolve({ ok: true, status: 200, json: async () => structuredClone(snapshot) }) })
  }
  const body = url.endsWith('/api/access/status') ? { authentication_required: true }
    : url.endsWith('/api/workspaces') ? history
    : url.includes('/sources/') ? { total: 1, blocks: [{ block_id: 'file_example:1', text: '原文营业收入 100 元', location: { page: 12 } }], next_offset: null }
    : url.endsWith('/timeline') ? []
    : url.endsWith('/data-services') ? {}
    : url.endsWith('/prevaluation-review') ? review
    : snapshot
  return { ok: true, status: 200, json: async () => structuredClone(body) }
}
const host = document.createElement('div')
document.body.append(host)
const root = createRoot(host)
const button = text => [...host.querySelectorAll('button')].find(element => element.textContent.includes(text))
const flush = () => act(async () => { await new Promise(resolve => setTimeout(resolve, 30)) })
try {
  const { default: WorkspaceApp } = await vite.ssrLoadModule('/src/WorkspaceApp.jsx')
  await act(async () => { root.render(createElement(WorkspaceApp)) })
  await flush()
  assert.match(host.textContent, /统一消息历史/)
  assert.match(host.querySelector('.ws-access-session').textContent, /登录者共享全部资料/)
  assert.ok(button('退出访问'))
  assert.doesNotMatch(host.textContent, /旧分流不应显示/)
  assert.match(host.textContent, /workspace-agent-2026-09-30.7/)
  assert.match(host.querySelector('[aria-label="文件交付"]').textContent, /研究笔记 · 未审阅/)
  assert.match(host.querySelector('[aria-label="文件交付"]').textContent, /执行中断报告 · 估值未完成/)
  assert.match(host.querySelector('[aria-label="文件交付"]').textContent, /SHA-256/)
  assert.match(host.querySelector('[aria-label="文档理解与复核"]').textContent, /1 项待复核.*1 项已获 LLM 支持/)
  assert.match(host.querySelector('[aria-label="文档理解与复核"]').textContent, /不等于独立审计/)
  assert.match(host.querySelector('[aria-label="文档理解与复核"]').textContent, /pdf_layout/)
  assert.ok(host.querySelector('[aria-label="Agent 任务计划"]'))
  assert.equal(host.querySelector('.ws-composer textarea').disabled, false)
  assert.ok(host.querySelector('[aria-label="方案选择"]'))
  assert.match(host.querySelector('.ws-year-coverage').textContent, /2025/)
  assert.match(host.querySelector('.ws-year-coverage').textContent, /2 个原文已绑定.*1 个通过字段准入/)
  assert.match(host.querySelector('.ws-year-coverage').textContent, /原文已定位，语义或模型处理仍待审查/)
  assert.match(host.querySelector('.ws-year-coverage').textContent, /补读原文或换视图后修正解释/)
  assert.match(host.querySelector('.ws-year-coverage').textContent, /PE · 输入准备通过.*DCF · 输入未就绪/)
  const beforeChoice = requests.filter(request => request.method === 'POST').length
  await act(async () => { button('A · 合并口径').click() })
  assert.match(host.querySelector('.ws-composer textarea').value, /我选择合并口径/)
  assert.equal(requests.filter(request => request.method === 'POST').length, beforeChoice)
  await act(async () => { host.querySelector('.ws-send').click() })
  await flush()
  assert.ok(requests.some(request => request.url.endsWith('/messages') && request.body?.content.includes('我选择合并口径')))
  assert.ok(!requests.some(request => request.url.endsWith('/approvals')))
  await act(async () => { button('来源').click() })
  await act(async () => { button('测试原文.txt').click() })
  await flush()
  assert.match(host.querySelector('dialog').textContent, /原文营业收入 100 元/)
  assert.match(host.querySelector('dialog').textContent, /page.*12/)
  await act(async () => { host.querySelector('dialog .icon-button').click() })
  await act(async () => { button('查看原文与口径').click() })
  await flush()
  assert.match(host.querySelector('[aria-label="字段证据质量"]').textContent, /C 级来源.*草案输入/)
  assert.match(host.querySelector('dialog').textContent, /跨来源一致不证明上游独立/)
  await act(async () => { host.querySelector('dialog .icon-button').click() })
  await act(async () => { button('计算历史').click() })
  assert.doesNotMatch(host.textContent, /切换查看|从此版分支/)
  await act(async () => { button('复核估值方案').click() })
  await flush()
  assert.ok(requests.some(request => request.url.endsWith('/prevaluation-review') && request.method === 'POST'))
  assert.doesNotMatch(host.querySelector('dialog').textContent, /\[object Object\]/)
  assert.match(host.querySelector('dialog').textContent, /0.05/)
  await act(async () => { host.querySelector('dialog .icon-button').click() })
  const deletionButton = title => host.querySelector(`button[aria-label="删除对话：${title}"]`)
  await act(async () => { deletionButton(otherWorkspace.title).click() })
  assert.match(host.querySelector('dialog').textContent, /无法撤销/)
  assert.ok(!requests.some(request => request.method === 'DELETE'))
  await act(async () => { button('保留对话').click() })
  assert.ok(deletionButton(otherWorkspace.title))
  assert.ok(!requests.some(request => request.method === 'DELETE'))
  deletionBlocked = true
  await act(async () => { deletionButton(otherWorkspace.title).click(); })
  await act(async () => { button('确认永久删除').click() })
  await flush()
  assert.match(host.querySelector('dialog').textContent, /任务仍在执行/)
  assert.ok(deletionButton(otherWorkspace.title))
  deletionBlocked = false
  await act(async () => { button('确认永久删除').click() })
  await flush()
  assert.equal(deletionButton(otherWorkspace.title), null)
  assert.match(host.querySelector('.ws-header').textContent, /统一工作区测试/)
  assert.ok(requests.some(request => request.method === 'DELETE' && request.url.endsWith('workspace_bbbbbbbb?revision=3')))
  delaySnapshot = true
  await act(async () => { host.querySelector('.ws-history-open').click() })
  await flush()
  assert.equal(typeof releaseSnapshot, 'function')
  await act(async () => { deletionButton(workspace.title).click() })
  await act(async () => { button('确认永久删除').click() })
  await flush()
  await act(async () => { releaseSnapshot() })
  await flush()
  assert.equal(host.querySelector('.ws-header'), null)
  assert.equal(host.querySelectorAll('.ws-history-item').length, 0)
  assert.equal(window.location.hash, '')
  console.log('PASS: confirmed deletion, cancel, conflict preservation, non-current/current deletion and stale response isolation.')
  console.log('PASS: unified messages, editable decisions, annual coverage, task plan, source drilldown, audit-only history, POST review and nested assumptions.')
} finally {
  await act(async () => { root.unmount() })
  await vite.close()
  window.happyDOM.abort()
}
