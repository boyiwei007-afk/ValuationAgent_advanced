import { useCallback, useEffect, useRef, useState } from 'react'
import Icon from './Icons'
import MessageBody from './MessageBody'
import { Modal, ErrorNotice } from './ui'
import { api, post, uploadFile, downloadWorkspace, downloadRun, downloadSavedArtifact } from './api'
import { MODEL_PRESETS } from './presets'
import './workspace.css'

const phaseOrder = ['scope', 'evidence', 'model_design', 'approval', 'calculation', 'challenge', 'decision', 'reporting']
const phaseNames = {
  scope: '任务范围', evidence: '数据取证', model_design: '建立模型', approval: '方案复核',
  calculation: '确定性计算', challenge: '风险检查', decision: '综合决策', reporting: '结果报告',
}
const statusNames = {
  setup: '等待开始', researching: '正在推进', awaiting_user: '需要确认', awaiting_approval: '方案待复核',
  ready: '可以计算', calculating: '正在计算', completed: '估值完成',
  completed_with_warnings: '完成，附风险提示', degraded: '降级交付', failed: '执行未完成',
  paused: '已暂停', cancelled: '已取消',
}
const tabs = [
  ['overview', '结果'], ['inputs', '输入'], ['evidence', '来源'],
  ['risk', '风险'], ['versions', '计算历史'], ['audit', '记录'],
]

function AccessSession() {
  const [enabled, setEnabled] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    api('/api/access/status').then(result => { if (active) setEnabled(Boolean(result.authentication_required)) }).catch(() => {})
    return () => { active = false }
  }, [])
  async function logout() {
    setBusy(true); setError('')
    try { await post('/api/access/logout', {}); window.location.reload() }
    catch (err) { setError(err.message); setBusy(false) }
  }
  if (!enabled) return null
  return <div className="ws-access-session"><small>受保护的单操作者空间 · 登录者共享全部资料</small><button className="button ghost" disabled={busy} onClick={logout}>退出访问</button>{error && <span role="alert">{error}</span>}</div>
}

function EvidenceQuality({ fact }) {
  const assessment = fact?.verification?.source_assessment
  if (!assessment) return null
  const admission = { blocked: '尚不可入模', eligible: '通过当前准入检查', corroborated_draft: '跨源一致 · 草案输入' }
  return <div className="ws-evidence-quality" aria-label="字段证据质量"><b>{assessment.source_tier || 'D'} 级来源 · {admission[assessment.admission] || assessment.admission}</b><p>原文绑定：{assessment.binding === 'verified' ? '通过' : '待修复'} · 披露日期：{assessment.publication_date_known ? '已记录且未晚于截止日' : '待确认'}</p>{assessment.model_issues?.map(item => <p key={item}>模型审查：{item}</p>)}{assessment.limitations?.map(item => <small key={item}>{item}</small>)}</div>
}

function EvidenceProgress({ plan }) {
  if (!plan?.annual_coverage?.length) return null
  const priorities = { baseline: '估值基期', core_history: '近期趋势', long_term_trend: '长期趋势' }
  return <details className="ws-year-coverage"><summary>年度取证覆盖 · 目标 {plan.history_policy.target_years} 年</summary><p>{plan.history_policy.instruction}</p><div>{plan.annual_coverage.map(row => <section key={row.year}><b>{row.year} · {priorities[row.priority] || '历史取证'}</b><span>{row.candidate_metrics?.length || 0} 个候选科目 · {row.bound_metrics?.length || 0} 个原文已绑定 · {row.confirmed_metrics.length} 个通过字段准入</span><small>{row.not_directly_verified.length} 个本阶段重点字段尚未直接核验；推导项由计算器检查，不是每年同一份清单。</small>{row.binding_issues?.length > 0 && <small>已有原文，绑定待修复：{row.binding_issues.join('；')}</small>}{row.model_review_count > 0 && <small>{row.model_review_count} 条原文已定位，语义或模型处理仍待审查</small>}</section>)}</div>{plan.table_repairs?.slice(0, 3).map(group => <p key={`${group.file_id}-${group.issue}`}><strong>共性解析问题 · {group.affected_count} 条</strong> {group.issue}。补读原文或换视图后修正解释。</p>)}{plan.method_readiness?.map(item => <p key={item.method}><strong>{item.method.toUpperCase()} · {item.status === 'inputs_ready' ? '输入准备通过' : '输入未就绪'}</strong> {item.reason}</p>)}<small>原文绑定、字段准入、方法可计算是独立状态；原文下载不等于完整年度已提取，中报和季报不替代年度收入。</small></details>
}

function InterpretationProgress({ plan }) {
  const counts = plan?.evidence_counts || {}
  const recovery = plan?.extraction_recovery?.files || []
  if (!counts.semantic_review_pending && !counts.semantic_review_supported && !recovery.length) return null
  return <section className="ws-artifacts" aria-label="文档理解与复核"><h3>文档理解与复核</h3><p>{counts.semantic_review_pending || 0} 项待复核 · {counts.semantic_review_supported || 0} 项已获 LLM 支持</p><small>原文定位 ≠ 语义正确 ≠ 模型准入；同一 LLM 的复核不等于独立审计。</small>{recovery.slice(-3).map(item => <div key={item.file_id}><div><b>读取恢复 · {item.failure_count} 次失败</b><small>{item.latest_issues.join('；')}</small><small>下一步候选：{item.next_choices.map(choice => choice.arguments?.view || choice.tool).join(' / ')}。由 Agent 选择，不重复下载。</small></div></div>)}</section>
}

function ArtifactShelf({ artifacts, onDownload }) {
  if (!artifacts.length) return null
  return <section className="ws-artifacts" aria-label="文件交付"><h3>文件交付 · 可追溯版本</h3><p>研究笔记不是原始证据；敏感性试算不改原模型；结果报告的计算状态由系统标记。</p>{artifacts.slice(0, 12).map(artifact => <div key={artifact.artifact_id}><div><b>{artifact.kind === 'research_note' ? '研究笔记 · 未审阅' : artifact.kind === 'interruption_report' ? '执行中断报告 · 估值未完成' : artifact.kind === 'sensitivity_analysis' ? '敏感性分析 · 假设试算' : '结果 / 缺口报告'} · {artifact.filename}</b><small>v{artifact.number} · {artifact.status} · 输入修订 {artifact.source_revision} · {Math.ceil(artifact.size_bytes / 1024)} KB</small><small>SHA-256 {artifact.sha256.slice(0, 20)}</small></div><button type="button" onClick={() => onDownload(artifact)}>下载</button></div>)}</section>
}

function ModelModal({ current, onClose, onConnected }) {
  const [form, setForm] = useState({ provider: 'openai_compatible', base_url: 'https://api.openai.com/v1', model: '', api_key: '', thinking: 'auto', reasoning_protocol: 'auto', temperature: 0, tool_call_format: 'native', supports_images: false })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const chooseModel = model => {
    const preset = MODEL_PRESETS.find(item => item.value === model)
    setForm(old => ({ ...old, model, ...(preset ? { provider: preset.provider, base_url: preset.base_url } : {}) }))
  }
  const submit = async event => {
    event.preventDefault(); setBusy(true); setError('')
    try {
      await post('/api/model-connections/test', form)
      const session = await post('/api/model-sessions', form)
      await onConnected(session)
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  return <Modal title="连接核心推理模型" onClose={busy ? () => {} : onClose}>
    <p className="modal-intro">模型负责理解需求、制定取证计划和解释结果；所有估值数字仍由确定性工具计算。密钥只保存在当前服务进程内。</p>
    {current && <div className="ws-connected"><Icon name="check" size={16}/>{current.model} 已连接</div>}
    <form onSubmit={submit}>
      <label className="field">模型名称<input list="ws-models" required value={form.model} onChange={e => chooseModel(e.target.value)} placeholder="选择或输入模型名称"/></label>
      <datalist id="ws-models">{MODEL_PRESETS.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</datalist>
      <details><summary>自定义接口</summary>
        <label className="field">接口地址<input type="url" required value={form.base_url} onChange={e => setForm(old => ({ ...old, base_url: e.target.value }))}/></label>
        <label className="field">采样温度<input type="number" min="0" max="2" step="0.05" value={form.temperature ?? ''} onChange={e => setForm(old => ({ ...old, temperature: e.target.value === '' ? null : Number(e.target.value) }))}/><small>按模型部署建议填写；留空不发送此参数。固定温度不保证推理可复现。</small></label>
        <label className="field">思考模式<select value={form.thinking} onChange={e => setForm(old => ({ ...old, thinking: e.target.value }))}><option value="auto">自动</option><option value="enabled">开启</option><option value="disabled">关闭</option></select></label>
        <label className="field">推理参数协议<select value={form.reasoning_protocol} onChange={e => setForm(old => ({ ...old, reasoning_protocol: e.target.value }))}><option value="auto">自动 / 默认接口</option><option value="chat_template">chat_template（SGLang / vLLM）</option></select><small>部署支持 enable_thinking 时选择 chat_template，思考模式才会传给该网关；不会修改远端启动参数。</small></label>
        <label className="field">工具调用格式<select value={form.tool_call_format} onChange={e => setForm(old => ({ ...old, tool_call_format: e.target.value }))}><option value="native">标准 tool_calls（默认）</option><option value="json_content">JSON 内容工具调用（网关兼容）</option></select><small>仅当服务把工具调用返回在文本内容中时选择 JSON；不解析普通聊天文字，也不改变财务准入。</small></label>
      </details>
      <label className="field">API Key<input type="password" autoComplete="off" required value={form.api_key} onChange={e => setForm(old => ({ ...old, api_key: e.target.value }))}/></label>
      <label className="field"><span><input type="checkbox" checked={form.supports_images} onChange={e => setForm(old => ({ ...old, supports_images: e.target.checked }))}/> 此接口支持图片，并允许发送选定页图（可选，可能增加计费）</span><small>不会自动切换模型。只支持文本的接口请保持关闭。</small></label>
      <ErrorNotice message={error}/>
      <div className="modal-actions"><button type="button" className="button secondary" onClick={onClose}>取消</button><button className="button primary" disabled={busy}>{busy ? '正在验证…' : '验证并连接'}</button></div>
    </form>
  </Modal>
}

function DataModal({ current, onClose, onConnected }) {
  const [tavily, setTavily] = useState('')
  const [tushare, setTushare] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const submit = async event => {
    event.preventDefault(); setBusy(true); setError('')
    try {
      await onConnected({ ...(tavily.trim() ? { tavily_api_key: tavily.trim(), verify_search: true } : {}), ...(tushare.trim() ? { tushare_token: tushare.trim() } : {}) })
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  return <Modal title="连接数据服务" onClose={busy ? () => {} : onClose}>
    <p className="modal-intro">Tavily 用于公开网页与行业资料，交易所公告检索无需密钥；Tushare 用于 A 股结构化取数。没有完整数据时系统会有界搜索、降级方法或输出缺口说明，不会无限轮询。</p>
    <div className="ws-service-state"><span className={current?.search?.available ? 'on' : ''}>网页检索 {current?.search?.available ? '可用' : '未连接'}</span><span className={current?.market?.available ? 'on' : ''}>结构化行情 {current?.market?.available ? '可用' : '未连接'}</span></div>
    <form onSubmit={submit}>
      <label className="field">Tavily API Key<input type="password" autoComplete="off" value={tavily} onChange={e => setTavily(e.target.value)}/></label>
      <label className="field">Tushare Token（可选）<input type="password" autoComplete="off" value={tushare} onChange={e => setTushare(e.target.value)}/></label>
      <ErrorNotice message={error}/>
      <div className="modal-actions"><button type="button" className="button secondary" onClick={onClose}>取消</button><button className="button primary" disabled={busy || (!tavily.trim() && !tushare.trim())}>{busy ? '正在连接…' : '连接服务'}</button></div>
    </form>
  </Modal>
}

function ReviewModal({ review, onClose, onApprove, busy }) {
  if (!review) return null
  const ready = review.executable_methods?.length > 0 && !review.unresolved_items?.length
  return <Modal title={ready ? '估值方案集中复核' : '当前模型准备度'} onClose={busy ? () => {} : onClose} className="wide-modal">
    <div className="ws-review-scope"><div><small>公司 / 证券主体</small><b>{review.company || review.ticker || '待确认'} {review.ticker}</b></div><div><small>行业 / 公司类型</small><b>{review.industry || '待确认'} · {review.company_type || '待判断'}</b></div><div><small>估值日 / 信息截止日</small><b>{review.valuation_date || '待确认'} / {review.information_cutoff_date || '待确认'}</b></div><div><small>方法 / 币种</small><b>{review.executable_methods?.map(x => x.toUpperCase()).join(' / ') || '暂无'} · {review.currency}</b></div><div><small>财务基期</small><b>{review.baseline_period || '待确认'}</b></div><div><small>证据覆盖</small><b>{Math.round((review.evidence_coverage || 0) * 100)}% · {review.evidence_grade} 级</b></div></div>
    {Object.keys(review.excluded_methods || {}).length > 0 && <section className="ws-review-section warning"><h3>本次排除的方法</h3>{Object.entries(review.excluded_methods).map(([method, reason]) => <p key={method}><b>{method.toUpperCase()}</b> · {reason}</p>)}</section>}
    <section className="ws-review-section"><h3>进入计算的基期输入</h3><div className="ws-key-values">{Object.entries(review.inputs || {}).filter(([, value]) => value != null && typeof value !== 'object').map(([key, value]) => <div key={key}><span>{key}</span><b>{String(value)}</b></div>)}</div>{!Object.keys(review.inputs || {}).length && <p>尚未形成可提交的完整基期。</p>}</section>
    <section className="ws-review-section"><h3>主要假设</h3><div className="ws-key-values">{Object.entries(review.assumptions || {}).map(([key, value]) => <div key={key}><span>{key}</span><b>{typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}</b></div>)}</div></section>
    {Object.keys(review.wacc_components || {}).length > 0 && <section className="ws-review-section"><h3>WACC 组成</h3><div className="ws-key-values">{Object.entries(review.wacc_components).map(([key, value]) => <div key={key}><span>{key}</span><b>{String(value)}</b></div>)}</div></section>}
    {review.peers?.length > 0 && <section className="ws-review-section"><h3>可比公司</h3>{review.peers.map(peer => { const multiples = peer.multiples || { pe: peer.pe, ps: peer.ps, ev_ebitda: peer.ev_ebitda }; return <p key={peer.ticker}><b>{peer.name || peer.ticker}</b> · {peer.ticker} · {Object.entries(multiples).filter(([, value]) => value != null).map(([key, value]) => `${key} ${value}`).join(' / ')}</p> })}</section>}
    {review.user_overrides?.length > 0 && <section className="ws-review-section warning"><h3>用户提供或覆盖值</h3>{review.user_overrides.map((item, index) => <p key={`${item.metric}-${index}`}>{item.metric} · {item.period} · {item.value}</p>)}</section>}
    {review.preflight_findings?.length > 0 && <section className="ws-review-section blocked"><h3>估值前挑战发现</h3>{review.preflight_findings.map((item, index) => <p key={`${item.title}-${index}`}><b>{item.title}</b> · {item.action}</p>)}</section>}
    {review.risks?.length > 0 && <section className="ws-review-section warning"><h3>风险与限制</h3>{review.risks.map(item => <p key={item}>· {item}</p>)}</section>}
    {review.unresolved_items?.length > 0 && <section className="ws-review-section blocked"><h3>尚不能进入数值计算</h3>{review.unresolved_items.map(item => <p key={item}>· {item}</p>)}</section>}
    <p className="ws-hash">复核快照 {review.state_hash.slice(0, 16)} · 研究版本 v{review.research_revision} · {review.fact_ids.length} 个事实 / {review.evidence_ids.length} 条来源</p>
    <div className="modal-actions"><button className="button secondary" onClick={onClose}>返回修改</button><button className="button primary" disabled={!ready || busy} onClick={() => onApprove(review)}>{busy ? '正在创建任务…' : '确认方案并开始计算'}</button></div>
  </Modal>
}

function RevisionModal({ run, onClose, onSubmit, busy }) {
  const current = run?.result?.assumptions || run?.request?.assumptions || {}
  const [wacc, setWacc] = useState(current.wacc != null ? String(Number(current.wacc) * 100) : '')
  const [growth, setGrowth] = useState(current.terminal_growth != null ? String(Number(current.terminal_growth) * 100) : '')
  const [methods, setMethods] = useState(run?.request?.methods || [])
  const [excludedPeers, setExcludedPeers] = useState('')
  const [reason, setReason] = useState('根据用户要求调整估值参数并重算')
  const submit = event => {
    event.preventDefault()
    const changes = {}
    const assumptions = {}
    if (wacc !== '') assumptions.wacc = String(Number(wacc) / 100)
    if (growth !== '') assumptions.terminal_growth = String(Number(growth) / 100)
    if (Object.keys(assumptions).length) changes.assumptions = assumptions
    if (methods.length) changes.methods = methods
    const excludes = new Set(excludedPeers.split(/[,，\s]+/).map(x => x.trim().toUpperCase()).filter(Boolean))
    if (excludes.size) changes.peers = (run?.request?.peers || []).filter(peer => !excludes.has(peer.ticker.toUpperCase()))
    onSubmit({ reason, changes })
  }
  return <Modal title="调整参数并重新计算" onClose={busy ? () => {} : onClose} className="wide-modal">
    <form onSubmit={submit}>
      <div className="ws-form-grid"><label className="field">WACC（%）<input type="number" step="0.01" value={wacc} onChange={e => setWacc(e.target.value)}/></label><label className="field">永续增长率（%）<input type="number" step="0.01" value={growth} onChange={e => setGrowth(e.target.value)}/></label></div>
      <fieldset className="ws-methods"><legend>本版本采用的方法</legend>{['dcf', 'pe', 'ps', 'ev_ebitda'].map(method => <label key={method}><input type="checkbox" checked={methods.includes(method)} onChange={e => setMethods(old => e.target.checked ? [...new Set([...old, method])] : old.filter(x => x !== method))}/>{method.toUpperCase()}</label>)}</fieldset>
      <label className="field">排除可比公司代码（可选）<input value={excludedPeers} onChange={e => setExcludedPeers(e.target.value)} placeholder="例如 000001.SZ, 600000.SH"/></label>
      <label className="field">修改原因<textarea required value={reason} onChange={e => setReason(e.target.value)} rows={3}/></label>
      <div className="modal-actions"><button type="button" className="button secondary" onClick={onClose}>取消</button><button className="button primary" disabled={busy || !methods.length}>{busy ? '正在创建…' : '冻结新输入并重算'}</button></div>
    </form>
  </Modal>
}

function Workflow({ phase, status, completed = [], skipped = [], failed = [] }) {
  const terminal = ['completed', 'completed_with_warnings', 'degraded', 'failed', 'cancelled'].includes(status)
  const completedSet = new Set(completed)
  const skippedSet = new Set(skipped)
  const failedSet = new Set(failed)
  const stateLabel = { done: '已完成', active: '正在进行', skipped: '未执行', failed: '执行失败', pending: '待进行' }
  return <div className="ws-workflow" aria-label="估值工作流">{phaseOrder.map((item, index) => {
    const state = completedSet.has(item) ? 'done'
      : failedSet.has(item) ? 'failed'
        : skippedSet.has(item) ? 'skipped'
          : !terminal && item === phase ? 'active' : 'pending'
    const marker = state === 'done' ? '✓' : state === 'skipped' ? '—' : state === 'failed' ? '!' : index + 1
    return <div key={item} className={state} title={`${phaseNames[item]}：${stateLabel[state]}`}><i>{marker}</i><span>{phaseNames[item]}</span></div>
  })}<strong>{statusNames[status] || status}</strong></div>
}

function EmptyStart({ onStart, onConnect, model }) {
  const [objective, setObjective] = useState('')
  const [policy, setPolicy] = useState('web')
  const [runPolicy, setRunPolicy] = useState('automatic')
  return <div className="ws-empty-start"><div className="ws-hero-mark"><Icon name="spark" size={28}/></div><p className="eyebrow">VALUATION-FIRST AGENT</p><h1>给我一家公司，交付一套可复核的估值</h1><p>自动判断公司类型与行业，按模型需要检索和核验证据，完成 DCF / 相对估值、敏感性分析、风险检查和结果报告。没有完整数据时会明确降级，不会反复空转。</p>
    <form onSubmit={event => { event.preventDefault(); if (objective.trim()) onStart(objective.trim(), policy, runPolicy) }}><textarea autoFocus value={objective} onChange={e => setObjective(e.target.value)} placeholder="例如：估值贵州茅台，以今天为基准日，使用 DCF 和 P/E；请自动查找公开资料" rows={4}/><div><div className="ws-start-options"><select value={policy} onChange={e => setPolicy(e.target.value)}><option value="web">公开资料检索 + 我的附件</option><option value="upload">仅使用我上传的资料</option></select><select value={runPolicy} onChange={e => setRunPolicy(e.target.value)}><option value="review">审阅模式 · 估值前确认</option><option value="automatic">自动模式 · 生成估值草案</option></select></div><button className="button primary" disabled={!objective.trim()}>开始估值 <Icon name="arrow" size={16}/></button></div></form>
    <button className="ws-model-link" onClick={onConnect}><Icon name={model ? 'check' : 'link'} size={16}/>{model ? `${model.model} 已连接` : '先连接核心推理模型（推荐）'}</button>
  </div>
}

function ResultOverview({ run, report, findings, decision, onRevise, onDownload, status, working }) {
  const result = run?.result
  const terminalReport = ['degraded', 'failed'].includes(status) ? report : null
  if (!result) return <div className="ws-empty-panel"><Icon name="chart" size={32}/><h3>{working ? '正在形成估值结果' : terminalReport?.status_label || '尚未形成数值估值'}</h3><p>{working ? 'Agent 正在按模型缺口自动取证与核验；完成后只呈现可计算结果，或一次性说明无法计算的原因。' : terminalReport?.conclusion || 'Agent 会先完成取证和模型准备，集中复核后才进入确定性计算。'}</p>{terminalReport && <button className="button secondary" onClick={() => onDownload('pdf')}>下载当前说明报告</button>}</div>
  const relative = result.relative?.filter(item => item.status === 'success') || []
  return <div className="ws-result"><div className="ws-result-hero"><div><small>综合结论</small><h2>{result.executive_summary}</h2><p>{result.reconciliation?.conclusion}</p></div><span className={`grade grade-${result.data_quality?.result_grade}`}>质量 {result.data_quality?.result_grade}<small>{result.data_quality?.confidence} confidence</small></span></div>
    <div className="ws-value-cards">{result.dcf && <div><span>DCF 每股价值</span><b>{Number(result.dcf.per_share_value).toFixed(2)}</b><small>{Number(result.dcf.range_low).toFixed(2)} – {Number(result.dcf.range_high).toFixed(2)} {result.currency}</small></div>}{relative.map(item => <div key={item.method}><span>{item.method.toUpperCase()} 每股价值</span><b>{Number(item.per_share_value).toFixed(2)}</b><small>{Number(item.range_low).toFixed(2)} – {Number(item.range_high).toFixed(2)} · {item.sample_size} 家样本</small></div>)}</div>
    <div className="ws-summary-grid"><section><h3>核心假设</h3><p>WACC <b>{(Number(result.assumptions.wacc) * 100).toFixed(2)}%</b></p><p>永续增长率 <b>{(Number(result.assumptions.terminal_growth) * 100).toFixed(2)}%</b></p><p>证据覆盖率 <b>{(Number(result.data_quality.evidence_coverage) * 100).toFixed(0)}%</b></p></section><section><h3>挑战层结论</h3><p>{decision?.selected_action || '挑战层将在结果完成后独立检查数据、终值、方法与敏感性。'}</p><small>{findings?.filter(item => ['high', 'blocking'].includes(item.severity)).length || 0} 项高风险 / 阻塞发现</small></section></div>
    <div className="ws-inline-actions"><button className="button primary" onClick={onRevise}>调整参数并重算</button><button className="button secondary" onClick={() => onDownload('xlsx')}>下载 Excel 底稿</button><button className="button secondary" onClick={() => onDownload('pdf')}>下载 PDF 报告</button></div>
  </div>
}

export default function WorkspaceApp() {
  const [workspaceId, updateWorkspaceId] = useState(() => window.location.hash.match(/^#(workspace_[a-f0-9]+)$/)?.[1] || '')
  const selectedWorkspace = useRef(workspaceId)
  const deletedWorkspaces = useRef(new Set())
  const setWorkspaceId = useCallback(id => { selectedWorkspace.current = id; updateWorkspaceId(id) }, [])
  const [history, setHistory] = useState([])
  const [snapshot, setSnapshot] = useState(null)
  const [model, setModel] = useState(null)
  const [services, setServices] = useState(null)
  const [modal, setModal] = useState('')
  const [review, setReview] = useState(null)
  const [source, setSource] = useState(null)
  const [tab, setTab] = useState('overview')
  const [input, setInput] = useState('')
  const [files, setFiles] = useState([])
  const [role, setRole] = useState('historical_financials')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [timeline, setTimeline] = useState([])
  const [comparison, setComparison] = useState(null)
  const [compareIds, setCompareIds] = useState(['', ''])
  const [pendingObjective, setPendingObjective] = useState(null)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')
  const scroll = useRef(null)
  const fileInput = useRef(null)
  const workspace = snapshot?.workspace
  const research = snapshot?.research
  const session = research?.session
  const execution = snapshot?.execution
  const run = snapshot?.active_run
  const messages = snapshot?.messages || []
  const resultDocument = research?.result_document
  const findings = (snapshot?.findings || []).filter(item => item.run_id === run?.run_id)
  const dispositions = snapshot?.finding_dispositions || []
  const decision = snapshot?.decisions?.filter(item => item.run_id === run?.run_id).at(-1)

  const loadHistory = useCallback(async () => {
    try { const items = await api('/api/workspaces'); setHistory(items.filter(item => !deletedWorkspaces.current.has(item.workspace_id))) } catch { /* health banner handles it */ }
  }, [])
  const load = useCallback(async (id = workspaceId, silent = false) => {
    if (!id) return
    try {
      const data = await api(`/api/workspaces/${id}`)
      if (selectedWorkspace.current !== id || deletedWorkspaces.current.has(id)) return
      setSnapshot(data)
      if (!silent) setError('')
    } catch (err) {
      if (selectedWorkspace.current !== id || deletedWorkspaces.current.has(id)) return
      if (err.status === 404) {
        setSnapshot(null); setServices(null)
        setWorkspaceId('')
        setPendingObjective(current => current?.workspaceId === id ? null : current)
        if (window.location.hash === `#${id}`) {
          window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`)
        }
        if (!silent) setError('')
        return
      }
      if (!silent) setError(err.message)
    }
  }, [workspaceId, setWorkspaceId])
  // eslint-disable-next-line react/set-state-in-effect -- synchronize persisted workspaces
  useEffect(() => { void loadHistory() }, [loadHistory])
  useEffect(() => {
    if (!workspaceId) return
    window.location.hash = workspaceId
    let live = true
    let timer = null
    const shouldPoll = execution?.active || ['created', 'running'].includes(run?.status)
    const pull = async () => {
      if (!live) return
      await load(workspaceId, true)
      if (live) timer = setTimeout(pull, 1600)
    }
    // eslint-disable-next-line react/set-state-in-effect -- fetch persisted workspace state
    void load(workspaceId)
    if (shouldPoll) timer = setTimeout(pull, 1600)
    return () => { live = false; if (timer) clearTimeout(timer) }
  }, [workspaceId, load, execution?.active, run?.status])
  useEffect(() => {
    // eslint-disable-next-line react/set-state-in-effect -- refresh external task index after state transitions
    if (workspaceId && workspace?.status) void loadHistory()
  }, [workspaceId, workspace?.status, workspace?.active_version_id, loadHistory])
  useEffect(() => {
    if (!workspaceId) return
    let live = true
    api(`/api/workspaces/${workspaceId}/data-services`).then(data => { if (live && selectedWorkspace.current === workspaceId) setServices(data) }).catch(() => {})
    return () => { live = false }
  }, [workspaceId])
  useEffect(() => { if (scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight }, [messages.length, execution?.stage])
  useEffect(() => {
    let live = true
    if (tab === 'audit' && workspaceId) api(`/api/workspaces/${workspaceId}/timeline`).then(data => { if (live && selectedWorkspace.current === workspaceId) setTimeline(data) }).catch(err => { if (live) setError(err.message) })
    return () => { live = false }
  }, [tab, workspaceId, snapshot?.workspace?.revision])

  const selectWorkspace = id => {
    setWorkspaceId(id); setSnapshot(null); setServices(null); setTimeline([]); setTab('overview')
    setSource(null); setReview(null); setComparison(null); setCompareIds(['', ''])
    setInput(''); setFiles([]); setPendingObjective(null); setError(''); setModal('')
    if (!id) window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`)
  }

  const deleteWorkspace = async () => {
    if (!deleteTarget || deleting) return
    const id = deleteTarget.workspace_id
    setDeleting(true); setDeleteError('')
    try {
      await api(`/api/workspaces/${id}?revision=${deleteTarget.revision}`, { method: 'DELETE' })
      deletedWorkspaces.current.add(id)
      setHistory(items => items.filter(item => item.workspace_id !== id))
      if (selectedWorkspace.current === id) selectWorkspace('')
      setDeleteTarget(null)
      await loadHistory()
    } catch (err) {
      setDeleteError(err.message)
      await loadHistory()
    } finally { setDeleting(false) }
  }

  const createWorkspace = async (objective, policy = 'web', runPolicy = 'review') => {
    setBusy(true); setError('')
    try {
      const created = await post('/api/workspaces', { title: objective.slice(0, 40), objective, model_session_id: model?.session_id || null, data_source_preference: policy, run_policy: runPolicy })
      const id = created.workspace.workspace_id
      setSnapshot(created); setWorkspaceId(id); await loadHistory()
      if (!model) {
        setPendingObjective({ workspaceId: id, content: objective })
        setModal('model')
        return
      }
      await post(`/api/workspaces/${id}/messages`, { content: objective })
      await load(id)
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  const send = async payload => {
    if (busy) return
    setBusy(true); setError('')
    try {
      let id = workspaceId
      if (!id) {
        await createWorkspace(payload.content || '读取上传资料并建立估值模型')
        return
      }
      const uploaded = []
      for (const file of files) uploaded.push(await uploadFile(file, role))
      await post(`/api/workspaces/${id}/messages`, { ...payload, file_ids: uploaded.map(item => item.file_id) })
      setInput(''); setFiles([]); await load(id); await loadHistory()
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  const requestReview = async () => {
    setBusy(true); setError('')
    try { const item = await post(`/api/workspaces/${workspaceId}/prevaluation-review`, {}); setReview(item); setModal('review') } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  const approve = async item => {
    setBusy(true); setError('')
    try { await post(`/api/workspaces/${workspaceId}/approvals`, { checkpoint_id: item.checkpoint_id, note: '用户在估值工作区确认整套模型输入与假设' }); setModal(''); setReview(null); await load(); } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  const revise = async body => {
    setBusy(true); setError('')
    try { await post(`/api/workspaces/${workspaceId}/versions`, body); setModal(''); await load() } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  const connectModel = async next => {
    const targetId = pendingObjective?.workspaceId || workspaceId
    if (targetId) await post(`/api/workspaces/${targetId}/model-session`, { model_session_id: next.session_id })
    setModel(next); setModal('')
    const persistedObjective = !pendingObjective && targetId && workspace?.objective && !messages.some(item => item.role === 'user')
      ? { workspaceId: targetId, content: workspace.objective }
      : null
    const objectiveToSend = pendingObjective || persistedObjective
    if (objectiveToSend) {
      await post(`/api/workspaces/${objectiveToSend.workspaceId}/messages`, { content: objectiveToSend.content })
      setPendingObjective(null)
      await load(objectiveToSend.workspaceId)
    } else {
      await load()
    }
  }
  const connectData = async body => { const next = await post(`/api/workspaces/${workspaceId}/data-services`, body); setServices(next); setModal('') }
  const download = async format => {
    try {
      if (run?.result && ['xlsx', 'pdf', 'json'].includes(format)) await downloadRun(run.run_id, format)
      else await downloadWorkspace(workspaceId, format)
      await load()
    } catch (err) { setError(err.message) }
  }
  const downloadSaved = async artifact => {
    try { await downloadSavedArtifact(workspaceId, artifact) } catch (err) { setError(err.message) }
  }
  const openSource = async (fileId, blockId = '', fact = null) => {
    setSource({ fileId, fact, blocks: [], loading: true }); setModal('source')
    try {
      const data = await api(`/api/workspaces/${workspaceId}/sources/${fileId}?block_id=${encodeURIComponent(blockId)}`)
      setSource({ ...data, fileId, fact, loading: false })
    } catch (err) { setSource({ fileId, fact, blocks: [], loading: false, error: err.message }) }
  }
  const moreSource = async () => {
    try {
      const data = await api(`/api/workspaces/${workspaceId}/sources/${source.fileId}?offset=${source.next_offset}`)
      setSource(old => ({ ...old, ...data, blocks: [...old.blocks, ...data.blocks] }))
    } catch (err) { setError(err.message) }
  }
  const compare = async () => {
    if (!compareIds[0] || !compareIds[1]) return
    try { setComparison(await api(`/api/workspaces/${workspaceId}/versions/compare?left=${encodeURIComponent(compareIds[0])}&right=${encodeURIComponent(compareIds[1])}`)) } catch (err) { setError(err.message) }
  }
  const controlWorkspace = async action => {
    setBusy(true); setError('')
    try { await post(`/api/workspaces/${workspaceId}/${action}`, {}); await load() } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  const requirements = snapshot?.requirements || []
  const completedRequirements = requirements.filter(item => item.status === 'satisfied').length
  const versions = snapshot?.versions || []
  const currentPhase = workspace?.current_phase || 'scope'
  const modelAvailable = Boolean(model || snapshot?.connections?.model?.available)
  const isWorking = busy || execution?.active || run && ['created', 'running'].includes(run.status)
  const canReview = session && !isWorking && workspace?.status === 'awaiting_approval'
  const sourceCount = session?.documents?.length || 0
  const leadCount = session?.documents?.filter(item => ['search_snippet', 'official_index'].includes(item.provenance_type)).length || 0
  const confirmedFacts = session?.facts?.filter(item => item.status === 'confirmed').length || 0
  const candidateFacts = session?.facts?.filter(item => item.status === 'proposed').length || 0
  const boundFacts = snapshot?.research_plan?.evidence_counts?.observations_verified || 0
  const scopeComplete = Boolean((session?.draft?.company || session?.draft?.ticker) && session?.draft?.valuation_date && session?.draft?.methods?.length)
  const approvedCheckpoint = snapshot?.checkpoints?.some(item => item.status === 'approved')
  const calculationComplete = Boolean(run?.result && ['completed', 'completed_with_warnings'].includes(run?.status))
  const reportComplete = Boolean(resultDocument || run?.result) && ['completed', 'completed_with_warnings', 'degraded'].includes(workspace?.status)
  const workflowCompleted = [
    scopeComplete && 'scope',
    (sourceCount > 0 || confirmedFacts > 0) && 'evidence',
    requirements.length > 0 && 'model_design',
    approvedCheckpoint && 'approval',
    calculationComplete && 'calculation',
    findings.length > 0 && 'challenge',
    decision && 'decision',
    reportComplete && 'reporting',
  ].filter(Boolean)
  const workflowFailed = workspace?.status === 'failed' && !calculationComplete ? [currentPhase] : []
  const workflowSkipped = ['completed', 'completed_with_warnings', 'degraded', 'failed', 'cancelled'].includes(workspace?.status)
    ? phaseOrder.filter(item => !workflowCompleted.includes(item) && !workflowFailed.includes(item))
    : []

  return <div className="ws-app">
    <aside className="ws-sidebar"><div className="ws-brand"><span><Icon name="chart" size={20}/></span><div><b>ValuationAgent</b><small>可追溯自动估值</small></div></div><button className="ws-new" onClick={() => selectWorkspace('')}><Icon name="plus" size={16}/>新建估值</button><div className="ws-history"><small>估值任务</small>{history.map(item => <div className="ws-history-item" key={item.workspace_id}><button className={`ws-history-open ${item.workspace_id === workspaceId ? 'active' : ''}`} onClick={() => selectWorkspace(item.workspace_id)}><span>{item.title}</span><small>{statusNames[item.status] || item.status}</small></button><button className="ws-history-delete" aria-label={`删除对话：${item.title}`} title="删除此对话" disabled={deleting || (item.workspace_id === workspaceId && isWorking)} onClick={() => { setDeleteTarget(item.workspace_id === workspaceId && workspace ? workspace : item); setDeleteError('') }}><Icon name="trash" size={16}/></button></div>)}</div><div className="ws-sidebar-bottom"><button onClick={() => setModal('model')}><Icon name={model ? 'check' : 'link'} size={16}/><span>{model ? model.model : '连接推理模型'}</span></button><button disabled={!workspaceId} onClick={() => setModal('data')}><Icon name="globe" size={16}/><span>数据服务</span></button></div></aside>
    <main className="ws-main">
      <AccessSession/>
      {!workspaceId ? <EmptyStart onStart={createWorkspace} onConnect={() => setModal('model')} model={model}/> : <>
        <header className="ws-header"><div><p className="eyebrow">VALUATION WORKSPACE · {workspace?.run_policy === 'automatic' ? '自动模式' : '审阅模式'}</p><h1>{workspace?.title || '正在载入…'}</h1><p>{session?.draft?.company || session?.draft?.ticker || workspace?.objective}</p></div><div className="ws-header-actions"><span className={`ws-status status-${workspace?.status}`}><i/>{statusNames[workspace?.status] || '载入中'}</span>{canReview && <button className="button primary" onClick={requestReview}>复核估值方案</button>}{workspace?.status === 'paused' ? <button className="button secondary" onClick={() => controlWorkspace('resume')}>继续</button> : isWorking && <button className="button secondary" onClick={() => controlWorkspace('pause')}>暂停</button>} {!['cancelled', 'completed', 'completed_with_warnings'].includes(workspace?.status) && <button className="button ghost danger" onClick={() => controlWorkspace('cancel')}>取消</button>}<button className="button secondary" onClick={() => download('pdf')} disabled={!snapshot}>导出报告</button></div></header>
        <Workflow phase={currentPhase} status={workspace?.status} completed={workflowCompleted} skipped={workflowSkipped} failed={workflowFailed}/>
        <div className="ws-grid"><section className="ws-chat"><div className="ws-chat-title"><span><Icon name="chat" size={18}/>与估值 Agent 协作</span><small title={snapshot?.runtime?.agent_version}>{model?.model || session?.model_name || '未连接 LLM'}</small></div><div className="ws-chat-scroll" ref={scroll}>
          <div className="ws-context-card"><div><span>任务进度</span><b>{completedRequirements}/{requirements.length || 4} 项准备完成</b></div><div><span>证据与事实</span><b>{sourceCount - leadCount} 份原文 · {leadCount} 条线索 · {boundFacts} 条原文已绑定 · {confirmedFacts} 条通过字段准入{candidateFacts > 0 ? ` · ${candidateFacts} 条候选待处理` : ''}</b></div></div>
          {snapshot?.runtime?.agent_version && <small className="ws-runtime">执行引擎 {snapshot.runtime.agent_version} · 来源等级与字段准入分别展示</small>}
          {messages.filter(item => ['user', 'assistant'].includes(item.role)).map(message => <article key={message.message_id} className={`ws-message ${message.role}`}><span>{message.role === 'user' ? '我' : <Icon name="spark" size={16}/>}</span><div><small>{message.role === 'user' ? '你' : 'ValuationAgent'}</small>{message.role === 'assistant' ? <MessageBody content={message.content}/> : <p>{message.content}</p>}</div></article>)}
          {snapshot?.plan?.length > 0 && <ol className="ws-agent-plan" aria-label="Agent 任务计划">{snapshot.plan.map((step, index) => <li key={index} data-status={step.status}><span>{step.status === 'completed' ? '✓' : step.status === 'in_progress' ? '◉' : '○'}</span>{step.title}</li>)}</ol>}
          {session?.pending_decision && !isWorking && <section className="ws-decision-prompt" aria-label="方案选择"><h3>{session.pending_decision.question}</h3><div>{session.pending_decision.options.map((option, index) => <button key={option.label} disabled={!modelAvailable} onClick={() => setInput(`关于“${session.pending_decision.question}”，我选择${option.label}。${option.description ? `具体方案：${option.description}` : ''}`)}><strong>{String.fromCharCode(65 + index)} · {option.label}</strong><span>{option.description}</span></button>)}</div><small>点击方案填入下方输入框，补充或修改后发送；也可以直接输入自己的方案。选择不等于批准数值计算。</small></section>}
          {session?.resume_context?.reason && !isWorking && <section className="ws-resume-card"><b>已保存续做检查点 · {session.resume_context.reason}</b><p>下一轮先核验已有资料与字段修复，再继续尚未处理的年度或可比样本，不重跑上一轮检索。</p><button className="button secondary" disabled={!modelAvailable} onClick={() => send({ content: '从保存的检查点继续，先检查已有事实修复和逐年覆盖，再处理其他未完成目标；不要重复上轮无进展检索。' })}>从检查点继续</button></section>}
          <EvidenceProgress plan={snapshot?.research_plan}/>
          <InterpretationProgress plan={snapshot?.research_plan}/>
          <ArtifactShelf artifacts={snapshot?.artifacts || []} onDownload={downloadSaved}/>
          {isWorking && <div className="ws-working"><span className="mini-spinner"/><div><b>{run && ['created', 'running'].includes(run.status) ? '确定性金融模型正在计算' : 'Agent 正在推进估值任务'}</b><small>{execution?.stage ? `当前步骤：${execution.stage}` : '正在根据模型需要取证、核验或生成结果'}</small></div></div>}
          {run?.result && <div className="ws-suggestions"><button onClick={() => send({ content: '解释本次 WACC 的计算依据和每个组成部分' })}>为什么 WACC 是这个数？</button><button onClick={() => send({ content: '下钻解释企业价值到股权价值的桥接过程' })}>下钻权益桥</button><button onClick={() => setModal('revision')}>调参重算</button></div>}
        </div><ErrorNotice message={error} onDismiss={() => setError('')}/><form className="ws-composer" onSubmit={event => { event.preventDefault(); if (!modelAvailable) { setModal('model'); return } if (input.trim() || files.length) send({ content: input.trim() }) }}>{!modelAvailable && <button type="button" className="ws-connect-inline" onClick={() => setModal('model')}><Icon name="link" size={14}/>连接核心推理模型后继续任务</button>}<div className="ws-files">{files.map((file, index) => <span key={`${file.name}-${index}`}><Icon name="file" size={13}/>{file.name}<button type="button" onClick={() => setFiles(old => old.filter((_, i) => i !== index))}>×</button></span>)}</div><textarea value={input} onChange={e => setInput(e.target.value)} placeholder={!modelAvailable ? '请先连接核心推理模型…' : run?.result ? '追问依据、质疑结果，或说“把 WACC 改为 8%”…' : '补充公司信息、估值目标、业务假设，或上传你已有的资料…'} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); e.currentTarget.form.requestSubmit() } }}/><div><div><button type="button" onClick={() => fileInput.current?.click()} disabled={isWorking || !modelAvailable}><Icon name="plus" size={18}/></button><input ref={fileInput} hidden multiple type="file" accept=".pdf,.docx,.xlsx,.csv,.tsv,.html,.htm,.json,.txt,.md" onChange={e => { setFiles(old => [...old, ...Array.from(e.target.files || [])].slice(0, 8)); e.target.value = '' }}/><select value={role} disabled={!modelAvailable} onChange={e => setRole(e.target.value)}><option value="historical_financials">财务资料</option><option value="assumptions">经营假设</option><option value="comparables">可比公司</option><option value="evidence">其他证据</option></select></div><small>Enter 发送 · Shift + Enter 换行</small><button className="ws-send" disabled={isWorking || !modelAvailable || (!input.trim() && !files.length)}><Icon name="send" size={17}/></button></div></form></section>
          <aside className="ws-notebook"><nav>{tabs.map(([key, label]) => <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{label}{key === 'risk' && findings.some(item => ['high', 'blocking'].includes(item.severity)) && <i/>}</button>)}</nav><div className="ws-notebook-body">
            {tab === 'overview' && <ResultOverview run={run} report={resultDocument} findings={findings} decision={decision} status={workspace?.status} working={isWorking} onRevise={() => setModal('revision')} onDownload={download}/ >}
            {tab === 'inputs' && <div className="ws-list"><h2>模型准备清单</h2>{requirements.map(item => <div className={`ws-requirement ${item.status}`} key={item.requirement_id}><span>{item.status === 'satisfied' ? '✓' : item.status === 'unavailable' ? '!' : '○'}</span><div><b>{item.label}</b><small>{item.status === 'satisfied' ? item.resolution : item.status === 'unavailable' ? item.resolution : '等待 Agent 完成或用户确认'}{item.attempt_count > 0 && ` · 已检索 ${item.attempt_count}/${item.attempt_limit} 次`}</small></div></div>)}{session?.forecast_proposal && <section className="ws-card"><h3>预测假设</h3><p>{session.forecast_proposal.rationale}</p>{session.forecast_proposal.risks.map(item => <small key={item}>· {item}</small>)}</section>}{canReview && <button className="button primary full" onClick={requestReview}>查看集中复核方案</button>}</div>}
            {tab === 'evidence' && <div className="ws-list"><div className="ws-section-heading"><div><h2>证据账本</h2><p>等级是来源与匹配质量分层，不是“真实性概率”；每条事实可回到引文、位置和哈希。</p></div><button onClick={() => fileInput.current?.click()}>补充文件</button></div>{session?.documents?.map(doc => <div className="ws-source" key={doc.file_id}><Icon name="file"/><div><b><button className="ws-source-link" onClick={() => openSource(doc.file_id)}>{doc.name}</button> <span className={`authority authority-${doc.authority_tier || 'D'}`}>{doc.authority_tier || 'D'}级 · 来源评分 {Math.round((doc.source_confidence ?? .5) * 100)}/100</span></b><small>{doc.provenance_type || doc.role} · {doc.provider || '未标明发布方'} · {doc.block_count} 个原文块 · SHA-256 {doc.sha256?.slice(0, 16)}</small>{doc.warnings.map(item => <p key={item}>{item}</p>)}</div></div>)}{!sourceCount && <div className="ws-empty-panel"><Icon name="folder" size={28}/><h3>暂无已取得的原始资料</h3><p>Agent 会按模型缺口检索；你的附件是补充，不是启动估值的前提。</p></div>}<h2>标准化事实</h2>{session?.facts?.slice().reverse().slice(0, 40).map(fact => <div className={`ws-fact ${fact.status}`} key={fact.fact_id}><div><b>{fact.standard_metric || fact.metric}</b><span>{fact.status}</span></div><p>{fact.period} · {fact.raw_value} {fact.unit}</p><small>{fact.metric} · 来源 {fact.block_id}</small><button onClick={() => fact.block_id.startsWith('message:') ? (setSource({ fact, blocks: [] }), setModal('source')) : openSource(fact.block_id.split(':')[0], fact.block_id, fact)}>查看原文与口径</button></div>)}</div>}
            {tab === 'risk' && <div className="ws-list"><h2>确定性风险检查与处置</h2>{decision && <section className={`ws-decision ${decision.outcome}`}><small>综合决策层</small><h3>{decision.selected_action}</h3><p>{decision.rationale}</p></section>}{findings.map(item => { const disposition = dispositions.find(row => row.finding_id === item.finding_id); return <div className={`ws-finding ${item.severity}`} key={item.finding_id}><div><span>{item.severity}</span><b>{item.title}</b></div><p>{item.analysis}</p><small>建议：{item.recommendation}</small>{disposition && <small className="ws-disposition">处置：{disposition.decision} · {disposition.resulting_action}</small>}</div> })}{!findings.length && <div className="ws-empty-panel"><Icon name="shield" size={28}/><h3>计算完成后自动挑刺</h3><p>独立检查证据覆盖、终值占比、WACC 与增长率间距、可比样本和敏感性。</p></div>}</div>}
            {tab === 'versions' && <div className="ws-list"><div className="ws-section-heading"><div><h2>估值版本</h2><p>每次计算冻结独立快照，仅供审计与对比，不切换旧工作流。</p></div>{run?.result && <button onClick={() => setModal('revision')}>创建版本</button>}</div>{versions.map(item => <div className={`ws-version ${item.version_id === workspace?.active_version_id ? 'active' : ''}`} key={item.version_id}><div><b>v{item.number} · {item.reason}</b><small>{item.status} · 输入 {item.request_hash.slice(0, 12)} · 结果 {item.result_hash?.slice(0, 12) || '待生成'}</small></div></div>)}{versions.length >= 2 && <section className="ws-compare"><h3>版本对比</h3><div><select value={compareIds[0]} onChange={e => setCompareIds(old => [e.target.value, old[1]])}><option value="">选择左侧版本</option>{versions.map(item => <option key={item.version_id} value={item.version_id}>v{item.number}</option>)}</select><select value={compareIds[1]} onChange={e => setCompareIds(old => [old[0], e.target.value])}><option value="">选择右侧版本</option>{versions.map(item => <option key={item.version_id} value={item.version_id}>v{item.number}</option>)}</select><button onClick={compare}>比较</button></div>{comparison && <pre>{JSON.stringify(comparison.output_changes, null, 2)}</pre>}</section>}</div>}
            {tab === 'audit' && <div className="ws-list"><div className="ws-section-heading"><div><h2>行动与计算账本</h2><p>只显示可核验的动作、工具、引用和结果摘要；不展示模型私有思维链，也不保存 API Key。</p></div><button onClick={() => api(`/api/workspaces/${workspaceId}/reproducibility`).then(data => { const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `${workspaceId}-replay.json`; a.click(); URL.revokeObjectURL(url) })}>复现清单</button></div>{timeline.slice().reverse().map(item => <div className={`ws-audit ${item.status}`} key={item.id}><i/><div><b>{item.summary}</b><small>{item.actor} · {item.type}{item.tool ? ` · ${item.tool}` : ''}{item.duration_ms != null ? ` · ${item.duration_ms}ms` : ''} · {new Date(item.timestamp).toLocaleString()}</small></div></div>)}</div>}
          </div></aside></div>
      </>}
    </main>
    {deleteTarget && <Modal title="删除历史对话" onClose={() => { if (!deleting) setDeleteTarget(null) }}>
      <p>确定删除“{deleteTarget.title}”吗？</p><p>将永久删除该任务的对话、事实与审计记录、计算版本及工作区内生成的报告，无法撤销。其他任务、共享原始文件和已下载到电脑的报告不受影响。</p>
      <ErrorNotice message={deleteError}/><div className="ws-delete-actions"><button className="button secondary" disabled={deleting} onClick={() => setDeleteTarget(null)}>保留对话</button><button className="button ghost danger" disabled={deleting} onClick={deleteWorkspace}>{deleting ? '正在删除…' : '确认永久删除'}</button></div>
    </Modal>}
    {modal === 'source' && source && <Modal title="原文与证据链" className="wide-modal" onClose={() => setModal('')}>
      {source.fact && <section className="ws-card"><h3>{source.fact.metric} · {source.fact.period}</h3><p>{source.fact.raw_value} {source.fact.unit} · {source.fact.scope}</p><EvidenceQuality fact={source.fact}/><blockquote>{source.fact.quote}</blockquote><small>{source.fact.block_id} · SHA-256 {source.fact.source_sha256 || '未记录'}</small>{source.fact.warnings?.map(warning => <p key={warning}>{warning}</p>)}</section>}
      {source.loading && <p>正在读取保存的原文快照…</p>}<ErrorNotice message={source.error}/>
      {source.blocks.map(block => <section className="ws-source-block" key={block.block_id}><small>{block.block_id} · {JSON.stringify(block.location)}</small><pre>{block.text}</pre></section>)}
      {source.next_offset != null && <button className="button secondary" onClick={moreSource}>读取更多原文</button>}
    </Modal>}
    {modal === 'model' && <ModelModal current={model} onClose={() => setModal('')} onConnected={connectModel}/ >}
    {modal === 'data' && <DataModal current={services} onClose={() => setModal('')} onConnected={connectData}/ >}
    {modal === 'review' && <ReviewModal review={review} busy={busy} onClose={() => setModal('')} onApprove={approve}/ >}
    {modal === 'revision' && <RevisionModal run={run} busy={busy} onClose={() => setModal('')} onSubmit={revise}/ >}
    {busy && !workspaceId && <div className="ws-global-loading"><span className="mini-spinner"/>正在创建估值工作区…</div>}
  </div>
}
