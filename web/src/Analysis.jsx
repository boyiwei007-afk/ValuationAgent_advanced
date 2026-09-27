import { useState, useId } from 'react'
import Icon from './Icons'
import { money, percent, methodLabel, valuationRanges, sensitivityGrid, studyLabel } from './domain'
import { Card, Empty } from './ui'

export function RangeChart({ result, t }) {
  const rows = valuationRanges(result)
  const max = Math.max(1, ...rows.map(row => Number(row.range_high))) * 1.12
  const min = Math.min(0, ...rows.map(row => Number(row.range_low)))
  const position = value => `${(Number(value) - min) / (max - min) * 100}%`
  return <Card title={t('估值区间对照', 'Valuation ranges')} subtitle={`${result.currency} / ${t('股', 'share')} · ${t('区间与基准值', 'Ranges and base values')}`} extra={<Icon name="layers" size={18}/>}>
    {rows.map((row, index) => <div className="range-row" key={row.method}><div><b>{methodLabel(row.method)}</b><strong>{money(row.per_share_value)}</strong></div><div className="range-track"><span className={`range-band tone-${index}`} style={{ left: position(row.range_low), width: `${(Number(row.range_high) - Number(row.range_low)) / (max - min) * 100}%` }}/><i className={`range-point tone-${index}`} style={{ left: position(row.per_share_value) }}/></div><small>{money(row.range_low)} — {money(row.range_high)}</small></div>)}
    <div className="range-axis"><span>{money(min, 0)}</span><span>{money((max + min) / 2, 0)}</span><span>{money(max, 0)}</span></div>
    {result.relative.filter(row => row.status !== 'success').map(row => <p className="field-hint warning-text" key={row.method}>{methodLabel(row.method)} · {row.reason}</p>)}
    <p className="chart-note"><Icon name="shield" size={15}/>{result.reconciliation.conclusion}</p>
  </Card>
}

export function ForecastChart({ result, t }) {
  const gradient = useId().replaceAll(':', '')
  const data = result.forecast || []
  if (!data.length) return <Card title={t('经营预测', 'Operating forecast')}><Empty icon="chart" title={t('未运行现金流预测', 'No cash flow forecast')}>{t('当前所选方法不需要 DCF 预测。', 'The selected methods do not require a DCF forecast.')}</Empty></Card>
  const high = Math.max(1, ...data.flatMap(row => [Number(row.revenue), Number(row.fcff)])) / 1e8 * 1.14
  const low = Math.min(0, ...data.map(row => Number(row.fcff))) / 1e8 * 1.14
  const x = index => 52 + index * (410 / Math.max(1, data.length - 1))
  const y = value => 182 - (Number(value) / 1e8 - low) / (high - low) * 151
  const revenue = data.map((row, index) => `${index ? 'L' : 'M'}${x(index)},${y(row.revenue)}`).join(' ')
  const cash = data.map((row, index) => `${index ? 'L' : 'M'}${x(index)},${y(row.fcff)}`).join(' ')
  return <Card title={t('收入与自由现金流', 'Revenue & free cash flow')} subtitle={`${t('年度预测', 'Annual forecast')} · ${result.currency} ${t('亿元', '× 100 million')}`}>
    <div className="chart-legend"><span><i className="teal-key"/>{t('营业收入', 'Revenue')}</span><span><i className="blue-key"/>FCFF</span></div>
    <svg className="forecast-chart" viewBox="0 0 500 224" role="img" aria-label={t('年度收入与自由现金流预测图，下方提供数据表。', 'Revenue and free cash flow forecast. Data table available below.')}><defs><linearGradient id={gradient} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2aafa1" stopOpacity=".19"/><stop offset="100%" stopColor="#2aafa1" stopOpacity=".01"/></linearGradient></defs>{[0,1,2,3].map(index => { const value = low + (high - low) * index / 3; const yy = y(value * 1e8); return <g key={index}><line x1="52" x2="474" y1={yy} y2={yy} stroke="#e8eef3" strokeDasharray="3 4"/><text x="42" y={yy + 4} textAnchor="end">{money(value, 0)}</text></g> })}<path d={`${revenue} L${x(data.length - 1)},${y(0)} L52,${y(0)} Z`} fill={`url(#${gradient})`}/><path d={revenue} fill="none" stroke="#169c8f" strokeWidth="2.5"/><path d={cash} fill="none" stroke="#5987ce" strokeWidth="2.5"/>{data.map((row, index) => <g key={row.year}><circle cx={x(index)} cy={y(row.revenue)} r="3.5" fill="#fff" stroke="#169c8f" strokeWidth="2"/><circle cx={x(index)} cy={y(row.fcff)} r="3" fill="#5987ce"/>{(data.length <= 7 || index % 2 === 0 || index === data.length - 1) && <text x={x(index)} y="211" textAnchor="middle">{row.year}E</text>}</g>)}</svg>
    <details className="data-details"><summary>{t('查看预测数据', 'View forecast data')}</summary><div className="table-scroll"><table><thead><tr><th>{t('年度', 'Year')}</th><th>{t('收入', 'Revenue')}</th><th>FCFF</th><th>{t('增长率', 'Growth')}</th></tr></thead><tbody>{data.map(row => <tr key={row.year}><td>{row.year}</td><td>{money(Number(row.revenue) / 1e8)}</td><td>{money(Number(row.fcff) / 1e8)}</td><td>{percent(row.revenue_growth)}</td></tr>)}</tbody></table></div></details>
  </Card>
}

export function Sensitivity({ result, t }) {
  const [selected, setSelected] = useState(null)
  const grid = sensitivityGrid(result.sensitivity || [])
  if (!grid.rows.length) return <RelativeSensitivity result={result} t={t}/>
  const values = grid.valid.map(cell => Number(cell.per_share_value))
  const low = Math.min(...values), high = Math.max(...values)
  const studies = result.sensitivity_studies || []
  const completedStudies = studies.filter(study => study.status === 'completed' && study.max_relative_change != null).sort((a, b) => Number(b.max_relative_change) - Number(a.max_relative_change))
  const maxImpact = Math.max(0.0001, ...completedStudies.map(study => Number(study.max_relative_change)))
  const unavailableCount = studies.filter(study => study.status === 'not_available').length
  const selectedCell = grid.byKey.get(selected) || grid.valid.find(cell => Math.abs(Number(cell.wacc) - Number(result.assumptions.wacc)) < 1e-8 && Math.abs(Number(cell.terminal_growth) - Number(result.assumptions.terminal_growth)) < 1e-8) || grid.valid[0]
  return <Card title={t('敏感性分析', 'Sensitivity analysis')} subtitle={`WACC × ${t('永续增长率', 'terminal growth')} · ${result.currency} / ${t('股', 'share')}`}>
    <div className="heat-caption">{t('列：永续增长率 g · 行：WACC', 'Columns: terminal growth g · Rows: WACC')}</div><div className="table-scroll"><table className="heatmap"><thead><tr><th>WACC / g</th>{grid.columns.map(value => <th key={value}>{percent(value)}</th>)}</tr></thead><tbody>{grid.rows.map(wacc => <tr key={wacc}><th>{percent(wacc)}</th>{grid.columns.map(growth => {
      const key = `${wacc}:${growth}`; const cell = grid.byKey.get(key)
      const valid = cell?.valid && cell.per_share_value != null
      const ratio = valid ? (Number(cell.per_share_value) - low) / (high - low || 1) : 0
      const background = valid ? `rgb(${Math.round(235 - ratio * 218)}, ${Math.round(248 - ratio * 112)}, ${Math.round(245 - ratio * 119)})` : '#edf0f4'
      const active = selectedCell === cell
      return <td key={key}><button style={{ background, color: ratio > .58 ? '#fff' : '#28544f' }} className={active ? 'selected' : ''} aria-pressed={active} aria-label={`WACC ${percent(wacc)}, g ${percent(growth)}, ${valid ? money(cell.per_share_value) : t('无效组合', 'Invalid combination')}`} onClick={() => setSelected(key)}>{valid ? money(cell.per_share_value) : '—'}</button></td>
    })}</tr>)}</tbody></table></div><div className="heat-legend"><span>{t('低', 'Low')}</span><i/><span>{t('高', 'High')}</span><small>{t('颜色代表每股估值', 'Color = value per share')}</small></div>
    {selectedCell && <div className="heat-selection"><div><small>WACC {percent(selectedCell.wacc)} · g {percent(selectedCell.terminal_growth)}</small><strong>{selectedCell.valid ? `${money(selectedCell.per_share_value)} ${result.currency}` : t('无效组合', 'Invalid combination')}</strong></div><span>{t('情景值 / 股', 'Scenario / share')}</span></div>}
    <p className="field-hint">{t('点击格子查看情景，不会改动本次假设。修改假设请在对话中提出。', 'Select a cell to inspect a scenario. To change assumptions, use the conversation.')}</p>
    {!!studies.length && <div className="sensitivity-catalogue"><div className="catalogue-heading"><div><b>{t('敏感性项目覆盖', 'Sensitivity study coverage')}</b><small>{t(`${completedStudies.length} 项已重算 · ${unavailableCount} 项缺少输入`, `${completedStudies.length} recalculated · ${unavailableCount} missing inputs`)}</small></div><span>{t('影响按绝对变动排序', 'Sorted by absolute impact')}</span></div>{completedStudies.map(study => <div className="impact-row" key={study.study_id}><span>{study.study_id}</span><div><b>{studyLabel(study.parameter, t)}</b><i><em style={{ width: `${Math.max(2, Number(study.max_relative_change) / maxImpact * 100)}%` }}/></i></div><strong>{percent(study.max_relative_change)}</strong><small className={`impact-${study.classification}`}>{{ high: t('高敏感', 'High'), medium: t('中敏感', 'Medium'), low: t('低敏感', 'Low'), unknown: t('未判定', 'Unknown') }[study.classification] || study.classification}</small></div>)}{unavailableCount > 0 && <details className="data-details unavailable-studies"><summary>{t('查看未测试项目及原因', 'View unavailable studies')}</summary>{studies.filter(study => study.status === 'not_available').map(study => <p key={study.study_id}><b>{study.study_id} · {studyLabel(study.parameter, t)}</b><span>{study.rationale}</span></p>)}</details>}</div>}
  </Card>
}

function RelativeSensitivity({ result, t }) {
  const studies = (result.sensitivity_studies || []).filter(study => study.study_id.startsWith('R-'))
  if (!studies.length) return null
  return <Card title={t('相对估值敏感性', 'Relative valuation sensitivity')} subtitle={t('单因素 ±10% · 每股价值', 'One factor ±10% · value per share')}><div className="table-scroll"><table><thead><tr><th>{t('因素', 'Factor')}</th><th>−10%</th><th>{t('基准', 'Base')}</th><th>+10%</th></tr></thead><tbody>{studies.map(s => <tr key={s.study_id}><td>{studyLabel(s.parameter, t)}</td><td>{money(s.low_per_share)}</td><td>{money(s.baseline_per_share)}</td><td>{money(s.high_per_share)}</td></tr>)}</tbody></table></div><p className="field-hint">{t('仅改变财务指标或可比倍数，其他输入不变；压力情景不代表概率。', 'Changes only the financial metric or comparable multiples. Stress scenarios are not probabilities.')}</p></Card>
}

function QualityCard({ result, t }) {
  const quality = result.data_quality
  if (!quality) return null
  const confidence = {
    high: t('高', 'High'), medium: t('中', 'Medium'), low: t('低', 'Low'),
  }[quality.confidence] || quality.confidence
  return <Card title={t('数据质量与适用边界', 'Data quality & scope')} subtitle={t('结果等级综合历史可比性、证据、市场参数和降级路径', 'Grade reflects history, evidence, market inputs and fallback paths')} extra={<Icon name="shield" size={18}/>}> <dl className="fact-list"><div><dt>{t('结果等级', 'Result grade')}</dt><dd>{quality.result_grade || 'C'}</dd></div><div><dt>{t('总体置信度', 'Overall confidence')}</dt><dd>{confidence}</dd></div><div><dt>{t('市场参数', 'Market inputs')}</dt><dd>{{ verified_current: t('已核验且新鲜', 'Verified/current'), verified_stale: t('已核验但过期', 'Verified/stale'), unverified: t('未完整核验', 'Unverified'), not_applicable: t('不适用', 'N/A') }[quality.market_input_quality] || quality.market_input_quality}</dd></div><div><dt>{t('可比历史', 'Comparable history')}</dt><dd>{quality.comparable_years} / {quality.historical_years} {t('年', 'years')}</dd></div><div><dt>{t('证据覆盖', 'Evidence coverage')}</dt><dd>{percent(quality.evidence_coverage)}</dd></div><div><dt>{t('同业样本', 'Peer sample')}</dt><dd>{{ adequate: t('充足', 'Adequate'), limited: t('有限', 'Limited'), insufficient: t('不足', 'Insufficient'), not_requested: t('未选用', 'Not requested') }[quality.peer_sample_quality] || quality.peer_sample_quality}</dd></div></dl>{(quality.degraded_fields || []).length > 0 && <p className="warning-line"><Icon name="alert" size={16}/>{t('已降级字段', 'Fallback fields')}：{quality.degraded_fields.join(' / ')}</p>}{(quality.notes || []).map((note, index) => <p className="warning-line" key={index}><Icon name="alert" size={16}/>{note}</p>)}</Card>
}

export function Results({ record, t, compact = false }) {
  if (!record?.result) return <Empty title={t('估值结果将在这里呈现', 'Your valuation will appear here')}>{t('开始研究后，可比较估值区间、查看经营预测和敏感性分析。', 'Start a study to compare ranges, forecasts and sensitivity.')}</Empty>
  const result = record.result
  const formalModel = result.model_version.includes('finance-team')
  return <div className={compact ? 'analysis-stack' : 'analysis-grid'}><div className="result-metrics"><div><small>{result.dcf ? t('DCF 基准 / 股', 'DCF base / share') : t('相对估值基准 / 股', 'Relative base / share')}</small><strong>{money(result.dcf?.per_share_value ?? result.relative.find(item => item.status === 'success')?.per_share_value)}</strong><span>{result.currency} · {formalModel ? t('金融小组模型', 'Finance-team model') : t('参考模型', 'Reference model')}</span></div><div><small>{result.dcf ? 'WACC / g' : t('适用方法', 'Applicable methods')}</small><strong className="small-number">{result.dcf ? <>{percent(result.assumptions.wacc)} <em>/</em> {percent(result.assumptions.terminal_growth)}</> : record.request.methods.map(methodLabel).join(' / ')}</strong><span>{result.dcf ? `${result.forecast.length} ${t('年预测期', 'forecast years')} · ${t('稳定期ROIC', 'Stable ROIC')} ${percent(result.dcf.stable_roic)}` : t('不使用DCF折现率与增长假设', 'DCF assumptions are not used')}</span></div></div><RangeChart result={result} t={t}/><QualityCard result={result} t={t}/><ForecastChart result={result} t={t}/><Sensitivity result={result} t={t}/><Card title={t('研究说明', 'Research notes')}><p className="report-text">{result.executive_summary}</p>{result.warnings.map((warning, index) => <p className="warning-line" key={index}><Icon name="alert" size={16}/>{warning}</p>)}<div className="source-footer"><span>{result.mode.toUpperCase()} · {result.model_version}</span><span>v{record.revision}</span></div></Card></div>
}

const financialLabels = [
  ['revenue', '营业收入', 'Revenue'], ['net_income_parent', '归母净利润', 'Net income to parent'],
  ['ebitda', 'EBITDA', 'EBITDA'], ['common_shares', '普通股股数', 'Common shares'],
  ['diluted_shares', '稀释后股数', 'Diluted shares'],
  ['cash_and_non_operating_assets', '现金及非经营资产', 'Cash & non-operating assets'],
  ['interest_bearing_debt', '有息负债', 'Interest-bearing debt'],
  ['lease_liabilities', '租赁负债', 'Lease liabilities'],
  ['minority_interest', '少数股东权益', 'Minority interest'],
]
export function Evidence({ record, events, artifacts, t }) {
  if (!record) return <Empty icon="shield" title={t('让数据来源清晰可查', 'Keep data sources visible')}>{t('本次财务、假设依据和审核结果会随研究一起保存。', 'Financial data, assumption sources and review results are saved with the study.')}</Empty>
  const result = record.result
  const data = result?.effective_financials || record.request.financials
  const validationId = [...events].reverse().find(e => e.tool === 'validate_financials' && e.payload?.artifact_id)?.payload.artifact_id
  const validation = artifacts.find(a => a.artifact_id === validationId)?.output
  const requestedMethods = record.request.requested_methods?.length ? record.request.requested_methods : record.request.methods
  const excludedMethods = Object.entries(record.request.excluded_methods || {})
  return <div className="analysis-stack"><Card title={t('本次研究参数', 'Study parameters')}><dl className="fact-list">{[[t('企业', 'Company'), record.request.company.name || record.request.company.ticker], [t('估值日', 'Valuation date'), record.request.valuation_date], [t('数据来源', 'Data source'), record.request.data_source], [t('假设来源', 'Assumption source'), record.request.assumption_source], [t('原请求方法', 'Requested methods'), requestedMethods.map(methodLabel).join(' / ')], [t('实际估值方法', 'Effective methods'), record.request.methods.map(methodLabel).join(' / ')], [t('语言', 'Language'), record.request.language], [t('版本', 'Revision'), `v${record.revision}`]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>{!!excludedMethods.length && <div className="valuation-method-fallback"><b>{t('未采用方法', 'Excluded methods')}</b><ul>{excludedMethods.map(([method, reason]) => <li key={method}><strong>{methodLabel(method)}</strong>：{reason}</li>)}</ul></div>}</Card>{data && <Card title={t('财务快照', 'Financial snapshot')} subtitle={`${data.period_end} · ${data.currency} · ${t('原始金额单位：元 / 股数单位：股', 'Amounts in currency units; shares in units')}`}><p className="source-label"><Icon name="file" size={15}/>{data.source_label}</p><dl className="fact-list">{financialLabels.map(([key, zh, en]) => <div key={key}><dt>{t(zh, en)}</dt><dd>{money(data[key], 0)}</dd></div>)}</dl><details className="data-details"><summary>{t('展开逐字段来源', 'Field-level evidence')}</summary>{Object.entries(data.evidence || {}).length ? Object.entries(data.evidence).map(([key, refs]) => <div className="evidence-entry" key={key}><b>{key}</b>{refs.map((ref, index) => <p key={index}>{ref.source}{ref.page ? ` · p.${ref.page}` : ''}{ref.sheet ? ` · ${ref.sheet} ${ref.cell || ''}` : ''}{ref.note ? ` · ${ref.note}` : ''}</p>)}</div>) : <p className="field-hint">{t('当前快照未附逐字段证据。', 'No field-level evidence was supplied.')}</p>}</details></Card>}
    <Card title={t('财务审核', 'Financial review')}>{validation ? <><p className="field-hint">{t('仅覆盖本次已提供字段与已配置规则。', 'Covers the supplied fields and configured rules.')}</p>{validation.map((finding, index) => <div className={`finding ${finding.severity}`} key={finding.rule_id + index}><Icon name={finding.severity === 'info' ? 'check' : 'alert'} size={17}/><div><b>{finding.rule_id}</b><p>{finding.message}</p></div></div>)}{validation.length === 0 && <p className="field-hint">{t('已运行的规则未返回异常项。', 'Executed rules returned no findings.')}</p>}</> : <p className="field-hint">{t('财务审核尚未生成结果。', 'Financial review results are not available yet.')}</p>}</Card>
    {result && <Card title={t('假设依据', 'Assumption sources')}><p className="source-label">{result.assumptions.source}</p>{Object.entries(result.assumptions.rationale || {}).map(([key, value]) => <div className="evidence-entry" key={key}><b>{key}</b><p>{value}</p></div>)}</Card>}
  </div>
}
