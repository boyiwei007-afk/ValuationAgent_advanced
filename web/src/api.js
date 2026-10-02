const base = (import.meta.env?.VITE_API_BASE_URL || '').replace(/\/$/, '')

export async function api(path, options = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), options.method ? 180000 : 30000)
  const abort = () => controller.abort()
  options.signal?.addEventListener('abort', abort, { once: true })
  if (options.signal?.aborted) controller.abort()
  try {
    const response = await fetch(base + path, { ...options, signal: controller.signal, headers: { ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...options.headers } })
    if (!response.ok) {
      const body = await response.json().catch(() => null)
      const detail = body?.detail
      const error = new Error(Array.isArray(detail) ? detail.map(e => `${e.loc?.slice(1).join('.') || ''}: ${e.msg}`).join('\n') : typeof detail === 'string' ? detail : detail?.message || `服务请求失败 / Request failed (${response.status})`)
      error.status = response.status
      throw error
    }
    // Keep cancellation active while reading the response body as well.
    return response.status === 204 ? null : await response.json()
  } catch (error) {
    if (error.status) throw error
    throw new Error('连接中断，请检查服务后重试；未发送内容已保留。 / Connection interrupted. Check the service and retry; your draft is preserved.')
  } finally {
    clearTimeout(timer)
    options.signal?.removeEventListener('abort', abort)
  }
}
export const post = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body) })
async function downloadArtifact(path, filename) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 60000)
  try {
    const response = await fetch(base + path, { signal: controller.signal })
    if (!response.ok) {
      const body = await response.json().catch(() => null)
      throw new Error(typeof body?.detail === 'string' ? body.detail : `报告下载失败 (${response.status}) / Download failed`)
    }
    const blob = await response.blob()
    if (!blob.size) throw new Error('报告内容为空，请重试 / Empty report. Please retry.')
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url; link.download = filename
    document.body.append(link); link.click(); link.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('报告生成超时，已保存的进度不受影响；可重试或先下载 HTML。 / Export timed out. Retry or download HTML.')
    throw err
  } finally { clearTimeout(timer) }
}
export const downloadWorkspace = (id, format) => downloadArtifact(`/api/workspaces/${id}/export?format=${format}`, `${id}.${format}`)
export const downloadSavedArtifact = (id, artifact) => downloadArtifact(`/api/workspaces/${id}/artifacts/${artifact.artifact_id}`, `${artifact.artifact_id}-${artifact.filename}`)
export const downloadRun = (id, format) => downloadArtifact(`/api/runs/${id}/export?format=${format}`, `valuation-${id}.${format}`)
export async function uploadFile(file, role) {
  if (file.size > 50 * 1024 * 1024) throw new Error('文件不能超过 50 MB / Maximum file size: 50 MB')
  const data = new FormData()
  data.append('file', file)
  data.append('role', role)
  return api('/api/files', { method: 'POST', body: data })
}
