import assert from 'node:assert/strict'

const origin = process.env.VALUATION_TEST_URL || 'http://127.0.0.1:8001'
async function request(path, body) {
  const response = await fetch(origin + path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  assert.ok(response.ok, `${path}: ${response.status}`)
  return response.json()
}

const page = await fetch(origin + '/')
assert.equal(page.status, 200)
const html = await page.text()
const bundle = html.match(/src="(\/assets\/[^"]+\.js)"/)
assert.ok(bundle)
assert.equal((await fetch(origin + bundle[1])).status, 200)
assert.equal((await request('/health')).status, 'ok')
const created = await request('/api/workspaces', { title: 'Workspace integration fixture', run_policy: 'automatic' })
const workspaceId = created.workspace.workspace_id
assert.equal(created.workspace.run_policy, 'automatic')
await request(`/api/workspaces/${workspaceId}/messages`, { content: 'No model is connected; preserve this message.', request_id: 'integration_turn_001' })
let snapshot
for (let index = 0; index < 50; index++) {
  snapshot = await request(`/api/workspaces/${workspaceId}`)
  if (!snapshot.execution.active) break
  await new Promise(resolve => setTimeout(resolve, 150))
}
assert.ok(snapshot.messages.some(message => message.content.includes('No model is connected')))
assert.equal(snapshot.active_run, null)
assert.equal(snapshot.execution.status, 'failed')
assert.match(snapshot.messages.at(-1).content, /MODEL_CONNECTION_REQUIRED/)
const exportResponse = await fetch(`${origin}/api/workspaces/${workspaceId}/export?format=html`)
assert.equal(exportResponse.status, 200)
assert.match(await exportResponse.text(), /MODEL_CONNECTION_REQUIRED/)
assert.ok([404, 405].includes((await fetch(origin + '/api/research-sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })).status))
console.log('PASS: production assets, workspace creation, durable conversation, honest missing-model outcome, export and retired route rejection.')
