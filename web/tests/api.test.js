import assert from 'node:assert/strict'
import { test } from 'node:test'
import { api, post, uploadFile } from '../src/api.js'

test('workspace message uses the unified API', async context => {
  context.mock.method(globalThis, 'fetch', async (path, options) => {
    assert.equal(path, '/api/workspaces/current/messages')
    assert.equal(options.method, 'POST')
    assert.deepEqual(JSON.parse(options.body), { content: 'Explain DCF' })
    return new Response(JSON.stringify({ queued: true }), { status: 202 })
  })
  assert.deepEqual(await post('/api/workspaces/current/messages', { content: 'Explain DCF' }), { queued: true })
})

test('empty success response is accepted', async context => {
  context.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 204 }))
  assert.equal(await post('/api/workspaces/current/model-session', { model_session_id: 'model' }), null)
})

test('validation error keeps safe server detail and status', async context => {
  context.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    detail: [{ loc: ['body', 'content'], msg: 'Input is too long' }],
  }), { status: 422 }))
  await assert.rejects(api('/api/workspaces/current'), error => error.status === 422 && error.message === 'content: Input is too long')
})

test('connection errors never echo raw network messages', async context => {
  context.mock.method(globalThis, 'fetch', async () => { throw new Error('private-network-detail') })
  await assert.rejects(api('/api/workspaces/current'), error => !error.message.includes('private-network-detail'))
})

test('attachment metadata uses multipart upload', async context => {
  context.mock.method(globalThis, 'fetch', async (path, options) => {
    assert.equal(path, '/api/files')
    assert.equal(options.body.get('role'), 'evidence')
    assert.equal(options.body.get('file').name, 'source.txt')
    assert.equal(options.headers['Content-Type'], undefined)
    return new Response(JSON.stringify({ file_id: 'file' }), { status: 201 })
  })
  assert.deepEqual(await uploadFile(new File(['source'], 'source.txt'), 'evidence'), { file_id: 'file' })
})

test('oversized attachments are rejected before fetching', async context => {
  const fetch = context.mock.method(globalThis, 'fetch', async () => assert.fail('must not send'))
  await assert.rejects(uploadFile({ size: 50 * 1024 * 1024 + 1 }, 'evidence'))
  assert.equal(fetch.mock.callCount(), 0)
})
