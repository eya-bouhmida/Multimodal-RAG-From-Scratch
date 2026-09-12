import { API_URL } from './config'

export async function streamChat(message, history, sessionId, { onToken, onDone, onError, signal }) {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, session_id: sessionId }),
    signal,
  })

  if (!response.ok || !response.body) {
    throw new Error(`Request failed with status ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop()

    for (const rawEvent of events) {
      const line = rawEvent.trim()
      if (!line.startsWith('data:')) continue

      const event = JSON.parse(line.slice(5).trim())

      if (event.type === 'token') {
        onToken?.(event.content)
      } else if (event.type === 'done') {
        onDone?.(event)
      } else if (event.type === 'error') {
        onError?.(event.message)
      }
    }
  }
}
