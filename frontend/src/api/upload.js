import { API_URL } from './config'

export async function uploadDocument(file, sessionId) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('session_id', sessionId)

  const response = await fetch(`${API_URL}/api/upload`, { method: 'POST', body: formData })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `Upload failed with status ${response.status}`)
  }

  return response.json()
}

export async function removeDocument(sessionId) {
  await fetch(`${API_URL}/api/upload/${sessionId}`, { method: 'DELETE' })
}
