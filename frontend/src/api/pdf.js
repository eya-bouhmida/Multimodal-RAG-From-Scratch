import { API_URL } from './config'

export async function exportPdf(title, text) {
  const response = await fetch(`${API_URL}/api/export-pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, text }),
  })

  if (!response.ok) {
    throw new Error(`PDF export failed with status ${response.status}`)
  }

  const blob = await response.blob()
  const url = URL.createObjectURL(blob)

  const link = document.createElement('a')
  link.href = url
  link.download = 'medlens.pdf'
  link.click()

  URL.revokeObjectURL(url)
}
