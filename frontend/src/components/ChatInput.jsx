import { useState } from 'react'
import DocumentUpload from './DocumentUpload'

export default function ChatInput({ onSend, disabled, sessionId, document, setDocument }) {
  const [value, setValue] = useState('')

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="chat-input">
      <DocumentUpload sessionId={sessionId} document={document} setDocument={setDocument} />
      <div className="chat-input-row">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask MedLens... / Posez votre question..."
          rows={1}
          disabled={disabled}
        />
        <button onClick={submit} disabled={disabled || !value.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}
