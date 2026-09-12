import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

const SUGGESTIONS = [
  'What are the symptoms of diabetes?',
  'Quels sont les traitements pour l’hypertension?',
  'What lifestyle changes help manage high blood pressure?',
]

export default function ChatWindow({ messages, streamingIndex, onSuggestion }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-icon">🌿</div>
          <h2>Welcome to MedLens</h2>
          <p>Ask a medical question in French or English.</p>
          <p className="empty-state-sub">Posez une question médicale en français ou en anglais.</p>
          <div className="suggestions">
            {SUGGESTIONS.map((suggestion) => (
              <button key={suggestion} onClick={() => onSuggestion(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}
      {messages.map((message, i) => (
        <MessageBubble key={i} {...message} isStreaming={i === streamingIndex} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
