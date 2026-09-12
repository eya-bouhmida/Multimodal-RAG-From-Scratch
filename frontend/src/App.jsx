import { useRef, useState } from 'react'
import './App.css'
import ChatInput from './components/ChatInput'
import ChatWindow from './components/ChatWindow'
import { streamChat } from './api/chat'

function App() {
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [activeDocument, setActiveDocument] = useState(null)
  const abortRef = useRef(null)
  const sessionIdRef = useRef(crypto.randomUUID())

  const handleSend = async (text) => {
    const history = messages.map(({ role, content }) => ({ role, content }))

    setMessages((prev) => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    setIsStreaming(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamChat(text, history, sessionIdRef.current, {
        signal: controller.signal,
        onToken: (token) => {
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            next[next.length - 1] = { ...last, content: last.content + token }
            return next
          })
        },
        onDone: (event) => {
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            next[next.length - 1] = { ...last, images: event.images || [] }
            return next
          })
          setIsStreaming(false)
        },
        onError: (message) => {
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            next[next.length - 1] = { ...last, content: `⚠️ ${message}` }
            return next
          })
          setIsStreaming(false)
        },
      })
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev]
        const last = next[next.length - 1]
        next[next.length - 1] = { ...last, content: `⚠️ ${err.message}` }
        return next
      })
      setIsStreaming(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-logo">🌿</div>
        <div>
          <h1>MedLens</h1>
          <p className="disclaimer">
            Informational only — not a substitute for medical advice. / À titre informatif uniquement, ne remplace
            pas un avis médical.
          </p>
        </div>
      </header>
      <ChatWindow
        messages={messages}
        streamingIndex={isStreaming ? messages.length - 1 : -1}
        onSuggestion={handleSend}
      />
      <ChatInput
        onSend={handleSend}
        disabled={isStreaming}
        sessionId={sessionIdRef.current}
        document={activeDocument}
        setDocument={setActiveDocument}
      />
    </div>
  )
}

export default App
