import { API_URL } from '../api/config'
import { exportPdf } from '../api/pdf'

export default function MessageBubble({ role, content, isStreaming, images }) {
  const isUser = role === 'user'
  const canExport = !isUser && !isStreaming && content.trim().length > 0
  const hasImages = !isUser && !isStreaming && images && images.length > 0

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      {!isUser && <div className="avatar">🌿</div>}
      <div className="message-bubble-wrap">
        <div className="message-bubble">
          <p>
            {content}
            {isStreaming && <span className="cursor" />}
          </p>
        </div>
        {hasImages && (
          <div className="figure-list">
            {images.map((img) => (
              <figure key={img.filename}>
                <img
                  src={`${API_URL}/api/images/${img.filename}`}
                  alt={img.caption}
                  loading="lazy"
                  onError={(e) => {
                    e.currentTarget.closest('figure').style.display = 'none'
                  }}
                />
                <figcaption>{img.caption}</figcaption>
              </figure>
            ))}
          </div>
        )}
        {canExport && (
          <button className="pdf-button" onClick={() => exportPdf('MedLens', content)}>
            📄 Download as PDF
          </button>
        )}
      </div>
      {isUser && <div className="avatar">🧑</div>}
    </div>
  )
}
