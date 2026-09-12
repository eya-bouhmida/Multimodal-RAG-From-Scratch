import { useRef, useState } from 'react'
import { removeDocument, uploadDocument } from '../api/upload'

export default function DocumentUpload({ sessionId, document, setDocument }) {
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const handleFileChange = async (e) => {
    const file = e.target.files[0]
    e.target.value = ''
    if (!file) return

    setIsUploading(true)
    setError(null)
    try {
      const result = await uploadDocument(file, sessionId)
      setDocument({ filename: result.filename, numChunks: result.num_chunks })
    } catch (err) {
      setError(err.message)
    } finally {
      setIsUploading(false)
    }
  }

  const handleRemove = async () => {
    await removeDocument(sessionId)
    setDocument(null)
  }

  return (
    <div className="document-upload">
      {document ? (
        <div className="document-chip">
          <span>📄 {document.filename}</span>
          <button onClick={handleRemove} aria-label="Remove document">
            ✕
          </button>
        </div>
      ) : (
        <button
          className="upload-button"
          onClick={() => inputRef.current?.click()}
          disabled={isUploading}
        >
          {isUploading ? 'Uploading...' : '📎 Add a document (PDF)'}
        </button>
      )}
      <input ref={inputRef} type="file" accept=".pdf" hidden onChange={handleFileChange} />
      {error && <span className="upload-error">{error}</span>}
    </div>
  )
}
