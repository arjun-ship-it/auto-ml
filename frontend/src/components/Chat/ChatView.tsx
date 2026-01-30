import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, Upload } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { useWebSocket } from '../../services/websocket'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolInfo?: {
    tool: string
    status: 'running' | 'done'
    result?: string
  }
}

interface ChatViewProps {
  projectId: string | null
}

export default function ChatView({ projectId }: ChatViewProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const { sendMessage, isConnected } = useWebSocket(projectId, {
    onMessage: (data) => {
      if (data.type === 'text') {
        setMessages((prev) => {
          const last = prev[prev.length - 1]
          if (last && last.role === 'assistant' && !last.toolInfo) {
            // Append to existing assistant message
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + data.content },
            ]
          }
          return [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: 'assistant',
              content: data.content,
            },
          ]
        })
      } else if (data.type === 'tool_start') {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: `Running: ${data.tool}`,
            toolInfo: { tool: data.tool, status: 'running' },
          },
        ])
      } else if (data.type === 'tool_result') {
        setMessages((prev) => {
          const updated = [...prev]
          const toolMsg = updated.findLast((m) => m.toolInfo?.status === 'running')
          if (toolMsg) {
            toolMsg.toolInfo = { tool: data.tool, status: 'done', result: JSON.stringify(data.result, null, 2) }
          }
          return updated
        })
      } else if (data.type === 'done') {
        setIsLoading(false)
      }
    },
  })

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    if (import.meta.env.DEV) { console.log('[ChatView] handleSend called:', { inputLength: input.length, projectId, isLoading, isConnected }) }
    if (!input.trim() || !projectId || isLoading) return

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    sendMessage(input)
  }

  if (!projectId) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <h2 className="text-xl font-medium mb-2">Welcome to AutoML Agent</h2>
          <p className="text-sm">Create or select a project to start building your ML pipeline.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-slate-500 mt-20">
            <p className="text-lg mb-2">Start by describing your ML problem</p>
            <p className="text-sm">
              Example: "I want to predict how much product I need to purchase for the next 5 months"
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : msg.toolInfo
                  ? 'bg-slate-700 border border-slate-600'
                  : 'bg-slate-800 text-slate-200'
              }`}
            >
              {msg.toolInfo ? (
                <div className="text-sm">
                  <div className="flex items-center gap-2 text-yellow-400 mb-1">
                    {msg.toolInfo.status === 'running' ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <span className="text-green-400">Done</span>
                    )}
                    <span className="font-mono">{msg.toolInfo.tool}</span>
                  </div>
                  {msg.toolInfo.result && (
                    <pre className="text-xs text-slate-400 mt-2 max-h-40 overflow-auto">
                      {msg.toolInfo.result}
                    </pre>
                  )}
                </div>
              ) : (
                <div className="prose prose-invert prose-sm max-w-none">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && messages[messages.length - 1]?.role === 'user' && (
          <div className="flex justify-start">
            <div className="bg-slate-800 rounded-lg px-4 py-3">
              <Loader2 size={16} className="animate-spin text-blue-400" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-slate-700 p-4 bg-slate-800">
        <div className="flex items-center gap-2 max-w-4xl mx-auto">
          <button className="p-2 text-slate-400 hover:text-white transition-colors">
            <Upload size={20} />
          </button>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Describe your ML problem or ask a question..."
            className="flex-1 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="p-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white transition-colors"
          >
            <Send size={20} />
          </button>
        </div>
        <div className="flex items-center justify-center mt-2">
          <span className={`text-xs ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
    </div>
  )
}
