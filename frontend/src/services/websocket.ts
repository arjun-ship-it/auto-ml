import { useEffect, useRef, useState, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

interface WebSocketMessage {
  type: string
  content?: string
  tool?: string
  result?: unknown
  input?: unknown
}

interface UseWebSocketOptions {
  onMessage: (data: WebSocketMessage) => void
  onError?: (error: Event) => void
  onConnect?: () => void
  onDisconnect?: () => void
}

export function useWebSocket(
  projectId: string | null,
  options: UseWebSocketOptions
) {
  const wsRef = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()

  const connect = useCallback(() => {
    if (!projectId) return

    const ws = new WebSocket(`${WS_URL}/ws/chat/${projectId}`)

    ws.onopen = () => {
      setIsConnected(true)
      options.onConnect?.()
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        options.onMessage(data)
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      options.onError?.(error)
    }

    ws.onclose = () => {
      setIsConnected(false)
      options.onDisconnect?.()

      // Attempt reconnection after 3 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        if (projectId) connect()
      }, 3000)
    }

    wsRef.current = ws
  }, [projectId])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      wsRef.current?.close()
    }
  }, [connect])

  const sendMessage = useCallback((message: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ message }))
    } else {
      console.error('WebSocket is not connected')
    }
  }, [])

  return { sendMessage, isConnected }
}
