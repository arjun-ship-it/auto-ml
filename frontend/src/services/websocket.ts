import { useEffect, useRef, useState, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

// Debug logging helper - only logs in development mode
const debugLog = (...args: unknown[]) => {
  if (import.meta.env.DEV) {
    console.log(...args)
  }
}

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

    debugLog('[WS] Connecting to:', `${WS_URL}/ws/chat/${projectId}`)
    const ws = new WebSocket(`${WS_URL}/ws/chat/${projectId}`)

    ws.onopen = () => {
      debugLog('[WS] Connection opened')
      setIsConnected(true)
      options.onConnect?.()
    }

    ws.onmessage = (event) => {
      debugLog('[WS] Message received, length:', event.data?.length)
      try {
        const data = JSON.parse(event.data)
        options.onMessage(data)
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err)
      }
    }

    ws.onerror = (error) => {
      console.error('[WS] WebSocket error:', error)
      options.onError?.(error)
    }

    ws.onclose = () => {
      debugLog('[WS] Connection closed')
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
    debugLog('[WS] sendMessage called:', {
      messageLength: message.length,
      readyState: wsRef.current?.readyState,
      isOpen: wsRef.current?.readyState === WebSocket.OPEN
    })
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const payload = JSON.stringify({ message })
      debugLog('[WS] Sending payload of length:', payload.length)
      wsRef.current.send(payload)
    } else {
      console.error('[WS] WebSocket is not connected, readyState:', wsRef.current?.readyState)
    }
  }, [])

  return { sendMessage, isConnected }
}
