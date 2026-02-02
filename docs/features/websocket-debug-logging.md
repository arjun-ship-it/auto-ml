# Feature: WebSocket Debug Logging

## Metadata

- **Slug**: `websocket-debug-logging`
- **Status**: In Progress
- **Owner**: Claude
- **Created**: 2026-01-30
- **Last Updated**: 2026-01-30

## References

- **Jira/Tracker**: N/A
- **Figma**: N/A
- **Screenshots**: N/A
- **Loom**: N/A
- **Notes**: User reported "When I am sending any message into chat module no API is calling from frontend"

## Objective

Add comprehensive debug logging to WebSocket communication layer to diagnose why chat messages are not being sent from the frontend to the backend. This will help identify whether the issue is in:
1. WebSocket connection establishment
2. Message sending from frontend
3. Message receiving on backend
4. Agent processing

## Scope

### In Scope
- Add console.log statements to frontend WebSocket hook
- Add console.log statements to frontend ChatView component
- Add print statements to backend WebSocket handler
- Track WebSocket connection state changes
- Track message send/receive events

### Out of Scope
- Fixing the underlying WebSocket issue (this is for diagnosis only)
- Production-ready logging infrastructure
- Log persistence or aggregation

## Impacted Files/Modules

| Module | Files | Changes |
|--------|-------|---------|
| Frontend Services | `frontend/src/services/websocket.ts` | Add debug logging for connect, send, receive |
| Frontend Components | `frontend/src/components/Chat/ChatView.tsx` | Add debug logging for handleSend |
| Backend API | `backend/app/api/websocket.py` | Add print statements for message flow |

## Data Model Impact

None - this is logging only, no database changes.

## Implementation Plan

1. Add logging to `websocket.ts`:
   - Log WebSocket URL on connect
   - Log connection open/close events
   - Log sendMessage calls with payload
   - Log received messages

2. Add logging to `ChatView.tsx`:
   - Log handleSend call with state values
   - Log if send is blocked and why

3. Add logging to `websocket.py`:
   - Log new connections
   - Log agent initialization
   - Log received messages
   - Log sent responses
   - Log disconnections and errors

## Test Plan

### Unit Tests
- N/A (debug logging only)

### Integration Tests
- N/A (debug logging only)

### Manual Testing
- Open browser Developer Tools Console
- Send a chat message
- Verify logs appear showing message flow
- Check backend terminal for corresponding logs

## Rollout Notes

- Debug logging should be removed or made configurable before production
- These are temporary changes for diagnosis

## Acceptance Criteria

- [x] Frontend logs WebSocket connection events to browser console
- [x] Frontend logs sendMessage calls with message content
- [x] Backend logs incoming WebSocket messages to terminal
- [ ] Logs help identify where message flow is breaking
