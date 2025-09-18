# Go Token Server

Simple Go server that provides OpenAI Realtime API tokens and serves static files.

## Features
- OpenAI token generation endpoint (`/token`)
- Static file serving for the realtime lip sync console
- CORS enabled for browser access
- Environment variable configuration

## Usage

```bash
# Install dependencies
go mod tidy

# Run the server
go run main.go
```

The server will:
- Start on port 3000 (configurable with PORT env var)
- Load `.env` from `../realtime_lipsync/.env`
- Serve static files from `../realtime_lipsync/`
- Provide token endpoint at `http://localhost:3000/token`

## Endpoints

### POST /token
Generates ephemeral tokens for OpenAI Realtime API.

**Response:**
```json
{
  "token": "rtctk_xxx...",
  "expires_at": "2024-01-01T12:00:00Z"
}
```

### GET /
Serves static files from the realtime_lipsync directory.

## Environment Variables

- `OPENAI_API_KEY` - Required. Your OpenAI API key.
- `PORT` - Optional. Server port (default: 3000).
