# Conversation Insights Pipeline

An AI-powered conversation analysis system that processes customer support transcripts to extract insights, generate embeddings, and enable semantic search with natural language queries.

## Features

- **Automated Processing**: Upload conversation transcripts and automatically extract intents, entities, sentiment, action items, and compliance flags
- **Vector Search**: Fast semantic search powered by Pinecone vector database
- **Batch Processing**: Cost-effective processing using OpenAI's Batch API (50% cheaper than real-time)
- **AI-Powered Insights**: Natural language queries synthesized by GPT-4o-mini
- **Real-time Status**: Track processing progress with polling
- **Modern UI**: Clean React interface with TailwindCSS

## Architecture

```
┌─────────────┐
│   React UI  │
│  (Frontend) │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│   FastAPI API    │
│    (Backend)     │
└────┬────┬────┬───┘
     │    │    │
     ▼    ▼    ▼
┌─────┐ ┌──────┐ ┌─────────┐
│OpenAI│ │Pinecone│ │tiktoken│
│Batch │ │Vector  │ │Chunking│
│ API  │ │  DB    │ │        │
└─────┘ └────────┘ └─────────┘
```

## Tech Stack

### Backend
- **Python 3.11**
- **FastAPI** - High-performance async API framework
- **OpenAI SDK** - Batch API for embeddings and extraction
- **Pinecone** - Vector database for semantic search
- **tiktoken** - Token-based text chunking

### Frontend
- **React 18** with TypeScript
- **Vite** - Fast build tool
- **TailwindCSS** - Utility-first CSS
- **Axios** - HTTP client

## Project Structure

```
conversation-insights/
├── backend/
│   ├── main.py                    # FastAPI app with all endpoints
│   ├── services/
│   │   ├── processor.py           # Text chunking + batch request creation
│   │   ├── openai_client.py       # OpenAI Batch API wrapper
│   │   └── pinecone_client.py     # Pinecone vector operations
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main UI component
│   │   ├── api.ts                 # API client
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── test.json                      # Sample conversation
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Pinecone API key ([Get one here](https://www.pinecone.io/))

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd conversation-insights/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   PINECONE_API_KEY=your-key-here
   PINECONE_INDEX=conversations
   PINECONE_ENVIRONMENT=us-east-1-aws
   ```

5. **Run the backend**:
   ```bash
   uvicorn main:app --reload
   ```

   Backend will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd conversation-insights/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```

   Frontend will be available at `http://localhost:3000`

## Usage

### 1. Upload a Conversation

The conversation file should be in JSON format:

```json
{
  "id": "conv_001",
  "transcript": "Agent: How can I help?\nCustomer: I want to cancel...",
  "metadata": {
    "account_id": "acct_123",
    "date": "2025-10-20",
    "agent_id": "agent_5"
  }
}
```

**Steps**:
1. Open the web interface at `http://localhost:3000`
2. Click "Choose File" and select your JSON file (or use the provided `test.json`)
3. Click "Upload & Process"
4. Watch the status update in real-time (polls every 5 seconds)
5. Wait for status to show "completed_and_stored"

### 2. Query Conversations

Once conversations are processed:

1. Enter a natural language question in the query box
2. Click "Ask AI"
3. View the AI-generated answer and source citations

**Example queries**:
- "What are the main reasons customers want to cancel?"
- "What discounts were offered to retain customers?"
- "What is the overall sentiment of the conversations?"
- "What action items were mentioned?"

## API Endpoints

### POST `/upload`
Upload and process a conversation

**Request**:
```json
{
  "id": "conv_001",
  "transcript": "...",
  "metadata": {}
}
```

**Response**:
```json
{
  "batch_id": "batch_abc123",
  "status": "submitted",
  "chunks": 4,
  "conversation_id": "conv_001"
}
```

### GET `/status/{batch_id}`
Check batch processing status

**Response**:
```json
{
  "batch_id": "batch_abc123",
  "status": "completed_and_stored",
  "progress": "8/8 requests",
  "conversation_id": "conv_001"
}
```

### POST `/query`
Query conversations with natural language

**Request**:
```json
{
  "query": "What are billing issues?",
  "filters": {"date": "2025-10-20"},
  "top_k": 50
}
```

**Response**:
```json
{
  "answer": "Based on the conversations...",
  "sources": [...],
  "processing_time_ms": 1234.5
}
```

### GET `/conversations`
List all processed conversations

**Response**:
```json
{
  "conversations": [
    {
      "id": "conv_001",
      "date": "2025-10-20",
      "sentiment": 0.65,
      "intents": ["cancel_subscription"]
    }
  ],
  "total": 1
}
```

## How It Works

### Upload & Processing Pipeline

1. **Chunking**: Text is split into 1500-token chunks with 200-token overlap using tiktoken
2. **Batch Creation**: Two requests per chunk:
   - Embedding with `text-embedding-3-small` (1536 dimensions)
   - Extraction with `gpt-4o-mini` in JSON mode
3. **Batch Submission**: Requests sent to OpenAI Batch API (24-hour SLA)
4. **Polling**: Frontend polls `/status` every 5 seconds
5. **Result Processing**: When complete, embeddings + extractions are combined
6. **Storage**: Data upserted to Pinecone with metadata

### Query Pipeline

1. **Embed Query**: Convert question to embedding vector
2. **Vector Search**: Find top 50 similar chunks in Pinecone
3. **Synthesis**: Send top 20 results to GPT-4o-mini for answer generation
4. **Response**: Return answer with source citations

### Extracted Data

For each conversation chunk, the system extracts:

- **Intents**: Customer intentions (e.g., "cancel_subscription", "request_refund")
- **Entities**: Named entities with types (e.g., {"type": "product", "value": "Pro Plan"})
- **Sentiment**: Score from -1 (negative) to 1 (positive)
- **Action Items**: Tasks or follow-ups mentioned
- **Compliance Flags**: Potential compliance issues

## Testing

### Using the Sample Conversation

A sample conversation is provided in `test.json`:

```bash
# Upload via UI or API
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d @test.json
```

### Expected Timeline

For a 5-chunk conversation:
- Upload: < 1 second
- Batch submission: 2-3 seconds
- OpenAI processing: 5-30 seconds (varies by API load)
- Pinecone storage: < 1 second
- **Total**: < 30 seconds for POC-sized conversations

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `PINECONE_API_KEY` | Pinecone API key | Required |
| `PINECONE_INDEX` | Pinecone index name | `conversations` |
| `PINECONE_ENVIRONMENT` | Pinecone region | `us-east-1-aws` |

### Chunking Parameters

In `backend/services/processor.py`:

```python
chunk_text(text, max_tokens=1500, overlap=200)
```

- `max_tokens`: Maximum tokens per chunk
- `overlap`: Token overlap between chunks

### Query Parameters

```python
query(
    embedding=...,
    top_k=50,           # Results to retrieve
    filter={...},       # Metadata filters
    namespace="conversations"
)
```

## Development

### Running Tests

```bash
# Backend
cd backend
python -m pytest

# Frontend
cd frontend
npm test
```

### Building for Production

```bash
# Frontend
cd frontend
npm run build

# Backend - use production WSGI server
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

## Troubleshooting

### Pinecone Index Not Created

If you see errors about missing index:
```bash
# The app auto-creates it on startup, but you can manually create:
# Go to Pinecone console and create index with:
# - Name: conversations
# - Dimensions: 1536
# - Metric: cosine
```

### OpenAI Batch API Delays

Batch API has 24-hour SLA but usually completes in 5-30 seconds. For faster results, consider switching to real-time API (2x cost).

### CORS Errors

Make sure backend is running on port 8000 and frontend on port 3000. Check `vite.config.ts` proxy settings.

### No Results When Querying

- Ensure conversations are in "completed_and_stored" status
- Check Pinecone index has data: visit Pinecone console
- Verify API keys are correct

## Cost Estimation

### OpenAI Costs (Batch API - 50% cheaper)

- Embeddings: ~$0.00001 per 1K tokens
- GPT-4o-mini extraction: ~$0.00015 per 1K tokens
- GPT-4o-mini synthesis: ~$0.00015 per 1K tokens

**Example**: 100 conversations × 5 chunks × 1500 tokens = ~$0.12

### Pinecone Costs

- Serverless: Pay per request + storage
- ~$0.10 per 1M read requests
- Storage: negligible for POC

## Future Enhancements

- [ ] Add authentication/authorization
- [ ] Implement conversation deletion
- [ ] Add advanced filters (date range, sentiment range)
- [ ] Export results to CSV/JSON
- [ ] Real-time processing option
- [ ] Batch upload multiple conversations
- [ ] Analytics dashboard
- [ ] Conversation comparison
- [ ] Custom extraction schemas
- [ ] Webhook notifications

## License

MIT License - feel free to use for commercial or personal projects

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API logs in terminal
3. Check Pinecone and OpenAI dashboards
4. Open an issue on GitHub

## Acknowledgments

- OpenAI for Batch API and embeddings
- Pinecone for vector database
- FastAPI team for the excellent framework
- React and Vite teams

---

Built with ❤️ using OpenAI, Pinecone, FastAPI, and React
