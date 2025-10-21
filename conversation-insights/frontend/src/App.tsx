import { useState, useEffect, useRef } from 'react';
import conversationApi, { QueryResponse, Source } from './api';

interface UploadStatus {
  batchId: string;
  conversationId: string;
  status: string;
  progress: string;
  isPolling: boolean;
}

function App() {
  // Upload state
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null);
  const [uploadError, setUploadError] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);

  // Query state
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [queryError, setQueryError] = useState<string>('');
  const [isQuerying, setIsQuerying] = useState(false);

  // Polling interval ref
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Poll batch status
  const pollStatus = async (batchId: string) => {
    try {
      const status = await conversationApi.getStatus(batchId);

      setUploadStatus(prev => prev ? {
        ...prev,
        status: status.status,
        progress: status.progress
      } : null);

      // Stop polling if completed or failed
      if (status.status === 'completed_and_stored' ||
          status.status === 'processing_failed' ||
          status.status === 'failed') {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
          setUploadStatus(prev => prev ? { ...prev, isPolling: false } : null);
        }
      }
    } catch (error: any) {
      console.error('Error polling status:', error);
      setUploadError(error.response?.data?.detail || 'Failed to check status');
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    }
  };

  // Start polling when batch is submitted
  useEffect(() => {
    if (uploadStatus?.isPolling && uploadStatus.batchId) {
      pollingIntervalRef.current = setInterval(() => {
        pollStatus(uploadStatus.batchId);
      }, 5000);

      return () => {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
        }
      };
    }
  }, [uploadStatus?.isPolling, uploadStatus?.batchId]);

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.type !== 'application/json') {
        setUploadError('Please select a JSON file');
        return;
      }
      setFile(selectedFile);
      setUploadError('');
    }
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!file) {
      setUploadError('Please select a file first');
      return;
    }

    setIsUploading(true);
    setUploadError('');

    try {
      // Read file content
      const fileContent = await file.text();
      const conversation = JSON.parse(fileContent);

      // Validate structure
      if (!conversation.id || !conversation.transcript) {
        setUploadError('Invalid conversation format. Must have "id" and "transcript" fields.');
        setIsUploading(false);
        return;
      }

      // Upload
      const response = await conversationApi.upload({
        id: conversation.id,
        transcript: conversation.transcript,
        metadata: conversation.metadata || {}
      });

      // Set upload status and start polling
      setUploadStatus({
        batchId: response.batch_id,
        conversationId: response.conversation_id,
        status: response.status,
        progress: `0/${response.chunks} chunks`,
        isPolling: true
      });

      setFile(null);
      // Reset file input
      const fileInput = document.getElementById('file-upload') as HTMLInputElement;
      if (fileInput) fileInput.value = '';

    } catch (error: any) {
      console.error('Upload error:', error);
      setUploadError(error.response?.data?.detail || 'Failed to upload conversation');
    } finally {
      setIsUploading(false);
    }
  };

  // Handle query submission
  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!query.trim()) {
      setQueryError('Please enter a query');
      return;
    }

    setIsQuerying(true);
    setQueryError('');
    setQueryResult(null);

    try {
      const response = await conversationApi.query({
        query: query.trim(),
        top_k: 50
      });

      setQueryResult(response);
    } catch (error: any) {
      console.error('Query error:', error);
      setQueryError(error.response?.data?.detail || 'Failed to process query');
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Conversation Insights
          </h1>
          <p className="text-gray-600">
            Upload conversations and query them with AI-powered insights
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Upload Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Upload Conversation
            </h2>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select JSON File
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".json"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-indigo-50 file:text-indigo-700
                  hover:file:bg-indigo-100
                  cursor-pointer"
              />
              {file && (
                <p className="mt-2 text-sm text-gray-600">
                  Selected: {file.name}
                </p>
              )}
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || isUploading}
              className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md
                hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed
                transition-colors duration-200 font-medium"
            >
              {isUploading ? 'Uploading...' : 'Upload & Process'}
            </button>

            {uploadError && (
              <div className="mt-4 p-3 bg-red-100 border border-red-300 rounded-md text-red-700 text-sm">
                {uploadError}
              </div>
            )}

            {uploadStatus && (
              <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
                <h3 className="font-semibold text-blue-900 mb-2">
                  Processing Status
                </h3>
                <div className="space-y-2 text-sm">
                  <p>
                    <span className="font-medium">Batch ID:</span>{' '}
                    <span className="font-mono text-xs">{uploadStatus.batchId}</span>
                  </p>
                  <p>
                    <span className="font-medium">Conversation:</span>{' '}
                    {uploadStatus.conversationId}
                  </p>
                  <p>
                    <span className="font-medium">Status:</span>{' '}
                    <span className={`px-2 py-1 rounded ${
                      uploadStatus.status === 'completed_and_stored'
                        ? 'bg-green-200 text-green-800'
                        : uploadStatus.status === 'processing_failed'
                        ? 'bg-red-200 text-red-800'
                        : 'bg-yellow-200 text-yellow-800'
                    }`}>
                      {uploadStatus.status}
                    </span>
                  </p>
                  <p>
                    <span className="font-medium">Progress:</span>{' '}
                    {uploadStatus.progress}
                  </p>
                  {uploadStatus.isPolling && (
                    <div className="flex items-center text-blue-600 mt-2">
                      <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full mr-2"></div>
                      Checking status...
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="mt-6 p-4 bg-gray-50 rounded-md text-sm text-gray-600">
              <p className="font-medium mb-2">Expected JSON format:</p>
              <pre className="text-xs overflow-x-auto">
{`{
  "id": "conv_001",
  "transcript": "...",
  "metadata": {
    "account_id": "acct_123",
    "date": "2025-10-20"
  }
}`}
              </pre>
            </div>
          </div>

          {/* Query Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Query Conversations
            </h2>

            <form onSubmit={handleQuery} className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Ask a question
              </label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., What are the main reasons customers want to cancel?"
                className="w-full px-3 py-2 border border-gray-300 rounded-md
                  focus:outline-none focus:ring-2 focus:ring-indigo-500
                  resize-none"
                rows={3}
              />

              <button
                type="submit"
                disabled={isQuerying || !query.trim()}
                className="mt-3 w-full bg-green-600 text-white py-2 px-4 rounded-md
                  hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed
                  transition-colors duration-200 font-medium"
              >
                {isQuerying ? 'Processing...' : 'Ask AI'}
              </button>
            </form>

            {queryError && (
              <div className="mt-4 p-3 bg-red-100 border border-red-300 rounded-md text-red-700 text-sm">
                {queryError}
              </div>
            )}

            {queryResult && (
              <div className="mt-6 space-y-4">
                {/* Answer */}
                <div className="p-4 bg-green-50 border border-green-200 rounded-md">
                  <h3 className="font-semibold text-green-900 mb-2">Answer</h3>
                  <p className="text-gray-800 whitespace-pre-wrap">
                    {queryResult.answer}
                  </p>
                  <p className="text-xs text-gray-500 mt-2">
                    Processing time: {queryResult.processing_time_ms.toFixed(0)}ms
                  </p>
                </div>

                {/* Sources */}
                {queryResult.sources.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-2">
                      Sources ({queryResult.sources.length})
                    </h3>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {queryResult.sources.map((source: Source, idx: number) => (
                        <div
                          key={idx}
                          className="p-3 bg-gray-50 border border-gray-200 rounded-md text-sm"
                        >
                          <div className="flex justify-between items-start mb-1">
                            <span className="font-medium text-gray-700">
                              {source.conversation_id}
                            </span>
                            <span className="text-xs text-gray-500">
                              Score: {source.score.toFixed(3)}
                            </span>
                          </div>
                          <p className="text-gray-600 text-xs mb-2">
                            Date: {source.date} | Sentiment: {source.sentiment.toFixed(2)}
                          </p>
                          <p className="text-gray-700 italic">"{source.text}"</p>
                          {source.intents.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                              {source.intents.map((intent: string, i: number) => (
                                <span
                                  key={i}
                                  className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded"
                                >
                                  {intent}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {!queryResult && !queryError && (
              <div className="mt-6 p-4 bg-gray-50 rounded-md text-sm text-gray-600">
                <p className="mb-2">Try asking questions like:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li>What are the main issues customers are facing?</li>
                  <li>What are the common reasons for cancellation?</li>
                  <li>What discounts or solutions were offered?</li>
                  <li>What is the overall sentiment of conversations?</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
