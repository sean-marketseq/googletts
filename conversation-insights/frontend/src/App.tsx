import { useState, useEffect, useRef } from 'react';
import conversationApi from './api';

interface UploadStatus {
  batchId: string;
  conversationId: string;
  status: string;
  progress: string;
  isPolling: boolean;
  filename: string;
}

interface QueryResponse {
  answer: string;
  sources: Source[];
  processing_time_ms: number;
}

interface Source {
  conversation_id: string;
  date: string;
  score: number;
  text: string;
  sentiment: number;
  intents: string[];
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

  // Purge state
  const [isPurging, setIsPurging] = useState(false);
  const [showPurgeConfirm, setShowPurgeConfirm] = useState(false);

  // Polling interval ref
  const pollingIntervalRef = useRef<number | null>(null);

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
      const allowedExtensions = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'];

      const fileExt = selectedFile.name.toLowerCase().slice(selectedFile.name.lastIndexOf('.'));

      if (!allowedExtensions.includes(fileExt)) {
        setUploadError(`Please select an audio file (${allowedExtensions.join(', ')})`);
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
      const response = await conversationApi.uploadAudio(file);

      setUploadStatus({
        batchId: response.batch_id,
        conversationId: response.conversation_id,
        status: response.status,
        progress: `0/${response.chunks} chunks`,
        isPolling: true,
        filename: file.name
      });

      setFile(null);
      const fileInput = document.getElementById('file-upload') as HTMLInputElement;
      if (fileInput) fileInput.value = '';

    } catch (error: any) {
      console.error('Upload error:', error);
      setUploadError(error.response?.data?.detail || 'Failed to upload audio file');
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

  // Handle purge confirmation
  const handlePurge = async () => {
    setIsPurging(true);

    try {
      await conversationApi.purgeIndex();

      // Clear local state
      setUploadStatus(null);
      setQueryResult(null);
      setUploadError('');
      setQueryError('');

      alert('✓ All data purged successfully!');
    } catch (error: any) {
      console.error('Purge error:', error);
      alert('Failed to purge index: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsPurging(false);
      setShowPurgeConfirm(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <header className="text-center mb-16">
          <h1 className="text-6xl font-bold text-white mb-4 tracking-tight">
            Conversation Insights
          </h1>
          <p className="text-xl text-purple-200">
            Upload audio conversations and search them with AI
          </p>
        </header>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 max-w-7xl mx-auto">

          {/* LEFT SIDE: Add to Index */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 p-8">
            <div className="flex items-center justify-center mb-6">
              <div className="bg-gradient-to-r from-blue-500 to-purple-500 rounded-full p-3 mr-4">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold text-white">
                Add to Index
              </h2>
            </div>

            <p className="text-purple-200 text-center mb-8">
              Upload audio files to transcribe and index for searching
            </p>

            <div className="space-y-6">
              {/* File Input */}
              <div>
                <label className="block text-sm font-semibold text-purple-200 mb-3">
                  Select Audio File
                </label>
                <div className="relative">
                  <input
                    id="file-upload"
                    type="file"
                    accept=".mp3,.wav,.m4a,.mp4,.mpeg,.mpga,.webm,audio/*"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-purple-200
                      file:mr-4 file:py-3 file:px-6
                      file:rounded-full file:border-0
                      file:text-sm file:font-semibold
                      file:bg-gradient-to-r file:from-blue-500 file:to-purple-500
                      file:text-white
                      hover:file:from-blue-600 hover:file:to-purple-600
                      file:cursor-pointer
                      cursor-pointer
                      bg-white/5 rounded-lg p-3 border border-white/20"
                  />
                </div>
                {file && (
                  <div className="mt-3 p-3 bg-green-500/20 border border-green-400/30 rounded-lg">
                    <p className="text-sm text-green-200 flex items-center">
                      <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
                        <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd"/>
                      </svg>
                      Selected: {file.name}
                    </p>
                  </div>
                )}
                <p className="mt-2 text-xs text-purple-300">
                  Supported formats: MP3, WAV, M4A, MP4, MPEG, WebM
                </p>
              </div>

              {/* Upload Button */}
              <button
                onClick={handleUpload}
                disabled={!file || isUploading}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-4 px-6 rounded-xl
                  hover:from-blue-600 hover:to-purple-600
                  disabled:from-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed
                  transition-all duration-200 font-bold text-lg shadow-lg
                  transform hover:scale-105 disabled:transform-none"
              >
                {isUploading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Transcribing & Processing...
                  </span>
                ) : 'Upload & Add to Index'}
              </button>

              {/* Upload Error */}
              {uploadError && (
                <div className="p-4 bg-red-500/20 border border-red-400/30 rounded-xl text-red-200 text-sm">
                  {uploadError}
                </div>
              )}

              {/* Upload Status */}
              {uploadStatus && (
                <div className="p-6 bg-blue-500/20 border border-blue-400/30 rounded-xl space-y-3">
                  <h3 className="font-bold text-blue-100 text-lg flex items-center">
                    <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd"/>
                    </svg>
                    Processing Status
                  </h3>
                  <div className="space-y-2 text-sm">
                    <p className="text-blue-200">
                      <span className="font-semibold">File:</span> {uploadStatus.filename}
                    </p>
                    <p className="text-blue-200">
                      <span className="font-semibold">Conversation ID:</span> {uploadStatus.conversationId}
                    </p>
                    <p className="text-blue-200 flex items-center justify-between">
                      <span>
                        <span className="font-semibold">Status:</span>
                        <span className={`ml-2 px-3 py-1 rounded-full text-xs font-bold ${
                          uploadStatus.status === 'completed_and_stored'
                            ? 'bg-green-500 text-white'
                            : uploadStatus.status === 'processing_failed'
                            ? 'bg-red-500 text-white'
                            : 'bg-yellow-500 text-gray-900'
                        }`}>
                          {uploadStatus.status.replace(/_/g, ' ').toUpperCase()}
                        </span>
                      </span>
                    </p>
                    <p className="text-blue-200">
                      <span className="font-semibold">Progress:</span> {uploadStatus.progress}
                    </p>
                    {uploadStatus.isPolling && (
                      <div className="flex items-center text-blue-300 mt-3 pt-3 border-t border-blue-400/30">
                        <div className="animate-spin h-4 w-4 border-2 border-blue-400 border-t-transparent rounded-full mr-2"></div>
                        Checking status every 5 seconds...
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT SIDE: Search Index */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl border border-white/20 p-8">
            <div className="flex items-center justify-center mb-6">
              <div className="bg-gradient-to-r from-green-500 to-emerald-500 rounded-full p-3 mr-4">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold text-white">
                Search Index
              </h2>
            </div>

            <p className="text-purple-200 text-center mb-8">
              Ask questions about your indexed conversations
            </p>

            <form onSubmit={handleQuery} className="space-y-6">
              <div>
                <label className="block text-sm font-semibold text-purple-200 mb-3">
                  Your Question
                </label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g., What are the main reasons customers want to cancel?"
                  className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl
                    text-white placeholder-purple-300
                    focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent
                    resize-none"
                  rows={3}
                />
              </div>

              <button
                type="submit"
                disabled={isQuerying || !query.trim()}
                className="w-full bg-gradient-to-r from-green-500 to-emerald-500 text-white py-4 px-6 rounded-xl
                  hover:from-green-600 hover:to-emerald-600
                  disabled:from-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed
                  transition-all duration-200 font-bold text-lg shadow-lg
                  transform hover:scale-105 disabled:transform-none"
              >
                {isQuerying ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Searching...
                  </span>
                ) : 'Search Index'}
              </button>

              {/* Query Error */}
              {queryError && (
                <div className="p-4 bg-red-500/20 border border-red-400/30 rounded-xl text-red-200 text-sm">
                  {queryError}
                </div>
              )}

              {/* Query Result */}
              {queryResult && (
                <div className="space-y-4 animate-fadeIn">
                  {/* Answer */}
                  <div className="p-6 bg-green-500/20 border border-green-400/30 rounded-xl">
                    <h3 className="font-bold text-green-100 mb-3 text-lg flex items-center">
                      <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd"/>
                      </svg>
                      Answer
                    </h3>
                    <p className="text-green-50 whitespace-pre-wrap leading-relaxed">
                      {queryResult.answer}
                    </p>
                    <p className="text-xs text-green-300 mt-3 pt-3 border-t border-green-400/30">
                      Processing time: {queryResult.processing_time_ms.toFixed(0)}ms
                    </p>
                  </div>

                  {/* Sources */}
                  {queryResult.sources.length > 0 && (
                    <div>
                      <h3 className="font-bold text-purple-100 mb-3 text-lg">
                        Sources ({queryResult.sources.length})
                      </h3>
                      <div className="space-y-3 max-h-96 overflow-y-auto pr-2 custom-scrollbar">
                        {queryResult.sources.map((source: Source, idx: number) => (
                          <div
                            key={idx}
                            className="p-4 bg-white/5 border border-white/20 rounded-xl text-sm hover:bg-white/10 transition-colors"
                          >
                            <div className="flex justify-between items-start mb-2">
                              <span className="font-semibold text-purple-200">
                                {source.conversation_id}
                              </span>
                              <span className="text-xs text-purple-300 bg-purple-500/20 px-2 py-1 rounded-full">
                                Score: {source.score.toFixed(3)}
                              </span>
                            </div>
                            <p className="text-purple-300 text-xs mb-2">
                              Date: {source.date} | Sentiment: {source.sentiment.toFixed(2)}
                            </p>
                            <p className="text-purple-100 italic text-xs leading-relaxed">
                              "{source.text}"
                            </p>
                            {source.intents.length > 0 && (
                              <div className="mt-3 flex flex-wrap gap-1">
                                {source.intents.map((intent: string, i: number) => (
                                  <span
                                    key={i}
                                    className="px-2 py-1 bg-blue-500/30 text-blue-200 text-xs rounded-full"
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

              {/* Example Questions */}
              {!queryResult && !queryError && (
                <div className="p-5 bg-white/5 rounded-xl border border-white/10">
                  <p className="text-purple-200 text-sm mb-3 font-semibold">Try asking:</p>
                  <ul className="list-disc list-inside space-y-2 text-xs text-purple-300">
                    <li>What are the main issues customers are facing?</li>
                    <li>What are the common reasons for cancellation?</li>
                    <li>What discounts or solutions were offered?</li>
                    <li>What is the overall sentiment of conversations?</li>
                  </ul>
                </div>
              )}
            </form>
          </div>
        </div>

        {/* Danger Zone - Purge Index */}
        <div className="max-w-7xl mx-auto mt-12">
          <div className="bg-red-900/20 backdrop-blur-lg rounded-2xl border-2 border-red-500/50 p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xl font-bold text-red-300 mb-2 flex items-center">
                  <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd"/>
                  </svg>
                  Danger Zone
                </h3>
                <p className="text-red-200 text-sm">
                  Testing only: Delete all conversations from the index. This cannot be undone.
                </p>
              </div>

              {!showPurgeConfirm ? (
                <button
                  onClick={() => setShowPurgeConfirm(true)}
                  className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-semibold
                    transition-colors duration-200 flex items-center"
                >
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Purge All Data
                </button>
              ) : (
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowPurgeConfirm(false)}
                    className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-3 rounded-lg font-semibold
                      transition-colors duration-200"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handlePurge}
                    disabled={isPurging}
                    className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-bold
                      transition-colors duration-200 disabled:bg-gray-500 flex items-center"
                  >
                    {isPurging ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Purging...
                      </>
                    ) : (
                      '⚠️ Yes, Delete Everything'
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Custom Scrollbar Styles */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(139, 92, 246, 0.5);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(139, 92, 246, 0.7);
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
      `}</style>
    </div>
  );
}

export default App;
