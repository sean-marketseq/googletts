import { useState, useEffect, useRef } from 'react';
import conversationApi from './api';
import EmotionChart from './components/EmotionChart';

interface BatchDiagnostics {
  embedding_batch?: {
    id: string;
    status: string;
    progress: string;
    failed: number;
    time_elapsed_min: number | null;
  };
  extraction_batch?: {
    id: string;
    status: string;
    progress: string;
    failed: number;
    time_elapsed_min: number | null;
  };
  hume_job?: {
    id: string;
    status: string;
    message?: string;
  };
}

interface FileStatus {
  file: File;
  id: string;
  status: 'queued' | 'uploading' | 'transcribing' | 'processing' | 'completed' | 'error';
  batchId?: string;
  conversationId?: string;
  progress?: string;
  error?: string;
  isPolling?: boolean;
  batchDiagnostics?: BatchDiagnostics;
  emotionData?: {
    speaker_a: {
      top_5_emotions: string[];
      emotion_vector: number[];
    };
    speaker_b: {
      top_5_emotions: string[];
      emotion_vector: number[];
    };
    timelines: {
      speaker_a: Array<{time: number, emotions: Record<string, number>}>;
      speaker_b: Array<{time: number, emotions: Record<string, number>}>;
    };
    speaker_labels: Record<string, string>;
    labeling_confidence: boolean;
  };
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

const MAX_CONCURRENT_UPLOADS = 2; // Process 2 files at a time

function App() {
  // Upload state
  const [fileStatuses, setFileStatuses] = useState<FileStatus[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadQueue, setUploadQueue] = useState<string[]>([]); // Queue of file IDs to process
  const [activeUploads, setActiveUploads] = useState<Set<string>>(new Set());
  const [expandedDetails, setExpandedDetails] = useState<Set<string>>(new Set());

  // Query state
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [queryError, setQueryError] = useState<string>('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [copied, setCopied] = useState(false);

  // Purge state
  const [isPurging, setIsPurging] = useState(false);
  const [showPurgeConfirm, setShowPurgeConfirm] = useState(false);

  // Polling intervals ref (now a Map)
  const pollingIntervalsRef = useRef<Map<string, number>>(new Map());

  // Allowed file extensions
  const allowedExtensions = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'];

  // Poll batch status for a specific file
  const pollStatus = async (fileId: string, batchId: string) => {
    try {
      const status = await conversationApi.getStatus(batchId);

      // Extract emotion data from status details
      const emotionDataRaw = status.details?.emotion_data;
      const emotionData = emotionDataRaw ? {
        speaker_a: emotionDataRaw.speaker_a || { top_5_emotions: [], emotion_vector: [] },
        speaker_b: emotionDataRaw.speaker_b || { top_5_emotions: [], emotion_vector: [] },
        timelines: emotionDataRaw.timelines || { speaker_a: [], speaker_b: [] },
        speaker_labels: status.details?.speaker_labels || {},
        labeling_confidence: status.details?.labeling_confidence || false
      } : undefined;

      // Extract batch diagnostics
      const batchDiagnostics = status.details?.batch_diagnostics;

      setFileStatuses(prev => prev.map(fs =>
        fs.id === fileId ? {
          ...fs,
          status: status.status === 'completed_and_stored' || status.status === 'completed' ? 'completed' :
                  status.status === 'failed' || status.status === 'processing_failed' ? 'error' :
                  'processing',
          progress: status.progress,
          error: status.status === 'failed' || status.status === 'processing_failed' ?
                 'Processing failed' : undefined,
          batchDiagnostics: batchDiagnostics,
          emotionData: emotionData
        } : fs
      ));

      // Stop polling if completed or failed
      if (status.status === 'completed_and_stored' ||
          status.status === 'completed' ||
          status.status === 'processing_failed' ||
          status.status === 'failed') {
        const intervalId = pollingIntervalsRef.current.get(fileId);
        if (intervalId) {
          clearInterval(intervalId);
          pollingIntervalsRef.current.delete(fileId);
        }

        // Remove from active uploads
        setActiveUploads(prev => {
          const next = new Set(prev);
          next.delete(fileId);
          return next;
        });
      }
    } catch (error: any) {
      console.error('Error polling status:', error);
      setFileStatuses(prev => prev.map(fs =>
        fs.id === fileId ? {
          ...fs,
          status: 'error',
          error: 'Failed to check status'
        } : fs
      ));

      const intervalId = pollingIntervalsRef.current.get(fileId);
      if (intervalId) {
        clearInterval(intervalId);
        pollingIntervalsRef.current.delete(fileId);
      }

      setActiveUploads(prev => {
        const next = new Set(prev);
        next.delete(fileId);
        return next;
      });
    }
  };

  // Process next file in queue
  const processNextFile = async (fileId: string) => {
    const fileStatus = fileStatuses.find(fs => fs.id === fileId);
    if (!fileStatus) return;

    setActiveUploads(prev => new Set(prev).add(fileId));

    setFileStatuses(prev => prev.map(fs =>
      fs.id === fileId ? { ...fs, status: 'uploading' as const } : fs
    ));

    try {
      const response = await conversationApi.uploadAudio(fileStatus.file);

      setFileStatuses(prev => prev.map(fs =>
        fs.id === fileId ? {
          ...fs,
          status: 'processing' as const,
          batchId: response.batch_id,
          conversationId: response.conversation_id,
          progress: `0/${response.chunks} chunks`,
          isPolling: true
        } : fs
      ));

      // Start polling for this file
      const intervalId = setInterval(() => {
        pollStatus(fileId, response.batch_id);
      }, 5000);

      pollingIntervalsRef.current.set(fileId, intervalId);

    } catch (error: any) {
      console.error('Upload error:', error);
      setFileStatuses(prev => prev.map(fs =>
        fs.id === fileId ? {
          ...fs,
          status: 'error' as const,
          error: error.response?.data?.detail || 'Failed to upload audio file'
        } : fs
      ));

      setActiveUploads(prev => {
        const next = new Set(prev);
        next.delete(fileId);
        return next;
      });
    }
  };

  // Watch upload queue and process files
  useEffect(() => {
    if (uploadQueue.length === 0) return;
    if (activeUploads.size >= MAX_CONCURRENT_UPLOADS) return;

    const nextFileId = uploadQueue[0];
    setUploadQueue(prev => prev.slice(1));
    processNextFile(nextFileId);
  }, [uploadQueue, activeUploads.size]);

  // Handle file selection
  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const validFiles: FileStatus[] = [];
    const errors: string[] = [];

    Array.from(files).forEach(file => {
      const fileExt = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));

      if (!allowedExtensions.includes(fileExt)) {
        errors.push(`${file.name}: Invalid file type`);
        return;
      }

      validFiles.push({
        file,
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        status: 'queued'
      });
    });

    if (errors.length > 0) {
      alert('Some files were skipped:\n' + errors.join('\n'));
    }

    if (validFiles.length > 0) {
      setFileStatuses(prev => [...prev, ...validFiles]);
      setUploadQueue(prev => [...prev, ...validFiles.map(f => f.id)]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    e.target.value = ''; // Reset input so same file can be added again
  };

  // Drag and drop handlers
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget === e.target) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    handleFiles(e.dataTransfer.files);
  };

  // Remove file from list
  const handleRemoveFile = (fileId: string) => {
    // Stop polling if active
    const intervalId = pollingIntervalsRef.current.get(fileId);
    if (intervalId) {
      clearInterval(intervalId);
      pollingIntervalsRef.current.delete(fileId);
    }

    setFileStatuses(prev => prev.filter(fs => fs.id !== fileId));
    setUploadQueue(prev => prev.filter(id => id !== fileId));
    setActiveUploads(prev => {
      const next = new Set(prev);
      next.delete(fileId);
      return next;
    });
  };

  // Clear all completed/errored files
  const handleClearCompleted = () => {
    fileStatuses.forEach(fs => {
      if (fs.status === 'completed' || fs.status === 'error') {
        const intervalId = pollingIntervalsRef.current.get(fs.id);
        if (intervalId) {
          clearInterval(intervalId);
          pollingIntervalsRef.current.delete(fs.id);
        }
      }
    });

    setFileStatuses(prev => prev.filter(fs =>
      fs.status !== 'completed' && fs.status !== 'error'
    ));
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
      setFileStatuses([]);
      setUploadQueue([]);
      setActiveUploads(new Set());
      setQueryResult(null);
      setQueryError('');

      // Clear all polling intervals
      pollingIntervalsRef.current.forEach(intervalId => clearInterval(intervalId));
      pollingIntervalsRef.current.clear();

      alert('✓ All data purged successfully!');
    } catch (error: any) {
      console.error('Purge error:', error);
      alert('Failed to purge index: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsPurging(false);
      setShowPurgeConfirm(false);
    }
  };

  // Handle copy to clipboard
  const handleCopyAnswer = async () => {
    if (!queryResult?.answer) return;

    try {
      await navigator.clipboard.writeText(queryResult.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  // Toggle details expansion
  const toggleDetails = (fileId: string) => {
    setExpandedDetails(prev => {
      const next = new Set(prev);
      if (next.has(fileId)) {
        next.delete(fileId);
      } else {
        next.add(fileId);
      }
      return next;
    });
  };

  // Get status counts
  const statusCounts = {
    queued: fileStatuses.filter(fs => fs.status === 'queued').length,
    processing: fileStatuses.filter(fs => ['uploading', 'transcribing', 'processing'].includes(fs.status)).length,
    completed: fileStatuses.filter(fs => fs.status === 'completed').length,
    error: fileStatuses.filter(fs => fs.status === 'error').length,
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
              Upload multiple audio files to transcribe and index for searching
            </p>

            <div className="space-y-6">
              {/* Drag and Drop Zone */}
              <div
                onDragEnter={handleDragEnter}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
                  isDragging
                    ? 'border-blue-400 bg-blue-500/20 scale-105'
                    : 'border-white/30 bg-white/5 hover:border-white/50'
                }`}
              >
                <input
                  id="file-upload"
                  type="file"
                  accept=".mp3,.wav,.m4a,.mp4,.mpeg,.mpga,.webm,audio/*"
                  onChange={handleFileChange}
                  multiple
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />

                <div className="pointer-events-none">
                  <svg className="w-16 h-16 mx-auto mb-4 text-purple-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="text-lg font-semibold text-purple-100 mb-2">
                    {isDragging ? 'Drop files here' : 'Drag & drop audio files'}
                  </p>
                  <p className="text-sm text-purple-300 mb-4">or click to browse</p>
                  <p className="text-xs text-purple-400">
                    Supported: MP3, WAV, M4A, MP4, MPEG, WebM
                  </p>
                </div>
              </div>

              {/* Status Summary */}
              {fileStatuses.length > 0 && (
                <div className="grid grid-cols-4 gap-3">
                  <div className="bg-yellow-500/20 border border-yellow-400/30 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-yellow-200">{statusCounts.queued}</div>
                    <div className="text-xs text-yellow-300">Queued</div>
                  </div>
                  <div className="bg-blue-500/20 border border-blue-400/30 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-blue-200">{statusCounts.processing}</div>
                    <div className="text-xs text-blue-300">Processing</div>
                  </div>
                  <div className="bg-green-500/20 border border-green-400/30 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-green-200">{statusCounts.completed}</div>
                    <div className="text-xs text-green-300">Completed</div>
                  </div>
                  <div className="bg-red-500/20 border border-red-400/30 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-red-200">{statusCounts.error}</div>
                    <div className="text-xs text-red-300">Errors</div>
                  </div>
                </div>
              )}

              {/* File List */}
              {fileStatuses.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-purple-100">
                      Files ({fileStatuses.length})
                    </h3>
                    {(statusCounts.completed > 0 || statusCounts.error > 0) && (
                      <button
                        onClick={handleClearCompleted}
                        className="text-xs text-purple-300 hover:text-purple-100 underline"
                      >
                        Clear completed/errors
                      </button>
                    )}
                  </div>

                  <div className="max-h-96 overflow-y-auto space-y-2 pr-2 custom-scrollbar">
                    {fileStatuses.map(fs => (
                      <div
                        key={fs.id}
                        className="bg-white/5 border border-white/20 rounded-lg p-4 hover:bg-white/10 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold text-purple-100 truncate">
                              {fs.file.name}
                            </p>
                            <div className="flex items-center gap-2 mt-1">
                              <span className={`text-xs px-2 py-1 rounded-full font-semibold ${
                                fs.status === 'queued' ? 'bg-yellow-500/30 text-yellow-200' :
                                fs.status === 'uploading' ? 'bg-blue-500/30 text-blue-200' :
                                fs.status === 'processing' ? 'bg-blue-500/30 text-blue-200' :
                                fs.status === 'completed' ? 'bg-green-500/30 text-green-200' :
                                'bg-red-500/30 text-red-200'
                              }`}>
                                {fs.status === 'uploading' && '⏫ Uploading'}
                                {fs.status === 'queued' && '⏳ Queued'}
                                {fs.status === 'processing' && '⚙️ Processing'}
                                {fs.status === 'completed' && '✓ Complete'}
                                {fs.status === 'error' && '✗ Error'}
                              </span>
                              {fs.progress && (
                                <span className="text-xs text-purple-300">{fs.progress}</span>
                              )}
                            </div>
                            {fs.conversationId && (
                              <p className="text-xs text-purple-400 mt-1">ID: {fs.conversationId}</p>
                            )}
                            {fs.error && (
                              <p className="text-xs text-red-300 mt-1">{fs.error}</p>
                            )}
                          </div>

                          <div className="flex items-center gap-2 flex-shrink-0">
                            {/* Details Toggle Button */}
                            {(fs.batchId || fs.batchDiagnostics) && (
                              <button
                                onClick={() => toggleDetails(fs.id)}
                                className="text-purple-300 hover:text-purple-100 transition-colors p-1"
                                title="Toggle details"
                              >
                                <svg
                                  className={`w-5 h-5 transition-transform ${expandedDetails.has(fs.id) ? 'rotate-180' : ''}`}
                                  fill="none"
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                >
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                </svg>
                              </button>
                            )}

                            {/* Remove Button */}
                            <button
                              onClick={() => handleRemoveFile(fs.id)}
                              className="text-purple-300 hover:text-red-400 transition-colors"
                              title="Remove"
                            >
                              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"/>
                              </svg>
                            </button>
                          </div>
                        </div>

                        {/* Processing Details Panel */}
                        {expandedDetails.has(fs.id) && (fs.batchId || fs.batchDiagnostics) && (
                          <div className="mt-3 pt-3 border-t border-white/10">
                            <div className="space-y-2 text-xs">
                              {/* Batch ID */}
                              {fs.batchId && (
                                <div className="flex items-start gap-2">
                                  <span className="text-purple-400 font-semibold min-w-[80px]">Batch ID:</span>
                                  <span className="text-purple-200 font-mono text-[10px] break-all">{fs.batchId}</span>
                                </div>
                              )}

                              {/* Conversation ID */}
                              {fs.conversationId && (
                                <div className="flex items-start gap-2">
                                  <span className="text-purple-400 font-semibold min-w-[80px]">Conv ID:</span>
                                  <span className="text-purple-200 font-mono text-[10px]">{fs.conversationId}</span>
                                </div>
                              )}

                              {/* Batch Diagnostics */}
                              {fs.batchDiagnostics && (
                                <div className="space-y-2 mt-3">
                                  {/* Embedding Batch */}
                                  {fs.batchDiagnostics.embedding_batch && (
                                    <div className="bg-blue-500/10 border border-blue-400/20 rounded p-2">
                                      <div className="font-semibold text-blue-300 mb-1">Embeddings</div>
                                      <div className="space-y-1 text-[10px]">
                                        <div className="flex justify-between">
                                          <span className="text-blue-200/70">Status:</span>
                                          <span className="text-blue-200 font-mono">{String(fs.batchDiagnostics.embedding_batch.status)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                          <span className="text-blue-200/70">Progress:</span>
                                          <span className="text-blue-200 font-mono">{String(fs.batchDiagnostics.embedding_batch.progress)}</span>
                                        </div>
                                        {fs.batchDiagnostics.embedding_batch.time_elapsed_min && (
                                          <div className="flex justify-between">
                                            <span className="text-blue-200/70">Time:</span>
                                            <span className="text-blue-200 font-mono">{String(fs.batchDiagnostics.embedding_batch.time_elapsed_min)} min</span>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Hume Job */}
                                  {fs.batchDiagnostics.hume_job && (
                                    <div className="bg-purple-500/10 border border-purple-400/20 rounded p-2">
                                      <div className="font-semibold text-purple-300 mb-1">Hume Analysis</div>
                                      <div className="space-y-1 text-[10px]">
                                        <div className="flex justify-between">
                                          <span className="text-purple-200/70">Status:</span>
                                          <span className="text-purple-200 font-mono">{String(fs.batchDiagnostics.hume_job.status)}</span>
                                        </div>
                                        {fs.batchDiagnostics.hume_job.message && (
                                          <div className="text-purple-200/70 italic">{String(fs.batchDiagnostics.hume_job.message)}</div>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}

                              {/* Progress Message */}
                              {fs.progress && !fs.batchDiagnostics && (
                                <div className="text-purple-300 italic">{fs.progress}</div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* BATCH DIAGNOSTICS TEMPORARILY DISABLED - CAUSING REACT CRASH
                        {fs.status === 'processing' && fs.batchDiagnostics && (
                          <div>...</div>
                        )}
                        */}

                        {/* Emotion Analysis Section */}
                        {fs.status === 'completed' && fs.emotionData && (
                          <div className="mt-4 pt-4 border-t border-white/20">
                            <h4 className="text-sm font-semibold text-purple-200 mb-3 flex items-center gap-2">
                              Emotion Analysis
                              {!fs.emotionData.labeling_confidence && (
                                <span className="text-xs text-yellow-300 bg-yellow-500/20 px-2 py-1 rounded-full">
                                  ⚠ Low confidence labeling
                                </span>
                              )}
                            </h4>

                            {/* Top Emotions Badges */}
                            <div className="grid grid-cols-2 gap-3 mb-4">
                              <div>
                                <p className="text-xs text-purple-300 mb-1">
                                  {Object.values(fs.emotionData.speaker_labels)[0] || 'Speaker A'}:
                                </p>
                                <div className="flex flex-wrap gap-1">
                                  {fs.emotionData.speaker_a.top_5_emotions.map((emotion, idx) => (
                                    <span
                                      key={idx}
                                      className="px-2 py-1 bg-red-500/30 text-red-200 text-xs rounded-full"
                                    >
                                      {emotion}
                                    </span>
                                  ))}
                                </div>
                              </div>

                              <div>
                                <p className="text-xs text-purple-300 mb-1">
                                  {Object.values(fs.emotionData.speaker_labels)[1] || 'Speaker B'}:
                                </p>
                                <div className="flex flex-wrap gap-1">
                                  {fs.emotionData.speaker_b.top_5_emotions.map((emotion, idx) => (
                                    <span
                                      key={idx}
                                      className="px-2 py-1 bg-blue-500/30 text-blue-200 text-xs rounded-full"
                                    >
                                      {emotion}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            </div>

                            {/* Emotion Timeline Chart */}
                            {fs.emotionData.timelines && fs.emotionData.timelines.speaker_a.length > 0 && (
                              <EmotionChart
                                speakerATimeline={fs.emotionData.timelines.speaker_a}
                                speakerBTimeline={fs.emotionData.timelines.speaker_b}
                                speakerALabel={Object.values(fs.emotionData.speaker_labels)[0] || 'SPEAKER_A'}
                                speakerBLabel={Object.values(fs.emotionData.speaker_labels)[1] || 'SPEAKER_B'}
                                topEmotions={[
                                  ...fs.emotionData.speaker_a.top_5_emotions,
                                  ...fs.emotionData.speaker_b.top_5_emotions
                                ].filter((v, i, a) => a.indexOf(v) === i).slice(0, 5)}
                              />
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {fileStatuses.length === 0 && (
                <div className="text-center py-8 text-purple-300 text-sm">
                  No files added yet. Drag & drop or click to browse.
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
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-bold text-green-100 text-lg flex items-center">
                        <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd"/>
                        </svg>
                        Answer
                      </h3>
                      <button
                        onClick={handleCopyAnswer}
                        className="flex items-center gap-2 px-3 py-1.5 bg-green-500/30 hover:bg-green-500/50 text-green-100 rounded-lg transition-colors text-sm font-medium"
                        title="Copy to clipboard"
                      >
                        {copied ? (
                          <>
                            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                            </svg>
                            Copied!
                          </>
                        ) : (
                          <>
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                            Copy
                          </>
                        )}
                      </button>
                    </div>
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
