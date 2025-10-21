/**
 * API client for Conversation Insights backend
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface ConversationMetadata {
  account_id?: string;
  date?: string;
  agent_id?: string;
  [key: string]: any;
}

export interface ConversationUpload {
  id: string;
  transcript: string;
  metadata: ConversationMetadata;
}

export interface UploadResponse {
  batch_id: string;
  status: string;
  chunks: number;
  conversation_id: string;
}

export interface StatusResponse {
  batch_id: string;
  status: string;
  progress: string;
  conversation_id: string;
  details?: any;
}

export interface QueryRequest {
  query: string;
  filters?: Record<string, any>;
  top_k?: number;
}

export interface Source {
  conversation_id: string;
  date: string;
  score: number;
  text: string;
  sentiment: number;
  intents: string[];
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  processing_time_ms: number;
}

export interface Conversation {
  id: string;
  date: string;
  account_id: string;
  agent_id: string;
  sentiment: number;
  intents: string[];
}

export interface ConversationsResponse {
  conversations: Conversation[];
  total: number;
}

// API methods
export const conversationApi = {
  /**
   * Upload a conversation for processing
   */
  upload: async (conversation: ConversationUpload): Promise<UploadResponse> => {
    const response = await api.post<UploadResponse>('/upload', conversation);
    return response.data;
  },

  /**
   * Upload an audio file for transcription and processing
   */
  uploadAudio: async (file: File, conversationId?: string, metadata?: Record<string, any>): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    if (conversationId) {
      formData.append('conversation_id', conversationId);
    }
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    const response = await api.post<UploadResponse>('/upload-audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Check the status of a batch job
   */
  getStatus: async (batchId: string): Promise<StatusResponse> => {
    const response = await api.get<StatusResponse>(`/status/${batchId}`);
    return response.data;
  },

  /**
   * Query conversations
   */
  query: async (request: QueryRequest): Promise<QueryResponse> => {
    const response = await api.post<QueryResponse>('/query', request);
    return response.data;
  },

  /**
   * Get all conversations
   */
  getConversations: async (): Promise<ConversationsResponse> => {
    const response = await api.get<ConversationsResponse>('/conversations');
    return response.data;
  },

  /**
   * Health check
   */
  healthCheck: async (): Promise<any> => {
    const response = await api.get('/');
    return response.data;
  },

  /**
   * Purge all data from index (DANGER - for testing only!)
   */
  purgeIndex: async (): Promise<any> => {
    const response = await api.delete('/purge-index');
    return response.data;
  },
};

export default conversationApi;
