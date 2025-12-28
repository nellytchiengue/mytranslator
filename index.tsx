import React, { useState, useEffect, useCallback, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";
import { create } from 'zustand';
import { persist, createJSONStorage, StateStorage } from 'zustand/middleware';
import { get, set, del } from 'idb-keyval';
import { jsPDF } from 'jspdf';
import {
  Upload,
  FileText,
  BookOpen,
  ArrowRight,
  Download,
  Play,
  Pause,
  CheckCircle,
  Clock,
  Settings,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Trash2,
  Languages,
  Eye,
  TestTube,
  ScrollText,
  AlertCircle,
  FileDown,
  Coins,
  RefreshCw,
  RotateCcw
} from 'lucide-react';

// --- Logger Service ---
type LogLevel = 'INFO' | 'WARN' | 'ERROR' | 'DEBUG' | 'SUCCESS';

interface LogEntry {
  id: string;
  timestamp: Date;
  level: LogLevel;
  category: string;
  message: string;
  data?: any;
}

class Logger {
  private logs: LogEntry[] = [];
  private subscribers: Set<(logs: LogEntry[]) => void> = new Set();
  private maxLogs = 500;

  log(level: LogLevel, category: string, message: string, data?: any) {
    const entry: LogEntry = {
      id: crypto.randomUUID(),
      timestamp: new Date(),
      level,
      category,
      message,
      data
    };

    this.logs.unshift(entry);
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(0, this.maxLogs);
    }

    // Console output avec couleurs
    const colors: Record<LogLevel, string> = {
      INFO: '\x1b[36m',
      WARN: '\x1b[33m',
      ERROR: '\x1b[31m',
      DEBUG: '\x1b[90m',
      SUCCESS: '\x1b[32m'
    };
    console.log(`${colors[level]}[${level}] [${category}] ${message}\x1b[0m`, data || '');

    // Notifier les subscribers
    this.subscribers.forEach(cb => cb([...this.logs]));

    // Persister dans localStorage
    this.persistLogs();
  }

  info(category: string, message: string, data?: any) {
    this.log('INFO', category, message, data);
  }

  warn(category: string, message: string, data?: any) {
    this.log('WARN', category, message, data);
  }

  error(category: string, message: string, data?: any) {
    this.log('ERROR', category, message, data);
  }

  debug(category: string, message: string, data?: any) {
    this.log('DEBUG', category, message, data);
  }

  success(category: string, message: string, data?: any) {
    this.log('SUCCESS', category, message, data);
  }

  subscribe(callback: (logs: LogEntry[]) => void) {
    this.subscribers.add(callback);
    callback([...this.logs]);
    return () => this.subscribers.delete(callback);
  }

  getLogs() {
    return [...this.logs];
  }

  getLogsByCategory(category: string) {
    return this.logs.filter(l => l.category === category);
  }

  clear() {
    this.logs = [];
    this.subscribers.forEach(cb => cb([]));
    localStorage.removeItem('linguist-logs');
  }

  exportLogs(): string {
    return this.logs.map(l =>
      `[${l.timestamp.toISOString()}] [${l.level}] [${l.category}] ${l.message}${l.data ? ' | Data: ' + JSON.stringify(l.data) : ''}`
    ).join('\n');
  }

  private persistLogs() {
    try {
      localStorage.setItem('linguist-logs', JSON.stringify(this.logs.slice(0, 100)));
    } catch (e) {
      // Storage full, ignore
    }
  }

  loadPersistedLogs() {
    try {
      const saved = localStorage.getItem('linguist-logs');
      if (saved) {
        this.logs = JSON.parse(saved).map((l: any) => ({
          ...l,
          timestamp: new Date(l.timestamp)
        }));
      }
    } catch (e) {
      // Ignore
    }
  }
}

const logger = new Logger();
logger.loadPersistedLogs();

// --- Types ---
interface Chapter {
  id: string;
  title: string;
  originalText: string;
  translatedText?: string;
  status: 'pending' | 'translating' | 'completed' | 'error';
  progress: number;
}

interface TranslationProject {
  id: string;
  title: string;
  author?: string;
  sourceLanguage: 'zh' | 'en';
  chapters: Chapter[];
  createdAt: number;
  lastAccessed: number;
}

interface AppState {
  projects: TranslationProject[];
  currentProjectId: string | null;
  isTranslating: boolean;
  addProject: (project: TranslationProject) => void;
  deleteProject: (id: string) => void;
  setCurrentProject: (id: string | null) => void;
  updateChapter: (projectId: string, chapterId: string, updates: Partial<Chapter>) => void;
  setTranslating: (status: boolean) => void;
}

// --- Storage ---
const storage: StateStorage = {
  getItem: async (name: string): Promise<string | null> => {
    return (await get(name)) || null;
  },
  setItem: async (name: string, value: string): Promise<void> => {
    await set(name, value);
  },
  removeItem: async (name: string): Promise<void> => {
    await del(name);
  },
};

// --- Store ---
const useStore = create<AppState>()(
  persist(
    (set) => ({
      projects: [],
      currentProjectId: null,
      isTranslating: false,
      addProject: (project) => set((state) => ({
        projects: [project, ...state.projects],
        currentProjectId: project.id
      })),
      deleteProject: (id) => set((state) => ({
        projects: state.projects.filter(p => p.id !== id),
        currentProjectId: state.currentProjectId === id ? null : state.currentProjectId
      })),
      setCurrentProject: (id) => set({ currentProjectId: id }),
      updateChapter: (projectId, chapterId, updates) => set((state) => ({
        projects: state.projects.map((p) =>
          p.id === projectId
            ? {
              ...p,
              chapters: p.chapters.map((c) =>
                c.id === chapterId ? { ...c, ...updates } : c
              )
            }
            : p
        )
      })),
      setTranslating: (status) => set({ isTranslating: status }),
    }),
    {
      name: 'linguist-storage',
      storage: createJSONStorage(() => storage),
    }
  )
);

// --- Utils ---
const chunkText = (text: string, size: number = 3000): string[] => {
  const chunks = [];
  let current = 0;
  while (current < text.length) {
    chunks.push(text.substring(current, current + size));
    current += size;
  }
  return chunks;
};

// --- Translation Providers ---
type TranslationProvider = 'gemini' | 'huggingface' | 'ollama';

interface ProviderConfig {
  id: TranslationProvider;
  name: string;
  description: string;
  models: { id: string; name: string; description: string }[];
  isFree: boolean;
}

const PROVIDERS: ProviderConfig[] = [
  {
    id: 'gemini',
    name: 'Google Gemini',
    description: 'API Google (quota limité gratuit)',
    isFree: false,
    models: [
      { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', description: 'Dernier modèle, très performant' },
      { id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash Lite', description: 'Économique, haute capacité' },
      { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash', description: 'Rapide, bon pour la traduction' },
      { id: 'gemini-2.0-flash-lite', name: 'Gemini 2.0 Flash Lite', description: 'Très rapide, économique' },
      { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro', description: 'Haute qualité, raisonnement avancé' },
    ]
  },
  {
    id: 'huggingface',
    name: 'HuggingFace (Gratuit)',
    description: 'Modèles open source gratuits',
    isFree: true,
    models: [
      { id: 'Qwen/Qwen2.5-72B-Instruct', name: 'Qwen 2.5 72B', description: 'Excellent pour la traduction' },
      { id: 'meta-llama/Llama-3.3-70B-Instruct', name: 'Llama 3.3 70B', description: 'Très performant, gratuit' },
      { id: 'mistralai/Mistral-Small-24B-Instruct-2501', name: 'Mistral Small 24B', description: 'Rapide et efficace' },
      { id: 'microsoft/Phi-3-mini-4k-instruct', name: 'Phi-3 Mini', description: 'Léger mais capable' },
    ]
  },
  {
    id: 'ollama',
    name: 'Ollama (Local)',
    description: 'Modèles locaux via Ollama',
    isFree: true,
    models: [
      { id: 'qwen2.5:7b', name: 'Qwen 2.5 7B', description: 'Excellent pour traduction (recommandé)' },
      { id: 'llama3.2', name: 'Llama 3.2 3B', description: 'Petit, rapide mais qualité limitée' },
      { id: 'mistral', name: 'Mistral 7B', description: 'Rapide, bonne qualité' },
      { id: 'gemma2', name: 'Gemma 2', description: 'Google open source' },
    ]
  }
];

// État global
let selectedProvider: TranslationProvider = 'gemini';
let selectedModel = 'gemini-2.0-flash';
let autoFallback = true; // Basculer auto vers HuggingFace si quota dépassé

const getSelectedProvider = () => selectedProvider;
const setSelectedProvider = (provider: TranslationProvider) => {
  selectedProvider = provider;
  // Sélectionner le premier modèle du provider
  const providerConfig = PROVIDERS.find(p => p.id === provider);
  if (providerConfig && providerConfig.models.length > 0) {
    selectedModel = providerConfig.models[0].id;
  }
  logger.info('Config', `Provider changé: ${provider}, modèle: ${selectedModel}`);
};

const getAutoFallback = () => autoFallback;
const setAutoFallback = (enabled: boolean) => {
  autoFallback = enabled;
  logger.info('Config', `Auto-fallback ${enabled ? 'activé' : 'désactivé'}`);
};

// --- Abort Controller pour annuler les requêtes en cours ---
let currentAbortController: AbortController | null = null;

const getAbortSignal = (): AbortSignal => {
  if (!currentAbortController) {
    currentAbortController = new AbortController();
  }
  return currentAbortController.signal;
};

const abortCurrentTranslation = () => {
  if (currentAbortController) {
    logger.warn('Abort', 'Annulation de la traduction en cours...');
    currentAbortController.abort();
    currentAbortController = null;
  }
};

const resetAbortController = () => {
  currentAbortController = new AbortController();
};

// --- Gemini Service ---
// Instance singleton pour éviter de recréer le client à chaque appel
let aiInstance: GoogleGenAI | null = null;

// Pour compatibilité avec l'ancien code
const AVAILABLE_MODELS = PROVIDERS.find(p => p.id === 'gemini')?.models || [];

const getSelectedModel = () => selectedModel;
const setSelectedModel = (model: string) => {
  selectedModel = model;
  logger.info('Config', `Modèle changé: ${model}`);
};

const getAIInstance = (): GoogleGenAI => {
  if (!aiInstance) {
    const apiKey = process.env.GEMINI_API_KEY || process.env.API_KEY || '';
    if (!apiKey) {
      logger.error('API', 'Clé API manquante! Vérifiez votre fichier .env.local');
      throw new Error('API_KEY_MISSING');
    }
    logger.info('API', 'Initialisation du client Gemini', { keyPreview: apiKey.substring(0, 10) + '...' });
    aiInstance = new GoogleGenAI({ apiKey });
  }
  return aiInstance;
};

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// --- HuggingFace Translation Service ---
// Utilise l'API Inference Serverless avec support CORS
const translateWithHuggingFace = async (text: string, sourceLang: string, model: string): Promise<string> => {
  const startTime = Date.now();
  logger.info('HuggingFace', `Traduction avec ${model}`, { textLength: text.length });

  // Clé API HuggingFace (gratuite, à créer sur huggingface.co/settings/tokens)
  const hfApiKey = process.env.HF_API_KEY || process.env.HUGGINGFACE_API_KEY || '';

  if (!hfApiKey) {
    logger.error('HuggingFace', 'Clé API HuggingFace manquante!');
    throw new Error('HF_API_KEY manquante. Créez un token gratuit sur https://huggingface.co/settings/tokens');
  }

  // Format de message pour les modèles chat/instruct
  const messages = [
    {
      role: "user",
      content: `Tu es un traducteur littéraire expert. Traduis ce texte ${sourceLang === 'zh' ? 'Chinois' : 'Anglais'} en français.

Règles :
- Préserve le style littéraire et les nuances émotionnelles
- Garde la mise en forme (paragraphes, dialogues avec guillemets français « »)
- Ne traduis pas les noms propres s'ils sont déjà romanisés ou courants
- Adapte les expressions idiomatiques au français naturel
- Réponds UNIQUEMENT avec la traduction, sans commentaires ni explication

Texte à traduire :
${text}`
    }
  ];

  try {
    // Utiliser l'endpoint serverless chat/completions (supporte CORS)
    const response = await fetch(`https://router.huggingface.co/hf-inference/models/${model}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${hfApiKey}`
      },
      body: JSON.stringify({
        messages: messages,
        max_tokens: Math.min(text.length * 2, 4096),
        temperature: 0.3,
        stream: false
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));

      // Gestion des erreurs spécifiques
      if (response.status === 503) {
        logger.warn('HuggingFace', 'Modèle en cours de chargement, attente 20s...');
        await sleep(20000);
        return translateWithHuggingFace(text, sourceLang, model);
      }

      if (response.status === 429) {
        logger.warn('HuggingFace', 'Rate limit atteint, attente 10s...');
        await sleep(10000);
        return translateWithHuggingFace(text, sourceLang, model);
      }

      throw new Error(`HuggingFace error ${response.status}: ${JSON.stringify(errorData)}`);
    }

    const data = await response.json();

    // Extraire le contenu de la réponse (format OpenAI-compatible)
    const result = data.choices?.[0]?.message?.content || '';
    const duration = Date.now() - startTime;

    if (!result) {
      logger.warn('HuggingFace', 'Réponse vide', { data });
      throw new Error('Réponse vide de HuggingFace');
    }

    logger.success('HuggingFace', `Traduction terminée en ${duration}ms`, {
      inputLength: text.length,
      outputLength: result.length
    });

    return result.trim();
  } catch (error: any) {
    logger.error('HuggingFace', 'Erreur', { error: error.message });
    throw error;
  }
};

// --- Ollama Translation Service (Local) ---
const translateWithOllama = async (text: string, sourceLang: string, model: string, signal?: AbortSignal): Promise<string> => {
  const startTime = Date.now();
  logger.info('Ollama', `Traduction locale avec ${model}`, { textLength: text.length });

  const prompt = `Tu es un traducteur littéraire expert. Traduis ce texte ${sourceLang === 'zh' ? 'Chinois' : 'Anglais'} en français.

Règles :
- Préserve le style littéraire et les nuances émotionnelles
- Garde la mise en forme (paragraphes, dialogues avec guillemets français « »)
- Ne traduis pas les noms propres s'ils sont déjà romanisés ou courants
- Adapte les expressions idiomatiques au français naturel
- Réponds UNIQUEMENT avec la traduction, sans commentaires

Texte à traduire :
${text}`;

  try {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.3,
          num_predict: Math.min(text.length * 2, 4096)
        }
      }),
      signal // Support de l'annulation
    });

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.status} - Vérifiez que Ollama est lancé (ollama serve)`);
    }

    const data = await response.json();
    const result = data.response || '';
    const duration = Date.now() - startTime;

    logger.success('Ollama', `Traduction terminée en ${duration}ms`, {
      inputLength: text.length,
      outputLength: result.length
    });

    return result.trim();
  } catch (error: any) {
    if (error.name === 'AbortError') {
      logger.warn('Ollama', 'Requête annulée par l\'utilisateur');
      throw new Error('CANCELLED');
    }
    if (error.message.includes('fetch')) {
      logger.error('Ollama', 'Ollama non accessible. Lancez: ollama serve');
    }
    throw error;
  }
};

interface TranslateOptions {
  text: string;
  sourceLang: string;
  isPreview?: boolean;
  retryCount?: number;
  forceProvider?: TranslationProvider;
  signal?: AbortSignal;
}

// Traduction avec Gemini
const translateWithGemini = async (text: string, sourceLang: string, model: string): Promise<string> => {
  const ai = getAIInstance();

  const prompt = `Tu es un traducteur littéraire expert. Traduis ce texte ${sourceLang === 'zh' ? 'Chinois' : 'Anglais'} en français.

Règles :
- Préserve le style littéraire et les nuances émotionnelles
- Garde la mise en forme (paragraphes, dialogues avec guillemets français « »)
- Ne traduis pas les noms propres s'ils sont déjà romanisés ou courants, garde la romanisation pinyin si c'est du chinois
- Adapte les expressions idiomatiques au français naturel
- Pas de commentaires, uniquement la traduction

Texte à traduire :
${text}`;

  const response = await ai.models.generateContent({
    model: model,
    contents: [{ parts: [{ text: prompt }] }],
  });

  return response.text || "";
};

const translateChunk = async (options: TranslateOptions): Promise<string> => {
  const { text, sourceLang, isPreview = false, retryCount = 0, forceProvider, signal } = options;
  const maxRetries = 3;
  const startTime = Date.now();

  const provider = forceProvider || selectedProvider;
  const modelName = getSelectedModel();

  // Vérifier si annulé avant de commencer
  if (signal?.aborted) {
    throw new Error('CANCELLED');
  }

  logger.info('Translation', `Début traduction ${isPreview ? '(PREVIEW)' : ''} [${provider}] ${modelName}`, {
    textLength: text.length,
    sourceLang,
    attempt: retryCount + 1,
    provider,
    model: modelName
  });

  try {
    let result: string;

    switch (provider) {
      case 'huggingface':
        result = await translateWithHuggingFace(text, sourceLang, modelName);
        break;
      case 'ollama':
        result = await translateWithOllama(text, sourceLang, modelName, signal);
        break;
      case 'gemini':
      default:
        result = await translateWithGemini(text, sourceLang, modelName);
        break;
    }

    const duration = Date.now() - startTime;
    logger.success('Translation', `Traduction terminée en ${duration}ms [${provider}]`, {
      inputLength: text.length,
      outputLength: result.length,
      preview: result.substring(0, 100) + '...'
    });

    return result;
  } catch (error: any) {
    const duration = Date.now() - startTime;
    const errorMessage = error.message || String(error);
    const isRateLimitError = errorMessage.includes('429') || errorMessage.includes('RESOURCE_EXHAUSTED') || errorMessage.includes('quota');

    logger.error('API', `Erreur ${provider} après ${duration}ms`, {
      error: errorMessage,
      statusCode: error.status,
      retryCount,
      isRateLimitError
    });

    // Auto-fallback vers HuggingFace si quota Gemini dépassé
    if (isRateLimitError && autoFallback && provider === 'gemini') {
      logger.warn('Fallback', '⚡ Quota Gemini épuisé - Basculement automatique vers HuggingFace (gratuit)');

      // Changer le provider global pour les prochains appels
      selectedProvider = 'huggingface';
      selectedModel = 'Qwen/Qwen2.5-72B-Instruct';

      // Retry immédiat avec HuggingFace
      return translateChunk({ ...options, retryCount: 0, forceProvider: 'huggingface' });
    }

    // Retry logic avec backoff exponentiel (seulement pour Gemini)
    if (retryCount < maxRetries && provider === 'gemini') {
      const backoffTime = isRateLimitError
        ? Math.max(60000, Math.pow(2, retryCount) * 30000)
        : Math.pow(2, retryCount) * 1000;

      logger.warn('API', `${isRateLimitError ? 'Quota dépassé - ' : ''}Retry dans ${Math.round(backoffTime/1000)}s...`, { attempt: retryCount + 2 });
      await sleep(backoffTime);
      return translateChunk({ ...options, retryCount: retryCount + 1 });
    }

    throw error;
  }
};

// Fonction pour tester un extrait avant de lancer la traduction complète
const testTranslation = async (text: string, sourceLang: string): Promise<{ success: boolean; result?: string; error?: string; duration: number }> => {
  const startTime = Date.now();
  logger.info('Test', 'Lancement du test de traduction...', { textLength: text.length });

  try {
    const result = await translateChunk({ text, sourceLang, isPreview: true });
    const duration = Date.now() - startTime;
    logger.success('Test', `Test réussi en ${duration}ms`);
    return { success: true, result, duration };
  } catch (error: any) {
    const duration = Date.now() - startTime;
    logger.error('Test', 'Test échoué', { error: error.message });
    return { success: false, error: error.message, duration };
  }
};

// Fonction de traduction avec streaming (affichage progressif)
// Supporte tous les providers (Gemini, HuggingFace, Ollama)
const translateWithStreaming = async (
  text: string,
  sourceLang: string,
  onChunk: (partialText: string) => void
): Promise<{ success: boolean; result?: string; error?: string; duration: number }> => {
  const startTime = Date.now();
  const provider = selectedProvider;
  const modelName = getSelectedModel();

  logger.info('Streaming', `Début traduction streaming avec ${modelName} [${provider}]`, { textLength: text.length });

  const prompt = `Tu es un traducteur littéraire expert. Traduis ce texte ${sourceLang === 'zh' ? 'Chinois' : 'Anglais'} en français.

Règles :
- Préserve le style littéraire et les nuances émotionnelles
- Garde la mise en forme (paragraphes, dialogues avec guillemets français « »)
- Ne traduis pas les noms propres s'ils sont déjà romanisés ou courants, garde la romanisation pinyin si c'est du chinois
- Adapte les expressions idiomatiques au français naturel
- Pas de commentaires, uniquement la traduction

Texte à traduire :
${text}`;

  try {
    // Ollama avec streaming
    if (provider === 'ollama') {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelName,
          prompt: prompt,
          stream: true,
          options: { temperature: 0.3 }
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      let fullText = '';
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const json = JSON.parse(line);
            if (json.response) {
              fullText += json.response;
              onChunk(fullText);
            }
          } catch {
            // Ignore parsing errors
          }
        }
      }

      const duration = Date.now() - startTime;
      logger.success('Streaming', `Traduction streaming terminée en ${duration}ms [ollama]`, { outputLength: fullText.length });
      return { success: true, result: fullText, duration };
    }

    // HuggingFace (pas de vrai streaming, mais on simule)
    if (provider === 'huggingface') {
      const result = await translateWithHuggingFace(text, sourceLang, modelName);
      onChunk(result);
      const duration = Date.now() - startTime;
      return { success: true, result, duration };
    }

    // Gemini avec streaming (par défaut)
    const ai = getAIInstance();
    const response = await ai.models.generateContentStream({
      model: modelName,
      contents: [{ parts: [{ text: prompt }] }],
    });

    let fullText = '';

    for await (const chunk of response) {
      const chunkText = chunk.text || '';
      fullText += chunkText;
      onChunk(fullText);
    }

    const duration = Date.now() - startTime;
    logger.success('Streaming', `Traduction streaming terminée en ${duration}ms`, {
      outputLength: fullText.length
    });

    return { success: true, result: fullText, duration };
  } catch (error: any) {
    const duration = Date.now() - startTime;
    logger.error('Streaming', 'Erreur streaming', { error: error.message });
    return { success: false, error: error.message, duration };
  }
};

// --- Token Estimation ---
// Estimation approximative : ~4 caractères = 1 token pour le français/anglais, ~2 caractères = 1 token pour le chinois
const estimateTokens = (text: string, lang: 'zh' | 'en' = 'en'): number => {
  if (!text) return 0;
  const charsPerToken = lang === 'zh' ? 2 : 4;
  return Math.ceil(text.length / charsPerToken);
};

interface TokenStats {
  used: number;           // Tokens déjà utilisés (chapitres traduits)
  remaining: number;      // Tokens restants à traduire
  total: number;          // Total estimé pour le projet
  promptOverhead: number; // Overhead pour les prompts système
}

const calculateTokenStats = (chapters: Chapter[], sourceLang: 'zh' | 'en'): TokenStats => {
  const PROMPT_OVERHEAD_PER_CHAPTER = 150; // ~150 tokens pour le prompt système par chapitre

  let used = 0;
  let remaining = 0;

  chapters.forEach(chapter => {
    const inputTokens = estimateTokens(chapter.originalText, sourceLang);
    const outputTokens = chapter.translatedText ? estimateTokens(chapter.translatedText, 'en') : inputTokens; // Output en français

    if (chapter.status === 'completed') {
      used += inputTokens + outputTokens + PROMPT_OVERHEAD_PER_CHAPTER;
    } else {
      remaining += inputTokens + inputTokens + PROMPT_OVERHEAD_PER_CHAPTER; // Input + output estimé
    }
  });

  const promptOverhead = chapters.length * PROMPT_OVERHEAD_PER_CHAPTER;
  const total = used + remaining;

  return { used, remaining, total, promptOverhead };
};

// --- PDF Export ---
const exportToPDF = (project: TranslationProject): void => {
  logger.info('Export', 'Génération du PDF...', { title: project.title });

  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4'
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 20;
  const maxWidth = pageWidth - 2 * margin;
  let yPosition = margin;

  // Titre du livre
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(24);
  doc.text(project.title, pageWidth / 2, yPosition, { align: 'center' });
  yPosition += 15;

  // Sous-titre
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(12);
  doc.setTextColor(128);
  const completedCount = project.chapters.filter(c => c.status === 'completed').length;
  doc.text(`Traduction ${project.sourceLanguage === 'zh' ? 'Chinois' : 'Anglais'} → Français`, pageWidth / 2, yPosition, { align: 'center' });
  yPosition += 8;
  doc.text(`${completedCount}/${project.chapters.length} chapitres traduits`, pageWidth / 2, yPosition, { align: 'center' });
  yPosition += 20;

  doc.setTextColor(0);

  // Contenu des chapitres
  project.chapters.forEach((chapter, index) => {
    // Vérifier si on a besoin d'une nouvelle page
    if (yPosition > pageHeight - 40) {
      doc.addPage();
      yPosition = margin;
    }

    // Titre du chapitre
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.text(`${index + 1}. ${chapter.title}`, margin, yPosition);
    yPosition += 10;

    // Contenu (traduit ou original)
    const content = chapter.translatedText || chapter.originalText;
    const status = chapter.status === 'completed' ? '' : ' [Non traduit]';

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(11);

    if (chapter.status !== 'completed') {
      doc.setTextColor(150);
    }

    // Découper le texte en lignes
    const lines = doc.splitTextToSize(content + status, maxWidth);

    lines.forEach((line: string) => {
      if (yPosition > pageHeight - margin) {
        doc.addPage();
        yPosition = margin;
      }
      doc.text(line, margin, yPosition);
      yPosition += 6;
    });

    doc.setTextColor(0);
    yPosition += 10; // Espace entre chapitres
  });

  // Footer
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFontSize(9);
    doc.setTextColor(150);
    doc.text(`Page ${i}/${totalPages}`, pageWidth / 2, pageHeight - 10, { align: 'center' });
    doc.text('Généré par Linguist AI', pageWidth - margin, pageHeight - 10, { align: 'right' });
  }

  // Télécharger
  const filename = `${project.title.replace(/[^a-zA-Z0-9]/g, '_')}_traduction.pdf`;
  doc.save(filename);

  logger.success('Export', `PDF généré: ${filename}`, { pages: totalPages });
};

// --- Components ---

const FileUploader: React.FC = () => {
  const { addProject } = useStore();
  const [isParsing, setIsParsing] = useState(false);
  const [sourceLang, setSourceLang] = useState<'zh' | 'en'>('en');

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsParsing(true);
    const id = crypto.randomUUID();
    const chapters: Chapter[] = [];

    try {
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        const arrayBuffer = await file.arrayBuffer();

        // Configure PDF.js worker
        // @ts-ignore
        if (window.pdfjsLib) {
          // @ts-ignore
          window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        }

        // @ts-ignore
        const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;

        let fullText = '';
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const textContent = await page.getTextContent();
          const pageText = textContent.items.map((item: any) => item.str).join(' ');
          fullText += pageText + '\n\n';
        }

        // Split by estimated chapter sizes or page-based logic
        const chunks = chunkText(fullText, 5000);
        chunks.forEach((chunk, index) => {
          chapters.push({
            id: crypto.randomUUID(),
            title: `Segment ${index + 1}`,
            originalText: chunk,
            status: 'pending',
            progress: 0
          });
        });

      } else if (file.name.endsWith('.epub')) {
        const arrayBuffer = await file.arrayBuffer();
        // @ts-ignore
        const book = window.ePub(arrayBuffer);
        const navigation = await book.loaded.navigation;
        const spine = await book.loaded.spine;

        // Extract content for each spine item
        for (const item of spine.spineItems) {
          const doc = await item.load(book.load.bind(book));
          const text = doc.body.innerText || doc.body.textContent || '';
          if (text.trim()) {
            chapters.push({
              id: crypto.randomUUID(),
              title: item.idref || `Chapter ${chapters.length + 1}`,
              originalText: text,
              status: 'pending',
              progress: 0
            });
          }
        }
      }

      addProject({
        id,
        title: file.name.replace(/\.[^/.]+$/, ""),
        sourceLanguage: sourceLang,
        chapters,
        createdAt: Date.now(),
        lastAccessed: Date.now()
      });
    } catch (err: any) {
      console.error(err);
      alert(`Erreur lors du parsing du fichier: ${err.message || err}`);
    } finally {
      setIsParsing(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center p-12 border-2 border-dashed border-slate-700 rounded-2xl bg-slate-900/30 hover:bg-slate-900/50 transition-all group">
      <Upload className="w-12 h-12 text-slate-500 mb-4 group-hover:text-blue-500 transition-colors" />
      <h3 className="text-xl font-semibold mb-2">Importer un livre</h3>
      <p className="text-slate-400 mb-6 text-center max-w-sm">
        Glissez-déposez votre fichier PDF ou EPUB pour commencer la traduction.
      </p>

      <div className="flex gap-4 mb-8">
        <button
          onClick={() => setSourceLang('en')}
          className={`px-4 py-2 rounded-lg flex items-center gap-2 border transition-all ${sourceLang === 'en' ? 'bg-blue-600 border-blue-500 text-white' : 'bg-slate-800 border-slate-700 text-slate-400'}`}
        >
          <span className="text-sm font-medium">Anglais</span>
        </button>
        <button
          onClick={() => setSourceLang('zh')}
          className={`px-4 py-2 rounded-lg flex items-center gap-2 border transition-all ${sourceLang === 'zh' ? 'bg-blue-600 border-blue-500 text-white' : 'bg-slate-800 border-slate-700 text-slate-400'}`}
        >
          <span className="text-sm font-medium">Chinois</span>
        </button>
      </div>

      <label className="relative cursor-pointer bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-8 rounded-xl transition-all shadow-lg shadow-blue-900/20 active:scale-95">
        <span>{isParsing ? "Analyse en cours..." : "Sélectionner un fichier"}</span>
        <input type="file" className="hidden" accept=".pdf,.epub" onChange={onFileChange} disabled={isParsing} />
      </label>
    </div>
  );
};

const ProjectList: React.FC = () => {
  const { projects, setCurrentProject, deleteProject } = useStore();

  if (projects.length === 0) return null;

  return (
    <div className="mt-12 w-full max-w-4xl">
      <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
        <BookOpen className="w-5 h-5 text-blue-500" /> Projets récents
      </h2>
      <div className="grid gap-4">
        {projects.map((p) => (
          <div
            key={p.id}
            className="glass-panel p-5 rounded-xl flex items-center justify-between group hover:border-blue-500/50 transition-all cursor-pointer"
            onClick={() => setCurrentProject(p.id)}
          >
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-slate-800 rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-slate-400" />
              </div>
              <div>
                <h4 className="font-medium group-hover:text-blue-400 transition-colors">{p.title}</h4>
                <div className="flex items-center gap-3 mt-1 text-sm text-slate-500">
                  <span className="flex items-center gap-1 uppercase">
                    <Languages className="w-3 h-3" /> {p.sourceLanguage} → FR
                  </span>
                  <span>•</span>
                  <span>{p.chapters.length} chapitres</span>
                  <span>•</span>
                  <span>{Math.round((p.chapters.filter(c => c.status === 'completed').length / p.chapters.length) * 100)}%</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={(e) => { e.stopPropagation(); deleteProject(p.id); }}
                className="p-2 text-slate-500 hover:text-red-400 transition-colors"
              >
                <Trash2 className="w-5 h-5" />
              </button>
              <ArrowRight className="w-5 h-5 text-slate-600 group-hover:text-blue-500 group-hover:translate-x-1 transition-all" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// --- Log Panel Component ---
const LogPanel: React.FC<{ isOpen: boolean; onClose: () => void }> = ({ isOpen, onClose }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<LogLevel | 'ALL'>('ALL');

  useEffect(() => {
    return logger.subscribe(setLogs);
  }, []);

  const filteredLogs = filter === 'ALL' ? logs : logs.filter(l => l.level === filter);

  const getLevelColor = (level: LogLevel) => {
    switch (level) {
      case 'INFO': return 'text-cyan-400';
      case 'WARN': return 'text-amber-400';
      case 'ERROR': return 'text-red-400';
      case 'DEBUG': return 'text-slate-500';
      case 'SUCCESS': return 'text-emerald-400';
    }
  };

  const exportLogs = () => {
    const content = logger.exportLogs();
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `linguist-logs-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-slate-700">
          <h3 className="text-lg font-bold flex items-center gap-2">
            <ScrollText className="w-5 h-5 text-blue-500" /> Journal d'activité
          </h3>
          <div className="flex items-center gap-2">
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as LogLevel | 'ALL')}
              className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1 text-sm"
            >
              <option value="ALL">Tous</option>
              <option value="INFO">Info</option>
              <option value="SUCCESS">Succès</option>
              <option value="WARN">Warnings</option>
              <option value="ERROR">Erreurs</option>
              <option value="DEBUG">Debug</option>
            </select>
            <button onClick={exportLogs} className="px-3 py-1 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm flex items-center gap-1">
              <Download className="w-4 h-4" /> Export
            </button>
            <button onClick={() => { logger.clear(); }} className="px-3 py-1 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm flex items-center gap-1">
              <Trash2 className="w-4 h-4" /> Clear
            </button>
            <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-lg">✕</button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-4 font-mono text-sm custom-scrollbar">
          {filteredLogs.length === 0 ? (
            <div className="text-center text-slate-500 py-8">Aucun log disponible</div>
          ) : (
            filteredLogs.map((log) => (
              <div key={log.id} className="py-1 border-b border-slate-800 hover:bg-slate-800/50">
                <span className="text-slate-600">[{log.timestamp.toLocaleTimeString()}]</span>
                <span className={`ml-2 font-bold ${getLevelColor(log.level)}`}>[{log.level}]</span>
                <span className="ml-2 text-slate-400">[{log.category}]</span>
                <span className="ml-2 text-slate-200">{log.message}</span>
                {log.data && <span className="ml-2 text-slate-500 text-xs">{JSON.stringify(log.data)}</span>}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

// --- Token Stats Modal Component ---
const TokenStatsModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  project: TranslationProject;
}> = ({ isOpen, onClose, project }) => {
  const [stats, setStats] = useState<TokenStats | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    if (isOpen && project) {
      refreshStats();
    }
  }, [isOpen, project]);

  const refreshStats = () => {
    setIsRefreshing(true);
    const newStats = calculateTokenStats(project.chapters, project.sourceLanguage);
    setStats(newStats);
    logger.info('Tokens', 'Stats mises à jour', newStats);
    setTimeout(() => setIsRefreshing(false), 300);
  };

  if (!isOpen || !stats) return null;

  const completedChapters = project.chapters.filter(c => c.status === 'completed').length;
  const totalChapters = project.chapters.length;
  const percentComplete = Math.round((completedChapters / totalChapters) * 100);

  // Estimer le coût (prix approximatifs Gemini 2.0 Flash)
  const costPerMillionInput = 0.10; // $0.10 per 1M input tokens
  const costPerMillionOutput = 0.40; // $0.40 per 1M output tokens
  const estimatedCost = ((stats.used / 1000000) * (costPerMillionInput + costPerMillionOutput) / 2).toFixed(4);
  const remainingCost = ((stats.remaining / 1000000) * (costPerMillionInput + costPerMillionOutput) / 2).toFixed(4);

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-lg">
        <div className="flex items-center justify-between p-4 border-b border-slate-700">
          <h3 className="text-lg font-bold flex items-center gap-2">
            <Coins className="w-5 h-5 text-amber-500" /> Utilisation des Tokens
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={refreshStats}
              disabled={isRefreshing}
              className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
              title="Actualiser"
            >
              <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            </button>
            <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-lg">✕</button>
          </div>
        </div>

        <div className="p-6 space-y-6">
          {/* Progress visuel */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-slate-400">Progression</span>
              <span className="font-medium">{completedChapters}/{totalChapters} chapitres ({percentComplete}%)</span>
            </div>
            <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-blue-500 transition-all duration-500"
                style={{ width: `${percentComplete}%` }}
              />
            </div>
          </div>

          {/* Stats en grille */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-emerald-900/20 border border-emerald-500/30 rounded-xl p-4">
              <div className="text-emerald-400 text-xs font-medium uppercase tracking-wide mb-1">Tokens utilisés</div>
              <div className="text-2xl font-bold text-emerald-300">{stats.used.toLocaleString()}</div>
              <div className="text-xs text-emerald-400/60 mt-1">≈ ${estimatedCost} estimé</div>
            </div>

            <div className="bg-amber-900/20 border border-amber-500/30 rounded-xl p-4">
              <div className="text-amber-400 text-xs font-medium uppercase tracking-wide mb-1">Tokens restants</div>
              <div className="text-2xl font-bold text-amber-300">{stats.remaining.toLocaleString()}</div>
              <div className="text-xs text-amber-400/60 mt-1">≈ ${remainingCost} estimé</div>
            </div>

            <div className="bg-blue-900/20 border border-blue-500/30 rounded-xl p-4">
              <div className="text-blue-400 text-xs font-medium uppercase tracking-wide mb-1">Total projet</div>
              <div className="text-2xl font-bold text-blue-300">{stats.total.toLocaleString()}</div>
            </div>

            <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
              <div className="text-slate-400 text-xs font-medium uppercase tracking-wide mb-1">Overhead prompts</div>
              <div className="text-2xl font-bold text-slate-300">{stats.promptOverhead.toLocaleString()}</div>
            </div>
          </div>

          {/* Info */}
          <div className="bg-slate-800/50 rounded-lg p-4 text-sm text-slate-400">
            <p className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0 text-slate-500" />
              <span>
                Estimations basées sur ~4 caractères/token (latin) et ~2 caractères/token (chinois).
                Les coûts réels peuvent varier selon le modèle utilisé.
              </span>
            </p>
          </div>
        </div>

        <div className="p-4 border-t border-slate-700 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm font-medium"
          >
            Fermer
          </button>
        </div>
      </div>
    </div>
  );
};

// --- Preview Modal Component ---
const PreviewModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  chapter: Chapter | undefined;
  sourceLang: string;
  onTranslateChapter: () => void;
}> = ({ isOpen, onClose, chapter, sourceLang, onTranslateChapter }) => {
  const [previewText, setPreviewText] = useState('');
  const [previewResult, setPreviewResult] = useState<string | null>(null);
  const [isTestingPreview, setIsTestingPreview] = useState(false);
  const [testStatus, setTestStatus] = useState<{ success: boolean; duration: number } | null>(null);
  const [useStreaming, setUseStreaming] = useState(true); // Mode streaming par défaut

  useEffect(() => {
    if (chapter && isOpen) {
      // Prendre les 500 premiers caractères comme extrait par défaut
      setPreviewText(chapter.originalText.substring(0, 500));
      setPreviewResult(null);
      setTestStatus(null);
    }
  }, [chapter, isOpen]);

  const runPreviewTest = async () => {
    if (!previewText.trim()) return;
    setIsTestingPreview(true);
    setTestStatus(null);
    setPreviewResult(null);

    logger.info('Preview', `Test de prévisualisation lancé (${useStreaming ? 'streaming' : 'normal'})`, { textLength: previewText.length });

    if (useStreaming) {
      // Mode streaming - affichage progressif
      const result = await translateWithStreaming(
        previewText,
        sourceLang,
        (partialText) => {
          setPreviewResult(partialText);
        }
      );
      setTestStatus({ success: result.success, duration: result.duration });
      if (!result.success) {
        setPreviewResult(`Erreur: ${result.error}`);
      }
    } else {
      // Mode normal
      const result = await testTranslation(previewText, sourceLang);
      setTestStatus({ success: result.success, duration: result.duration });
      if (result.success && result.result) {
        setPreviewResult(result.result);
      } else {
        setPreviewResult(`Erreur: ${result.error}`);
      }
    }
    setIsTestingPreview(false);
  };

  if (!isOpen || !chapter) return null;

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-6xl max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-slate-700">
          <div>
            <h3 className="text-lg font-bold flex items-center gap-2">
              <Eye className="w-5 h-5 text-blue-500" /> Prévisualisation - {chapter.title}
            </h3>
            <p className="text-sm text-slate-400 mt-1">
              Testez un extrait avant de lancer la traduction complète
            </p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-lg">✕</button>
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Texte original / extrait à tester */}
          <div className="flex-1 flex flex-col border-r border-slate-700">
            <div className="p-4 border-b border-slate-700 bg-slate-800/50">
              <h4 className="text-sm font-medium text-slate-400 mb-2">Extrait à tester (modifiable)</h4>
              <textarea
                value={previewText}
                onChange={(e) => setPreviewText(e.target.value)}
                className="w-full h-32 bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm resize-none custom-scrollbar"
                placeholder="Collez ou modifiez le texte à tester..."
              />
              <div className="flex items-center gap-3 mt-2">
                <button
                  onClick={runPreviewTest}
                  disabled={isTestingPreview || !previewText.trim()}
                  className="flex items-center gap-2 px-4 py-2 bg-amber-600 hover:bg-amber-700 disabled:bg-slate-700 rounded-lg text-sm font-medium transition-all"
                >
                  {isTestingPreview ? (
                    <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Test en cours...</>
                  ) : (
                    <><TestTube className="w-4 h-4" /> Tester cet extrait</>
                  )}
                </button>
                {/* Toggle streaming */}
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <div className="relative">
                    <input
                      type="checkbox"
                      checked={useStreaming}
                      onChange={(e) => setUseStreaming(e.target.checked)}
                      className="sr-only"
                      disabled={isTestingPreview}
                    />
                    <div className={`w-10 h-5 rounded-full transition-colors ${useStreaming ? 'bg-blue-600' : 'bg-slate-700'}`}>
                      <div className={`w-4 h-4 rounded-full bg-white shadow-md transform transition-transform mt-0.5 ${useStreaming ? 'translate-x-5 ml-0.5' : 'translate-x-0.5'}`} />
                    </div>
                  </div>
                  <span className="text-xs text-slate-400">Streaming</span>
                </label>
                {testStatus && (
                  <span className={`text-sm ${testStatus.success ? 'text-emerald-400' : 'text-red-400'}`}>
                    {testStatus.success ? `✓ Succès en ${testStatus.duration}ms` : '✗ Échec'}
                  </span>
                )}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Chapitre complet ({chapter.originalText.length} caractères)</h4>
              <div className="whitespace-pre-wrap text-sm text-slate-400 leading-relaxed">
                {chapter.originalText}
              </div>
            </div>
          </div>

          {/* Résultat du test */}
          <div className="flex-1 flex flex-col">
            <div className="p-4 border-b border-slate-700 bg-blue-900/20 flex items-center justify-between">
              <h4 className="text-sm font-medium text-blue-400">Résultat de la prévisualisation</h4>
              {isTestingPreview && useStreaming && (
                <span className="text-xs text-blue-400 flex items-center gap-2 animate-pulse">
                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-ping" />
                  Streaming en cours...
                </span>
              )}
            </div>
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
              {previewResult ? (
                <div className="whitespace-pre-wrap text-sm text-slate-200 leading-relaxed">
                  {previewResult}
                  {isTestingPreview && useStreaming && (
                    <span className="inline-block w-2 h-4 bg-blue-500 animate-pulse ml-1" />
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-slate-600">
                  <TestTube className="w-12 h-12 mb-4 opacity-50" />
                  <p>Lancez un test pour voir la traduction</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-slate-700 flex items-center justify-between bg-slate-800/50">
          <div className="text-sm text-slate-400">
            {testStatus?.success ? (
              <span className="text-emerald-400 flex items-center gap-2">
                <CheckCircle className="w-4 h-4" /> Test réussi - La traduction fonctionne correctement
              </span>
            ) : testStatus ? (
              <span className="text-red-400 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" /> Le test a échoué - Vérifiez votre clé API
              </span>
            ) : (
              <span>Testez d'abord un extrait avant de traduire le chapitre complet</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button onClick={onClose} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm">
              Annuler
            </button>
            <button
              onClick={() => { onClose(); onTranslateChapter(); }}
              disabled={!testStatus?.success}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg text-sm font-medium flex items-center gap-2"
            >
              <Play className="w-4 h-4" /> Traduire ce chapitre
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const ProjectDetail: React.FC = () => {
  const { currentProjectId, projects, setCurrentProject, updateChapter, isTranslating, setTranslating } = useStore();
  const project = projects.find(p => p.id === currentProjectId);
  const [activeChapterId, setActiveChapterId] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [showTokenStats, setShowTokenStats] = useState(false);
  const [currentProvider, setCurrentProviderState] = useState<TranslationProvider>(getSelectedProvider());
  const [currentModel, setCurrentModelState] = useState(getSelectedModel());
  const [isAutoFallback, setIsAutoFallbackState] = useState(getAutoFallback());

  // Mettre à jour les modèles disponibles quand le provider change
  const currentProviderConfig = PROVIDERS.find(p => p.id === currentProvider);
  const availableModels = currentProviderConfig?.models || [];

  const activeChapter = project?.chapters.find(c => c.id === activeChapterId) || project?.chapters[0];

  const startTranslation = useCallback(async () => {
    if (!project || isTranslating) return;

    // Créer un nouveau AbortController pour cette session
    resetAbortController();
    const signal = getAbortSignal();

    setTranslating(true);
    logger.info('Translation', `Démarrage traduction projet "${project.title}"`, { chaptersCount: project.chapters.length });

    const pendingChapters = project.chapters.filter(c => c.status !== 'completed');

    for (const chapter of pendingChapters) {
      // Vérifier si annulé ou mis en pause
      if (!useStore.getState().isTranslating || signal.aborted) {
        logger.warn('Translation', 'Traduction arrêtée par l\'utilisateur');
        // Remettre le chapitre en cours en pending s'il était en translating
        const currentChapter = useStore.getState().projects.find(p => p.id === project.id)?.chapters.find(c => c.id === chapter.id);
        if (currentChapter?.status === 'translating') {
          updateChapter(project.id, chapter.id, { status: 'pending', progress: 0 });
        }
        break;
      }

      logger.info('Chapter', `Début du chapitre : "${chapter.title}"`, { chapterId: chapter.id });
      updateChapter(project.id, chapter.id, { status: 'translating', progress: 10 });

      try {
        const result = await translateChunk({
          text: chapter.originalText,
          sourceLang: project.sourceLanguage,
          signal
        });

        // Vérifier si annulé pendant la traduction
        if (signal.aborted) {
          logger.warn('Chapter', `Chapitre "${chapter.title}" annulé`);
          updateChapter(project.id, chapter.id, { status: 'pending', progress: 0 });
          break;
        }

        logger.success('Chapter', `Chapitre "${chapter.title}" terminé`, {
          inputLength: chapter.originalText.length,
          outputLength: result.length
        });

        updateChapter(project.id, chapter.id, {
          translatedText: result,
          status: 'completed',
          progress: 100
        });
      } catch (err: any) {
        if (err.message === 'CANCELLED') {
          logger.warn('Chapter', `Chapitre "${chapter.title}" annulé par l'utilisateur`);
          updateChapter(project.id, chapter.id, { status: 'pending', progress: 0 });
          break;
        }
        logger.error('Chapter', `Erreur chapitre "${chapter.title}"`, { error: err.message });
        updateChapter(project.id, chapter.id, { status: 'error' });
      }
    }
    logger.info('Translation', 'Session de traduction terminée');
    setTranslating(false);
  }, [project, isTranslating, updateChapter, setTranslating]);

  // Arrêter/Annuler la traduction en cours
  const stopTranslation = () => {
    logger.info('Control', 'Arrêt de la traduction demandé');
    abortCurrentTranslation();
    setTranslating(false);
  };

  // Réinitialiser un chapitre (effacer la traduction et remettre en pending)
  const resetChapter = (chapterId: string) => {
    if (!project || isTranslating) return;
    const chapter = project.chapters.find(c => c.id === chapterId);
    if (!chapter) return;

    logger.info('Reset', `Réinitialisation du chapitre "${chapter.title}"`, { chapterId });
    updateChapter(project.id, chapterId, {
      translatedText: '',
      status: 'pending',
      progress: 0
    });
  };

  const exportProject = (format: 'md' | 'txt') => {
    if (!project) return;
    const content = project.chapters
      .map(c => `# ${c.title}\n\n${c.translatedText || c.originalText}`)
      .join('\n\n---\n\n');

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${project.title}.${format}`;
    a.click();
  };

  if (!project) return null;

  const progress = Math.round((project.chapters.filter(c => c.status === 'completed').length / project.chapters.length) * 100);

  return (
    <div className="w-full flex flex-col h-screen max-h-screen overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 glass-panel border-b-0">
        <div className="flex items-center gap-4">
          <button onClick={() => setCurrentProject(null)} className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
            <ChevronLeft className="w-6 h-6" />
          </button>
          <div>
            <h1 className="text-xl font-bold">{project.title}</h1>
            <p className="text-sm text-slate-400 flex items-center gap-2">
              Traduction {project.sourceLanguage === 'zh' ? 'Chinoise' : 'Anglaise'} → Française • {progress}% traduit
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Sélecteur de provider + modèle */}
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-slate-400" />
            {/* Provider */}
            <select
              value={currentProvider}
              onChange={(e) => {
                const newProvider = e.target.value as TranslationProvider;
                setCurrentProviderState(newProvider);
                setSelectedProvider(newProvider);
                // Mettre à jour le modèle avec le premier du nouveau provider
                const providerModels = PROVIDERS.find(p => p.id === newProvider)?.models || [];
                if (providerModels.length > 0) {
                  setCurrentModelState(providerModels[0].id);
                }
              }}
              disabled={isTranslating}
              className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-2 text-sm disabled:opacity-50"
              title="Choisir le provider"
            >
              {PROVIDERS.map((provider) => (
                <option key={provider.id} value={provider.id}>
                  {provider.name}
                </option>
              ))}
            </select>
            {/* Modèle */}
            <select
              value={currentModel}
              onChange={(e) => {
                setCurrentModelState(e.target.value);
                setSelectedModel(e.target.value);
              }}
              disabled={isTranslating}
              className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-2 text-sm disabled:opacity-50 max-w-[180px]"
              title="Choisir le modèle"
            >
              {availableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
            {/* Auto-fallback toggle */}
            <label className="flex items-center gap-1 cursor-pointer select-none" title="Basculer auto vers HuggingFace si quota Gemini épuisé">
              <div className="relative">
                <input
                  type="checkbox"
                  checked={isAutoFallback}
                  onChange={(e) => {
                    setIsAutoFallbackState(e.target.checked);
                    setAutoFallback(e.target.checked);
                  }}
                  className="sr-only"
                  disabled={isTranslating}
                />
                <div className={`w-8 h-4 rounded-full transition-colors ${isAutoFallback ? 'bg-emerald-600' : 'bg-slate-700'}`}>
                  <div className={`w-3 h-3 rounded-full bg-white shadow-md transform transition-transform mt-0.5 ${isAutoFallback ? 'translate-x-4 ml-0.5' : 'translate-x-0.5'}`} />
                </div>
              </div>
              <span className="text-xs text-slate-400">Auto</span>
            </label>
          </div>
          <button
            onClick={() => setShowTokenStats(true)}
            className="flex items-center gap-2 px-3 py-2 bg-amber-600 hover:bg-amber-700 border border-amber-500 rounded-lg transition-colors text-sm"
            title="Voir l'utilisation des tokens"
          >
            <Coins className="w-4 h-4" /> Tokens
          </button>
          <button
            onClick={() => setShowLogs(true)}
            className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors text-sm"
            title="Voir les logs"
          >
            <ScrollText className="w-4 h-4" /> Logs
          </button>
          <button
            onClick={() => setShowPreview(true)}
            disabled={!activeChapter || activeChapter.status === 'completed'}
            className="flex items-center gap-2 px-3 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg transition-colors text-sm"
            title="Prévisualiser et tester"
          >
            <Eye className="w-4 h-4" /> Preview
          </button>
          <button
            onClick={isTranslating ? stopTranslation : startTranslation}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${isTranslating ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'}`}
          >
            {isTranslating ? <><Pause className="w-4 h-4" /> Arrêter</> : <><Play className="w-4 h-4" /> {progress === 100 ? 'Relancer' : 'Traduire'}</>}
          </button>
          {!isTranslating && (
            <>
              <button
                onClick={() => exportToPDF(project)}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 border border-red-500 rounded-lg transition-colors"
                title="Exporter en PDF (même partiel)"
              >
                <FileDown className="w-4 h-4" /> PDF
              </button>
              <button
                onClick={() => exportProject('md')}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" /> Export MD
              </button>
            </>
          )}
        </div>
      </header>

      {/* Modals */}
      <LogPanel isOpen={showLogs} onClose={() => setShowLogs(false)} />
      <TokenStatsModal
        isOpen={showTokenStats}
        onClose={() => setShowTokenStats(false)}
        project={project}
      />
      <PreviewModal
        isOpen={showPreview}
        onClose={() => setShowPreview(false)}
        chapter={activeChapter}
        sourceLang={project.sourceLanguage}
        onTranslateChapter={startTranslation}
      />

      {/* Progress Bar */}
      <div className="w-full h-1 bg-slate-800">
        <div
          className="h-full bg-blue-500 transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 glass-panel border-y-0 border-l-0 custom-scrollbar overflow-y-auto">
          <div className="p-4 space-y-2">
            {project.chapters.map((c, idx) => (
              <div
                key={c.id}
                className={`w-full p-3 rounded-xl transition-all flex items-center justify-between group ${activeChapter?.id === c.id ? 'bg-blue-600/20 border border-blue-500/50' : 'hover:bg-slate-800 border border-transparent'}`}
              >
                <button
                  onClick={() => setActiveChapterId(c.id)}
                  className="flex items-center gap-3 min-w-0 flex-1 text-left"
                >
                  <span className="text-xs text-slate-500 font-mono flex-shrink-0">{(idx + 1).toString().padStart(2, '0')}</span>
                  <span className={`text-sm font-medium truncate ${activeChapter?.id === c.id ? 'text-blue-400' : 'text-slate-300'}`}>
                    {c.title}
                  </span>
                </button>
                <div className="flex items-center gap-2">
                  {/* Bouton reset - visible si complété ou erreur */}
                  {(c.status === 'completed' || c.status === 'error') && !isTranslating && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Réinitialiser "${c.title}" ?`)) {
                          resetChapter(c.id);
                        }
                      }}
                      className="p-1 rounded hover:bg-slate-700 opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Réinitialiser ce chapitre"
                    >
                      <RotateCcw className="w-3.5 h-3.5 text-slate-400 hover:text-amber-400" />
                    </button>
                  )}
                  {/* Status indicator */}
                  {c.status === 'completed' ? (
                    <CheckCircle className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                  ) : c.status === 'translating' ? (
                    <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse flex-shrink-0" />
                  ) : c.status === 'error' ? (
                    <div className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" />
                  ) : (
                    <div className="w-2 h-2 rounded-full bg-slate-700 flex-shrink-0" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* Main View - Split Screen */}
        <main className="flex-1 flex overflow-hidden bg-slate-950">
          <div className="flex-1 h-full overflow-y-auto custom-scrollbar p-12 border-r border-slate-800">
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-8 border-b border-slate-800 pb-2">Texte Original</h2>
            <div className="prose prose-invert prose-slate max-w-none">
              <div className="whitespace-pre-wrap leading-relaxed text-slate-300 text-lg">
                {activeChapter?.originalText}
              </div>
            </div>
          </div>
          <div className="flex-1 h-full overflow-y-auto custom-scrollbar p-12 bg-slate-900/10">
            <h2 className="text-xs font-bold text-blue-500 uppercase tracking-widest mb-8 border-b border-blue-900/30 pb-2">Traduction Française</h2>
            <div className="prose prose-invert prose-blue max-w-none">
              <div className="whitespace-pre-wrap leading-relaxed text-slate-200 text-lg">
                {activeChapter?.translatedText || (
                  <div className="flex flex-col items-center justify-center h-64 text-slate-600 italic">
                    {activeChapter?.status === 'translating' ? (
                      <div className="flex flex-col items-center gap-4">
                        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                        <p>Traduction en cours...</p>
                      </div>
                    ) : (
                      <p>En attente de traduction</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

const Dashboard: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col items-center py-20 px-6 max-w-6xl mx-auto">
      <div className="text-center mb-16">
        <div className="inline-block p-3 bg-blue-600/10 rounded-2xl mb-4 border border-blue-500/20">
          <BookOpen className="w-10 h-10 text-blue-500" />
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight mb-4 text-white">Linguist AI</h1>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto">
          Transformez vos livres étrangers en chefs-d'œuvre français grâce à la puissance du modèle Gemini de Google.
        </p>
      </div>

      <div className="w-full max-w-4xl">
        <FileUploader />
        <ProjectList />
      </div>

      <footer className="mt-auto py-12 text-slate-500 text-sm flex items-center gap-6">
        <span className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-emerald-500" /> 100% Client-side</span>
        <span className="flex items-center gap-2"><Clock className="w-4 h-4" /> Autosave enabled</span>
        <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" className="flex items-center gap-1 hover:text-blue-400 transition-colors">
          <ExternalLink className="w-4 h-4" /> Billing Info
        </a>
      </footer>
    </div>
  );
};

const App: React.FC = () => {
  const { currentProjectId } = useStore();

  return (
    <div className="min-h-screen">
      {currentProjectId ? <ProjectDetail /> : <Dashboard />}
    </div>
  );
};

// --- Initialization ---
const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}