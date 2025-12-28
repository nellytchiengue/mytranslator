import React, { useState, useEffect, useCallback, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
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
  Languages
} from 'lucide-react';

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
    { name: 'linguist-storage' }
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

// --- Gemini Service ---
const translateChunk = async (text: string, sourceLang: string): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
  const modelName = 'gemini-3-pro-preview';
  
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
    const response = await ai.models.generateContent({
      model: modelName,
      contents: [{ parts: [{ text: prompt }] }],
      config: {
        thinkingConfig: { thinkingBudget: 2000 }
      }
    });
    return response.text || "";
  } catch (error) {
    console.error("Gemini Error:", error);
    throw error;
  }
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
    } catch (err) {
      console.error(err);
      alert("Erreur lors du parsing du fichier.");
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

const ProjectDetail: React.FC = () => {
  const { currentProjectId, projects, setCurrentProject, updateChapter, isTranslating, setTranslating } = useStore();
  const project = projects.find(p => p.id === currentProjectId);
  const [activeChapterId, setActiveChapterId] = useState<string | null>(null);

  const activeChapter = project?.chapters.find(c => c.id === activeChapterId) || project?.chapters[0];

  const startTranslation = useCallback(async () => {
    if (!project || isTranslating) return;
    setTranslating(true);

    const pendingChapters = project.chapters.filter(c => c.status !== 'completed');
    
    for (const chapter of pendingChapters) {
      if (!useStore.getState().isTranslating) break; // Check for pause
      
      updateChapter(project.id, chapter.id, { status: 'translating', progress: 10 });
      
      try {
        const result = await translateChunk(chapter.originalText, project.sourceLanguage);
        updateChapter(project.id, chapter.id, { 
          translatedText: result, 
          status: 'completed', 
          progress: 100 
        });
      } catch (err) {
        updateChapter(project.id, chapter.id, { status: 'error' });
      }
    }
    setTranslating(false);
  }, [project, isTranslating, updateChapter, setTranslating]);

  const togglePause = () => {
    setTranslating(!isTranslating);
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
          <button 
            onClick={togglePause}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${isTranslating ? 'bg-amber-600 hover:bg-amber-700' : 'bg-blue-600 hover:bg-blue-700'}`}
          >
            {isTranslating ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> {progress === 100 ? 'Relancer' : 'Reprendre'}</>}
          </button>
          {!isTranslating && (
            <button 
              onClick={() => exportProject('md')}
              className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors"
            >
              <Download className="w-4 h-4" /> Export MD
            </button>
          )}
        </div>
      </header>

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
              <button
                key={c.id}
                onClick={() => setActiveChapterId(c.id)}
                className={`w-full text-left p-3 rounded-xl transition-all flex items-center justify-between ${activeChapter?.id === c.id ? 'bg-blue-600/20 border border-blue-500/50' : 'hover:bg-slate-800 border border-transparent'}`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span className="text-xs text-slate-500 font-mono flex-shrink-0">{(idx + 1).toString().padStart(2, '0')}</span>
                  <span className={`text-sm font-medium truncate ${activeChapter?.id === c.id ? 'text-blue-400' : 'text-slate-300'}`}>
                    {c.title}
                  </span>
                </div>
                {c.status === 'completed' ? (
                  <CheckCircle className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                ) : c.status === 'translating' ? (
                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse flex-shrink-0" />
                ) : c.status === 'error' ? (
                  <div className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" />
                ) : (
                  <div className="w-2 h-2 rounded-full bg-slate-700 flex-shrink-0" />
                )}
              </button>
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