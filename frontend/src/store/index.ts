import { create } from 'zustand'

interface Project {
  id: string
  name: string
  description?: string
}

interface AppState {
  // Projects
  currentProject: Project | null
  projects: Project[]
  setCurrentProject: (project: Project | null) => void
  setProjects: (projects: Project[]) => void

  // Pipeline state
  pipelineSteps: PipelineStep[]
  updateStep: (stepId: string, status: PipelineStep['status']) => void

  // Results
  predictions: unknown[] | null
  metrics: Record<string, number> | null
  setPredictions: (predictions: unknown[]) => void
  setMetrics: (metrics: Record<string, number>) => void
}

interface PipelineStep {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'error'
  details?: string
}

export const useAppStore = create<AppState>((set) => ({
  // Projects
  currentProject: null,
  projects: [],
  setCurrentProject: (project) => set({ currentProject: project }),
  setProjects: (projects) => set({ projects }),

  // Pipeline
  pipelineSteps: [],
  updateStep: (stepId, status) =>
    set((state) => ({
      pipelineSteps: state.pipelineSteps.map((step) =>
        step.id === stepId ? { ...step, status } : step
      ),
    })),

  // Results
  predictions: null,
  metrics: null,
  setPredictions: (predictions) => set({ predictions }),
  setMetrics: (metrics) => set({ metrics }),
}))
