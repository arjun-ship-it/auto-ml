import { useState, useEffect } from 'react'
import { Plus, FolderOpen } from 'lucide-react'
import { api } from '../../services/api'

interface Project {
  id: string
  name: string
  description?: string
  created_at: string
}

interface SidebarProps {
  currentProject: string | null
  onProjectSelect: (id: string) => void
}

export default function Sidebar({ currentProject, onProjectSelect }: SidebarProps) {
  const [projects, setProjects] = useState<Project[]>([])
  const [showNewProject, setShowNewProject] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')

  useEffect(() => {
    loadProjects()
  }, [])

  const loadProjects = async () => {
    try {
      const data = await api.getProjects()
      setProjects(data)
    } catch (err) {
      console.error('Failed to load projects:', err)
    }
  }

  const createProject = async () => {
    if (!newProjectName.trim()) return
    try {
      const project = await api.createProject(newProjectName)
      setProjects([project, ...projects])
      onProjectSelect(project.id)
      setNewProjectName('')
      setShowNewProject(false)
    } catch (err) {
      console.error('Failed to create project:', err)
    }
  }

  return (
    <div className="w-64 bg-slate-800 border-r border-slate-700 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <h1 className="text-lg font-bold text-white">AutoML Agent</h1>
        <p className="text-xs text-slate-400 mt-1">AI-Powered ML Pipeline</p>
      </div>

      {/* New Project Button */}
      <div className="p-3">
        <button
          onClick={() => setShowNewProject(true)}
          className="w-full flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm text-white transition-colors"
        >
          <Plus size={16} />
          New Project
        </button>
      </div>

      {/* New Project Form */}
      {showNewProject && (
        <div className="px-3 pb-3">
          <input
            type="text"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && createProject()}
            placeholder="Project name..."
            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-sm text-white placeholder-slate-400 focus:outline-none focus:border-blue-500"
            autoFocus
          />
          <div className="flex gap-2 mt-2">
            <button
              onClick={createProject}
              className="flex-1 px-2 py-1 bg-green-600 rounded text-xs text-white"
            >
              Create
            </button>
            <button
              onClick={() => setShowNewProject(false)}
              className="flex-1 px-2 py-1 bg-slate-600 rounded text-xs text-white"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Project List */}
      <div className="flex-1 overflow-y-auto p-2">
        {projects.map((project) => (
          <button
            key={project.id}
            onClick={() => onProjectSelect(project.id)}
            className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-left transition-colors mb-1 ${
              currentProject === project.id
                ? 'bg-slate-700 text-white'
                : 'text-slate-300 hover:bg-slate-700/50'
            }`}
          >
            <FolderOpen size={14} />
            <span className="truncate">{project.name}</span>
          </button>
        ))}

        {projects.length === 0 && (
          <p className="text-center text-slate-500 text-xs mt-8">
            No projects yet. Create one to get started.
          </p>
        )}
      </div>
    </div>
  )
}
