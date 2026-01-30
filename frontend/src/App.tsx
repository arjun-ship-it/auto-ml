import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useState } from 'react'
import Sidebar from './components/Chat/Sidebar'
import ChatView from './components/Chat/ChatView'
import DataPreview from './components/DataPreview/DataPreview'
import PipelineView from './components/Pipeline/PipelineView'
import ResultsView from './components/Results/ResultsView'

function App() {
  const [currentProject, setCurrentProject] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'chat' | 'data' | 'pipeline' | 'results'>('chat')

  return (
    <BrowserRouter>
      <div className="flex h-screen bg-slate-900">
        {/* Sidebar */}
        <Sidebar
          currentProject={currentProject}
          onProjectSelect={setCurrentProject}
        />

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Tab Navigation */}
          <nav className="flex border-b border-slate-700 bg-slate-800">
            {(['chat', 'data', 'pipeline', 'results'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-3 text-sm font-medium capitalize transition-colors ${
                  activeTab === tab
                    ? 'text-blue-400 border-b-2 border-blue-400'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {tab}
              </button>
            ))}
          </nav>

          {/* Tab Content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === 'chat' && <ChatView projectId={currentProject} />}
            {activeTab === 'data' && <DataPreview projectId={currentProject} />}
            {activeTab === 'pipeline' && <PipelineView projectId={currentProject} />}
            {activeTab === 'results' && <ResultsView projectId={currentProject} />}
          </div>
        </div>
      </div>
    </BrowserRouter>
  )
}

export default App
