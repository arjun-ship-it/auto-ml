import { BarChart3, TrendingUp, Award } from 'lucide-react'

interface ResultsViewProps {
  projectId: string | null
}

export default function ResultsView({ projectId }: ResultsViewProps) {
  if (!projectId) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        Select a project to view results.
      </div>
    )
  }

  return (
    <div className="flex-1 p-6 overflow-y-auto">
      <h2 className="text-xl font-semibold text-white mb-6">Results & Predictions</h2>

      {/* Placeholder - will be populated after model training */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 mb-2">
            <Award size={16} />
            <span className="text-sm">Best Model</span>
          </div>
          <p className="text-lg font-semibold text-white">--</p>
          <p className="text-xs text-slate-500 mt-1">Run the pipeline to see results</p>
        </div>

        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 mb-2">
            <BarChart3 size={16} />
            <span className="text-sm">Accuracy / R2</span>
          </div>
          <p className="text-lg font-semibold text-white">--</p>
          <p className="text-xs text-slate-500 mt-1">Performance metric</p>
        </div>

        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 mb-2">
            <TrendingUp size={16} />
            <span className="text-sm">Predictions</span>
          </div>
          <p className="text-lg font-semibold text-white">--</p>
          <p className="text-xs text-slate-500 mt-1">Generated predictions</p>
        </div>
      </div>

      {/* Predictions Table Placeholder */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-medium text-white mb-4">Prediction Results</h3>
        <div className="text-center py-12 text-slate-500">
          <BarChart3 size={48} className="mx-auto mb-3 opacity-30" />
          <p>No predictions yet.</p>
          <p className="text-sm mt-1">Chat with the agent to build your ML pipeline and generate predictions.</p>
        </div>
      </div>
    </div>
  )
}
