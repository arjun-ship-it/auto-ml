import { CheckCircle, Circle, Loader2, AlertCircle } from 'lucide-react'

interface PipelineViewProps {
  projectId: string | null
}

interface PipelineStep {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'error'
  details?: string
}

export default function PipelineView({ projectId }: PipelineViewProps) {
  // Pipeline steps will be populated by the agent as it works
  const steps: PipelineStep[] = [
    { id: '1', name: 'Requirement Gathering', status: 'pending', details: 'Collecting problem definition and constraints' },
    { id: '2', name: 'Data Ingestion', status: 'pending', details: 'Loading and validating data' },
    { id: '3', name: 'Exploratory Data Analysis', status: 'pending', details: 'Profiling, distributions, correlations' },
    { id: '4', name: 'Data Preprocessing', status: 'pending', details: 'Cleaning, encoding, scaling' },
    { id: '5', name: 'Feature Engineering', status: 'pending', details: 'Creating new features' },
    { id: '6', name: 'Model Selection', status: 'pending', details: 'AI-powered model recommendation' },
    { id: '7', name: 'Model Training', status: 'pending', details: 'Training with best parameters' },
    { id: '8', name: 'Evaluation & Validation', status: 'pending', details: 'Performance metrics and checks' },
    { id: '9', name: 'Prediction', status: 'pending', details: 'Generating final predictions' },
  ]

  if (!projectId) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        Select a project to view pipeline.
      </div>
    )
  }

  const getIcon = (status: PipelineStep['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle size={20} className="text-green-400" />
      case 'running':
        return <Loader2 size={20} className="text-blue-400 animate-spin" />
      case 'error':
        return <AlertCircle size={20} className="text-red-400" />
      default:
        return <Circle size={20} className="text-slate-600" />
    }
  }

  return (
    <div className="flex-1 p-6 overflow-y-auto">
      <h2 className="text-xl font-semibold text-white mb-6">ML Pipeline</h2>

      <div className="max-w-2xl">
        {steps.map((step, index) => (
          <div key={step.id} className="flex gap-4 mb-1">
            {/* Timeline */}
            <div className="flex flex-col items-center">
              {getIcon(step.status)}
              {index < steps.length - 1 && (
                <div className={`w-0.5 h-12 mt-1 ${
                  step.status === 'completed' ? 'bg-green-400' : 'bg-slate-700'
                }`} />
              )}
            </div>

            {/* Content */}
            <div className="pb-8">
              <h3 className={`font-medium ${
                step.status === 'completed' ? 'text-green-400' :
                step.status === 'running' ? 'text-blue-400' :
                'text-slate-300'
              }`}>
                {step.name}
              </h3>
              <p className="text-sm text-slate-500 mt-1">{step.details}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
