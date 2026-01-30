import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileSpreadsheet, Database } from 'lucide-react'
import { api } from '../../services/api'

interface DataPreviewProps {
  projectId: string | null
}

interface DatasetInfo {
  dataset_id: string
  rows: number
  columns: number
  column_names: string[]
  preview: Record<string, unknown>[]
}

export default function DataPreview({ projectId }: DataPreviewProps) {
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!projectId || acceptedFiles.length === 0) return

    setUploading(true)
    setError(null)

    try {
      const result = await api.uploadFile(projectId, acceptedFiles[0])
      setDataset(result)
    } catch (err) {
      setError('Failed to upload file. Please try again.')
    } finally {
      setUploading(false)
    }
  }, [projectId])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    },
    maxFiles: 1,
  })

  if (!projectId) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500">
        Select a project to manage data.
      </div>
    )
  }

  return (
    <div className="flex-1 p-6 overflow-y-auto">
      <h2 className="text-xl font-semibold text-white mb-6">Data Management</h2>

      {/* Upload Zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-blue-400 bg-blue-900/20'
            : 'border-slate-600 hover:border-slate-500'
        }`}
      >
        <input {...getInputProps()} />
        <Upload size={40} className="mx-auto mb-3 text-slate-400" />
        {uploading ? (
          <p className="text-slate-300">Uploading...</p>
        ) : isDragActive ? (
          <p className="text-blue-400">Drop your file here</p>
        ) : (
          <div>
            <p className="text-slate-300">Drag & drop your dataset here</p>
            <p className="text-sm text-slate-500 mt-1">Supports CSV, Excel (.xlsx, .xls)</p>
          </div>
        )}
      </div>

      {error && (
        <p className="mt-3 text-red-400 text-sm">{error}</p>
      )}

      {/* Dataset Preview */}
      {dataset && (
        <div className="mt-6">
          <div className="flex items-center gap-3 mb-4">
            <FileSpreadsheet size={20} className="text-green-400" />
            <div>
              <h3 className="text-white font-medium">{dataset.dataset_id}</h3>
              <p className="text-sm text-slate-400">
                {dataset.rows} rows x {dataset.columns} columns
              </p>
            </div>
          </div>

          {/* Column Names */}
          <div className="mb-4">
            <h4 className="text-sm font-medium text-slate-300 mb-2">Columns</h4>
            <div className="flex flex-wrap gap-2">
              {dataset.column_names.map((col) => (
                <span
                  key={col}
                  className="px-2 py-1 bg-slate-700 rounded text-xs text-slate-300"
                >
                  {col}
                </span>
              ))}
            </div>
          </div>

          {/* Data Preview Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-slate-700 text-slate-300">
                <tr>
                  {dataset.column_names.map((col) => (
                    <th key={col} className="px-3 py-2 font-medium whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataset.preview.map((row, i) => (
                  <tr key={i} className="border-b border-slate-700">
                    {dataset.column_names.map((col) => (
                      <td key={col} className="px-3 py-2 text-slate-400 whitespace-nowrap">
                        {String(row[col] ?? '')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Database Connection */}
      <div className="mt-8">
        <h3 className="text-lg font-medium text-white mb-3 flex items-center gap-2">
          <Database size={18} />
          Database Connections
        </h3>
        <button className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm text-slate-300 transition-colors">
          + Add Database Connection
        </button>
      </div>
    </div>
  )
}
