import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const client = axios.create({
  baseURL: `${BASE_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const api = {
  // Projects
  async getProjects() {
    const { data } = await client.get('/projects')
    return data
  },

  async createProject(name: string, description?: string) {
    const { data } = await client.post('/projects', { name, description })
    return data
  },

  async getProject(projectId: string) {
    const { data } = await client.get(`/projects/${projectId}`)
    return data
  },

  // Data
  async uploadFile(projectId: string, file: File) {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await client.post(
      `/projects/${projectId}/upload`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    return data
  },

  async getDatasets(projectId: string) {
    const { data } = await client.get(`/projects/${projectId}/datasets`)
    return data
  },

  // Database Connections
  async addConnection(projectId: string, connection: {
    name: string
    connection_string: string
    db_type: string
  }) {
    const { data } = await client.post(`/projects/${projectId}/connections`, connection)
    return data
  },

  // Chat (REST fallback)
  async sendMessage(projectId: string, message: string) {
    const { data } = await client.post(`/projects/${projectId}/chat`, { message })
    return data
  },
}
