import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import WorkspaceApp from './WorkspaceApp.jsx'
import './styles.css'
import './components.css'
import './readability.css'

createRoot(document.getElementById('root')).render(<StrictMode><WorkspaceApp/></StrictMode>)
