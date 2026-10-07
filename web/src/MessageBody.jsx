import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// Provider text is untrusted: raw HTML and remote images are never rendered.
const components = {
  a: ({ href, children }) => href && (/^(https?:\/\/|#)/i.test(href) || /^\/api\/workspaces\/workspace_[a-z0-9]+\/artifacts\/artifact_[a-z0-9]+$/i.test(href))
    ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
    : <span>{children}</span>,
  img: ({ alt }) => <span>{alt || ''}</span>,
  table: ({ children }) => <div className="message-table"><table>{children}</table></div>,
}

export default function MessageBody({ content }) {
  return <div className="message-markdown"><Markdown remarkPlugins={[remarkGfm]} components={components} skipHtml>{content}</Markdown></div>
}
