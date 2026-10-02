import { useEffect, useRef, useId } from 'react'
import Icon from './Icons'

export function Modal({ title, children, onClose, className = '' }) {
  const ref = useRef(null)
  const heading = useId()
  useEffect(() => { const dialog = ref.current; dialog.showModal(); return () => dialog.close() }, [])
  return <dialog ref={ref} className={`modal ${className}`} aria-labelledby={heading} onCancel={e => { e.preventDefault(); onClose() }}><div className="modal-heading"><h2 id={heading}>{title}</h2><button className="icon-button" onClick={onClose} aria-label="关闭 / Close"><Icon name="close"/></button></div>{children}</dialog>
}
export function ErrorNotice({ message, onDismiss }) { return message ? <div className="error-notice" role="alert"><Icon name="alert" size={18}/><span>{message}</span>{onDismiss && <button onClick={onDismiss} aria-label="关闭 / Dismiss"><Icon name="close" size={16}/></button>}</div> : null }
