import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Loader2, Image as ImageIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DropZoneProps {
  onFile: (file: File) => void
  loading: boolean
}

const ACCEPTED = { 'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] }

export default function DropZone({ onFile, loading }: DropZoneProps) {
  const [preview, setPreview] = useState<string | null>(null)

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (!accepted[0]) return
      setPreview(URL.createObjectURL(accepted[0]))
      onFile(accepted[0])
    },
    [onFile]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxFiles: 1,
    disabled: loading,
  })

  return (
    <div
      {...getRootProps()}
      className={cn(
        'relative cursor-pointer rounded-xl border-2 border-dashed transition-colors p-8',
        'flex flex-col items-center justify-center gap-4 min-h-[220px]',
        isDragActive
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/60 hover:bg-muted/30',
        loading && 'pointer-events-none opacity-70'
      )}
    >
      <input {...getInputProps()} />

      {preview ? (
        <img
          src={preview}
          alt="Preview"
          className="max-h-40 max-w-full rounded-lg object-contain shadow-sm"
        />
      ) : (
        <div className="p-4 rounded-full bg-muted">
          <ImageIcon className="h-10 w-10 text-muted-foreground" />
        </div>
      )}

      {loading ? (
        <div className="flex items-center gap-2 text-primary font-medium">
          <Loader2 className="h-5 w-5 animate-spin" />
          Classifying…
        </div>
      ) : (
        <div className="text-center">
          <p className="font-medium text-sm">
            {isDragActive ? 'Drop it here!' : 'Drag & drop a waste image'}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            or click to browse — JPG, PNG, WebP, BMP (max 10 MB)
          </p>
        </div>
      )}

      {!loading && (
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-md bg-primary text-primary-foreground px-4 py-2 text-sm font-medium hover:bg-primary/90 transition-colors"
        >
          <Upload className="h-4 w-4" />
          {preview ? 'Upload Another' : 'Choose File'}
        </button>
      )}
    </div>
  )
}
