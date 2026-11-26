import { useCallback, useEffect, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Camera,
  Image as ImageIcon,
  Loader2,
  StopCircle,
  Upload,
  Video,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface DropZoneProps {
  onFile: (file: File) => void
  loading: boolean
}

const ACCEPTED = { 'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] }

export default function DropZone({ onFile, loading }: DropZoneProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [mode, setMode] = useState<'upload' | 'camera'>('upload')
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCameraActive(false)
  }, [])

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [stopCamera])

  const startCamera = useCallback(async () => {
    setCameraError(null)
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError('Live camera access is not supported in this browser context.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = stream
      setCameraActive(true)
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => undefined)
      }
    } catch {
      setCameraError('Camera permission denied or no camera device detected.')
      stopCamera()
    }
  }, [stopCamera])

  const captureFrame = useCallback(() => {
    const video = videoRef.current
    if (!video || loading) return
    const width = video.videoWidth || 640
    const height = video.videoHeight || 480
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(video, 0, 0, width, height)
    canvas.toBlob(
      (blob) => {
        if (!blob) return
        const file = new File([blob], `camera_capture_${Date.now()}.jpg`, { type: 'image/jpeg' })
        setPreview(URL.createObjectURL(blob))
        stopCamera()
        setMode('upload')
        onFile(file)
      },
      'image/jpeg',
      0.92
    )
  }, [loading, onFile, stopCamera])

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
    disabled: loading || mode !== 'upload',
  })

  const switchMode = (next: 'upload' | 'camera') => {
    setMode(next)
    setCameraError(null)
    if (next === 'upload') {
      stopCamera()
    } else {
      startCamera()
    }
  }

  return (
    <div className="card-eco p-4 sm:p-5 space-y-4">
      {/* Mode Switcher: Upload vs Live Camera */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="tab-nav">
          <button
            type="button"
            onClick={() => switchMode('upload')}
            disabled={loading}
            className={mode === 'upload' ? 'tab-item-active text-xs' : 'tab-item-inactive text-xs'}
          >
            <Upload className="h-3.5 w-3.5" />
            <span>Upload Image</span>
          </button>
          <button
            type="button"
            onClick={() => switchMode('camera')}
            disabled={loading}
            className={mode === 'camera' ? 'tab-item-active text-xs' : 'tab-item-inactive text-xs'}
          >
            <Camera className="h-3.5 w-3.5" />
            <span>Live Camera</span>
          </button>
        </div>
        <span className="text-[11px] font-medium text-slate-400">
          JPG, PNG, WebP, BMP · Max 10 MB
        </span>
      </div>

      {mode === 'camera' ? (
        <div className="rounded-2xl border-2 border-dashed border-emerald-300 bg-slate-950 p-5 flex flex-col items-center gap-4 text-white">
          <div className="relative w-full max-w-lg overflow-hidden rounded-xl bg-slate-900 aspect-video flex items-center justify-center ring-1 ring-white/10">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={cn('w-full h-full object-cover', !cameraActive && 'hidden')}
            />
            {!cameraActive && (
              <div className="flex flex-col items-center gap-2 text-slate-400 p-4 text-center">
                <Video className="h-8 w-8 text-emerald-400" />
                <span className="text-xs">
                  {cameraError || 'Initializing camera stream…'}
                </span>
              </div>
            )}
          </div>

          {cameraError && (
            <p className="text-xs text-rose-300 text-center">{cameraError}</p>
          )}

          <div className="flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              onClick={captureFrame}
              disabled={!cameraActive || loading}
              className="btn-primary text-xs px-4 py-2"
            >
              <Camera className="h-4 w-4" />
              <span>Capture &amp; Analyze Frame</span>
            </button>
            {cameraActive ? (
              <button
                type="button"
                onClick={() => switchMode('upload')}
                className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800 px-3.5 py-2 text-xs font-semibold text-slate-200 hover:bg-slate-700 transition-colors"
              >
                <StopCircle className="h-4 w-4" />
                <span>Stop Camera</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={startCamera}
                className="inline-flex items-center gap-1.5 rounded-xl border border-emerald-500/60 bg-emerald-950/60 px-3.5 py-2 text-xs font-semibold text-emerald-200 hover:bg-emerald-900 transition-colors"
              >
                Retry Camera
              </button>
            )}
          </div>
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={cn(
            'relative cursor-pointer rounded-2xl border-2 border-dashed transition-all p-8',
            'flex flex-col items-center justify-center gap-4 min-h-[220px]',
            isDragActive
              ? 'border-ecosort-emerald bg-ecosort-mint/70 scale-[0.995]'
              : 'border-slate-200 bg-slate-50/50 hover:border-ecosort-emerald/60 hover:bg-ecosort-sage/60',
            loading && 'pointer-events-none opacity-70'
          )}
        >
          <input {...getInputProps()} />

          {preview ? (
            <div className="p-2 rounded-xl bg-white border border-slate-200 shadow-eco-sm">
              <img
                src={preview}
                alt="Preview"
                className="max-h-40 max-w-full rounded-lg object-contain"
              />
            </div>
          ) : (
            <div className="p-4 rounded-2xl bg-ecosort-mint text-ecosort-emerald border border-emerald-200/70 shadow-eco-sm">
              <ImageIcon className="h-9 w-9" />
            </div>
          )}

          {loading ? (
            <div className="flex items-center gap-2 text-ecosort-emerald font-semibold text-sm">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>Analyzing waste image…</span>
            </div>
          ) : (
            <div className="text-center space-y-1">
              <p className="font-heading font-bold text-sm text-slate-900">
                {isDragActive ? 'Drop your waste image here!' : 'Drag & drop a waste image'}
              </p>
              <p className="text-xs text-slate-500">
                or click anywhere inside this box to browse local files
              </p>
            </div>
          )}

          {!loading && (
            <button
              type="button"
              className="btn-primary text-xs px-4 py-2"
            >
              <Upload className="h-3.5 w-3.5" />
              <span>{preview ? 'Upload Another' : 'Choose File'}</span>
            </button>
          )}
        </div>
      )}
    </div>
  )
}
