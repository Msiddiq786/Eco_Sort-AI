import { useCallback, useEffect, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Camera, Image as ImageIcon, Loader2, StopCircle, Upload, Video } from 'lucide-react'
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
    <div className="space-y-3">
      {/* Mode Switcher: Upload vs Live Camera */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => switchMode('upload')}
          disabled={loading}
          className={cn(
            'inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold border transition-colors',
            mode === 'upload'
              ? 'bg-emerald-600 text-white border-emerald-600 shadow-sm'
              : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
          )}
        >
          <Upload className="h-3.5 w-3.5" />
          Upload Image
        </button>
        <button
          type="button"
          onClick={() => switchMode('camera')}
          disabled={loading}
          className={cn(
            'inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold border transition-colors',
            mode === 'camera'
              ? 'bg-emerald-600 text-white border-emerald-600 shadow-sm'
              : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
          )}
        >
          <Camera className="h-3.5 w-3.5" />
          Live Camera
        </button>
      </div>

      {mode === 'camera' ? (
        <div className="rounded-xl border-2 border-dashed border-emerald-300 bg-slate-950/95 p-5 flex flex-col items-center gap-4 text-white">
          <div className="relative w-full max-w-lg overflow-hidden rounded-lg bg-slate-900 aspect-video flex items-center justify-center">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={cn('w-full h-full object-cover', !cameraActive && 'hidden')}
            />
            {!cameraActive && (
              <div className="flex flex-col items-center gap-2 text-slate-400 p-4 text-center">
                <Video className="h-8 w-8" />
                <span className="text-xs">
                  {cameraError || 'Initializing camera stream…'}
                </span>
              </div>
            )}
          </div>

          {cameraError && (
            <p className="text-xs text-rose-300 text-center">{cameraError}</p>
          )}

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={captureFrame}
              disabled={!cameraActive || loading}
              className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
            >
              <Camera className="h-4 w-4" />
              Capture & Analyze Frame
            </button>
            {cameraActive ? (
              <button
                type="button"
                onClick={() => switchMode('upload')}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-200 hover:bg-slate-700"
              >
                <StopCircle className="h-4 w-4" />
                Stop Camera
              </button>
            ) : (
              <button
                type="button"
                onClick={startCamera}
                className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/60 bg-emerald-950/60 px-3 py-2 text-xs font-semibold text-emerald-200 hover:bg-emerald-900"
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
              Analyzing…
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
      )}
    </div>
  )
}
