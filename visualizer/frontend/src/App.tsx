'use client';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  Loader2Icon,
  PauseIcon,
  PlayIcon,
  SkipBackIcon,
  SkipForwardIcon,
  UploadIcon,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

const MODEL_TYPES = [
  { id: 'effnetb2', name: 'EfficientNet-B2' },
  { id: 'resnet50', name: 'ResNet-50' },
  { id: 'convnext_tiny', name: 'ConvNeXt-Tiny' },
  { id: 'clip_vitb16', name: 'CLIP ViT-B/16' },
] as const;

const MUPPET_CHARACTERS = [
    { id: 'kermit', name: 'Kermit', color: 'bg-green-500' },
    { id: 'fozzie_bear', name: 'Fozzie Bear', color: 'bg-orange-500' },
    { id: 'miss_piggy', name: 'Miss Piggy', color: 'bg-pink-500' },
    { id: 'statler_waldorf', name: 'Statler & Waldorf', color: 'bg-purple-500' },
    { id: 'the_cook', name: 'The Swedish Chef', color: 'bg-red-500' },
    { id: 'rowlf_the_dog', name: 'Rowlf The Dog', color: 'bg-amber-700' },
] as const;

export function MuppetShowPredictor() {
    const [modelType, setModelType] = useState<string>('effnetb2');
    const [modelLoadLoading, setModelLoadLoading] = useState(false);
    const [loadedModelName, setLoadedModelName] = useState<string | null>(null);
    const [videoFile, setVideoFile] = useState<File | null>(null);
    const [videoUrl, setVideoUrl] = useState<string>('');

    // Server-side video state
    const [videoId, setVideoId] = useState<string | null>(null);
    const [serverFps, setServerFps] = useState<number>(25);
    const [serverFrameCount, setServerFrameCount] = useState<number>(0);
    const [videoUploading, setVideoUploading] = useState<boolean>(false);

    // Whether the browser can render the video natively (false for AVI, etc.)
    const [isVideoPlayable, setIsVideoPlayable] = useState<boolean>(true);
    // Frame image fetched from the server, used when the browser can't render the video
    const [serverFrameData, setServerFrameData] = useState<string>('');

    const [currentFrame, setCurrentFrame] = useState(0);
    const [totalFrames, setTotalFrames] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const [selectedCharacters, setSelectedCharacters] = useState<string[]>(MUPPET_CHARACTERS.map((c) => c.id));

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [predictionResult, setPredictionResult] = useState<any>(null);
    const [currentCharacterIndex, setCurrentCharacterIndex] = useState(0);
    const [loading, setLoading] = useState(false);

    const videoRef = useRef<HTMLVideoElement>(null);
    const videoInputRef = useRef<HTMLInputElement>(null);
    // Debounce server frame requests while scrubbing
    const frameRequestTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const handleVideoFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setVideoFile(file);
        setVideoId(null);
        setServerFps(25);
        setServerFrameCount(0);
        setIsVideoPlayable(true);
        setServerFrameData('');
        setPredictionResult(null);
        setCurrentFrame(0);
        setTotalFrames(0);

        const url = URL.createObjectURL(file);
        setVideoUrl(url);

        // Upload to server so OpenCV can extract frames (needed for AVI and other
        // formats that browsers cannot decode via the canvas API)
        setVideoUploading(true);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch(`${API_BASE}/api/upload-video`, {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) throw new Error('Upload failed');
            const data = await res.json();
            setVideoId(data.videoId);
            setServerFps(data.fps);
            setServerFrameCount(data.frameCount);
        } catch {
            toast('Warning', {
                description: 'Could not upload video to server. Prediction may not work.',
            });
        } finally {
            setVideoUploading(false);
        }
    };

    const handleLoadModel = async () => {
        setModelLoadLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ modelType }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail ?? 'Model load failed');
            }
            const displayName = MODEL_TYPES.find((m) => m.id === modelType)?.name ?? modelType;
            setLoadedModelName(displayName);
            toast('Model loaded', { description: displayName });
        } catch (err) {
            toast('Error', {
                description: err instanceof Error ? err.message : 'Model could not be loaded',
            });
        } finally {
            setModelLoadLoading(false);
        }
    };

    const toggleCharacter = (characterId: string) => {
        setSelectedCharacters((prev) =>
            prev.includes(characterId) ? prev.filter((id) => id !== characterId) : [...prev, characterId]
        );
    };

    const toggleAllCharacters = () => {
        if (selectedCharacters.length === MUPPET_CHARACTERS.length) {
            setSelectedCharacters([]);
        } else {
            setSelectedCharacters(MUPPET_CHARACTERS.map((c) => c.id));
        }
    };

    // Detect whether the browser can render this video format
    useEffect(() => {
        const video = videoRef.current;
        if (!video || !videoUrl) return;

        const handleLoadedMetadata = () => {
            const canPlay = video.videoWidth > 0 && video.videoHeight > 0;
            setIsVideoPlayable(canPlay);
            if (canPlay) {
                const fps = serverFps || 25;
                setTotalFrames(Math.floor(video.duration * fps));
            }
        };

        const handleError = () => {
            setIsVideoPlayable(false);
        };

        video.addEventListener('loadedmetadata', handleLoadedMetadata);
        video.addEventListener('error', handleError);
        return () => {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
            video.removeEventListener('error', handleError);
        };
    }, [videoUrl, serverFps]);

    // Keep total frames in sync with server metadata (OpenCV is more accurate than browser)
    useEffect(() => {
        if (serverFrameCount > 0) {
            setTotalFrames(serverFrameCount);
        }
    }, [serverFrameCount]);

    // When video can't play in browser, fetch the first frame to show something
    useEffect(() => {
        if (!isVideoPlayable && videoId && !serverFrameData) {
            fetch(`${API_BASE}/api/extract-frame`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ videoId, frameNumber: 0 }),
            })
                .then((r) => (r.ok ? r.json() : null))
                .then((data) => { if (data?.frameBase64) setServerFrameData(data.frameBase64); })
                .catch(() => {});
        }
    }, [isVideoPlayable, videoId, serverFrameData]);

    useEffect(() => {
        const video = videoRef.current;
        if (!video || !isPlaying) return;

        const fps = serverFps || 25;
        const interval = setInterval(() => {
            setCurrentFrame(Math.floor(video.currentTime * fps));
        }, 1000 / fps);

        return () => clearInterval(interval);
    }, [isPlaying, serverFps]);

    const goToFrame = (frame: number) => {
        const video = videoRef.current;
        if (video && isVideoPlayable) {
            video.currentTime = frame / (serverFps || 25);
        }
        setCurrentFrame(frame);

        // For non-playable formats, fetch the frame image from the server (debounced)
        if (!isVideoPlayable && videoId) {
            if (frameRequestTimeoutRef.current) clearTimeout(frameRequestTimeoutRef.current);
            frameRequestTimeoutRef.current = setTimeout(() => {
                fetch(`${API_BASE}/api/extract-frame`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ videoId, frameNumber: frame }),
                })
                    .then((r) => (r.ok ? r.json() : null))
                    .then((data) => { if (data?.frameBase64) setServerFrameData(data.frameBase64); })
                    .catch(() => {});
            }, 100);
        }
    };

    const handlePlayPause = () => {
        const video = videoRef.current;
        if (!video) return;

        if (isPlaying) {
            video.pause();
        } else {
            video.play();
        }
        setIsPlaying(!isPlaying);
    };

    const handleFrameStep = (direction: 'forward' | 'backward') => {
        const newFrame =
            direction === 'forward' ? Math.min(currentFrame + 1, totalFrames - 1) : Math.max(currentFrame - 1, 0);
        goToFrame(newFrame);
    };

    const navigateCharacter = (direction: 'prev' | 'next') => {
        if (!predictionResult) return;

        const maxIndex = predictionResult.predictions.length - 1;
        setCurrentCharacterIndex((prev) =>
            direction === 'next' ? Math.min(prev + 1, maxIndex) : Math.max(prev - 1, 0)
        );
    };

    const runPrediction = async () => {
        if (!loadedModelName || !videoFile) {
            toast('Error', { description: 'Please load a model and video' });
            return;
        }
        if (selectedCharacters.length === 0) {
            toast('Error', { description: 'Please select at least one character' });
            return;
        }
        if (!videoId) {
            toast('Error', { description: 'Video is still uploading to server, please wait' });
            return;
        }

        setLoading(true);
        setCurrentCharacterIndex(0);

        try {
            // Always use server-side frame extraction — works for AVI and all other
            // OpenCV-supported formats, unlike the browser canvas approach
            const extractRes = await fetch(`${API_BASE}/api/extract-frame`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ videoId, frameNumber: currentFrame }),
            });
            if (!extractRes.ok) {
                const err = await extractRes.json().catch(() => ({ detail: extractRes.statusText }));
                throw new Error(err.detail ?? 'Failed to extract frame');
            }
            const { frameBase64: frameData } = await extractRes.json();

            const res = await fetch(`${API_BASE}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    frameBase64: frameData,
                    characterIds: selectedCharacters,
                    frameNumber: currentFrame,
                }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail ?? 'Prediction failed');
            }

            const data = (await res.json()) as {
                predictions: Array<{
                    characterId: string;
                    characterName: string;
                    confidence: number;
                    gradCamImageBase64: string;
                }>;
                frameNumber?: number;
            };

            const result = {
                originalFrame: frameData,
                frameNumber: data.frameNumber ?? currentFrame,
                predictions: data.predictions.map((p) => ({
                    characterId: p.characterId,
                    characterName: p.characterName,
                    confidence: p.confidence,
                    gradCamImage: `data:image/png;base64,${p.gradCamImageBase64}`,
                })),
            };

            setPredictionResult(result);
            toast('Success', { description: 'Prediction successfully created' });
        } catch (error) {
            console.error('Prediction error:', error);
            toast('Error', {
                description: error instanceof Error ? error.message : 'Prediction failed',
            });
        } finally {
            setLoading(false);
        }
    };

    const currentPrediction = predictionResult?.predictions[currentCharacterIndex];

    return (
        <div className='space-y-6'>
            <Card className='p-6'>
                <div className='grid gap-6 md:grid-cols-2 lg:grid-cols-4'>
                    <div className='space-y-3'>
                        <Label className='text-base font-semibold'>Model</Label>
                        <div className='space-y-2'>
                            <Label className='text-muted-foreground text-xs'>Architecture</Label>
                            <Select value={modelType} onValueChange={setModelType}>
                                <SelectTrigger className='w-full'>
                                    <SelectValue placeholder='Select architecture' />
                                </SelectTrigger>
                                <SelectContent>
                                    {MODEL_TYPES.map((m) => (
                                        <SelectItem key={m.id} value={m.id}>
                                            {m.name}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                            <Button
                                variant='default'
                                className='w-full'
                                onClick={handleLoadModel}
                                disabled={modelLoadLoading}
                            >
                                {modelLoadLoading ? (
                                    <Loader2Icon className='mr-2 h-4 w-4 shrink-0 animate-spin' />
                                ) : null}
                                {modelLoadLoading ? 'Loading...' : 'Load model'}
                            </Button>
                            {loadedModelName && (
                                <p className='text-xs text-muted-foreground truncate'>
                                    Loaded: {loadedModelName}
                                </p>
                            )}
                        </div>
                    </div>

                    <div className='space-y-2'>
                        <Label>Load video</Label>
                        <input
                            ref={videoInputRef}
                            type='file'
                            accept='video/*,.avi'
                            onChange={handleVideoFileChange}
                            className='hidden'
                        />
                        <Button
                            variant='outline'
                            className='w-full justify-start text-sm'
                            onClick={() => videoInputRef.current?.click()}
                        >
                            {videoUploading ? (
                                <Loader2Icon className='mr-2 h-4 w-4 shrink-0 animate-spin' />
                            ) : (
                                <UploadIcon className='mr-2 h-4 w-4 shrink-0' />
                            )}
                            <span className='truncate'>
                                {videoUploading ? 'Uploading...' : videoFile ? videoFile.name : 'Select video...'}
                            </span>
                        </Button>
                    </div>

                    <div className='space-y-2'>
                        <div className='flex items-center justify-between'>
                            <Label>Characters</Label>
                            <Button
                                variant='ghost'
                                size='sm'
                                onClick={toggleAllCharacters}
                                className='h-auto p-1 text-xs'
                            >
                                {selectedCharacters.length === MUPPET_CHARACTERS.length ? 'None' : 'All'}
                            </Button>
                        </div>
                        <div className='grid grid-cols-2 gap-2'>
                            {MUPPET_CHARACTERS.map((character) => (
                                <div
                                    key={character.id}
                                    className='flex items-center space-x-2'
                                >
                                    <Checkbox
                                        id={character.id}
                                        checked={selectedCharacters.includes(character.id)}
                                        onCheckedChange={() => toggleCharacter(character.id)}
                                    />
                                    <Label
                                        htmlFor={character.id}
                                        className='flex items-center gap-1 cursor-pointer text-xs leading-tight'
                                    >
                                        <span className='truncate'>{character.name}</span>
                                    </Label>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className='flex items-end'>
                        <Button
                            onClick={runPrediction}
                            disabled={loading || videoUploading || !loadedModelName || !videoFile}
                            className='w-full'
                            size='lg'
                        >
                            {loading ? 'Running...' : 'Create prediction'}
                        </Button>
                    </div>
                </div>
            </Card>

            <Card className='p-6'>
                <div className='grid gap-6 lg:grid-cols-2'>
                    <div className='space-y-4'>
                        <h3 className='text-lg font-semibold'>Original Frame</h3>
                        <div className='aspect-video w-full overflow-hidden rounded-lg bg-black relative'>
                            {videoUrl ? (
                                <>
                                    {/* Always keep the video element in the DOM for metadata access */}
                                    <video
                                        ref={videoRef}
                                        src={videoUrl}
                                        className={`h-full w-full object-contain ${isVideoPlayable ? '' : 'hidden'}`}
                                        onEnded={() => setIsPlaying(false)}
                                    />
                                    {!isVideoPlayable && (
                                        serverFrameData ? (
                                            <img
                                                src={serverFrameData}
                                                alt={`Frame ${currentFrame}`}
                                                className='h-full w-full object-contain'
                                            />
                                        ) : (
                                            <div className='flex h-full items-center justify-center text-muted-foreground text-sm text-center p-4'>
                                                {videoUploading
                                                    ? 'Uploading video to server...'
                                                    : 'Video format not supported in browser — use frame controls to navigate'}
                                            </div>
                                        )
                                    )}
                                </>
                            ) : (
                                <div className='flex h-full items-center justify-center text-muted-foreground'>
                                    Load a video
                                </div>
                            )}
                        </div>

                        <div className='space-y-4'>
                            <div className='flex items-center justify-center gap-2'>
                                <Button
                                    variant='outline'
                                    size='icon'
                                    onClick={() => handleFrameStep('backward')}
                                    disabled={!videoUrl || currentFrame === 0}
                                >
                                    <SkipBackIcon className='h-4 w-4' />
                                </Button>

                                <Button
                                    variant='outline'
                                    size='icon'
                                    onClick={handlePlayPause}
                                    disabled={!videoUrl || !isVideoPlayable}
                                >
                                    {isPlaying ? <PauseIcon className='h-4 w-4' /> : <PlayIcon className='h-4 w-4' />}
                                </Button>

                                <Button
                                    variant='outline'
                                    size='icon'
                                    onClick={() => handleFrameStep('forward')}
                                    disabled={!videoUrl || currentFrame >= totalFrames - 1}
                                >
                                    <SkipForwardIcon className='h-4 w-4' />
                                </Button>
                            </div>

                            <div className='space-y-2'>
                                <div className='flex items-center justify-between text-sm'>
                                    <span>Frame: {currentFrame}</span>
                                    <span>Total: {totalFrames}</span>
                                </div>

                                <Slider
                                    value={[currentFrame]}
                                    onValueChange={([value]) => goToFrame(value)}
                                    max={Math.max(0, totalFrames - 1)}
                                    step={1}
                                    disabled={!videoUrl || totalFrames === 0}
                                    className='w-full'
                                />
                            </div>
                        </div>
                    </div>

                    <div className='space-y-4'>
                        <div className='flex items-center justify-between'>
                            <h3 className='text-lg font-semibold'>
                                {predictionResult ? 'Grad-CAM Heatmap' : 'Prediction result'}
                            </h3>
                            {predictionResult && (
                                <span className='text-sm text-muted-foreground'>
                                    Frame {predictionResult.frameNumber}
                                </span>
                            )}
                        </div>

                        {predictionResult ? (
                            <div className='space-y-4'>
                                <div className='aspect-video w-full overflow-hidden rounded-lg bg-black'>
                                    <img
                                        src={currentPrediction?.gradCamImage || '/placeholder.svg'}
                                        alt={`Grad-CAM for ${currentPrediction?.characterName}`}
                                        className='h-full w-full object-contain'
                                    />
                                </div>

                                <div className='flex items-center justify-between gap-4'>
                                    <Button
                                        variant='outline'
                                        size='icon'
                                        onClick={() => navigateCharacter('prev')}
                                        disabled={currentCharacterIndex === 0}
                                    >
                                        <ChevronLeftIcon className='h-4 w-4' />
                                    </Button>

                                    <div className='flex-1 text-center'>
                                        <h4 className='font-semibold'>{currentPrediction?.characterName}</h4>
                                        <p className='text-2xl font-bold text-primary'>
                                            {(currentPrediction?.confidence * 100).toFixed(1)}%
                                        </p>
                                    </div>

                                    <Button
                                        variant='outline'
                                        size='icon'
                                        onClick={() => navigateCharacter('next')}
                                        disabled={currentCharacterIndex === predictionResult.predictions.length - 1}
                                    >
                                        <ChevronRightIcon className='h-4 w-4' />
                                    </Button>
                                </div>

                                <div className='flex items-center justify-center gap-2'>
                                    {predictionResult.predictions.map((_: unknown, index: number) => (
                                        <button
                                            key={index}
                                            onClick={() => setCurrentCharacterIndex(index)}
                                            className={`h-2 w-2 rounded-full transition-all ${
                                                index === currentCharacterIndex
                                                    ? 'bg-primary w-4'
                                                    : 'bg-muted-foreground/30'
                                            }`}
                                        />
                                    ))}
                                </div>
                            </div>
                        ) : (
                            <div className='flex aspect-video items-center justify-center rounded-lg border-2 border-dashed text-muted-foreground'>
                                Create a prediction to see results
                            </div>
                        )}
                    </div>
                </div>
            </Card>
        </div>
    );
}
