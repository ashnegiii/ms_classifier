'use client';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  PauseIcon,
  PlayIcon,
  SkipBackIcon,
  SkipForwardIcon,
  UploadIcon,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';

const MUPPET_CHARACTERS = [
    { id: 'kermit', name: 'Kermit', color: 'bg-green-500' },
    { id: 'fozzie_bear', name: 'Fozzie Bear', color: 'bg-orange-500' },
    { id: 'miss_piggy', name: 'Miss Piggy', color: 'bg-pink-500' },
    { id: 'statler_waldorf', name: 'Statler & Waldorf', color: 'bg-purple-500' },
    { id: 'the_cook', name: 'The Swedish Chef', color: 'bg-red-500' },
    { id: 'rowlf_the_dog', name: 'Rowlf The Dog', color: 'bg-amber-700' },
] as const;

export function MuppetShowPredictor() {
    const [modelFile, setModelFile] = useState<File | null>(null);
    const [videoFile, setVideoFile] = useState<File | null>(null);
    const [videoUrl, setVideoUrl] = useState<string>('');

    const [currentFrame, setCurrentFrame] = useState(0);
    const [totalFrames, setTotalFrames] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const [selectedCharacters, setSelectedCharacters] = useState<string[]>(MUPPET_CHARACTERS.map((c) => c.id));

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [predictionResult, setPredictionResult] = useState<any>(null);
    const [currentCharacterIndex, setCurrentCharacterIndex] = useState(0);
    const [loading, setLoading] = useState(false);

    const videoRef = useRef<HTMLVideoElement>(null);
    const modelInputRef = useRef<HTMLInputElement>(null);
    const videoInputRef = useRef<HTMLInputElement>(null);

    const handleVideoFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setVideoFile(file);
            const url = URL.createObjectURL(file);
            setVideoUrl(url);
        }
    };

    const handleModelFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file && file.name.endsWith('.pth')) {
            setModelFile(file);
            toast('Model geladen', {
                description: file.name,
            });
        } else {
            toast('Fehler', {
                description: 'Bitte wähle eine .pth Datei',
            });
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

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const handleLoadedMetadata = () => {
            const fps = 25;
            setTotalFrames(Math.floor(video.duration * fps));
        };

        video.addEventListener('loadedmetadata', handleLoadedMetadata);
        return () => video.removeEventListener('loadedmetadata', handleLoadedMetadata);
    }, [videoUrl]);

    useEffect(() => {
        const video = videoRef.current;
        if (!video || !isPlaying) return;

        const interval = setInterval(() => {
            const fps = 25;
            setCurrentFrame(Math.floor(video.currentTime * fps));
        }, 1000 / 25);

        return () => clearInterval(interval);
    }, [isPlaying]);

    const goToFrame = (frame: number) => {
        const video = videoRef.current;
        if (!video) return;

        const fps = 25;
        video.currentTime = frame / fps;
        setCurrentFrame(frame);
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
        if (!modelFile || !videoFile) {
            toast('Fehler', {
                description: 'Bitte lade ein Model und Video hoch',
            });
            return;
        }

        if (selectedCharacters.length === 0) {
            toast('Fehler', {
                description: 'Bitte wähle mindestens einen Charakter',
            });
            return;
        }

        setLoading(true);
        setCurrentCharacterIndex(0);

        try {
            const video = videoRef.current;
            if (!video) return;

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx?.drawImage(video, 0, 0);
            const frameData = canvas.toDataURL('image/jpeg');

            const CHARACTER_NAMES: Record<string, string> = {
                kermit: 'Kermit',
                fozzie_bear: 'Fozzie Bear',
                miss_piggy: 'Miss Piggy',
                statler_waldorf: 'Statler & Waldorf',
                the_cook: 'The Swedish Chef',
                rowlf_the_dog: 'Rowlf The Dog',
            };

            const predictions = selectedCharacters.map((characterId) => ({
                characterId,
                characterName: CHARACTER_NAMES[characterId] || characterId,
                confidence: Math.random() * 0.5 + 0.5,
                gradCamImage: `/placeholder.svg?height=600&width=800&query=grad-cam-heatmap-${characterId}`,
            }));

            const result = {
                originalFrame: frameData,
                frameNumber: currentFrame,
                predictions,
            };

            setPredictionResult(result);

            toast('Erfolg', {
                description: 'Prediction erfolgreich erstellt',
            });
        } catch (error) {
            console.error('[v0] Prediction error:', error);
            toast('Fehler', {
                description: 'Prediction fehlgeschlagen',
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
                    <div className='space-y-2'>
                        <Label>Model laden (.pth)</Label>
                        <input
                            ref={modelInputRef}
                            type='file'
                            accept='.pth'
                            onChange={handleModelFileChange}
                            className='hidden'
                        />
                        <Button
                            variant='outline'
                            className='w-full justify-start text-sm'
                            onClick={() => modelInputRef.current?.click()}
                        >
                            <UploadIcon className='mr-2 h-4 w-4 shrink-0' />
                            <span className='truncate'>{modelFile ? modelFile.name : 'Model auswählen...'}</span>
                        </Button>
                    </div>

                    <div className='space-y-2'>
                        <Label>Video laden</Label>
                        <input
                            ref={videoInputRef}
                            type='file'
                            accept='video/*'
                            onChange={handleVideoFileChange}
                            className='hidden'
                        />
                        <Button
                            variant='outline'
                            className='w-full justify-start text-sm'
                            onClick={() => videoInputRef.current?.click()}
                        >
                            <UploadIcon className='mr-2 h-4 w-4 shrink-0' />
                            <span className='truncate'>{videoFile ? videoFile.name : 'Video auswählen...'}</span>
                        </Button>
                    </div>

                    <div className='space-y-2'>
                        <div className='flex items-center justify-between'>
                            <Label>Charaktere</Label>
                            <Button
                                variant='ghost'
                                size='sm'
                                onClick={toggleAllCharacters}
                                className='h-auto p-1 text-xs'
                            >
                                {selectedCharacters.length === MUPPET_CHARACTERS.length ? 'Keine' : 'Alle'}
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
                                        <div className={`h-2 w-2 shrink-0 rounded-full ${character.color}`} />
                                        <span className='truncate'>{character.name}</span>
                                    </Label>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className='flex items-end'>
                        <Button
                            onClick={runPrediction}
                            disabled={loading || !modelFile || !videoFile}
                            className='w-full'
                            size='lg'
                        >
                            {loading ? 'Läuft...' : 'Prediction erstellen'}
                        </Button>
                    </div>
                </div>
            </Card>

            <Card className='p-6'>
                <div className='grid gap-6 lg:grid-cols-2'>
                    <div className='space-y-4'>
                        <h3 className='text-lg font-semibold'>Original Frame</h3>
                        <div className='aspect-video w-full overflow-hidden rounded-lg bg-black'>
                            {videoUrl ? (
                                <video
                                    ref={videoRef}
                                    src={videoUrl}
                                    className='h-full w-full object-contain'
                                    onEnded={() => setIsPlaying(false)}
                                />
                            ) : (
                                <div className='flex h-full items-center justify-center text-muted-foreground'>
                                    Lade ein Video hoch
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
                                    disabled={!videoUrl}
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
                                    max={totalFrames - 1}
                                    step={1}
                                    disabled={!videoUrl}
                                    className='w-full'
                                />
                            </div>
                        </div>
                    </div>

                    <div className='space-y-4'>
                        <div className='flex items-center justify-between'>
                            <h3 className='text-lg font-semibold'>
                                {predictionResult ? 'Grad-CAM Heatmap' : 'Prediction Ergebnis'}
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
                                Erstelle eine Prediction um Ergebnisse zu sehen
                            </div>
                        )}
                    </div>
                </div>
            </Card>
        </div>
    );
}
