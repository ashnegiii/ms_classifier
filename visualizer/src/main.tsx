import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Toaster } from 'sonner';
import './index.css';
import { MuppetShowPredictor } from './App';

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <main className='min-h-screen bg-background p-6'>
            <div className='mx-auto max-w-7xl space-y-6'>
                <div className='space-y-2'>
                    <h1 className='text-3xl font-bold'>Muppet Show Prediction Viewer</h1>
                    <p>
                        Choose a model and video, navigate frames and create Predictions with optional Grad-CAM
                        activations
                    </p>
                </div>
                <MuppetShowPredictor />
            </div>
        </main>
        <Toaster />
    </StrictMode>
);
