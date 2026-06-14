import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, EffectComposer, Bloom } from '@react-three/drei';
import SudarshanGlobe from './components/globe/SudarshanGlobe';
import { TacticalPanel } from './components/panels/TacticalPanel';
import { OrbitalPanel } from './components/panels/OrbitalPanel';
import { useWebsocket } from './hooks/useWebsocket';
import { useSudarshanStore } from './store/sudarshanStore';

function App() {
  // Initialize WebSocket connection
  useWebsocket();
  const connected = useSudarshanStore(state => state.connected);

  const injectScenario = async (name) => {
    try {
      await fetch(`http://localhost:8000/api/scenario/${name}`, { method: 'POST' });
    } catch (e) {
      console.error("Failed to inject scenario", e);
    }
  };

  return (
    <div className="w-screen h-screen bg-black overflow-hidden relative font-sans">
      {/* 3D Viewport */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 2, 6], fov: 45 }}>
          <color attach="background" args={['#000510']} />
          <ambientLight intensity={0.2} />
          <pointLight position={[10, 10, 10]} intensity={1} color="#00aaff" />
          <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
          
          <SudarshanGlobe />
          
          <OrbitControls 
            enablePan={false}
            minDistance={3}
            maxDistance={10}
            autoRotate
            autoRotateSpeed={0.5}
          />
        </Canvas>
      </div>

      {/* Overlay UI */}
      <div className="absolute top-0 left-0 w-full p-4 pointer-events-none z-10 flex justify-center">
        <div className="flex flex-col items-center">
          <h1 className="text-4xl font-black tracking-[0.5em] text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-200 uppercase drop-shadow-[0_0_10px_rgba(0,170,255,0.8)]">
            Sudarshan
          </h1>
          <p className="text-blue-400/80 tracking-[0.2em] text-xs mt-1">QUAD-DOMAIN C4ISR NEXUS</p>
          <div className={`mt-2 px-3 py-1 rounded text-xs tracking-widest font-bold ${connected ? 'bg-green-900/50 text-green-400 border border-green-500/50' : 'bg-red-900/50 text-red-400 border border-red-500/50 animate-pulse'}`}>
            {connected ? 'UPLINK ESTABLISHED' : 'NO SIGNAL'}
          </div>
        </div>
      </div>

      {/* Side Panels */}
      <div className="pointer-events-auto">
        <TacticalPanel />
        <OrbitalPanel />
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex gap-4 pointer-events-auto z-10">
        <button 
          onClick={() => injectScenario('UAV_SWARM')}
          className="px-6 py-2 bg-blue-900/40 hover:bg-blue-800/60 border border-blue-500/50 text-blue-200 font-mono text-sm uppercase tracking-wider rounded transition-all hover:scale-105"
        >
          Inject: UAV Swarm
        </button>
        <button 
          onClick={() => injectScenario('RED_ALERT')}
          className="px-6 py-2 bg-red-900/40 hover:bg-red-800/60 border border-red-500/50 text-red-200 font-mono text-sm uppercase tracking-wider rounded transition-all hover:scale-105"
        >
          Inject: Red Alert
        </button>
      </div>
      
      {/* Scanline Overlay */}
      <div className="absolute inset-0 pointer-events-none z-50 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_4px,3px_100%] opacity-20 mix-blend-overlay"></div>
    </div>
  );
}

export default App;
