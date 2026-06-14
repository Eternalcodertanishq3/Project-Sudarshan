import React from 'react';
import { useSudarshanStore } from '../../store/sudarshanStore';

export function OrbitalPanel() {
  const satellites = useSudarshanStore(state => state.satellites);
  
  return (
    <div className="absolute top-4 left-4 w-72 bg-black/60 border border-blue-900/50 backdrop-blur-md p-4 rounded-lg font-mono text-sm text-blue-100 z-10">
      <h2 className="text-xl mb-4 font-bold tracking-widest border-b border-blue-800 pb-2">ORBITAL (SGP4)</h2>
      
      <div className="space-y-3">
        {satellites.map((sat, i) => (
          <div key={i} className="flex flex-col border-l-2 border-blue-500 pl-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-blue-300">{sat.name}</span>
              {sat.is_threat && <span className="text-red-500 animate-pulse text-[10px]">ISR THREAT</span>}
            </div>
            <div className="flex justify-between text-[10px] text-blue-400/70 mt-1">
              <span>EL: {sat.elevation.toFixed(1)}°</span>
              <span>AZ: {sat.azimuth.toFixed(1)}°</span>
            </div>
            <div className="w-full bg-blue-950 h-1 mt-1 rounded overflow-hidden">
              <div 
                className={`h-full ${sat.is_threat ? 'bg-red-500' : 'bg-blue-500'}`}
                style={{ width: \`\${sat.risk * 100}%\` }}
              />
            </div>
          </div>
        ))}
        {satellites.length === 0 && <div className="text-blue-500/50 text-center py-4">NO OVERHEAD ISR</div>}
      </div>
    </div>
  );
}
