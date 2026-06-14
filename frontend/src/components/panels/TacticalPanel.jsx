import React from 'react';
import { useSudarshanStore } from '../../store/sudarshanStore';

export function TacticalPanel() {
  const globalThreatLevel = useSudarshanStore(state => state.globalThreatLevel);
  const tracks = useSudarshanStore(state => state.tracks);
  
  const threatColor = globalThreatLevel > 0.8 ? 'text-red-500' : globalThreatLevel > 0.4 ? 'text-yellow-400' : 'text-blue-400';

  return (
    <div className="absolute top-4 right-4 w-80 bg-black/60 border border-blue-900/50 backdrop-blur-md p-4 rounded-lg font-mono text-sm text-blue-100 z-10">
      <h2 className="text-xl mb-4 font-bold tracking-widest border-b border-blue-800 pb-2">TACTICAL FUSION</h2>
      
      <div className="mb-6">
        <div className="flex justify-between mb-1">
          <span>GLOBAL THREAT PROB:</span>
          <span className={`font-bold ${threatColor}`}>{(globalThreatLevel * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-blue-950 h-2 rounded overflow-hidden">
          <div 
            className={`h-full ${globalThreatLevel > 0.8 ? 'bg-red-500' : 'bg-blue-500'} transition-all duration-300`}
            style={{ width: \`\${globalThreatLevel * 100}%\` }}
          />
        </div>
      </div>

      <div className="space-y-4">
        {tracks.map(track => (
          <div key={track.track_id} className="bg-blue-950/30 p-2 border border-blue-800/30 rounded">
            <div className="flex justify-between items-center text-xs text-blue-300 mb-1">
              <span>TRK-{track.track_id} {track.is_occluded && <span className="text-yellow-500 ml-2 animate-pulse">[OCCLUDED]</span>}</span>
              <span className={track.assessment.threat_probability > 0.5 ? 'text-red-400' : ''}>
                {(track.assessment.threat_probability * 100).toFixed(0)}% THREAT
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>SPD: {track.speed.toFixed(1)} m/s</div>
              <div>HDG: {track.heading.toFixed(0)}°</div>
            </div>
          </div>
        ))}
        {tracks.length === 0 && <div className="text-blue-500/50 text-center py-4">NO ACTIVE TRACKS</div>}
      </div>
    </div>
  );
}
