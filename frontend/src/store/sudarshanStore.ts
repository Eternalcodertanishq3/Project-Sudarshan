import { create } from 'zustand';

interface Track {
  track_id: number;
  position: [number, number];
  speed: number;
  heading: number;
  is_occluded: boolean;
  assessment: {
    threat_probability: number;
    threat_level: string;
    alert_state: string;
  };
}

interface Satellite {
  name: string;
  elevation: number;
  azimuth: number;
  risk: number;
  is_threat: boolean;
}

interface SudarshanState {
  connected: boolean;
  globalThreatLevel: number;
  tracks: Track[];
  satellites: Satellite[];
  setConnected: (status: boolean) => void;
  updateData: (data: any) => void;
}

export const useSudarshanStore = create<SudarshanState>((set) => ({
  connected: false,
  globalThreatLevel: 0,
  tracks: [],
  satellites: [],
  setConnected: (status) => set({ connected: status }),
  updateData: (data) => set({
    globalThreatLevel: data.global_threat_level || 0,
    tracks: data.tracks || [],
    satellites: data.satellites || [],
  })
}));
