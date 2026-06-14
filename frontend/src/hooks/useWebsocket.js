import React, { useEffect } from 'react';
import { useSudarshanStore } from '../store/sudarshanStore';

export function useWebsocket() {
  const { setConnected, updateData } = useSudarshanStore();

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/tactical-feed');
    
    ws.onopen = () => {
      console.log('Sudarshan uplink established');
      setConnected(true);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateData(data);
      } catch (e) {
        console.error("Payload parse error", e);
      }
    };
    
    ws.onclose = () => {
      console.log('Sudarshan uplink lost');
      setConnected(false);
      // Auto-reconnect logic could go here
    };

    return () => ws.close();
  }, [setConnected, updateData]);
}
