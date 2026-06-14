import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';
import { useSudarshanStore } from '../../store/sudarshanStore';
import { earthVertexShader, earthFragmentShader } from './shaders/earth.glsl.js';

export default function SudarshanGlobe() {
  const earthRef = useRef();
  const tracks = useSudarshanStore(state => state.tracks);
  const satellites = useSudarshanStore(state => state.satellites);
  const globalThreatLevel = useSudarshanStore(state => state.globalThreatLevel);
  
  // Custom shader material for the Earth
  const earthMaterial = useMemo(() => new THREE.ShaderMaterial({
    vertexShader: earthVertexShader,
    fragmentShader: earthFragmentShader,
    uniforms: {
      time: { value: 0 },
      threatLevel: { value: 0 }
    },
    transparent: true,
    side: THREE.DoubleSide
  }), []);

  useFrame((state) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.001;
      earthMaterial.uniforms.time.value = state.clock.elapsedTime;
      // Smoothly interpolate threat level into shader
      earthMaterial.uniforms.threatLevel.value += (globalThreatLevel - earthMaterial.uniforms.threatLevel.value) * 0.1;
    }
  });

  return (
    <group>
      {/* Central Cyberpunk Earth */}
      <Sphere ref={earthRef} args={[2, 64, 64]}>
        <primitive object={earthMaterial} attach="material" />
      </Sphere>

      {/* Render Satellite Orbits */}
      {satellites.map((sat, i) => {
        const radius = 2.5 + (sat.elevation / 90) * 0.5;
        const speed = 0.5;
        const color = sat.is_threat ? 'red' : '#00aaff';
        return (
          <group key={`sat-${i}`} rotation={[Math.PI/4 * i, 0, 0]}>
            <Line
              points={[...Array(64)].map((_, j) => {
                const angle = (j / 63) * Math.PI * 2;
                return new THREE.Vector3(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
              })}
              color={color}
              lineWidth={1}
              opacity={0.3}
              transparent
            />
            {/* Satellite marker */}
            <mesh position={[Math.cos(Date.now() * 0.001 * speed) * radius, 0, Math.sin(Date.now() * 0.001 * speed) * radius]}>
              <sphereGeometry args={[0.05, 16, 16]} />
              <meshBasicMaterial color={color} />
            </mesh>
          </group>
        );
      })}

      {/* Render Surface Tracks (UAVs/Vessels) */}
      {tracks.map((track) => {
        // Map 2D coordinate to 3D sphere surface
        const lat = (track.position[1] - 500) / 1000 * Math.PI; // Fake mapping for demo
        const lon = (track.position[0] - 500) / 1000 * Math.PI * 2;
        const radius = 2.05;
        
        const x = radius * Math.cos(lat) * Math.cos(lon);
        const y = radius * Math.sin(lat);
        const z = radius * Math.cos(lat) * Math.sin(lon);
        
        const isThreat = track.assessment.threat_probability > 0.5;
        
        return (
          <mesh key={`track-${track.track_id}`} position={[x, y, z]}>
            <sphereGeometry args={[0.03, 16, 16]} />
            <meshBasicMaterial color={isThreat ? 'red' : '#00ff00'} />
            {track.is_occluded && (
              <mesh scale={[1.5, 1.5, 1.5]}>
                <sphereGeometry args={[0.04, 16, 16]} />
                <meshBasicMaterial color="yellow" wireframe opacity={0.5} transparent />
              </mesh>
            )}
          </mesh>
        );
      })}
    </group>
  );
}
