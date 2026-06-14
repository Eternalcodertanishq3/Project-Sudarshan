import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// GLSL Vertex Shader — Holographic scanline effect
const HOLOGRAPHIC_VERT_SHADER = `
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  
  void main() {
    vUv = uv;
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// GLSL Fragment Shader — Sudarshan tactical display effect
const HOLOGRAPHIC_FRAG_SHADER = `
  uniform float uTime;
  uniform float uThreatLevel;  // 0.0=green, 1.0=red
  uniform sampler2D uEarthTexture;
  
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  
  // Scanline effect — cinematic CRT/holographic look
  float scanline(vec2 uv, float frequency) {
    return sin(uv.y * frequency + uTime * 2.0) * 0.5 + 0.5;
  }
  
  // Fresnel rim — glowing edges
  float fresnel(vec3 normal, float power) {
    vec3 viewDir = normalize(cameraPosition - vPosition);
    return pow(1.0 - abs(dot(viewDir, normal)), power);
  }
  
  void main() {
    // Base Earth texture
    vec4 earthColor = texture2D(uEarthTexture, vUv);
    
    // Tactical overlay color — green (safe) to red (threat)
    vec3 safeColor = vec3(0.0, 0.8, 0.3);    // #00CC4D — tactical green
    vec3 threatColor = vec3(1.0, 0.1, 0.1);  // #FF1A1A — red alert
    vec3 overlayColor = mix(safeColor, threatColor, uThreatLevel);
    
    // Scanline sweep (holographic effect)
    float scan = scanline(vUv, 200.0) * 0.08;
    
    // Fresnel rim glow
    float rim = fresnel(vNormal, 3.0) * 0.4;
    
    // Grid lines — tactical overlay
    float gridX = step(0.98, fract(vUv.x * 30.0));
    float gridY = step(0.98, fract(vUv.y * 15.0));
    float grid = (gridX + gridY) * 0.03 * (0.5 + 0.5 * sin(uTime * 3.0));
    
    // Combine
    vec3 finalColor = earthColor.rgb * 0.7          // Darkened Earth
                    + overlayColor * rim              // Rim glow
                    + overlayColor * scan             // Scanlines
                    + overlayColor * grid;            // Grid overlay
    
    // Pulse on RED ALERT
    if (uThreatLevel > 0.9) {
      float pulse = 0.5 + 0.5 * sin(uTime * 10.0);
      finalColor += vec3(0.3, 0.0, 0.0) * pulse;
    }
    
    gl_FragColor = vec4(finalColor, 0.92);
  }
`;

// Orbital track shader — neon glowing satellite paths
const ORBITAL_TRACK_FRAG = `
  uniform vec3 uColor;
  uniform float uThreatRisk;
  uniform float uTime;
  
  void main() {
    // Threat-colored track
    vec3 safeTrack = vec3(0.0, 0.6, 1.0);    // Blue = civilian
    vec3 threatTrack = vec3(1.0, 0.3, 0.0);  // Orange = ISR threat
    vec3 trackColor = mix(safeTrack, threatTrack, uThreatRisk);
    
    // Pulsing glow for high-threat satellites
    float glow = 0.7 + 0.3 * sin(uTime * 4.0 * uThreatRisk);
    
    gl_FragColor = vec4(trackColor * glow, 0.85);
  }
`;

export default function SudarshanGlobe({ threatLevel, satellites, tracks }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const uniformsRef = useRef(null);

  useEffect(() => {
    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000810);  // Deep tactical black

    const camera = new THREE.PerspectiveCamera(
      45,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      10000
    );
    camera.position.set(0, 0, 3.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1.5;
    controls.maxDistance = 8.0;

    // Load Earth texture
    const textureLoader = new THREE.TextureLoader();
    // We will need to provide these textures in public/textures/
    // using a placeholder for now or catching the error if missing
    const earthTexture = textureLoader.load('/textures/earth_8k.jpg');
    const normalMap = textureLoader.load('/textures/earth_normal.jpg');

    // Earth sphere with holographic GLSL shader
    const earthGeo = new THREE.SphereGeometry(1.0, 64, 64);
    const uniforms = {
      uTime: { value: 0 },
      uThreatLevel: { value: 0 },
      uEarthTexture: { value: earthTexture },
    };
    uniformsRef.current = uniforms;

    const earthMat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: HOLOGRAPHIC_VERT_SHADER,
      fragmentShader: HOLOGRAPHIC_FRAG_SHADER,
      transparent: true,
    });

    const earth = new THREE.Mesh(earthGeo, earthMat);
    scene.add(earth);

    // Atmosphere glow
    const atmGeo = new THREE.SphereGeometry(1.02, 64, 64);
    const atmMat = new THREE.MeshPhongMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0.04,
      side: THREE.BackSide,
    });
    scene.add(new THREE.Mesh(atmGeo, atmMat));

    // Starfield background
    const starGeo = new THREE.BufferGeometry();
    const starCount = 5000;
    const positions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i++) {
      positions[i] = (Math.random() - 0.5) * 2000;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.3 });
    scene.add(new THREE.Points(starGeo, starMat));

    // Ambient + directional lighting
    scene.add(new THREE.AmbientLight(0x111122, 1.0));
    const sunLight = new THREE.DirectionalLight(0xffffff, 2.0);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    sceneRef.current = scene;

    // Animation loop at 60fps
    let animId;
    const clock = new THREE.Clock();
    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      // Update time uniform for shaders
      if (uniformsRef.current) {
        uniformsRef.current.uTime.value = elapsed;
      }

      earth.rotation.y += 0.0005;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      renderer.dispose();
      if (mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  // React to threat level changes
  useEffect(() => {
    if (uniformsRef.current) {
      uniformsRef.current.uThreatLevel.value = threatLevel;
    }
  }, [threatLevel]);

  return (
    <div
      ref={mountRef}
      style={{ width: '100%', height: '100%', background: '#000810' }}
    />
  );
}
