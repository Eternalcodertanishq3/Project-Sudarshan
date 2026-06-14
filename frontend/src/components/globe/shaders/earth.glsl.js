export const earthVertexShader = `
varying vec2 vUv;
varying vec3 vNormal;

void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const earthFragmentShader = `
varying vec2 vUv;
varying vec3 vNormal;
uniform float time;
uniform float threatLevel;

void main() {
    // Cyberpunk grid pattern
    vec2 grid = fract(vUv * 40.0);
    float line = smoothstep(0.0, 0.05, grid.x) * smoothstep(0.0, 0.05, grid.y);
    
    // Base holographic blue
    vec3 color = vec3(0.0, 0.2, 0.5) * (1.0 - line) * 0.5;
    
    // Threat pulse (red interpolation)
    float pulse = (sin(time * 3.0) * 0.5 + 0.5) * threatLevel;
    color = mix(color, vec3(0.8, 0.0, 0.1), pulse * (1.0 - line));
    
    // Edge glow (Fresnel)
    float intensity = pow(0.6 - dot(vNormal, vec3(0, 0, 1.0)), 2.0);
    vec3 glow = mix(vec3(0.0, 0.5, 1.0), vec3(1.0, 0.0, 0.0), threatLevel) * intensity;
    
    gl_FragColor = vec4(color + glow, 0.8 + pulse * 0.2);
}
`;
