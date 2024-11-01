struct Ant {
    position: vec2<f32>,
    direction: vec2<f32>,
    @size(16) hue: f32,
}

fn rgb2hsv(rgb: vec3f) -> vec3f {
    let k = vec4(0., -1 / 3., 2. / 3., -1);
    let p = mix(vec4(rgb.zy, k.xy), vec4(rgb.yz, k.xy), step(rgb.z, rgb.y));
    let q = mix(vec4(p.xyw, rgb.x), vec4(rgb.x, p.yzx), step(p.x, rgb.x));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6. * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(hsv: vec3f) -> vec3f {
    let k = vec4(1., 2. / 3., 1. / 3., 3.);
    let p = abs(fract(hsv.xxx + k.xyz) * 6. - k.www);
    return hsv.z * mix(k.xxx, clamp(p - k.xxx, vec3(0.0), vec3(1.0)), hsv.y);
}