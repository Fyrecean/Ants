const leftSampleMatrix =  mat2x2(cos(sampleAngle), sin(sampleAngle), -sin(sampleAngle), cos(sampleAngle));
const rightSampleMatrix = mat2x2(cos(sampleAngle), -sin(sampleAngle), sin(sampleAngle), cos(sampleAngle));

fn getSamplePixels(position: vec2<f32>, direction: vec2<f32>, steps: u32) -> array<vec2<u32>,6> {
    let sampleStart = position + direction * 2;
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    
}