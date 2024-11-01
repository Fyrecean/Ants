struct Uniforms {
  viewMatrix : mat4x4f,
  ants: array<Ant, 2>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) uv : vec2f,
}

@vertex
fn vertex_environment(@location(0) position: vec2f,
               @location(1) color: vec4f,
                @location(2) uv : vec2f,

) -> VertexOut
{
  var output : VertexOut;
  output.position = uniforms.viewMatrix * vec4(position, 0., 1.);
  output.color = color;
  output.uv = uv;
  return output;
}

@group(1) @binding(0) var mySampler: sampler;
@group(1) @binding(1) var environmentTexture: texture_2d<f32>;
@group(1) @binding(2) var dirtTexture: texture_2d<f32>;

@fragment
fn fragment_environment(fragData: VertexOut) -> @location(0) vec4f
{
  let environmentSample = textureSample(environmentTexture, mySampler, fragData.uv);
  let sample = textureSample(dirtTexture, mySampler, fragData.uv);
  // Otherwise, rectangle texture
  if (environmentSample.x > 0.) {
    return sample;
  } else if (environmentSample.y > 0) {
    return vec4(0.,environmentSample.y,0.,1.);
  } else {
    discard;
  }
  return vec4(1.);
}

@vertex
fn vertex_ant(
  @builtin(instance_index) instance_index: u32,
  @location(0) position: vec2f,
  @location(1) uv: vec2f,
) -> VertexOut {
  var output: VertexOut;
  output.position = uniforms.viewMatrix * vec4(uniforms.ants[instance_index].position + (position * .005), 0., 1.);
  output.color = vec4(hsv2rgb(vec3(uniforms.ants[instance_index].hue, 1., 1.)), 1.);
  output.uv = uv;
  return output;
}

@fragment
fn fragment_ant(fragData: VertexOut) -> @location(0) vec4f {
  return fragData.color;
}