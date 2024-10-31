struct Uniforms {
  viewMatrix : mat4x4f,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) uv : vec2f,
  @location(2) shape: f32,
}

@vertex
fn vertex_main(@location(0) position: vec2f,
               @location(1) color: vec4f,
                @location(2) uv : vec2f,
                @location(3) shape: f32,

) -> VertexOut
{
  var output : VertexOut;
  output.position = uniforms.viewMatrix * vec4(position, 0., 1.);
  output.color = color;
  output.uv = uv;
  output.shape = shape;
  return output;
}

@group(1) @binding(0) var mySampler: sampler;
@group(1) @binding(1) var environmentTexture: texture_2d<f32>;
@group(1) @binding(2) var dirtTexture: texture_2d<f32>;

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
  let environmentSample = textureSample(environmentTexture, mySampler, fragData.uv);
  let sample = textureSample(dirtTexture, mySampler, fragData.uv);
  // Handle debug circles
  if (fragData.shape == 1) {
    let outOfCircle = fragData.shape == 1. && length(fragData.uv - vec2(.5)) >= .5;
    if (outOfCircle) {
      discard;
    }
    return fragData.color;
  }
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