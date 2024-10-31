import { mat4, Mat4, vec2, vec4 } from "wgpu-matrix";
import { PhysObject } from "./physics";
import shaderCode from "./render.wgsl";

const VERTEX_STRIDE = 9;

type TextureHandle = number;

export class Renderer {
    device: GPUDevice;
    canvasContext: GPUCanvasContext;
    vertexBuffer?: GPUBuffer;
    indexBuffer?: GPUBuffer;
    indexCount: number;
    vertexBufferDescriptor: GPUVertexBufferLayout[];
    pipelineDescriptor: GPURenderPipelineDescriptor;
    uniformBuffer: GPUBuffer;
    uniformBindGroup: GPUBindGroup;
    texturesBindGroup: GPUBindGroup;
    viewMatrix: Mat4;
    textures: GPUTexture[] = [];
    constructor(device: GPUDevice, canvasContext: GPUCanvasContext, textures: ImageBitmap[]) {
        this.device = device;
        this.canvasContext = canvasContext;

        this.addTexture(textures[0]); // environment
        this.addTexture(textures[1]); // dirt

        this.viewMatrix = mat4.identity();
        this.uniformBuffer = device.createBuffer({
          size: 4 * 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        })

        const uniformBindGroupLayout = device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.VERTEX,
              buffer: {
                type: "uniform",
              }
            },
          ]
        });
        this.uniformBindGroup = device.createBindGroup({
          layout: uniformBindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: {
                buffer: this.uniformBuffer,
              }
            },
          ]
        });
        
        const sampler = device.createSampler({
          magFilter: "nearest",
          minFilter: "nearest",
        });
        const texturesBindGroupLayout = device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.FRAGMENT,
              sampler: {}
            },
            {
              binding: 1,
              visibility: GPUShaderStage.FRAGMENT,
              texture: {},
            },
            {
              binding: 2,
              visibility: GPUShaderStage.FRAGMENT,
              texture: {},
            },

          ]
        });
        this.texturesBindGroup = device.createBindGroup({
          entries: [
            {
              binding: 0,
              resource: sampler,
            },
            {
              binding: 1,
              resource: this.textures[0].createView(),
            },
            {
              binding: 2,
              resource: this.textures[1].createView(),
            },
          ],
          layout: texturesBindGroupLayout,
        })

        this.vertexBufferDescriptor = [
            {
              attributes: [
                {
                  shaderLocation: 0, // position
                  offset: 0,
                  format: "float32x2",
                },
                {
                  shaderLocation: 1, // color
                  offset: 8,
                  format: "float32x4",
                },
                {
                  shaderLocation: 2, // uv
                  offset: 24,
                  format: "float32x2",
                },
                {
                  shaderLocation: 3, // shape
                  offset: 32,
                  format: "float32",
                },
              ],
              arrayStride: VERTEX_STRIDE * 4,
              stepMode: "vertex",
            },
          ];

          const shaderModule = device.createShaderModule({
            code: shaderCode
          })

          this.pipelineDescriptor = {
            vertex: {
              module: shaderModule,
              entryPoint: "vertex_main",
              buffers: this.vertexBufferDescriptor,
            },
            fragment: {
              module: shaderModule,
              entryPoint: "fragment_main",
              targets: [
                {
                  format: navigator.gpu.getPreferredCanvasFormat(),
                },
              ],
            },
            primitive: {
              topology: "triangle-list",
            },
            layout: device.createPipelineLayout({
              bindGroupLayouts: [uniformBindGroupLayout, texturesBindGroupLayout]
            }),
          };
    }

    addVertexBuffer = (objectList: PhysObject[]) => {
        let vertexList: number[] = [];
        let indexList: number[] = [];
        objectList.forEach(obj => {
            let startIndex = vertexList.length / VERTEX_STRIDE;
            vertexList.push(...obj.shape.vertices.flatMap(vertex => {
                return [...vec2.add(vertex.position, obj.position), ...vertex.color, ...vertex.uv, vertex.shape];
            }));
            indexList.push(...obj.shape.indices.map(index => index + startIndex));
        });

        const vertices = new Float32Array(vertexList);
        this.vertexBuffer = this.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices, 0, vertices.length);

        const indices = new Uint32Array(indexList);
        this.indexCount = indexList.length;
        this.indexBuffer = this.device.createBuffer({
          size: indices.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(this.indexBuffer, 0, indices, 0, indices.length);
    };

    render = () => {
        const renderPipeline = this.device.createRenderPipeline(this.pipelineDescriptor);
        const commandEncoder = this.device.createCommandEncoder();
        const clearColor = { r: 0.0, g: 0.5, b: 1.0, a: 1.0 };

        const renderPassDescriptor = {
        colorAttachments: [
                {
                    clearValue: clearColor,
                    loadOp: "clear",
                    storeOp: "store",
                    view: this.canvasContext.getCurrentTexture().createView(),
                },
            ] as GPURenderPassColorAttachment[],
        };

        this.device.queue.writeBuffer(
          this.uniformBuffer,
          0,
          this.viewMatrix.buffer,
          this.viewMatrix.byteOffset,
          this.viewMatrix.byteLength,
        );

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(renderPipeline);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.setBindGroup(1, this.texturesBindGroup);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.setIndexBuffer(this.indexBuffer, "uint32");
        passEncoder.drawIndexed(this.indexCount);
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        this.vertexBuffer.destroy();
        this.indexBuffer.destroy();
    }

    private addTexture(bitmap: ImageBitmap): TextureHandle {
      const texture = this.device.createTexture({
          size: [bitmap.width, bitmap.height],
          format:  'rgba8unorm',
          usage: 
              GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.STORAGE_BINDING |
              GPUTextureUsage.COPY_DST |
              GPUTextureUsage.RENDER_ATTACHMENT
      });
  
      this.device.queue.copyExternalImageToTexture(
          {source: bitmap},
          {texture: texture},
          [bitmap.width, bitmap.height]
      );
      return this.textures.push(texture) - 1 as TextureHandle;
  }
}

export async function requestRenderer(canvas: HTMLCanvasElement, textures: ImageBitmap[]): Promise<Renderer> {
    const device = await navigator.gpu.requestAdapter()
        .then(adapter => adapter.requestDevice());

    const canvasContext = canvas.getContext("webgpu") as unknown as GPUCanvasContext;
    canvasContext.configure({
        device: device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "premultiplied",
    });
    return new Renderer(device, canvasContext, textures);
}