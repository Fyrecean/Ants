import { mat4, Mat4, vec2, vec4 } from "wgpu-matrix";
import typesWGSL from "./types.wgsl";
import renderWGSL from "./render.wgsl";
import { Shape2D } from "main";

const shaderCode = typesWGSL + "\n" + renderWGSL;

const VERTEX_STRIDE = 8;

type TextureHandle = number;

export class Renderer {
    device: GPUDevice;
    canvasContext: GPUCanvasContext;
    antVertexBufferDescriptor: GPUVertexBufferLayout;
    antVertexBuffer?: GPUBuffer
    environmentVertexBufferDescriptor: GPUVertexBufferLayout;
    environmentVertexBuffer?: GPUBuffer;
    environmentIndexBuffer?: GPUBuffer;
    indexCount: number;
    environmentPiplineDescriptor: GPURenderPipelineDescriptor;
    antsPipelineDescriptor: GPURenderPipelineDescriptor;
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
            size: this.viewMatrix.byteLength + 64,
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

        this.environmentVertexBufferDescriptor ={
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
            ],
            arrayStride: VERTEX_STRIDE * 4,
            stepMode: "vertex",
        };
        this.antVertexBufferDescriptor = { // Ant triangle to instance
            attributes: [
                {
                    shaderLocation: 0, // position
                    offset: 0,
                    format: "float32x2",
                },
                {
                    shaderLocation: 1, // uv
                    offset: 8,
                    format: "float32x2",
                },
            ],
            arrayStride: 16,
            stepMode: "vertex",
        };

        const shaderModule = device.createShaderModule({
            code: shaderCode
        })

        this.environmentPiplineDescriptor = {
            vertex: {
                module: shaderModule,
                entryPoint: "vertex_environment",
                buffers: [this.environmentVertexBufferDescriptor],
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fragment_environment",
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

        this.antsPipelineDescriptor = {
            vertex: {
                module: shaderModule,
                entryPoint: "vertex_ant",
                buffers: [this.antVertexBufferDescriptor],
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fragment_ant",
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
        }

        const antBufferSize = this.antVertexBufferDescriptor.arrayStride * 3;
        this.antVertexBuffer = device.createBuffer({
            size: antBufferSize,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(this.antVertexBuffer.getMappedRange(0, antBufferSize)).set(
            [
                // Top
                0, 1, .5, 1,
                //Left
                -.5, -1, .25, 0,
                // Right
                .5, -1, .75, 0,
            ]
        );
        this.antVertexBuffer.unmap();
    }

    addVertexBuffer = (objectList: Shape2D[]) => {
        this.environmentIndexBuffer?.destroy();
        this.environmentVertexBuffer?.destroy();
        let vertexList: number[] = [];
        let indexList: number[] = [];
        objectList.forEach(obj => {
            let startIndex = vertexList.length / VERTEX_STRIDE;
            vertexList.push(...obj.vertices.flatMap(vertex => {
                return [...vec2.add(vertex.position, obj.position), ...vertex.color, ...vertex.uv];
            }));
            indexList.push(...obj.indices.map(index => index + startIndex));
        });

        const vertices = new Float32Array(vertexList);
        this.environmentVertexBuffer = this.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.environmentVertexBuffer, 0, vertices, 0, vertices.length);

        const indices = new Uint32Array(indexList);
        this.indexCount = indexList.length;
        this.environmentIndexBuffer = this.device.createBuffer({
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(this.environmentIndexBuffer, 0, indices, 0, indices.length);
    };

    render = () => {
        const environmentPipeline = this.device.createRenderPipeline(this.environmentPiplineDescriptor);
        const antPipeline = this.device.createRenderPipeline(this.antsPipelineDescriptor);
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

        const ant1Data = new Float32Array([
            -.5, 0, 0, 0, .5,
        ]);
        const ant2Data = new Float32Array([
            .5, 0, 0, 0, 1,
        ]);

        this.device.queue.writeBuffer(
            this.uniformBuffer,
            this.viewMatrix.byteLength,
            ant1Data,0,ant1Data.length
        );
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            this.viewMatrix.byteLength + 32,
            ant2Data,0,ant2Data.length
        );

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.setBindGroup(1, this.texturesBindGroup);

        passEncoder.setPipeline(environmentPipeline);
        passEncoder.setVertexBuffer(0, this.environmentVertexBuffer);
        passEncoder.setIndexBuffer(this.environmentIndexBuffer, "uint32");
        passEncoder.drawIndexed(this.indexCount);

        passEncoder.setPipeline(antPipeline);
        passEncoder.setVertexBuffer(0, this.antVertexBuffer);
        passEncoder.draw(3, 2);

        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    private addTexture(bitmap: ImageBitmap): TextureHandle {
        const texture = this.device.createTexture({
            size: [bitmap.width, bitmap.height],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: texture },
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