/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	var __webpack_modules__ = ({

/***/ "./src/render.wgsl":
/*!*************************!*\
  !*** ./src/render.wgsl ***!
  \*************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = ("struct Uniforms {\r\n  viewMatrix : mat4x4f,\r\n}\r\n\r\n@group(0) @binding(0) var<uniform> uniforms : Uniforms;\r\n\r\nstruct VertexOut {\r\n  @builtin(position) position : vec4f,\r\n  @location(0) color : vec4f,\r\n  @location(1) uv : vec2f,\r\n  @location(2) shape: f32,\r\n}\r\n\r\n@vertex\r\nfn vertex_main(@location(0) position: vec2f,\r\n               @location(1) color: vec4f,\r\n                @location(2) uv : vec2f,\r\n                @location(3) shape: f32,\r\n\r\n) -> VertexOut\r\n{\r\n  var output : VertexOut;\r\n  output.position = uniforms.viewMatrix * vec4(position, 0., 1.);\r\n  output.color = color;\r\n  output.uv = uv;\r\n  output.shape = shape;\r\n  return output;\r\n}\r\n\r\n@group(1) @binding(0) var mySampler: sampler;\r\n@group(1) @binding(1) var environmentTexture: texture_2d<f32>;\r\n@group(1) @binding(2) var dirtTexture: texture_2d<f32>;\r\n\r\n@fragment\r\nfn fragment_main(fragData: VertexOut) -> @location(0) vec4f\r\n{\r\n  let environmentSample = textureSample(environmentTexture, mySampler, fragData.uv);\r\n  let sample = textureSample(dirtTexture, mySampler, fragData.uv);\r\n  // Handle debug circles\r\n  if (fragData.shape == 1) {\r\n    let outOfCircle = fragData.shape == 1. && length(fragData.uv - vec2(.5)) >= .5;\r\n    if (outOfCircle) {\r\n      discard;\r\n    }\r\n    return fragData.color;\r\n  }\r\n  // Otherwise, rectangle texture\r\n  if (environmentSample.x > 0.) {\r\n    return sample;\r\n  } else if (environmentSample.y > 0) {\r\n    return vec4(0.,environmentSample.y,0.,1.);\r\n  } else {\r\n    discard;\r\n  }\r\n  return vec4(1.);\r\n}");

/***/ }),

/***/ "./src/physics.ts":
/*!************************!*\
  !*** ./src/physics.ts ***!
  \************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   CartControl: () => (/* binding */ CartControl),
/* harmony export */   Drag: () => (/* binding */ Drag),
/* harmony export */   GlobalGravity: () => (/* binding */ GlobalGravity),
/* harmony export */   Pivot: () => (/* binding */ Pivot),
/* harmony export */   Shapes: () => (/* binding */ Shapes),
/* harmony export */   black: () => (/* binding */ black),
/* harmony export */   blue: () => (/* binding */ blue),
/* harmony export */   green: () => (/* binding */ green),
/* harmony export */   red: () => (/* binding */ red),
/* harmony export */   simulate: () => (/* binding */ simulate),
/* harmony export */   white: () => (/* binding */ white)
/* harmony export */ });
/* harmony import */ var wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! wgpu-matrix */ "./node_modules/wgpu-matrix/dist/3.x/wgpu-matrix.module.js");

const white = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec4.fromValues(1, 1, 1, 1);
const red = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec4.fromValues(1, 0, 0, 1);
const blue = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec4.fromValues(0, 0, 1, 1);
const green = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec4.fromValues(0, 1, 0, 1);
const black = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec4.fromValues(0, 0, 0, 1);
var Shapes;
(function (Shapes) {
    Shapes[Shapes["rect"] = 0] = "rect";
    Shapes[Shapes["circle"] = 1] = "circle";
})(Shapes || (Shapes = {}));
function simulate(objects, deltaTime) {
    objects.forEach(obj => {
        let acceleration = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.zero();
        obj.forces.forEach(force => {
            const a = force.getAcceleration(obj);
            wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.add(acceleration, a, acceleration);
        });
        obj.constraints.forEach(constraint => {
            acceleration = constraint.applyConstraint(acceleration);
        });
        if (obj.constraints.length > 0) {
            // console.log("net accel", acceleration[0], acceleration[1], "vel", obj.velocity[0], obj.velocity[1]);
        }
        obj.velocity = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.addScaled(obj.velocity, acceleration, deltaTime);
        obj.position = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.addScaled(obj.position, obj.velocity, deltaTime);
    });
}
class GlobalGravity {
    acceleration;
    constructor(acceleration) {
        this.acceleration = acceleration;
    }
    getAcceleration() {
        return wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.fromValues(0, this.acceleration);
    }
}
var InputDirection;
(function (InputDirection) {
    InputDirection[InputDirection["left"] = -1] = "left";
    InputDirection[InputDirection["zero"] = 0] = "zero";
    InputDirection[InputDirection["right"] = 1] = "right";
})(InputDirection || (InputDirection = {}));
class CartControl {
    force;
    direction;
    constructor(force) {
        this.force = force;
        this.direction = InputDirection.zero;
        document.addEventListener("keypress", (event) => {
            if (event.key == "a" && this.direction > -1) {
                this.direction += InputDirection.left;
            }
            else if (event.key == "d" && this.direction < 1) {
                this.direction += InputDirection.right;
            }
        });
        document.addEventListener("keyup", (event) => {
            if (event.key == "a") {
                this.direction -= InputDirection.left;
            }
            else if (event.key == "d") {
                this.direction -= InputDirection.right;
            }
            // console.log(this.direction);
        });
    }
    getAcceleration(object) {
        return wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.fromValues(this.direction * (this.force / object.mass), 0);
    }
}
class Drag {
    coefficient;
    constructor(coefficient) {
        this.coefficient = coefficient;
    }
    getAcceleration(object) {
        return wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.scale(wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.negate(object.velocity), this.coefficient / object.mass);
    }
}
class Pivot {
    object;
    position;
    length;
    constructor(object, x, y) {
        this.position = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.fromValues(x, y);
        this.object = object;
        this.length = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.len(wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.sub(this.position, this.object.position));
    }
    applyConstraint(acceleration) {
        // console.log(acceleration);
        const tensionDirection = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.normalize(wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.sub(this.position, this.object.position));
        const tension = wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.scale(tensionDirection, -(wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.dot(acceleration, tensionDirection) + (length * wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.lenSq(this.object.velocity))));
        // console.log(tension);
        wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.add(this.position, wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.scale(tensionDirection, -this.length), this.object.position);
        return wgpu_matrix__WEBPACK_IMPORTED_MODULE_0__.vec2.add(acceleration, tension);
    }
}


/***/ }),

/***/ "./src/renderer.ts":
/*!*************************!*\
  !*** ./src/renderer.ts ***!
  \*************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   Renderer: () => (/* binding */ Renderer),
/* harmony export */   requestRenderer: () => (/* binding */ requestRenderer)
/* harmony export */ });
/* harmony import */ var wgpu_matrix__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! wgpu-matrix */ "./node_modules/wgpu-matrix/dist/3.x/wgpu-matrix.module.js");
/* harmony import */ var _render_wgsl__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./render.wgsl */ "./src/render.wgsl");


const VERTEX_STRIDE = 9;
class Renderer {
    device;
    canvasContext;
    vertexBuffer;
    indexBuffer;
    indexCount;
    vertexBufferDescriptor;
    pipelineDescriptor;
    uniformBuffer;
    uniformBindGroup;
    texturesBindGroup;
    viewMatrix;
    textures = [];
    constructor(device, canvasContext, textures) {
        this.device = device;
        this.canvasContext = canvasContext;
        this.addTexture(textures[0]); // environment
        this.addTexture(textures[1]); // dirt
        this.viewMatrix = wgpu_matrix__WEBPACK_IMPORTED_MODULE_1__.mat4.identity();
        this.uniformBuffer = device.createBuffer({
            size: 4 * 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
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
        });
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
            code: _render_wgsl__WEBPACK_IMPORTED_MODULE_0__["default"]
        });
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
    addVertexBuffer = (objectList) => {
        let vertexList = [];
        let indexList = [];
        objectList.forEach(obj => {
            let startIndex = vertexList.length / VERTEX_STRIDE;
            vertexList.push(...obj.shape.vertices.flatMap(vertex => {
                return [...wgpu_matrix__WEBPACK_IMPORTED_MODULE_1__.vec2.add(vertex.position, obj.position), ...vertex.color, ...vertex.uv, vertex.shape];
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
        });
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
            ],
        };
        this.device.queue.writeBuffer(this.uniformBuffer, 0, this.viewMatrix.buffer, this.viewMatrix.byteOffset, this.viewMatrix.byteLength);
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
    };
    addTexture(bitmap) {
        const texture = this.device.createTexture({
            size: [bitmap.width, bitmap.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT
        });
        this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: texture }, [bitmap.width, bitmap.height]);
        return this.textures.push(texture) - 1;
    }
}
async function requestRenderer(canvas, textures) {
    const device = await navigator.gpu.requestAdapter()
        .then(adapter => adapter.requestDevice());
    const canvasContext = canvas.getContext("webgpu");
    canvasContext.configure({
        device: device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "premultiplied",
    });
    return new Renderer(device, canvasContext, textures);
}


/***/ }),

/***/ "./node_modules/wgpu-matrix/dist/3.x/wgpu-matrix.module.js":
/*!*****************************************************************!*\
  !*** ./node_modules/wgpu-matrix/dist/3.x/wgpu-matrix.module.js ***!
  \*****************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   mat3: () => (/* binding */ mat3),
/* harmony export */   mat3d: () => (/* binding */ mat3d),
/* harmony export */   mat3n: () => (/* binding */ mat3n),
/* harmony export */   mat4: () => (/* binding */ mat4),
/* harmony export */   mat4d: () => (/* binding */ mat4d),
/* harmony export */   mat4n: () => (/* binding */ mat4n),
/* harmony export */   quat: () => (/* binding */ quat),
/* harmony export */   quatd: () => (/* binding */ quatd),
/* harmony export */   quatn: () => (/* binding */ quatn),
/* harmony export */   utils: () => (/* binding */ utils),
/* harmony export */   vec2: () => (/* binding */ vec2),
/* harmony export */   vec2d: () => (/* binding */ vec2d),
/* harmony export */   vec2n: () => (/* binding */ vec2n),
/* harmony export */   vec3: () => (/* binding */ vec3),
/* harmony export */   vec3d: () => (/* binding */ vec3d),
/* harmony export */   vec3n: () => (/* binding */ vec3n),
/* harmony export */   vec4: () => (/* binding */ vec4),
/* harmony export */   vec4d: () => (/* binding */ vec4d),
/* harmony export */   vec4n: () => (/* binding */ vec4n)
/* harmony export */ });
/* wgpu-matrix@3.2.0, license MIT */
function wrapConstructor(OriginalConstructor, modifier) {
    return class extends OriginalConstructor {
        constructor(...args) {
            super(...args);
            modifier(this);
        }
    }; // Type assertion is necessary here
}
const ZeroArray = wrapConstructor((Array), a => a.fill(0));

/*
 * Copyright 2022 Gregg Tavares
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
let EPSILON = 0.000001;
/**
 * Set the value for EPSILON for various checks
 * @param v - Value to use for EPSILON.
 * @returns previous value of EPSILON;
 */
function setEpsilon(v) {
    const old = EPSILON;
    EPSILON = v;
    return old;
}
/**
 * Convert degrees to radians
 * @param degrees - Angle in degrees
 * @returns angle converted to radians
 */
function degToRad(degrees) {
    return degrees * Math.PI / 180;
}
/**
 * Convert radians to degrees
 * @param radians - Angle in radians
 * @returns angle converted to degrees
 */
function radToDeg(radians) {
    return radians * 180 / Math.PI;
}
/**
 * Lerps between a and b via t
 * @param a - starting value
 * @param b - ending value
 * @param t - value where 0 = a and 1 = b
 * @returns a + (b - a) * t
 */
function lerp(a, b, t) {
    return a + (b - a) * t;
}
/**
 * Compute the opposite of lerp. Given a and b and a value between
 * a and b returns a value between 0 and 1. 0 if a, 1 if b.
 * Note: no clamping is done.
 * @param a - start value
 * @param b - end value
 * @param v - value between a and b
 * @returns (v - a) / (b - a)
 */
function inverseLerp(a, b, v) {
    const d = b - a;
    return (Math.abs(b - a) < EPSILON)
        ? a
        : (v - a) / d;
}
/**
 * Compute the euclidean modulo
 *
 * ```
 * // table for n / 3
 * -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5   <- n
 * ------------------------------------
 * -2  -1  -0  -2  -1   0,  1,  2,  0,  1,  2   <- n % 3
 *  1   2   0   1   2   0,  1,  2,  0,  1,  2   <- euclideanModule(n, 3)
 * ```
 *
 * @param n - dividend
 * @param m - divisor
 * @returns the euclidean modulo of n / m
 */
function euclideanModulo(n, m) {
    return ((n % m) + m) % m;
}

var utils = {
    __proto__: null,
    get EPSILON () { return EPSILON; },
    degToRad: degToRad,
    euclideanModulo: euclideanModulo,
    inverseLerp: inverseLerp,
    lerp: lerp,
    radToDeg: radToDeg,
    setEpsilon: setEpsilon
};

/*
 * Copyright 2022 Gregg Tavares
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/**
 * Generates am typed API for Vec3
 */
function getAPIImpl$5(Ctor) {
    /**
     * Creates a Vec2; may be called with x, y, z to set initial values.
     *
     * Note: Since passing in a raw JavaScript array
     * is valid in all circumstances, if you want to
     * force a JavaScript array into a Vec2's specified type
     * it would be faster to use
     *
     * ```
     * const v = vec2.clone(someJSArray);
     * ```
     *
     * @param x - Initial x value.
     * @param y - Initial y value.
     * @returns the created vector
     */
    function create(x = 0, y = 0) {
        const newDst = new Ctor(2);
        if (x !== undefined) {
            newDst[0] = x;
            if (y !== undefined) {
                newDst[1] = y;
            }
        }
        return newDst;
    }
    /**
     * Creates a Vec2; may be called with x, y, z to set initial values. (same as create)
     * @param x - Initial x value.
     * @param y - Initial y value.
     * @returns the created vector
     */
    const fromValues = create;
    /**
     * Sets the values of a Vec2
     * Also see {@link vec2.create} and {@link vec2.copy}
     *
     * @param x first value
     * @param y second value
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector with its elements set.
     */
    function set(x, y, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = x;
        newDst[1] = y;
        return newDst;
    }
    /**
     * Applies Math.ceil to each element of vector
     * @param v - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the ceil of each element of v.
     */
    function ceil(v, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = Math.ceil(v[0]);
        newDst[1] = Math.ceil(v[1]);
        return newDst;
    }
    /**
     * Applies Math.floor to each element of vector
     * @param v - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the floor of each element of v.
     */
    function floor(v, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = Math.floor(v[0]);
        newDst[1] = Math.floor(v[1]);
        return newDst;
    }
    /**
     * Applies Math.round to each element of vector
     * @param v - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the round of each element of v.
     */
    function round(v, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = Math.round(v[0]);
        newDst[1] = Math.round(v[1]);
        return newDst;
    }
    /**
     * Clamp each element of vector between min and max
     * @param v - Operand vector.
     * @param max - Min value, default 0
     * @param min - Max value, default 1
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that the clamped value of each element of v.
     */
    function clamp(v, min = 0, max = 1, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = Math.min(max, Math.max(min, v[0]));
        newDst[1] = Math.min(max, Math.max(min, v[1]));
        return newDst;
    }
    /**
     * Adds two vectors; assumes a and b have the same dimension.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the sum of a and b.
     */
    function add(a, b, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = a[0] + b[0];
        newDst[1] = a[1] + b[1];
        return newDst;
    }
    /**
     * Adds two vectors, scaling the 2nd; assumes a and b have the same dimension.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param scale - Amount to scale b
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the sum of a + b * scale.
     */
    function addScaled(a, b, scale, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = a[0] + b[0] * scale;
        newDst[1] = a[1] + b[1] * scale;
        return newDst;
    }
    /**
     * Returns the angle in radians between two vectors.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns The angle in radians between the 2 vectors.
     */
    function angle(a, b) {
        const ax = a[0];
        const ay = a[1];
        const bx = b[0];
        const by = b[1];
        const mag1 = Math.sqrt(ax * ax + ay * ay);
        const mag2 = Math.sqrt(bx * bx + by * by);
        const mag = mag1 * mag2;
        const cosine = mag && dot(a, b) / mag;
        return Math.acos(cosine);
    }
    /**
     * Subtracts two vectors.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the difference of a and b.
     */
    function subtract(a, b, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = a[0] - b[0];
        newDst[1] = a[1] - b[1];
        return newDst;
    }
    /**
     * Subtracts two vectors.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the difference of a and b.
     */
    const sub = subtract;
    /**
     * Check if 2 vectors are approximately equal
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns true if vectors are approximately equal
     */
    function equalsApproximately(a, b) {
        return Math.abs(a[0] - b[0]) < EPSILON &&
            Math.abs(a[1] - b[1]) < EPSILON;
    }
    /**
     * Check if 2 vectors are exactly equal
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns true if vectors are exactly equal
     */
    function equals(a, b) {
        return a[0] === b[0] && a[1] === b[1];
    }
    /**
     * Performs linear interpolation on two vectors.
     * Given vectors a and b and interpolation coefficient t, returns
     * a + t * (b - a).
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param t - Interpolation coefficient.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The linear interpolated result.
     */
    function lerp(a, b, t, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = a[0] + t * (b[0] - a[0]);
        newDst[1] = a[1] + t * (b[1] - a[1]);
        return newDst;
    }
    /**
     * Performs linear interpolation on two vectors.
     * Given vectors a and b and interpolation coefficient vector t, returns
     * a + t * (b - a).
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param t - Interpolation coefficients vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns the linear interpolated result.
     */
    function lerpV(a, b, t, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = a[0] + t[0] * (b[0] - a[0]);
        newDst[1] = a[1] + t[1] * (b[1] - a[1]);
        return newDst;
    }
    /**
     * Return max values of two vectors.
     * Given vectors a and b returns
     * [max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2])].
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The max components vector.
     */
    function max(a, b, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = Math.max(a[0], b[0]);
        newDst[1] = Math.max(a[1], b[1]);
        return newDst;
    }
    /**
     * Return min values of two vectors.
     * Given vectors a and b returns
     * [min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2])].
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The min components vector.
     */
    function min(a, b, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = Math.min(a[0], b[0]);
        newDst[1] = Math.min(a[1], b[1]);
        return newDst;
    }
    /**
     * Multiplies a vector by a scalar.
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The scaled vector.
     */
    function mulScalar(v, k, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = v[0] * k;
        newDst[1] = v[1] * k;
        return newDst;
    }
    /**
     * Multiplies a vector by a scalar. (same as mulScalar)
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The scaled vector.
     */
    const scale = mulScalar;
    /**
     * Divides a vector by a scalar.
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The scaled vector.
     */
    function divScalar(v, k, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = v[0] / k;
        newDst[1] = v[1] / k;
        return newDst;
    }
    /**
     * Inverse a vector.
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The inverted vector.
     */
    function inverse(v, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = 1 / v[0];
        newDst[1] = 1 / v[1];
        return newDst;
    }
    /**
     * Invert a vector. (same as inverse)
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The inverted vector.
     */
    const invert = inverse;
    /**
     * Computes the cross product of two vectors; assumes both vectors have
     * three entries.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of a cross b.
     */
    function cross(a, b, dst) {
        const newDst = (dst ?? new Ctor(3));
        const z = a[0] * b[1] - a[1] * b[0];
        newDst[0] = 0;
        newDst[1] = 0;
        newDst[2] = z;
        return newDst;
    }
    /**
     * Computes the dot product of two vectors; assumes both vectors have
     * three entries.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns dot product
     */
    function dot(a, b) {
        return a[0] * b[0] + a[1] * b[1];
    }
    /**
     * Computes the length of vector
     * @param v - vector.
     * @returns length of vector.
     */
    function length(v) {
        const v0 = v[0];
        const v1 = v[1];
        return Math.sqrt(v0 * v0 + v1 * v1);
    }
    /**
     * Computes the length of vector (same as length)
     * @param v - vector.
     * @returns length of vector.
     */
    const len = length;
    /**
     * Computes the square of the length of vector
     * @param v - vector.
     * @returns square of the length of vector.
     */
    function lengthSq(v) {
        const v0 = v[0];
        const v1 = v[1];
        return v0 * v0 + v1 * v1;
    }
    /**
     * Computes the square of the length of vector (same as lengthSq)
     * @param v - vector.
     * @returns square of the length of vector.
     */
    const lenSq = lengthSq;
    /**
     * Computes the distance between 2 points
     * @param a - vector.
     * @param b - vector.
     * @returns distance between a and b
     */
    function distance(a, b) {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        return Math.sqrt(dx * dx + dy * dy);
    }
    /**
     * Computes the distance between 2 points (same as distance)
     * @param a - vector.
     * @param b - vector.
     * @returns distance between a and b
     */
    const dist = distance;
    /**
     * Computes the square of the distance between 2 points
     * @param a - vector.
     * @param b - vector.
     * @returns square of the distance between a and b
     */
    function distanceSq(a, b) {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }
    /**
     * Computes the square of the distance between 2 points (same as distanceSq)
     * @param a - vector.
     * @param b - vector.
     * @returns square of the distance between a and b
     */
    const distSq = distanceSq;
    /**
     * Divides a vector by its Euclidean length and returns the quotient.
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The normalized vector.
     */
    function normalize(v, dst) {
        const newDst = (dst ?? new Ctor(2));
        const v0 = v[0];
        const v1 = v[1];
        const len = Math.sqrt(v0 * v0 + v1 * v1);
        if (len > 0.00001) {
            newDst[0] = v0 / len;
            newDst[1] = v1 / len;
        }
        else {
            newDst[0] = 0;
            newDst[1] = 0;
        }
        return newDst;
    }
    /**
     * Negates a vector.
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns -v.
     */
    function negate(v, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = -v[0];
        newDst[1] = -v[1];
        return newDst;
    }
    /**
     * Copies a vector. (same as {@link vec2.clone})
     * Also see {@link vec2.create} and {@link vec2.set}
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A copy of v.
     */
    function copy(v, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = v[0];
        newDst[1] = v[1];
        return newDst;
    }
    /**
     * Clones a vector. (same as {@link vec2.copy})
     * Also see {@link vec2.create} and {@link vec2.set}
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A copy of v.
     */
    const clone = copy;
    /**
     * Multiplies a vector by another vector (component-wise); assumes a and
     * b have the same length.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of products of entries of a and b.
     */
    function multiply(a, b, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = a[0] * b[0];
        newDst[1] = a[1] * b[1];
        return newDst;
    }
    /**
     * Multiplies a vector by another vector (component-wise); assumes a and
     * b have the same length. (same as mul)
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of products of entries of a and b.
     */
    const mul = multiply;
    /**
     * Divides a vector by another vector (component-wise); assumes a and
     * b have the same length.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of quotients of entries of a and b.
     */
    function divide(a, b, dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = a[0] / b[0];
        newDst[1] = a[1] / b[1];
        return newDst;
    }
    /**
     * Divides a vector by another vector (component-wise); assumes a and
     * b have the same length. (same as divide)
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of quotients of entries of a and b.
     */
    const div = divide;
    /**
     * Creates a random unit vector * scale
     * @param scale - Default 1
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The random vector.
     */
    function random(scale = 1, dst) {
        const newDst = (dst ?? new Ctor(2));
        const angle = Math.random() * 2 * Math.PI;
        newDst[0] = Math.cos(angle) * scale;
        newDst[1] = Math.sin(angle) * scale;
        return newDst;
    }
    /**
     * Zero's a vector
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The zeroed vector.
     */
    function zero(dst) {
        const newDst = (dst ?? new Ctor(2));
        newDst[0] = 0;
        newDst[1] = 0;
        return newDst;
    }
    /**
     * transform Vec2 by 4x4 matrix
     * @param v - the vector
     * @param m - The matrix.
     * @param dst - optional Vec2 to store result. If not passed a new one is created.
     * @returns the transformed vector
     */
    function transformMat4(v, m, dst) {
        const newDst = (dst ?? new Ctor(2));
        const x = v[0];
        const y = v[1];
        newDst[0] = x * m[0] + y * m[4] + m[12];
        newDst[1] = x * m[1] + y * m[5] + m[13];
        return newDst;
    }
    /**
     * Transforms vec4 by 3x3 matrix
     *
     * @param v - the vector
     * @param m - The matrix.
     * @param dst - optional Vec2 to store result. If not passed a new one is created.
     * @returns the transformed vector
     */
    function transformMat3(v, m, dst) {
        const newDst = (dst ?? new Ctor(2));
        const x = v[0];
        const y = v[1];
        newDst[0] = m[0] * x + m[4] * y + m[8];
        newDst[1] = m[1] * x + m[5] * y + m[9];
        return newDst;
    }
    /**
     * Rotate a 2D vector
     *
     * @param a The vec2 point to rotate
     * @param b The origin of the rotation
     * @param rad The angle of rotation in radians
     * @returns the rotated vector
     */
    function rotate(a, b, rad, dst) {
        const newDst = (dst ?? new Ctor(2));
        // Translate point to the origin
        const p0 = a[0] - b[0];
        const p1 = a[1] - b[1];
        const sinC = Math.sin(rad);
        const cosC = Math.cos(rad);
        //perform rotation and translate to correct position
        newDst[0] = p0 * cosC - p1 * sinC + b[0];
        newDst[1] = p0 * sinC + p1 * cosC + b[1];
        return newDst;
    }
    /**
     * Treat a 2D vector as a direction and set it's length
     *
     * @param a The vec2 to lengthen
     * @param len The length of the resulting vector
     * @returns The lengthened vector
     */
    function setLength(a, len, dst) {
        const newDst = (dst ?? new Ctor(2));
        normalize(a, newDst);
        return mulScalar(newDst, len, newDst);
    }
    /**
     * Ensure a vector is not longer than a max length
     *
     * @param a The vec2 to limit
     * @param maxLen The longest length of the resulting vector
     * @returns The vector, shortened to maxLen if it's too long
     */
    function truncate(a, maxLen, dst) {
        const newDst = (dst ?? new Ctor(2));
        if (length(a) > maxLen) {
            return setLength(a, maxLen, newDst);
        }
        return copy(a, newDst);
    }
    /**
     * Return the vector exactly between 2 endpoint vectors
     *
     * @param a Endpoint 1
     * @param b Endpoint 2
     * @returns The vector exactly residing between endpoints 1 and 2
     */
    function midpoint(a, b, dst) {
        const newDst = (dst ?? new Ctor(2));
        return lerp(a, b, 0.5, newDst);
    }
    return {
        create,
        fromValues,
        set,
        ceil,
        floor,
        round,
        clamp,
        add,
        addScaled,
        angle,
        subtract,
        sub,
        equalsApproximately,
        equals,
        lerp,
        lerpV,
        max,
        min,
        mulScalar,
        scale,
        divScalar,
        inverse,
        invert,
        cross,
        dot,
        length,
        len,
        lengthSq,
        lenSq,
        distance,
        dist,
        distanceSq,
        distSq,
        normalize,
        negate,
        copy,
        clone,
        multiply,
        mul,
        divide,
        div,
        random,
        zero,
        transformMat4,
        transformMat3,
        rotate,
        setLength,
        truncate,
        midpoint,
    };
}
const cache$5 = new Map();
function getAPI$5(Ctor) {
    let api = cache$5.get(Ctor);
    if (!api) {
        api = getAPIImpl$5(Ctor);
        cache$5.set(Ctor, api);
    }
    return api;
}

/*
 * Copyright 2022 Gregg Tavares
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/**
 * Generates am typed API for Vec3
 * */
function getAPIImpl$4(Ctor) {
    /**
     * Creates a vec3; may be called with x, y, z to set initial values.
     * @param x - Initial x value.
     * @param y - Initial y value.
     * @param z - Initial z value.
     * @returns the created vector
     */
    function create(x, y, z) {
        const newDst = new Ctor(3);
        if (x !== undefined) {
            newDst[0] = x;
            if (y !== undefined) {
                newDst[1] = y;
                if (z !== undefined) {
                    newDst[2] = z;
                }
            }
        }
        return newDst;
    }
    /**
     * Creates a vec3; may be called with x, y, z to set initial values. (same as create)
     * @param x - Initial x value.
     * @param y - Initial y value.
     * @param z - Initial z value.
     * @returns the created vector
     */
    const fromValues = create;
    /**
     * Sets the values of a Vec3
     * Also see {@link vec3.create} and {@link vec3.copy}
     *
     * @param x first value
     * @param y second value
     * @param z third value
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector with its elements set.
     */
    function set(x, y, z, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = x;
        newDst[1] = y;
        newDst[2] = z;
        return newDst;
    }
    /**
     * Applies Math.ceil to each element of vector
     * @param v - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the ceil of each element of v.
     */
    function ceil(v, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = Math.ceil(v[0]);
        newDst[1] = Math.ceil(v[1]);
        newDst[2] = Math.ceil(v[2]);
        return newDst;
    }
    /**
     * Applies Math.floor to each element of vector
     * @param v - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the floor of each element of v.
     */
    function floor(v, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = Math.floor(v[0]);
        newDst[1] = Math.floor(v[1]);
        newDst[2] = Math.floor(v[2]);
        return newDst;
    }
    /**
     * Applies Math.round to each element of vector
     * @param v - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the round of each element of v.
     */
    function round(v, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = Math.round(v[0]);
        newDst[1] = Math.round(v[1]);
        newDst[2] = Math.round(v[2]);
        return newDst;
    }
    /**
     * Clamp each element of vector between min and max
     * @param v - Operand vector.
     * @param max - Min value, default 0
     * @param min - Max value, default 1
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that the clamped value of each element of v.
     */
    function clamp(v, min = 0, max = 1, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = Math.min(max, Math.max(min, v[0]));
        newDst[1] = Math.min(max, Math.max(min, v[1]));
        newDst[2] = Math.min(max, Math.max(min, v[2]));
        return newDst;
    }
    /**
     * Adds two vectors; assumes a and b have the same dimension.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the sum of a and b.
     */
    function add(a, b, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = a[0] + b[0];
        newDst[1] = a[1] + b[1];
        newDst[2] = a[2] + b[2];
        return newDst;
    }
    /**
     * Adds two vectors, scaling the 2nd; assumes a and b have the same dimension.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param scale - Amount to scale b
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the sum of a + b * scale.
     */
    function addScaled(a, b, scale, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = a[0] + b[0] * scale;
        newDst[1] = a[1] + b[1] * scale;
        newDst[2] = a[2] + b[2] * scale;
        return newDst;
    }
    /**
     * Returns the angle in radians between two vectors.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns The angle in radians between the 2 vectors.
     */
    function angle(a, b) {
        const ax = a[0];
        const ay = a[1];
        const az = a[2];
        const bx = b[0];
        const by = b[1];
        const bz = b[2];
        const mag1 = Math.sqrt(ax * ax + ay * ay + az * az);
        const mag2 = Math.sqrt(bx * bx + by * by + bz * bz);
        const mag = mag1 * mag2;
        const cosine = mag && dot(a, b) / mag;
        return Math.acos(cosine);
    }
    /**
     * Subtracts two vectors.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the difference of a and b.
     */
    function subtract(a, b, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = a[0] - b[0];
        newDst[1] = a[1] - b[1];
        newDst[2] = a[2] - b[2];
        return newDst;
    }
    /**
     * Subtracts two vectors.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the difference of a and b.
     */
    const sub = subtract;
    /**
     * Check if 2 vectors are approximately equal
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns true if vectors are approximately equal
     */
    function equalsApproximately(a, b) {
        return Math.abs(a[0] - b[0]) < EPSILON &&
            Math.abs(a[1] - b[1]) < EPSILON &&
            Math.abs(a[2] - b[2]) < EPSILON;
    }
    /**
     * Check if 2 vectors are exactly equal
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns true if vectors are exactly equal
     */
    function equals(a, b) {
        return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
    }
    /**
     * Performs linear interpolation on two vectors.
     * Given vectors a and b and interpolation coefficient t, returns
     * a + t * (b - a).
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param t - Interpolation coefficient.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The linear interpolated result.
     */
    function lerp(a, b, t, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = a[0] + t * (b[0] - a[0]);
        newDst[1] = a[1] + t * (b[1] - a[1]);
        newDst[2] = a[2] + t * (b[2] - a[2]);
        return newDst;
    }
    /**
     * Performs linear interpolation on two vectors.
     * Given vectors a and b and interpolation coefficient vector t, returns
     * a + t * (b - a).
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param t - Interpolation coefficients vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns the linear interpolated result.
     */
    function lerpV(a, b, t, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = a[0] + t[0] * (b[0] - a[0]);
        newDst[1] = a[1] + t[1] * (b[1] - a[1]);
        newDst[2] = a[2] + t[2] * (b[2] - a[2]);
        return newDst;
    }
    /**
     * Return max values of two vectors.
     * Given vectors a and b returns
     * [max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2])].
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The max components vector.
     */
    function max(a, b, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = Math.max(a[0], b[0]);
        newDst[1] = Math.max(a[1], b[1]);
        newDst[2] = Math.max(a[2], b[2]);
        return newDst;
    }
    /**
     * Return min values of two vectors.
     * Given vectors a and b returns
     * [min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2])].
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The min components vector.
     */
    function min(a, b, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = Math.min(a[0], b[0]);
        newDst[1] = Math.min(a[1], b[1]);
        newDst[2] = Math.min(a[2], b[2]);
        return newDst;
    }
    /**
     * Multiplies a vector by a scalar.
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The scaled vector.
     */
    function mulScalar(v, k, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = v[0] * k;
        newDst[1] = v[1] * k;
        newDst[2] = v[2] * k;
        return newDst;
    }
    /**
     * Multiplies a vector by a scalar. (same as mulScalar)
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The scaled vector.
     */
    const scale = mulScalar;
    /**
     * Divides a vector by a scalar.
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The scaled vector.
     */
    function divScalar(v, k, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = v[0] / k;
        newDst[1] = v[1] / k;
        newDst[2] = v[2] / k;
        return newDst;
    }
    /**
     * Inverse a vector.
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The inverted vector.
     */
    function inverse(v, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = 1 / v[0];
        newDst[1] = 1 / v[1];
        newDst[2] = 1 / v[2];
        return newDst;
    }
    /**
     * Invert a vector. (same as inverse)
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The inverted vector.
     */
    const invert = inverse;
    /**
     * Computes the cross product of two vectors; assumes both vectors have
     * three entries.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of a cross b.
     */
    function cross(a, b, dst) {
        const newDst = (dst ?? new Ctor(3));
        const t1 = a[2] * b[0] - a[0] * b[2];
        const t2 = a[0] * b[1] - a[1] * b[0];
        newDst[0] = a[1] * b[2] - a[2] * b[1];
        newDst[1] = t1;
        newDst[2] = t2;
        return newDst;
    }
    /**
     * Computes the dot product of two vectors; assumes both vectors have
     * three entries.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns dot product
     */
    function dot(a, b) {
        return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
    }
    /**
     * Computes the length of vector
     * @param v - vector.
     * @returns length of vector.
     */
    function length(v) {
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        return Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    }
    /**
     * Computes the length of vector (same as length)
     * @param v - vector.
     * @returns length of vector.
     */
    const len = length;
    /**
     * Computes the square of the length of vector
     * @param v - vector.
     * @returns square of the length of vector.
     */
    function lengthSq(v) {
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        return v0 * v0 + v1 * v1 + v2 * v2;
    }
    /**
     * Computes the square of the length of vector (same as lengthSq)
     * @param v - vector.
     * @returns square of the length of vector.
     */
    const lenSq = lengthSq;
    /**
     * Computes the distance between 2 points
     * @param a - vector.
     * @param b - vector.
     * @returns distance between a and b
     */
    function distance(a, b) {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        const dz = a[2] - b[2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    /**
     * Computes the distance between 2 points (same as distance)
     * @param a - vector.
     * @param b - vector.
     * @returns distance between a and b
     */
    const dist = distance;
    /**
     * Computes the square of the distance between 2 points
     * @param a - vector.
     * @param b - vector.
     * @returns square of the distance between a and b
     */
    function distanceSq(a, b) {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        const dz = a[2] - b[2];
        return dx * dx + dy * dy + dz * dz;
    }
    /**
     * Computes the square of the distance between 2 points (same as distanceSq)
     * @param a - vector.
     * @param b - vector.
     * @returns square of the distance between a and b
     */
    const distSq = distanceSq;
    /**
     * Divides a vector by its Euclidean length and returns the quotient.
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The normalized vector.
     */
    function normalize(v, dst) {
        const newDst = (dst ?? new Ctor(3));
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        const len = Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2);
        if (len > 0.00001) {
            newDst[0] = v0 / len;
            newDst[1] = v1 / len;
            newDst[2] = v2 / len;
        }
        else {
            newDst[0] = 0;
            newDst[1] = 0;
            newDst[2] = 0;
        }
        return newDst;
    }
    /**
     * Negates a vector.
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns -v.
     */
    function negate(v, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = -v[0];
        newDst[1] = -v[1];
        newDst[2] = -v[2];
        return newDst;
    }
    /**
     * Copies a vector. (same as {@link vec3.clone})
     * Also see {@link vec3.create} and {@link vec3.set}
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A copy of v.
     */
    function copy(v, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = v[0];
        newDst[1] = v[1];
        newDst[2] = v[2];
        return newDst;
    }
    /**
     * Clones a vector. (same as {@link vec3.copy})
     * Also see {@link vec3.create} and {@link vec3.set}
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A copy of v.
     */
    const clone = copy;
    /**
     * Multiplies a vector by another vector (component-wise); assumes a and
     * b have the same length.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of products of entries of a and b.
     */
    function multiply(a, b, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = a[0] * b[0];
        newDst[1] = a[1] * b[1];
        newDst[2] = a[2] * b[2];
        return newDst;
    }
    /**
     * Multiplies a vector by another vector (component-wise); assumes a and
     * b have the same length. (same as mul)
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of products of entries of a and b.
     */
    const mul = multiply;
    /**
     * Divides a vector by another vector (component-wise); assumes a and
     * b have the same length.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of quotients of entries of a and b.
     */
    function divide(a, b, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = a[0] / b[0];
        newDst[1] = a[1] / b[1];
        newDst[2] = a[2] / b[2];
        return newDst;
    }
    /**
     * Divides a vector by another vector (component-wise); assumes a and
     * b have the same length. (same as divide)
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of quotients of entries of a and b.
     */
    const div = divide;
    /**
     * Creates a random vector
     * @param scale - Default 1
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The random vector.
     */
    function random(scale = 1, dst) {
        const newDst = (dst ?? new Ctor(3));
        const angle = Math.random() * 2 * Math.PI;
        const z = Math.random() * 2 - 1;
        const zScale = Math.sqrt(1 - z * z) * scale;
        newDst[0] = Math.cos(angle) * zScale;
        newDst[1] = Math.sin(angle) * zScale;
        newDst[2] = z * scale;
        return newDst;
    }
    /**
     * Zero's a vector
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The zeroed vector.
     */
    function zero(dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = 0;
        newDst[1] = 0;
        newDst[2] = 0;
        return newDst;
    }
    /**
     * transform vec3 by 4x4 matrix
     * @param v - the vector
     * @param m - The matrix.
     * @param dst - optional vec3 to store result. If not passed a new one is created.
     * @returns the transformed vector
     */
    function transformMat4(v, m, dst) {
        const newDst = (dst ?? new Ctor(3));
        const x = v[0];
        const y = v[1];
        const z = v[2];
        const w = (m[3] * x + m[7] * y + m[11] * z + m[15]) || 1;
        newDst[0] = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
        newDst[1] = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
        newDst[2] = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;
        return newDst;
    }
    /**
     * Transform vec3 by upper 3x3 matrix inside 4x4 matrix.
     * @param v - The direction.
     * @param m - The matrix.
     * @param dst - optional vec3 to store result. If not passed a new one is created.
     * @returns The transformed vector.
     */
    function transformMat4Upper3x3(v, m, dst) {
        const newDst = (dst ?? new Ctor(3));
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        newDst[0] = v0 * m[0 * 4 + 0] + v1 * m[1 * 4 + 0] + v2 * m[2 * 4 + 0];
        newDst[1] = v0 * m[0 * 4 + 1] + v1 * m[1 * 4 + 1] + v2 * m[2 * 4 + 1];
        newDst[2] = v0 * m[0 * 4 + 2] + v1 * m[1 * 4 + 2] + v2 * m[2 * 4 + 2];
        return newDst;
    }
    /**
     * Transforms vec3 by 3x3 matrix
     *
     * @param v - the vector
     * @param m - The matrix.
     * @param dst - optional vec3 to store result. If not passed a new one is created.
     * @returns the transformed vector
     */
    function transformMat3(v, m, dst) {
        const newDst = (dst ?? new Ctor(3));
        const x = v[0];
        const y = v[1];
        const z = v[2];
        newDst[0] = x * m[0] + y * m[4] + z * m[8];
        newDst[1] = x * m[1] + y * m[5] + z * m[9];
        newDst[2] = x * m[2] + y * m[6] + z * m[10];
        return newDst;
    }
    /**
     * Transforms vec3 by Quaternion
     * @param v - the vector to transform
     * @param q - the quaternion to transform by
     * @param dst - optional vec3 to store result. If not passed a new one is created.
     * @returns the transformed
     */
    function transformQuat(v, q, dst) {
        const newDst = (dst ?? new Ctor(3));
        const qx = q[0];
        const qy = q[1];
        const qz = q[2];
        const w2 = q[3] * 2;
        const x = v[0];
        const y = v[1];
        const z = v[2];
        const uvX = qy * z - qz * y;
        const uvY = qz * x - qx * z;
        const uvZ = qx * y - qy * x;
        newDst[0] = x + uvX * w2 + (qy * uvZ - qz * uvY) * 2;
        newDst[1] = y + uvY * w2 + (qz * uvX - qx * uvZ) * 2;
        newDst[2] = z + uvZ * w2 + (qx * uvY - qy * uvX) * 2;
        return newDst;
    }
    /**
     * Returns the translation component of a 4-by-4 matrix as a vector with 3
     * entries.
     * @param m - The matrix.
     * @param dst - vector to hold result. If not passed a new one is created.
     * @returns The translation component of m.
     */
    function getTranslation(m, dst) {
        const newDst = (dst ?? new Ctor(3));
        newDst[0] = m[12];
        newDst[1] = m[13];
        newDst[2] = m[14];
        return newDst;
    }
    /**
     * Returns an axis of a 4x4 matrix as a vector with 3 entries
     * @param m - The matrix.
     * @param axis - The axis 0 = x, 1 = y, 2 = z;
     * @returns The axis component of m.
     */
    function getAxis(m, axis, dst) {
        const newDst = (dst ?? new Ctor(3));
        const off = axis * 4;
        newDst[0] = m[off + 0];
        newDst[1] = m[off + 1];
        newDst[2] = m[off + 2];
        return newDst;
    }
    /**
     * Returns the scaling component of the matrix
     * @param m - The Matrix
     * @param dst - The vector to set. If not passed a new one is created.
     */
    function getScaling(m, dst) {
        const newDst = (dst ?? new Ctor(3));
        const xx = m[0];
        const xy = m[1];
        const xz = m[2];
        const yx = m[4];
        const yy = m[5];
        const yz = m[6];
        const zx = m[8];
        const zy = m[9];
        const zz = m[10];
        newDst[0] = Math.sqrt(xx * xx + xy * xy + xz * xz);
        newDst[1] = Math.sqrt(yx * yx + yy * yy + yz * yz);
        newDst[2] = Math.sqrt(zx * zx + zy * zy + zz * zz);
        return newDst;
    }
    /**
     * Rotate a 3D vector around the x-axis
     *
     * @param {ReadonlyVec3} a The vec3 point to rotate
     * @param {ReadonlyVec3} b The origin of the rotation
     * @param {Number} rad The angle of rotation in radians
     * @param dst - The vector to set. If not passed a new one is created.
     * @returns the rotated vector
     */
    function rotateX(a, b, rad, dst) {
        const newDst = (dst ?? new Ctor(3));
        const p = [];
        const r = [];
        //Translate point to the origin
        p[0] = a[0] - b[0];
        p[1] = a[1] - b[1];
        p[2] = a[2] - b[2];
        //perform rotation
        r[0] = p[0];
        r[1] = p[1] * Math.cos(rad) - p[2] * Math.sin(rad);
        r[2] = p[1] * Math.sin(rad) + p[2] * Math.cos(rad);
        //translate to correct position
        newDst[0] = r[0] + b[0];
        newDst[1] = r[1] + b[1];
        newDst[2] = r[2] + b[2];
        return newDst;
    }
    /**
     * Rotate a 3D vector around the y-axis
     *
     * @param {ReadonlyVec3} a The vec3 point to rotate
     * @param {ReadonlyVec3} b The origin of the rotation
     * @param {Number} rad The angle of rotation in radians
     * @param dst - The vector to set. If not passed a new one is created.
     * @returns the rotated vector
     */
    function rotateY(a, b, rad, dst) {
        const newDst = (dst ?? new Ctor(3));
        const p = [];
        const r = [];
        // translate point to the origin
        p[0] = a[0] - b[0];
        p[1] = a[1] - b[1];
        p[2] = a[2] - b[2];
        // perform rotation
        r[0] = p[2] * Math.sin(rad) + p[0] * Math.cos(rad);
        r[1] = p[1];
        r[2] = p[2] * Math.cos(rad) - p[0] * Math.sin(rad);
        // translate to correct position
        newDst[0] = r[0] + b[0];
        newDst[1] = r[1] + b[1];
        newDst[2] = r[2] + b[2];
        return newDst;
    }
    /**
     * Rotate a 3D vector around the z-axis
     *
     * @param {ReadonlyVec3} a The vec3 point to rotate
     * @param {ReadonlyVec3} b The origin of the rotation
     * @param {Number} rad The angle of rotation in radians
     * @param dst - The vector to set. If not passed a new one is created.
     * @returns {vec3} out
     */
    function rotateZ(a, b, rad, dst) {
        const newDst = (dst ?? new Ctor(3));
        const p = [];
        const r = [];
        // translate point to the origin
        p[0] = a[0] - b[0];
        p[1] = a[1] - b[1];
        p[2] = a[2] - b[2];
        // perform rotation
        r[0] = p[0] * Math.cos(rad) - p[1] * Math.sin(rad);
        r[1] = p[0] * Math.sin(rad) + p[1] * Math.cos(rad);
        r[2] = p[2];
        // translate to correct position
        newDst[0] = r[0] + b[0];
        newDst[1] = r[1] + b[1];
        newDst[2] = r[2] + b[2];
        return newDst;
    }
    /**
     * Treat a 3D vector as a direction and set it's length
     *
     * @param a The vec3 to lengthen
     * @param len The length of the resulting vector
     * @returns The lengthened vector
     */
    function setLength(a, len, dst) {
        const newDst = (dst ?? new Ctor(3));
        normalize(a, newDst);
        return mulScalar(newDst, len, newDst);
    }
    /**
     * Ensure a vector is not longer than a max length
     *
     * @param a The vec3 to limit
     * @param maxLen The longest length of the resulting vector
     * @returns The vector, shortened to maxLen if it's too long
     */
    function truncate(a, maxLen, dst) {
        const newDst = (dst ?? new Ctor(3));
        if (length(a) > maxLen) {
            return setLength(a, maxLen, newDst);
        }
        return copy(a, newDst);
    }
    /**
     * Return the vector exactly between 2 endpoint vectors
     *
     * @param a Endpoint 1
     * @param b Endpoint 2
     * @returns The vector exactly residing between endpoints 1 and 2
     */
    function midpoint(a, b, dst) {
        const newDst = (dst ?? new Ctor(3));
        return lerp(a, b, 0.5, newDst);
    }
    return {
        create,
        fromValues,
        set,
        ceil,
        floor,
        round,
        clamp,
        add,
        addScaled,
        angle,
        subtract,
        sub,
        equalsApproximately,
        equals,
        lerp,
        lerpV,
        max,
        min,
        mulScalar,
        scale,
        divScalar,
        inverse,
        invert,
        cross,
        dot,
        length,
        len,
        lengthSq,
        lenSq,
        distance,
        dist,
        distanceSq,
        distSq,
        normalize,
        negate,
        copy,
        clone,
        multiply,
        mul,
        divide,
        div,
        random,
        zero,
        transformMat4,
        transformMat4Upper3x3,
        transformMat3,
        transformQuat,
        getTranslation,
        getAxis,
        getScaling,
        rotateX,
        rotateY,
        rotateZ,
        setLength,
        truncate,
        midpoint,
    };
}
const cache$4 = new Map();
function getAPI$4(Ctor) {
    let api = cache$4.get(Ctor);
    if (!api) {
        api = getAPIImpl$4(Ctor);
        cache$4.set(Ctor, api);
    }
    return api;
}

/*
 * Copyright 2022 Gregg Tavares
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/**
 * Generates a typed API for Mat3
 * */
function getAPIImpl$3(Ctor) {
    const vec2 = getAPI$5(Ctor);
    const vec3 = getAPI$4(Ctor);
    /**
     * Create a Mat3 from values
     *
     * Note: Since passing in a raw JavaScript array
     * is valid in all circumstances, if you want to
     * force a JavaScript array into a Mat3's specified type
     * it would be faster to use
     *
     * ```
     * const m = mat3.clone(someJSArray);
     * ```
     *
     * @param v0 - value for element 0
     * @param v1 - value for element 1
     * @param v2 - value for element 2
     * @param v3 - value for element 3
     * @param v4 - value for element 4
     * @param v5 - value for element 5
     * @param v6 - value for element 6
     * @param v7 - value for element 7
     * @param v8 - value for element 8
     * @returns matrix created from values.
     */
    function create(v0, v1, v2, v3, v4, v5, v6, v7, v8) {
        const newDst = new Ctor(12);
        // to make the array homogenous
        newDst[3] = 0;
        newDst[7] = 0;
        newDst[11] = 0;
        if (v0 !== undefined) {
            newDst[0] = v0;
            if (v1 !== undefined) {
                newDst[1] = v1;
                if (v2 !== undefined) {
                    newDst[2] = v2;
                    if (v3 !== undefined) {
                        newDst[4] = v3;
                        if (v4 !== undefined) {
                            newDst[5] = v4;
                            if (v5 !== undefined) {
                                newDst[6] = v5;
                                if (v6 !== undefined) {
                                    newDst[8] = v6;
                                    if (v7 !== undefined) {
                                        newDst[9] = v7;
                                        if (v8 !== undefined) {
                                            newDst[10] = v8;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return newDst;
    }
    /**
     * Sets the values of a Mat3
     * Also see {@link mat3.create} and {@link mat3.copy}
     *
     * @param v0 - value for element 0
     * @param v1 - value for element 1
     * @param v2 - value for element 2
     * @param v3 - value for element 3
     * @param v4 - value for element 4
     * @param v5 - value for element 5
     * @param v6 - value for element 6
     * @param v7 - value for element 7
     * @param v8 - value for element 8
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns Mat3 set from values.
     */
    function set(v0, v1, v2, v3, v4, v5, v6, v7, v8, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = v0;
        newDst[1] = v1;
        newDst[2] = v2;
        newDst[3] = 0;
        newDst[4] = v3;
        newDst[5] = v4;
        newDst[6] = v5;
        newDst[7] = 0;
        newDst[8] = v6;
        newDst[9] = v7;
        newDst[10] = v8;
        newDst[11] = 0;
        return newDst;
    }
    /**
     * Creates a Mat3 from the upper left 3x3 part of a Mat4
     * @param m4 - source matrix
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns Mat3 made from m4
     */
    function fromMat4(m4, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = m4[0];
        newDst[1] = m4[1];
        newDst[2] = m4[2];
        newDst[3] = 0;
        newDst[4] = m4[4];
        newDst[5] = m4[5];
        newDst[6] = m4[6];
        newDst[7] = 0;
        newDst[8] = m4[8];
        newDst[9] = m4[9];
        newDst[10] = m4[10];
        newDst[11] = 0;
        return newDst;
    }
    /**
     * Creates a Mat3 rotation matrix from a quaternion
     * @param q - quaternion to create matrix from
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns Mat3 made from q
     */
    function fromQuat(q, dst) {
        const newDst = (dst ?? new Ctor(12));
        const x = q[0];
        const y = q[1];
        const z = q[2];
        const w = q[3];
        const x2 = x + x;
        const y2 = y + y;
        const z2 = z + z;
        const xx = x * x2;
        const yx = y * x2;
        const yy = y * y2;
        const zx = z * x2;
        const zy = z * y2;
        const zz = z * z2;
        const wx = w * x2;
        const wy = w * y2;
        const wz = w * z2;
        newDst[0] = 1 - yy - zz;
        newDst[1] = yx + wz;
        newDst[2] = zx - wy;
        newDst[3] = 0;
        newDst[4] = yx - wz;
        newDst[5] = 1 - xx - zz;
        newDst[6] = zy + wx;
        newDst[7] = 0;
        newDst[8] = zx + wy;
        newDst[9] = zy - wx;
        newDst[10] = 1 - xx - yy;
        newDst[11] = 0;
        return newDst;
    }
    /**
     * Negates a matrix.
     * @param m - The matrix.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns -m.
     */
    function negate(m, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = -m[0];
        newDst[1] = -m[1];
        newDst[2] = -m[2];
        newDst[4] = -m[4];
        newDst[5] = -m[5];
        newDst[6] = -m[6];
        newDst[8] = -m[8];
        newDst[9] = -m[9];
        newDst[10] = -m[10];
        return newDst;
    }
    /**
     * Copies a matrix. (same as {@link mat3.clone})
     * Also see {@link mat3.create} and {@link mat3.set}
     * @param m - The matrix.
     * @param dst - The matrix. If not passed a new one is created.
     * @returns A copy of m.
     */
    function copy(m, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = m[0];
        newDst[1] = m[1];
        newDst[2] = m[2];
        newDst[4] = m[4];
        newDst[5] = m[5];
        newDst[6] = m[6];
        newDst[8] = m[8];
        newDst[9] = m[9];
        newDst[10] = m[10];
        return newDst;
    }
    /**
     * Copies a matrix (same as {@link mat3.copy})
     * Also see {@link mat3.create} and {@link mat3.set}
     * @param m - The matrix.
     * @param dst - The matrix. If not passed a new one is created.
     * @returns A copy of m.
     */
    const clone = copy;
    /**
     * Check if 2 matrices are approximately equal
     * @param a Operand matrix.
     * @param b Operand matrix.
     * @returns true if matrices are approximately equal
     */
    function equalsApproximately(a, b) {
        return Math.abs(a[0] - b[0]) < EPSILON &&
            Math.abs(a[1] - b[1]) < EPSILON &&
            Math.abs(a[2] - b[2]) < EPSILON &&
            Math.abs(a[4] - b[4]) < EPSILON &&
            Math.abs(a[5] - b[5]) < EPSILON &&
            Math.abs(a[6] - b[6]) < EPSILON &&
            Math.abs(a[8] - b[8]) < EPSILON &&
            Math.abs(a[9] - b[9]) < EPSILON &&
            Math.abs(a[10] - b[10]) < EPSILON;
    }
    /**
     * Check if 2 matrices are exactly equal
     * @param a Operand matrix.
     * @param b Operand matrix.
     * @returns true if matrices are exactly equal
     */
    function equals(a, b) {
        return a[0] === b[0] &&
            a[1] === b[1] &&
            a[2] === b[2] &&
            a[4] === b[4] &&
            a[5] === b[5] &&
            a[6] === b[6] &&
            a[8] === b[8] &&
            a[9] === b[9] &&
            a[10] === b[10];
    }
    /**
     * Creates a 3-by-3 identity matrix.
     *
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns A 3-by-3 identity matrix.
     */
    function identity(dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = 1;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[4] = 0;
        newDst[5] = 1;
        newDst[6] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = 1;
        return newDst;
    }
    /**
     * Takes the transpose of a matrix.
     * @param m - The matrix.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The transpose of m.
     */
    function transpose(m, dst) {
        const newDst = (dst ?? new Ctor(12));
        if (newDst === m) {
            let t;
            // 0 1 2
            // 4 5 6
            // 8 9 10
            t = m[1];
            m[1] = m[4];
            m[4] = t;
            t = m[2];
            m[2] = m[8];
            m[8] = t;
            t = m[6];
            m[6] = m[9];
            m[9] = t;
            return newDst;
        }
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        newDst[0] = m00;
        newDst[1] = m10;
        newDst[2] = m20;
        newDst[4] = m01;
        newDst[5] = m11;
        newDst[6] = m21;
        newDst[8] = m02;
        newDst[9] = m12;
        newDst[10] = m22;
        return newDst;
    }
    /**
     * Computes the inverse of a 3-by-3 matrix.
     * @param m - The matrix.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The inverse of m.
     */
    function inverse(m, dst) {
        const newDst = (dst ?? new Ctor(12));
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        const b01 = m22 * m11 - m12 * m21;
        const b11 = -m22 * m10 + m12 * m20;
        const b21 = m21 * m10 - m11 * m20;
        const invDet = 1 / (m00 * b01 + m01 * b11 + m02 * b21);
        newDst[0] = b01 * invDet;
        newDst[1] = (-m22 * m01 + m02 * m21) * invDet;
        newDst[2] = (m12 * m01 - m02 * m11) * invDet;
        newDst[4] = b11 * invDet;
        newDst[5] = (m22 * m00 - m02 * m20) * invDet;
        newDst[6] = (-m12 * m00 + m02 * m10) * invDet;
        newDst[8] = b21 * invDet;
        newDst[9] = (-m21 * m00 + m01 * m20) * invDet;
        newDst[10] = (m11 * m00 - m01 * m10) * invDet;
        return newDst;
    }
    /**
     * Compute the determinant of a matrix
     * @param m - the matrix
     * @returns the determinant
     */
    function determinant(m) {
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        return m00 * (m11 * m22 - m21 * m12) -
            m10 * (m01 * m22 - m21 * m02) +
            m20 * (m01 * m12 - m11 * m02);
    }
    /**
     * Computes the inverse of a 3-by-3 matrix. (same as inverse)
     * @param m - The matrix.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The inverse of m.
     */
    const invert = inverse;
    /**
     * Multiplies two 3-by-3 matrices with a on the left and b on the right
     * @param a - The matrix on the left.
     * @param b - The matrix on the right.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The matrix product of a and b.
     */
    function multiply(a, b, dst) {
        const newDst = (dst ?? new Ctor(12));
        const a00 = a[0];
        const a01 = a[1];
        const a02 = a[2];
        const a10 = a[4 + 0];
        const a11 = a[4 + 1];
        const a12 = a[4 + 2];
        const a20 = a[8 + 0];
        const a21 = a[8 + 1];
        const a22 = a[8 + 2];
        const b00 = b[0];
        const b01 = b[1];
        const b02 = b[2];
        const b10 = b[4 + 0];
        const b11 = b[4 + 1];
        const b12 = b[4 + 2];
        const b20 = b[8 + 0];
        const b21 = b[8 + 1];
        const b22 = b[8 + 2];
        newDst[0] = a00 * b00 + a10 * b01 + a20 * b02;
        newDst[1] = a01 * b00 + a11 * b01 + a21 * b02;
        newDst[2] = a02 * b00 + a12 * b01 + a22 * b02;
        newDst[4] = a00 * b10 + a10 * b11 + a20 * b12;
        newDst[5] = a01 * b10 + a11 * b11 + a21 * b12;
        newDst[6] = a02 * b10 + a12 * b11 + a22 * b12;
        newDst[8] = a00 * b20 + a10 * b21 + a20 * b22;
        newDst[9] = a01 * b20 + a11 * b21 + a21 * b22;
        newDst[10] = a02 * b20 + a12 * b21 + a22 * b22;
        return newDst;
    }
    /**
     * Multiplies two 3-by-3 matrices with a on the left and b on the right (same as multiply)
     * @param a - The matrix on the left.
     * @param b - The matrix on the right.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The matrix product of a and b.
     */
    const mul = multiply;
    /**
     * Sets the translation component of a 3-by-3 matrix to the given
     * vector.
     * @param a - The matrix.
     * @param v - The vector.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The matrix with translation set.
     */
    function setTranslation(a, v, dst) {
        const newDst = (dst ?? identity());
        if (a !== newDst) {
            newDst[0] = a[0];
            newDst[1] = a[1];
            newDst[2] = a[2];
            newDst[4] = a[4];
            newDst[5] = a[5];
            newDst[6] = a[6];
        }
        newDst[8] = v[0];
        newDst[9] = v[1];
        newDst[10] = 1;
        return newDst;
    }
    /**
     * Returns the translation component of a 3-by-3 matrix as a vector with 3
     * entries.
     * @param m - The matrix.
     * @param dst - vector to hold result. If not passed a new one is created.
     * @returns The translation component of m.
     */
    function getTranslation(m, dst) {
        const newDst = (dst ?? vec2.create());
        newDst[0] = m[8];
        newDst[1] = m[9];
        return newDst;
    }
    /**
     * Returns an axis of a 3x3 matrix as a vector with 2 entries
     * @param m - The matrix.
     * @param axis - The axis 0 = x, 1 = y,
     * @returns The axis component of m.
     */
    function getAxis(m, axis, dst) {
        const newDst = (dst ?? vec2.create());
        const off = axis * 4;
        newDst[0] = m[off + 0];
        newDst[1] = m[off + 1];
        return newDst;
    }
    /**
     * Sets an axis of a 3x3 matrix as a vector with 2 entries
     * @param m - The matrix.
     * @param v - the axis vector
     * @param axis - The axis  0 = x, 1 = y;
     * @param dst - The matrix to set. If not passed a new one is created.
     * @returns The matrix with axis set.
     */
    function setAxis(m, v, axis, dst) {
        const newDst = (dst === m ? m : copy(m, dst));
        const off = axis * 4;
        newDst[off + 0] = v[0];
        newDst[off + 1] = v[1];
        return newDst;
    }
    /**
     * Returns the "2d" scaling component of the matrix
     * @param m - The Matrix
     * @param dst - The vector to set. If not passed a new one is created.
     */
    function getScaling(m, dst) {
        const newDst = (dst ?? vec2.create());
        const xx = m[0];
        const xy = m[1];
        const yx = m[4];
        const yy = m[5];
        newDst[0] = Math.sqrt(xx * xx + xy * xy);
        newDst[1] = Math.sqrt(yx * yx + yy * yy);
        return newDst;
    }
    /**
     * Returns the "3d" scaling component of the matrix
     * @param m - The Matrix
     * @param dst - The vector to set. If not passed a new one is created.
     */
    function get3DScaling(m, dst) {
        const newDst = (dst ?? vec3.create());
        const xx = m[0];
        const xy = m[1];
        const xz = m[2];
        const yx = m[4];
        const yy = m[5];
        const yz = m[6];
        const zx = m[8];
        const zy = m[9];
        const zz = m[10];
        newDst[0] = Math.sqrt(xx * xx + xy * xy + xz * xz);
        newDst[1] = Math.sqrt(yx * yx + yy * yy + yz * yz);
        newDst[2] = Math.sqrt(zx * zx + zy * zy + zz * zz);
        return newDst;
    }
    /**
     * Creates a 3-by-3 matrix which translates by the given vector v.
     * @param v - The vector by which to translate.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The translation matrix.
     */
    function translation(v, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = 1;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[4] = 0;
        newDst[5] = 1;
        newDst[6] = 0;
        newDst[8] = v[0];
        newDst[9] = v[1];
        newDst[10] = 1;
        return newDst;
    }
    /**
     * Translates the given 3-by-3 matrix by the given vector v.
     * @param m - The matrix.
     * @param v - The vector by which to translate.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The translated matrix.
     */
    function translate(m, v, dst) {
        const newDst = (dst ?? new Ctor(12));
        const v0 = v[0];
        const v1 = v[1];
        const m00 = m[0];
        const m01 = m[1];
        const m02 = m[2];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        if (m !== newDst) {
            newDst[0] = m00;
            newDst[1] = m01;
            newDst[2] = m02;
            newDst[4] = m10;
            newDst[5] = m11;
            newDst[6] = m12;
        }
        newDst[8] = m00 * v0 + m10 * v1 + m20;
        newDst[9] = m01 * v0 + m11 * v1 + m21;
        newDst[10] = m02 * v0 + m12 * v1 + m22;
        return newDst;
    }
    /**
     * Creates a 3-by-3 matrix which rotates  by the given angle.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotation matrix.
     */
    function rotation(angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(12));
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = c;
        newDst[1] = s;
        newDst[2] = 0;
        newDst[4] = -s;
        newDst[5] = c;
        newDst[6] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = 1;
        return newDst;
    }
    /**
     * Rotates the given 3-by-3 matrix  by the given angle.
     * @param m - The matrix.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotated matrix.
     */
    function rotate(m, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(12));
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = c * m00 + s * m10;
        newDst[1] = c * m01 + s * m11;
        newDst[2] = c * m02 + s * m12;
        newDst[4] = c * m10 - s * m00;
        newDst[5] = c * m11 - s * m01;
        newDst[6] = c * m12 - s * m02;
        if (m !== newDst) {
            newDst[8] = m[8];
            newDst[9] = m[9];
            newDst[10] = m[10];
        }
        return newDst;
    }
    /**
     * Creates a 3-by-3 matrix which rotates around the x-axis by the given angle.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotation matrix.
     */
    function rotationX(angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(12));
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = 1;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[4] = 0;
        newDst[5] = c;
        newDst[6] = s;
        newDst[8] = 0;
        newDst[9] = -s;
        newDst[10] = c;
        return newDst;
    }
    /**
     * Rotates the given 3-by-3 matrix around the x-axis by the given
     * angle.
     * @param m - The matrix.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotated matrix.
     */
    function rotateX(m, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(12));
        const m10 = m[4];
        const m11 = m[5];
        const m12 = m[6];
        const m20 = m[8];
        const m21 = m[9];
        const m22 = m[10];
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[4] = c * m10 + s * m20;
        newDst[5] = c * m11 + s * m21;
        newDst[6] = c * m12 + s * m22;
        newDst[8] = c * m20 - s * m10;
        newDst[9] = c * m21 - s * m11;
        newDst[10] = c * m22 - s * m12;
        if (m !== newDst) {
            newDst[0] = m[0];
            newDst[1] = m[1];
            newDst[2] = m[2];
        }
        return newDst;
    }
    /**
     * Creates a 3-by-3 matrix which rotates around the y-axis by the given angle.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotation matrix.
     */
    function rotationY(angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(12));
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = c;
        newDst[1] = 0;
        newDst[2] = -s;
        newDst[4] = 0;
        newDst[5] = 1;
        newDst[6] = 0;
        newDst[8] = s;
        newDst[9] = 0;
        newDst[10] = c;
        return newDst;
    }
    /**
     * Rotates the given 3-by-3 matrix around the y-axis by the given
     * angle.
     * @param m - The matrix.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotated matrix.
     */
    function rotateY(m, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(12));
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = c * m00 - s * m20;
        newDst[1] = c * m01 - s * m21;
        newDst[2] = c * m02 - s * m22;
        newDst[8] = c * m20 + s * m00;
        newDst[9] = c * m21 + s * m01;
        newDst[10] = c * m22 + s * m02;
        if (m !== newDst) {
            newDst[4] = m[4];
            newDst[5] = m[5];
            newDst[6] = m[6];
        }
        return newDst;
    }
    /**
     * Creates a 3-by-3 matrix which rotates around the z-axis by the given angle.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotation matrix.
     */
    const rotationZ = rotation;
    /**
     * Rotates the given 3-by-3 matrix around the z-axis by the given
     * angle.
     * @param m - The matrix.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotated matrix.
     */
    const rotateZ = rotate;
    /**
     * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
     * the corresponding entry in the given vector; assumes the vector has two
     * entries.
     * @param v - A vector of
     *     2 entries specifying the factor by which to scale in each dimension.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaling matrix.
     */
    function scaling(v, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = v[0];
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[4] = 0;
        newDst[5] = v[1];
        newDst[6] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = 1;
        return newDst;
    }
    /**
     * Scales the given 3-by-3 matrix in each dimension by an amount
     * given by the corresponding entry in the given vector; assumes the vector has
     * two entries.
     * @param m - The matrix to be modified.
     * @param v - A vector of 2 entries specifying the
     *     factor by which to scale in each dimension.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaled matrix.
     */
    function scale(m, v, dst) {
        const newDst = (dst ?? new Ctor(12));
        const v0 = v[0];
        const v1 = v[1];
        newDst[0] = v0 * m[0 * 4 + 0];
        newDst[1] = v0 * m[0 * 4 + 1];
        newDst[2] = v0 * m[0 * 4 + 2];
        newDst[4] = v1 * m[1 * 4 + 0];
        newDst[5] = v1 * m[1 * 4 + 1];
        newDst[6] = v1 * m[1 * 4 + 2];
        if (m !== newDst) {
            newDst[8] = m[8];
            newDst[9] = m[9];
            newDst[10] = m[10];
        }
        return newDst;
    }
    /**
     * Creates a 3-by-3 matrix which scales in each dimension by an amount given by
     * the corresponding entry in the given vector; assumes the vector has three
     * entries.
     * @param v - A vector of
     *     3 entries specifying the factor by which to scale in each dimension.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaling matrix.
     */
    function scaling3D(v, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = v[0];
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[4] = 0;
        newDst[5] = v[1];
        newDst[6] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = v[2];
        return newDst;
    }
    /**
     * Scales the given 3-by-3 matrix in each dimension by an amount
     * given by the corresponding entry in the given vector; assumes the vector has
     * three entries.
     * @param m - The matrix to be modified.
     * @param v - A vector of 3 entries specifying the
     *     factor by which to scale in each dimension.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaled matrix.
     */
    function scale3D(m, v, dst) {
        const newDst = (dst ?? new Ctor(12));
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        newDst[0] = v0 * m[0 * 4 + 0];
        newDst[1] = v0 * m[0 * 4 + 1];
        newDst[2] = v0 * m[0 * 4 + 2];
        newDst[4] = v1 * m[1 * 4 + 0];
        newDst[5] = v1 * m[1 * 4 + 1];
        newDst[6] = v1 * m[1 * 4 + 2];
        newDst[8] = v2 * m[2 * 4 + 0];
        newDst[9] = v2 * m[2 * 4 + 1];
        newDst[10] = v2 * m[2 * 4 + 2];
        return newDst;
    }
    /**
     * Creates a 3-by-3 matrix which scales uniformly in the X and Y dimensions
     * @param s - Amount to scale
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaling matrix.
     */
    function uniformScaling(s, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = s;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[4] = 0;
        newDst[5] = s;
        newDst[6] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = 1;
        return newDst;
    }
    /**
     * Scales the given 3-by-3 matrix in the X and Y dimension by an amount
     * given.
     * @param m - The matrix to be modified.
     * @param s - Amount to scale.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaled matrix.
     */
    function uniformScale(m, s, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = s * m[0 * 4 + 0];
        newDst[1] = s * m[0 * 4 + 1];
        newDst[2] = s * m[0 * 4 + 2];
        newDst[4] = s * m[1 * 4 + 0];
        newDst[5] = s * m[1 * 4 + 1];
        newDst[6] = s * m[1 * 4 + 2];
        if (m !== newDst) {
            newDst[8] = m[8];
            newDst[9] = m[9];
            newDst[10] = m[10];
        }
        return newDst;
    }
    /**
     * Creates a 3-by-3 matrix which scales uniformly in each dimension
     * @param s - Amount to scale
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaling matrix.
     */
    function uniformScaling3D(s, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = s;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[4] = 0;
        newDst[5] = s;
        newDst[6] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = s;
        return newDst;
    }
    /**
     * Scales the given 3-by-3 matrix in each dimension by an amount
     * given.
     * @param m - The matrix to be modified.
     * @param s - Amount to scale.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaled matrix.
     */
    function uniformScale3D(m, s, dst) {
        const newDst = (dst ?? new Ctor(12));
        newDst[0] = s * m[0 * 4 + 0];
        newDst[1] = s * m[0 * 4 + 1];
        newDst[2] = s * m[0 * 4 + 2];
        newDst[4] = s * m[1 * 4 + 0];
        newDst[5] = s * m[1 * 4 + 1];
        newDst[6] = s * m[1 * 4 + 2];
        newDst[8] = s * m[2 * 4 + 0];
        newDst[9] = s * m[2 * 4 + 1];
        newDst[10] = s * m[2 * 4 + 2];
        return newDst;
    }
    return {
        clone,
        create,
        set,
        fromMat4,
        fromQuat,
        negate,
        copy,
        equalsApproximately,
        equals,
        identity,
        transpose,
        inverse,
        invert,
        determinant,
        mul,
        multiply,
        setTranslation,
        getTranslation,
        getAxis,
        setAxis,
        getScaling,
        get3DScaling,
        translation,
        translate,
        rotation,
        rotate,
        rotationX,
        rotateX,
        rotationY,
        rotateY,
        rotationZ,
        rotateZ,
        scaling,
        scale,
        uniformScaling,
        uniformScale,
        scaling3D,
        scale3D,
        uniformScaling3D,
        uniformScale3D,
    };
}
const cache$3 = new Map();
function getAPI$3(Ctor) {
    let api = cache$3.get(Ctor);
    if (!api) {
        api = getAPIImpl$3(Ctor);
        cache$3.set(Ctor, api);
    }
    return api;
}

/**
 * Generates a typed API for Mat4
 * */
function getAPIImpl$2(Ctor) {
    const vec3 = getAPI$4(Ctor);
    /**
     * 4x4 Matrix math math functions.
     *
     * Almost all functions take an optional `newDst` argument. If it is not passed in the
     * functions will create a new matrix. In other words you can do this
     *
     *     const mat = mat4.translation([1, 2, 3]);  // Creates a new translation matrix
     *
     * or
     *
     *     const mat = mat4.create();
     *     mat4.translation([1, 2, 3], mat);  // Puts translation matrix in mat.
     *
     * The first style is often easier but depending on where it's used it generates garbage where
     * as there is almost never allocation with the second style.
     *
     * It is always save to pass any matrix as the destination. So for example
     *
     *     const mat = mat4.identity();
     *     const trans = mat4.translation([1, 2, 3]);
     *     mat4.multiply(mat, trans, mat);  // Multiplies mat * trans and puts result in mat.
     *
     */
    /**
     * Create a Mat4 from values
     *
     * Note: Since passing in a raw JavaScript array
     * is valid in all circumstances, if you want to
     * force a JavaScript array into a Mat4's specified type
     * it would be faster to use
     *
     * ```
     * const m = mat4.clone(someJSArray);
     * ```
     *
     * @param v0 - value for element 0
     * @param v1 - value for element 1
     * @param v2 - value for element 2
     * @param v3 - value for element 3
     * @param v4 - value for element 4
     * @param v5 - value for element 5
     * @param v6 - value for element 6
     * @param v7 - value for element 7
     * @param v8 - value for element 8
     * @param v9 - value for element 9
     * @param v10 - value for element 10
     * @param v11 - value for element 11
     * @param v12 - value for element 12
     * @param v13 - value for element 13
     * @param v14 - value for element 14
     * @param v15 - value for element 15
     * @returns created from values.
     */
    function create(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) {
        const newDst = new Ctor(16);
        if (v0 !== undefined) {
            newDst[0] = v0;
            if (v1 !== undefined) {
                newDst[1] = v1;
                if (v2 !== undefined) {
                    newDst[2] = v2;
                    if (v3 !== undefined) {
                        newDst[3] = v3;
                        if (v4 !== undefined) {
                            newDst[4] = v4;
                            if (v5 !== undefined) {
                                newDst[5] = v5;
                                if (v6 !== undefined) {
                                    newDst[6] = v6;
                                    if (v7 !== undefined) {
                                        newDst[7] = v7;
                                        if (v8 !== undefined) {
                                            newDst[8] = v8;
                                            if (v9 !== undefined) {
                                                newDst[9] = v9;
                                                if (v10 !== undefined) {
                                                    newDst[10] = v10;
                                                    if (v11 !== undefined) {
                                                        newDst[11] = v11;
                                                        if (v12 !== undefined) {
                                                            newDst[12] = v12;
                                                            if (v13 !== undefined) {
                                                                newDst[13] = v13;
                                                                if (v14 !== undefined) {
                                                                    newDst[14] = v14;
                                                                    if (v15 !== undefined) {
                                                                        newDst[15] = v15;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return newDst;
    }
    /**
     * Sets the values of a Mat4
     * Also see {@link mat4.create} and {@link mat4.copy}
     *
     * @param v0 - value for element 0
     * @param v1 - value for element 1
     * @param v2 - value for element 2
     * @param v3 - value for element 3
     * @param v4 - value for element 4
     * @param v5 - value for element 5
     * @param v6 - value for element 6
     * @param v7 - value for element 7
     * @param v8 - value for element 8
     * @param v9 - value for element 9
     * @param v10 - value for element 10
     * @param v11 - value for element 11
     * @param v12 - value for element 12
     * @param v13 - value for element 13
     * @param v14 - value for element 14
     * @param v15 - value for element 15
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns Mat4 created from values.
     */
    function set(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = v0;
        newDst[1] = v1;
        newDst[2] = v2;
        newDst[3] = v3;
        newDst[4] = v4;
        newDst[5] = v5;
        newDst[6] = v6;
        newDst[7] = v7;
        newDst[8] = v8;
        newDst[9] = v9;
        newDst[10] = v10;
        newDst[11] = v11;
        newDst[12] = v12;
        newDst[13] = v13;
        newDst[14] = v14;
        newDst[15] = v15;
        return newDst;
    }
    /**
     * Creates a Mat4 from a Mat3
     * @param m3 - source matrix
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns Mat4 made from m3
     */
    function fromMat3(m3, dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = m3[0];
        newDst[1] = m3[1];
        newDst[2] = m3[2];
        newDst[3] = 0;
        newDst[4] = m3[4];
        newDst[5] = m3[5];
        newDst[6] = m3[6];
        newDst[7] = 0;
        newDst[8] = m3[8];
        newDst[9] = m3[9];
        newDst[10] = m3[10];
        newDst[11] = 0;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = 0;
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Creates a Mat4 rotation matrix from a quaternion
     * @param q - quaternion to create matrix from
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns Mat4 made from q
     */
    function fromQuat(q, dst) {
        const newDst = (dst ?? new Ctor(16));
        const x = q[0];
        const y = q[1];
        const z = q[2];
        const w = q[3];
        const x2 = x + x;
        const y2 = y + y;
        const z2 = z + z;
        const xx = x * x2;
        const yx = y * x2;
        const yy = y * y2;
        const zx = z * x2;
        const zy = z * y2;
        const zz = z * z2;
        const wx = w * x2;
        const wy = w * y2;
        const wz = w * z2;
        newDst[0] = 1 - yy - zz;
        newDst[1] = yx + wz;
        newDst[2] = zx - wy;
        newDst[3] = 0;
        newDst[4] = yx - wz;
        newDst[5] = 1 - xx - zz;
        newDst[6] = zy + wx;
        newDst[7] = 0;
        newDst[8] = zx + wy;
        newDst[9] = zy - wx;
        newDst[10] = 1 - xx - yy;
        newDst[11] = 0;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = 0;
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Negates a matrix.
     * @param m - The matrix.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns -m.
     */
    function negate(m, dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = -m[0];
        newDst[1] = -m[1];
        newDst[2] = -m[2];
        newDst[3] = -m[3];
        newDst[4] = -m[4];
        newDst[5] = -m[5];
        newDst[6] = -m[6];
        newDst[7] = -m[7];
        newDst[8] = -m[8];
        newDst[9] = -m[9];
        newDst[10] = -m[10];
        newDst[11] = -m[11];
        newDst[12] = -m[12];
        newDst[13] = -m[13];
        newDst[14] = -m[14];
        newDst[15] = -m[15];
        return newDst;
    }
    /**
     * Copies a matrix. (same as {@link mat4.clone})
     * Also see {@link mat4.create} and {@link mat4.set}
     * @param m - The matrix.
     * @param dst - The matrix. If not passed a new one is created.
     * @returns A copy of m.
     */
    function copy(m, dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = m[0];
        newDst[1] = m[1];
        newDst[2] = m[2];
        newDst[3] = m[3];
        newDst[4] = m[4];
        newDst[5] = m[5];
        newDst[6] = m[6];
        newDst[7] = m[7];
        newDst[8] = m[8];
        newDst[9] = m[9];
        newDst[10] = m[10];
        newDst[11] = m[11];
        newDst[12] = m[12];
        newDst[13] = m[13];
        newDst[14] = m[14];
        newDst[15] = m[15];
        return newDst;
    }
    /**
     * Copies a matrix (same as {@link mat4.copy})
     * Also see {@link mat4.create} and {@link mat4.set}
     * @param m - The matrix.
     * @param dst - The matrix. If not passed a new one is created.
     * @returns A copy of m.
     */
    const clone = copy;
    /**
     * Check if 2 matrices are approximately equal
     * @param a - Operand matrix.
     * @param b - Operand matrix.
     * @returns true if matrices are approximately equal
     */
    function equalsApproximately(a, b) {
        return Math.abs(a[0] - b[0]) < EPSILON &&
            Math.abs(a[1] - b[1]) < EPSILON &&
            Math.abs(a[2] - b[2]) < EPSILON &&
            Math.abs(a[3] - b[3]) < EPSILON &&
            Math.abs(a[4] - b[4]) < EPSILON &&
            Math.abs(a[5] - b[5]) < EPSILON &&
            Math.abs(a[6] - b[6]) < EPSILON &&
            Math.abs(a[7] - b[7]) < EPSILON &&
            Math.abs(a[8] - b[8]) < EPSILON &&
            Math.abs(a[9] - b[9]) < EPSILON &&
            Math.abs(a[10] - b[10]) < EPSILON &&
            Math.abs(a[11] - b[11]) < EPSILON &&
            Math.abs(a[12] - b[12]) < EPSILON &&
            Math.abs(a[13] - b[13]) < EPSILON &&
            Math.abs(a[14] - b[14]) < EPSILON &&
            Math.abs(a[15] - b[15]) < EPSILON;
    }
    /**
     * Check if 2 matrices are exactly equal
     * @param a - Operand matrix.
     * @param b - Operand matrix.
     * @returns true if matrices are exactly equal
     */
    function equals(a, b) {
        return a[0] === b[0] &&
            a[1] === b[1] &&
            a[2] === b[2] &&
            a[3] === b[3] &&
            a[4] === b[4] &&
            a[5] === b[5] &&
            a[6] === b[6] &&
            a[7] === b[7] &&
            a[8] === b[8] &&
            a[9] === b[9] &&
            a[10] === b[10] &&
            a[11] === b[11] &&
            a[12] === b[12] &&
            a[13] === b[13] &&
            a[14] === b[14] &&
            a[15] === b[15];
    }
    /**
     * Creates a 4-by-4 identity matrix.
     *
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns A 4-by-4 identity matrix.
     */
    function identity(dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = 1;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = 1;
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = 1;
        newDst[11] = 0;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = 0;
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Takes the transpose of a matrix.
     * @param m - The matrix.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The transpose of m.
     */
    function transpose(m, dst) {
        const newDst = (dst ?? new Ctor(16));
        if (newDst === m) {
            let t;
            t = m[1];
            m[1] = m[4];
            m[4] = t;
            t = m[2];
            m[2] = m[8];
            m[8] = t;
            t = m[3];
            m[3] = m[12];
            m[12] = t;
            t = m[6];
            m[6] = m[9];
            m[9] = t;
            t = m[7];
            m[7] = m[13];
            m[13] = t;
            t = m[11];
            m[11] = m[14];
            m[14] = t;
            return newDst;
        }
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m03 = m[0 * 4 + 3];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const m13 = m[1 * 4 + 3];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        const m23 = m[2 * 4 + 3];
        const m30 = m[3 * 4 + 0];
        const m31 = m[3 * 4 + 1];
        const m32 = m[3 * 4 + 2];
        const m33 = m[3 * 4 + 3];
        newDst[0] = m00;
        newDst[1] = m10;
        newDst[2] = m20;
        newDst[3] = m30;
        newDst[4] = m01;
        newDst[5] = m11;
        newDst[6] = m21;
        newDst[7] = m31;
        newDst[8] = m02;
        newDst[9] = m12;
        newDst[10] = m22;
        newDst[11] = m32;
        newDst[12] = m03;
        newDst[13] = m13;
        newDst[14] = m23;
        newDst[15] = m33;
        return newDst;
    }
    /**
     * Computes the inverse of a 4-by-4 matrix.
     * @param m - The matrix.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The inverse of m.
     */
    function inverse(m, dst) {
        const newDst = (dst ?? new Ctor(16));
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m03 = m[0 * 4 + 3];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const m13 = m[1 * 4 + 3];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        const m23 = m[2 * 4 + 3];
        const m30 = m[3 * 4 + 0];
        const m31 = m[3 * 4 + 1];
        const m32 = m[3 * 4 + 2];
        const m33 = m[3 * 4 + 3];
        const tmp0 = m22 * m33;
        const tmp1 = m32 * m23;
        const tmp2 = m12 * m33;
        const tmp3 = m32 * m13;
        const tmp4 = m12 * m23;
        const tmp5 = m22 * m13;
        const tmp6 = m02 * m33;
        const tmp7 = m32 * m03;
        const tmp8 = m02 * m23;
        const tmp9 = m22 * m03;
        const tmp10 = m02 * m13;
        const tmp11 = m12 * m03;
        const tmp12 = m20 * m31;
        const tmp13 = m30 * m21;
        const tmp14 = m10 * m31;
        const tmp15 = m30 * m11;
        const tmp16 = m10 * m21;
        const tmp17 = m20 * m11;
        const tmp18 = m00 * m31;
        const tmp19 = m30 * m01;
        const tmp20 = m00 * m21;
        const tmp21 = m20 * m01;
        const tmp22 = m00 * m11;
        const tmp23 = m10 * m01;
        const t0 = (tmp0 * m11 + tmp3 * m21 + tmp4 * m31) -
            (tmp1 * m11 + tmp2 * m21 + tmp5 * m31);
        const t1 = (tmp1 * m01 + tmp6 * m21 + tmp9 * m31) -
            (tmp0 * m01 + tmp7 * m21 + tmp8 * m31);
        const t2 = (tmp2 * m01 + tmp7 * m11 + tmp10 * m31) -
            (tmp3 * m01 + tmp6 * m11 + tmp11 * m31);
        const t3 = (tmp5 * m01 + tmp8 * m11 + tmp11 * m21) -
            (tmp4 * m01 + tmp9 * m11 + tmp10 * m21);
        const d = 1 / (m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3);
        newDst[0] = d * t0;
        newDst[1] = d * t1;
        newDst[2] = d * t2;
        newDst[3] = d * t3;
        newDst[4] = d * ((tmp1 * m10 + tmp2 * m20 + tmp5 * m30) -
            (tmp0 * m10 + tmp3 * m20 + tmp4 * m30));
        newDst[5] = d * ((tmp0 * m00 + tmp7 * m20 + tmp8 * m30) -
            (tmp1 * m00 + tmp6 * m20 + tmp9 * m30));
        newDst[6] = d * ((tmp3 * m00 + tmp6 * m10 + tmp11 * m30) -
            (tmp2 * m00 + tmp7 * m10 + tmp10 * m30));
        newDst[7] = d * ((tmp4 * m00 + tmp9 * m10 + tmp10 * m20) -
            (tmp5 * m00 + tmp8 * m10 + tmp11 * m20));
        newDst[8] = d * ((tmp12 * m13 + tmp15 * m23 + tmp16 * m33) -
            (tmp13 * m13 + tmp14 * m23 + tmp17 * m33));
        newDst[9] = d * ((tmp13 * m03 + tmp18 * m23 + tmp21 * m33) -
            (tmp12 * m03 + tmp19 * m23 + tmp20 * m33));
        newDst[10] = d * ((tmp14 * m03 + tmp19 * m13 + tmp22 * m33) -
            (tmp15 * m03 + tmp18 * m13 + tmp23 * m33));
        newDst[11] = d * ((tmp17 * m03 + tmp20 * m13 + tmp23 * m23) -
            (tmp16 * m03 + tmp21 * m13 + tmp22 * m23));
        newDst[12] = d * ((tmp14 * m22 + tmp17 * m32 + tmp13 * m12) -
            (tmp16 * m32 + tmp12 * m12 + tmp15 * m22));
        newDst[13] = d * ((tmp20 * m32 + tmp12 * m02 + tmp19 * m22) -
            (tmp18 * m22 + tmp21 * m32 + tmp13 * m02));
        newDst[14] = d * ((tmp18 * m12 + tmp23 * m32 + tmp15 * m02) -
            (tmp22 * m32 + tmp14 * m02 + tmp19 * m12));
        newDst[15] = d * ((tmp22 * m22 + tmp16 * m02 + tmp21 * m12) -
            (tmp20 * m12 + tmp23 * m22 + tmp17 * m02));
        return newDst;
    }
    /**
     * Compute the determinant of a matrix
     * @param m - the matrix
     * @returns the determinant
     */
    function determinant(m) {
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m03 = m[0 * 4 + 3];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const m13 = m[1 * 4 + 3];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        const m23 = m[2 * 4 + 3];
        const m30 = m[3 * 4 + 0];
        const m31 = m[3 * 4 + 1];
        const m32 = m[3 * 4 + 2];
        const m33 = m[3 * 4 + 3];
        const tmp0 = m22 * m33;
        const tmp1 = m32 * m23;
        const tmp2 = m12 * m33;
        const tmp3 = m32 * m13;
        const tmp4 = m12 * m23;
        const tmp5 = m22 * m13;
        const tmp6 = m02 * m33;
        const tmp7 = m32 * m03;
        const tmp8 = m02 * m23;
        const tmp9 = m22 * m03;
        const tmp10 = m02 * m13;
        const tmp11 = m12 * m03;
        const t0 = (tmp0 * m11 + tmp3 * m21 + tmp4 * m31) -
            (tmp1 * m11 + tmp2 * m21 + tmp5 * m31);
        const t1 = (tmp1 * m01 + tmp6 * m21 + tmp9 * m31) -
            (tmp0 * m01 + tmp7 * m21 + tmp8 * m31);
        const t2 = (tmp2 * m01 + tmp7 * m11 + tmp10 * m31) -
            (tmp3 * m01 + tmp6 * m11 + tmp11 * m31);
        const t3 = (tmp5 * m01 + tmp8 * m11 + tmp11 * m21) -
            (tmp4 * m01 + tmp9 * m11 + tmp10 * m21);
        return m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3;
    }
    /**
     * Computes the inverse of a 4-by-4 matrix. (same as inverse)
     * @param m - The matrix.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The inverse of m.
     */
    const invert = inverse;
    /**
     * Multiplies two 4-by-4 matrices with a on the left and b on the right
     * @param a - The matrix on the left.
     * @param b - The matrix on the right.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The matrix product of a and b.
     */
    function multiply(a, b, dst) {
        const newDst = (dst ?? new Ctor(16));
        const a00 = a[0];
        const a01 = a[1];
        const a02 = a[2];
        const a03 = a[3];
        const a10 = a[4 + 0];
        const a11 = a[4 + 1];
        const a12 = a[4 + 2];
        const a13 = a[4 + 3];
        const a20 = a[8 + 0];
        const a21 = a[8 + 1];
        const a22 = a[8 + 2];
        const a23 = a[8 + 3];
        const a30 = a[12 + 0];
        const a31 = a[12 + 1];
        const a32 = a[12 + 2];
        const a33 = a[12 + 3];
        const b00 = b[0];
        const b01 = b[1];
        const b02 = b[2];
        const b03 = b[3];
        const b10 = b[4 + 0];
        const b11 = b[4 + 1];
        const b12 = b[4 + 2];
        const b13 = b[4 + 3];
        const b20 = b[8 + 0];
        const b21 = b[8 + 1];
        const b22 = b[8 + 2];
        const b23 = b[8 + 3];
        const b30 = b[12 + 0];
        const b31 = b[12 + 1];
        const b32 = b[12 + 2];
        const b33 = b[12 + 3];
        newDst[0] = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
        newDst[1] = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;
        newDst[2] = a02 * b00 + a12 * b01 + a22 * b02 + a32 * b03;
        newDst[3] = a03 * b00 + a13 * b01 + a23 * b02 + a33 * b03;
        newDst[4] = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
        newDst[5] = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
        newDst[6] = a02 * b10 + a12 * b11 + a22 * b12 + a32 * b13;
        newDst[7] = a03 * b10 + a13 * b11 + a23 * b12 + a33 * b13;
        newDst[8] = a00 * b20 + a10 * b21 + a20 * b22 + a30 * b23;
        newDst[9] = a01 * b20 + a11 * b21 + a21 * b22 + a31 * b23;
        newDst[10] = a02 * b20 + a12 * b21 + a22 * b22 + a32 * b23;
        newDst[11] = a03 * b20 + a13 * b21 + a23 * b22 + a33 * b23;
        newDst[12] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
        newDst[13] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
        newDst[14] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
        newDst[15] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;
        return newDst;
    }
    /**
     * Multiplies two 4-by-4 matrices with a on the left and b on the right (same as multiply)
     * @param a - The matrix on the left.
     * @param b - The matrix on the right.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The matrix product of a and b.
     */
    const mul = multiply;
    /**
     * Sets the translation component of a 4-by-4 matrix to the given
     * vector.
     * @param a - The matrix.
     * @param v - The vector.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The matrix with translation set.
     */
    function setTranslation(a, v, dst) {
        const newDst = (dst ?? identity());
        if (a !== newDst) {
            newDst[0] = a[0];
            newDst[1] = a[1];
            newDst[2] = a[2];
            newDst[3] = a[3];
            newDst[4] = a[4];
            newDst[5] = a[5];
            newDst[6] = a[6];
            newDst[7] = a[7];
            newDst[8] = a[8];
            newDst[9] = a[9];
            newDst[10] = a[10];
            newDst[11] = a[11];
        }
        newDst[12] = v[0];
        newDst[13] = v[1];
        newDst[14] = v[2];
        newDst[15] = 1;
        return newDst;
    }
    ///**
    // * Returns the translation component of a 4-by-4 matrix as a vector with 3
    // * entries.
    // * @param m - The matrix.
    // * @param dst - vector to hold result. If not passed a new one is created.
    // * @returns The translation component of m.
    // */
    function getTranslation(m, dst) {
        const newDst = (dst ?? vec3.create());
        newDst[0] = m[12];
        newDst[1] = m[13];
        newDst[2] = m[14];
        return newDst;
    }
    /**
     * Returns an axis of a 4x4 matrix as a vector with 3 entries
     * @param m - The matrix.
     * @param axis - The axis 0 = x, 1 = y, 2 = z;
     * @returns The axis component of m.
     */
    function getAxis(m, axis, dst) {
        const newDst = (dst ?? vec3.create());
        const off = axis * 4;
        newDst[0] = m[off + 0];
        newDst[1] = m[off + 1];
        newDst[2] = m[off + 2];
        return newDst;
    }
    /**
     * Sets an axis of a 4x4 matrix as a vector with 3 entries
     * @param m - The matrix.
     * @param v - the axis vector
     * @param axis - The axis  0 = x, 1 = y, 2 = z;
     * @param dst - The matrix to set. If not passed a new one is created.
     * @returns The matrix with axis set.
     */
    function setAxis(m, v, axis, dst) {
        const newDst = (dst === m) ? dst : copy(m, dst);
        const off = axis * 4;
        newDst[off + 0] = v[0];
        newDst[off + 1] = v[1];
        newDst[off + 2] = v[2];
        return newDst;
    }
    /**
     * Returns the "3d" scaling component of the matrix
     * @param m - The Matrix
     * @param dst - The vector to set. If not passed a new one is created.
     */
    function getScaling(m, dst) {
        const newDst = (dst ?? vec3.create());
        const xx = m[0];
        const xy = m[1];
        const xz = m[2];
        const yx = m[4];
        const yy = m[5];
        const yz = m[6];
        const zx = m[8];
        const zy = m[9];
        const zz = m[10];
        newDst[0] = Math.sqrt(xx * xx + xy * xy + xz * xz);
        newDst[1] = Math.sqrt(yx * yx + yy * yy + yz * yz);
        newDst[2] = Math.sqrt(zx * zx + zy * zy + zz * zz);
        return newDst;
    }
    /**
     * Computes a 4-by-4 perspective transformation matrix given the angular height
     * of the frustum, the aspect ratio, and the near and far clipping planes.  The
     * arguments define a frustum extending in the negative z direction.  The given
     * angle is the vertical angle of the frustum, and the horizontal angle is
     * determined to produce the given aspect ratio.  The arguments near and far are
     * the distances to the near and far clipping planes.  Note that near and far
     * are not z coordinates, but rather they are distances along the negative
     * z-axis.  The matrix generated sends the viewing frustum to the unit box.
     * We assume a unit box extending from -1 to 1 in the x and y dimensions and
     * from 0 to 1 in the z dimension.
     *
     * Note: If you pass `Infinity` for zFar then it will produce a projection matrix
     * returns -Infinity for Z when transforming coordinates with Z <= 0 and +Infinity for Z
     * otherwise.
     *
     * @param fieldOfViewYInRadians - The camera angle from top to bottom (in radians).
     * @param aspect - The aspect ratio width / height.
     * @param zNear - The depth (negative z coordinate)
     *     of the near clipping plane.
     * @param zFar - The depth (negative z coordinate)
     *     of the far clipping plane.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The perspective matrix.
     */
    function perspective(fieldOfViewYInRadians, aspect, zNear, zFar, dst) {
        const newDst = (dst ?? new Ctor(16));
        const f = Math.tan(Math.PI * 0.5 - 0.5 * fieldOfViewYInRadians);
        newDst[0] = f / aspect;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = f;
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[11] = -1;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[15] = 0;
        if (Number.isFinite(zFar)) {
            const rangeInv = 1 / (zNear - zFar);
            newDst[10] = zFar * rangeInv;
            newDst[14] = zFar * zNear * rangeInv;
        }
        else {
            newDst[10] = -1;
            newDst[14] = -zNear;
        }
        return newDst;
    }
    /**
     * Computes a 4-by-4 reverse-z perspective transformation matrix given the angular height
     * of the frustum, the aspect ratio, and the near and far clipping planes.  The
     * arguments define a frustum extending in the negative z direction.  The given
     * angle is the vertical angle of the frustum, and the horizontal angle is
     * determined to produce the given aspect ratio.  The arguments near and far are
     * the distances to the near and far clipping planes.  Note that near and far
     * are not z coordinates, but rather they are distances along the negative
     * z-axis.  The matrix generated sends the viewing frustum to the unit box.
     * We assume a unit box extending from -1 to 1 in the x and y dimensions and
     * from 1 (at -zNear) to 0 (at -zFar) in the z dimension.
     *
     * @param fieldOfViewYInRadians - The camera angle from top to bottom (in radians).
     * @param aspect - The aspect ratio width / height.
     * @param zNear - The depth (negative z coordinate)
     *     of the near clipping plane.
     * @param zFar - The depth (negative z coordinate)
     *     of the far clipping plane. (default = Infinity)
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The perspective matrix.
     */ function perspectiveReverseZ(fieldOfViewYInRadians, aspect, zNear, zFar = Infinity, dst) {
        const newDst = (dst ?? new Ctor(16));
        const f = 1 / Math.tan(fieldOfViewYInRadians * 0.5);
        newDst[0] = f / aspect;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = f;
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[11] = -1;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[15] = 0;
        if (zFar === Infinity) {
            newDst[10] = 0;
            newDst[14] = zNear;
        }
        else {
            const rangeInv = 1 / (zFar - zNear);
            newDst[10] = zNear * rangeInv;
            newDst[14] = zFar * zNear * rangeInv;
        }
        return newDst;
    }
    /**
     * Computes a 4-by-4 orthogonal transformation matrix that transforms from
     * the given the left, right, bottom, and top dimensions to -1 +1 in x, and y
     * and 0 to +1 in z.
     * @param left - Left side of the near clipping plane viewport.
     * @param right - Right side of the near clipping plane viewport.
     * @param bottom - Bottom of the near clipping plane viewport.
     * @param top - Top of the near clipping plane viewport.
     * @param near - The depth (negative z coordinate)
     *     of the near clipping plane.
     * @param far - The depth (negative z coordinate)
     *     of the far clipping plane.
     * @param dst - Output matrix. If not passed a new one is created.
     * @returns The orthographic projection matrix.
     */
    function ortho(left, right, bottom, top, near, far, dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = 2 / (right - left);
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = 2 / (top - bottom);
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = 1 / (near - far);
        newDst[11] = 0;
        newDst[12] = (right + left) / (left - right);
        newDst[13] = (top + bottom) / (bottom - top);
        newDst[14] = near / (near - far);
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Computes a 4-by-4 perspective transformation matrix given the left, right,
     * top, bottom, near and far clipping planes. The arguments define a frustum
     * extending in the negative z direction. The arguments near and far are the
     * distances to the near and far clipping planes. Note that near and far are not
     * z coordinates, but rather they are distances along the negative z-axis. The
     * matrix generated sends the viewing frustum to the unit box. We assume a unit
     * box extending from -1 to 1 in the x and y dimensions and from 0 to 1 in the z
     * dimension.
     * @param left - The x coordinate of the left plane of the box.
     * @param right - The x coordinate of the right plane of the box.
     * @param bottom - The y coordinate of the bottom plane of the box.
     * @param top - The y coordinate of the right plane of the box.
     * @param near - The negative z coordinate of the near plane of the box.
     * @param far - The negative z coordinate of the far plane of the box.
     * @param dst - Output matrix. If not passed a new one is created.
     * @returns The perspective projection matrix.
     */
    function frustum(left, right, bottom, top, near, far, dst) {
        const newDst = (dst ?? new Ctor(16));
        const dx = (right - left);
        const dy = (top - bottom);
        const dz = (near - far);
        newDst[0] = 2 * near / dx;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = 2 * near / dy;
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = (left + right) / dx;
        newDst[9] = (top + bottom) / dy;
        newDst[10] = far / dz;
        newDst[11] = -1;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = near * far / dz;
        newDst[15] = 0;
        return newDst;
    }
    /**
     * Computes a 4-by-4 reverse-z perspective transformation matrix given the left, right,
     * top, bottom, near and far clipping planes. The arguments define a frustum
     * extending in the negative z direction. The arguments near and far are the
     * distances to the near and far clipping planes. Note that near and far are not
     * z coordinates, but rather they are distances along the negative z-axis. The
     * matrix generated sends the viewing frustum to the unit box. We assume a unit
     * box extending from -1 to 1 in the x and y dimensions and from 1 (-near) to 0 (-far) in the z
     * dimension.
     * @param left - The x coordinate of the left plane of the box.
     * @param right - The x coordinate of the right plane of the box.
     * @param bottom - The y coordinate of the bottom plane of the box.
     * @param top - The y coordinate of the right plane of the box.
     * @param near - The negative z coordinate of the near plane of the box.
     * @param far - The negative z coordinate of the far plane of the box.
     * @param dst - Output matrix. If not passed a new one is created.
     * @returns The perspective projection matrix.
     */
    function frustumReverseZ(left, right, bottom, top, near, far = Infinity, dst) {
        const newDst = (dst ?? new Ctor(16));
        const dx = (right - left);
        const dy = (top - bottom);
        newDst[0] = 2 * near / dx;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = 2 * near / dy;
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = (left + right) / dx;
        newDst[9] = (top + bottom) / dy;
        newDst[11] = -1;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[15] = 0;
        if (far === Infinity) {
            newDst[10] = 0;
            newDst[14] = near;
        }
        else {
            const rangeInv = 1 / (far - near);
            newDst[10] = near * rangeInv;
            newDst[14] = far * near * rangeInv;
        }
        return newDst;
    }
    const xAxis = vec3.create();
    const yAxis = vec3.create();
    const zAxis = vec3.create();
    /**
     * Computes a 4-by-4 aim transformation.
     *
     * This is a matrix which positions an object aiming down positive Z.
     * toward the target.
     *
     * Note: this is **NOT** the inverse of lookAt as lookAt looks at negative Z.
     *
     * @param position - The position of the object.
     * @param target - The position meant to be aimed at.
     * @param up - A vector pointing up.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The aim matrix.
     */
    function aim(position, target, up, dst) {
        const newDst = (dst ?? new Ctor(16));
        vec3.normalize(vec3.subtract(target, position, zAxis), zAxis);
        vec3.normalize(vec3.cross(up, zAxis, xAxis), xAxis);
        vec3.normalize(vec3.cross(zAxis, xAxis, yAxis), yAxis);
        newDst[0] = xAxis[0];
        newDst[1] = xAxis[1];
        newDst[2] = xAxis[2];
        newDst[3] = 0;
        newDst[4] = yAxis[0];
        newDst[5] = yAxis[1];
        newDst[6] = yAxis[2];
        newDst[7] = 0;
        newDst[8] = zAxis[0];
        newDst[9] = zAxis[1];
        newDst[10] = zAxis[2];
        newDst[11] = 0;
        newDst[12] = position[0];
        newDst[13] = position[1];
        newDst[14] = position[2];
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Computes a 4-by-4 camera aim transformation.
     *
     * This is a matrix which positions an object aiming down negative Z.
     * toward the target.
     *
     * Note: this is the inverse of `lookAt`
     *
     * @param eye - The position of the object.
     * @param target - The position meant to be aimed at.
     * @param up - A vector pointing up.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The aim matrix.
     */
    function cameraAim(eye, target, up, dst) {
        const newDst = (dst ?? new Ctor(16));
        vec3.normalize(vec3.subtract(eye, target, zAxis), zAxis);
        vec3.normalize(vec3.cross(up, zAxis, xAxis), xAxis);
        vec3.normalize(vec3.cross(zAxis, xAxis, yAxis), yAxis);
        newDst[0] = xAxis[0];
        newDst[1] = xAxis[1];
        newDst[2] = xAxis[2];
        newDst[3] = 0;
        newDst[4] = yAxis[0];
        newDst[5] = yAxis[1];
        newDst[6] = yAxis[2];
        newDst[7] = 0;
        newDst[8] = zAxis[0];
        newDst[9] = zAxis[1];
        newDst[10] = zAxis[2];
        newDst[11] = 0;
        newDst[12] = eye[0];
        newDst[13] = eye[1];
        newDst[14] = eye[2];
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Computes a 4-by-4 view transformation.
     *
     * This is a view matrix which transforms all other objects
     * to be in the space of the view defined by the parameters.
     *
     * @param eye - The position of the object.
     * @param target - The position meant to be aimed at.
     * @param up - A vector pointing up.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The look-at matrix.
     */
    function lookAt(eye, target, up, dst) {
        const newDst = (dst ?? new Ctor(16));
        vec3.normalize(vec3.subtract(eye, target, zAxis), zAxis);
        vec3.normalize(vec3.cross(up, zAxis, xAxis), xAxis);
        vec3.normalize(vec3.cross(zAxis, xAxis, yAxis), yAxis);
        newDst[0] = xAxis[0];
        newDst[1] = yAxis[0];
        newDst[2] = zAxis[0];
        newDst[3] = 0;
        newDst[4] = xAxis[1];
        newDst[5] = yAxis[1];
        newDst[6] = zAxis[1];
        newDst[7] = 0;
        newDst[8] = xAxis[2];
        newDst[9] = yAxis[2];
        newDst[10] = zAxis[2];
        newDst[11] = 0;
        newDst[12] = -(xAxis[0] * eye[0] + xAxis[1] * eye[1] + xAxis[2] * eye[2]);
        newDst[13] = -(yAxis[0] * eye[0] + yAxis[1] * eye[1] + yAxis[2] * eye[2]);
        newDst[14] = -(zAxis[0] * eye[0] + zAxis[1] * eye[1] + zAxis[2] * eye[2]);
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Creates a 4-by-4 matrix which translates by the given vector v.
     * @param v - The vector by
     *     which to translate.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The translation matrix.
     */
    function translation(v, dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = 1;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = 1;
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = 1;
        newDst[11] = 0;
        newDst[12] = v[0];
        newDst[13] = v[1];
        newDst[14] = v[2];
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Translates the given 4-by-4 matrix by the given vector v.
     * @param m - The matrix.
     * @param v - The vector by
     *     which to translate.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The translated matrix.
     */
    function translate(m, v, dst) {
        const newDst = (dst ?? new Ctor(16));
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        const m00 = m[0];
        const m01 = m[1];
        const m02 = m[2];
        const m03 = m[3];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const m13 = m[1 * 4 + 3];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        const m23 = m[2 * 4 + 3];
        const m30 = m[3 * 4 + 0];
        const m31 = m[3 * 4 + 1];
        const m32 = m[3 * 4 + 2];
        const m33 = m[3 * 4 + 3];
        if (m !== newDst) {
            newDst[0] = m00;
            newDst[1] = m01;
            newDst[2] = m02;
            newDst[3] = m03;
            newDst[4] = m10;
            newDst[5] = m11;
            newDst[6] = m12;
            newDst[7] = m13;
            newDst[8] = m20;
            newDst[9] = m21;
            newDst[10] = m22;
            newDst[11] = m23;
        }
        newDst[12] = m00 * v0 + m10 * v1 + m20 * v2 + m30;
        newDst[13] = m01 * v0 + m11 * v1 + m21 * v2 + m31;
        newDst[14] = m02 * v0 + m12 * v1 + m22 * v2 + m32;
        newDst[15] = m03 * v0 + m13 * v1 + m23 * v2 + m33;
        return newDst;
    }
    /**
     * Creates a 4-by-4 matrix which rotates around the x-axis by the given angle.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotation matrix.
     */
    function rotationX(angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(16));
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = 1;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = c;
        newDst[6] = s;
        newDst[7] = 0;
        newDst[8] = 0;
        newDst[9] = -s;
        newDst[10] = c;
        newDst[11] = 0;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = 0;
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Rotates the given 4-by-4 matrix around the x-axis by the given
     * angle.
     * @param m - The matrix.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotated matrix.
     */
    function rotateX(m, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(16));
        const m10 = m[4];
        const m11 = m[5];
        const m12 = m[6];
        const m13 = m[7];
        const m20 = m[8];
        const m21 = m[9];
        const m22 = m[10];
        const m23 = m[11];
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[4] = c * m10 + s * m20;
        newDst[5] = c * m11 + s * m21;
        newDst[6] = c * m12 + s * m22;
        newDst[7] = c * m13 + s * m23;
        newDst[8] = c * m20 - s * m10;
        newDst[9] = c * m21 - s * m11;
        newDst[10] = c * m22 - s * m12;
        newDst[11] = c * m23 - s * m13;
        if (m !== newDst) {
            newDst[0] = m[0];
            newDst[1] = m[1];
            newDst[2] = m[2];
            newDst[3] = m[3];
            newDst[12] = m[12];
            newDst[13] = m[13];
            newDst[14] = m[14];
            newDst[15] = m[15];
        }
        return newDst;
    }
    /**
     * Creates a 4-by-4 matrix which rotates around the y-axis by the given angle.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotation matrix.
     */
    function rotationY(angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(16));
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = c;
        newDst[1] = 0;
        newDst[2] = -s;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = 1;
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = s;
        newDst[9] = 0;
        newDst[10] = c;
        newDst[11] = 0;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = 0;
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Rotates the given 4-by-4 matrix around the y-axis by the given
     * angle.
     * @param m - The matrix.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotated matrix.
     */
    function rotateY(m, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(16));
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m03 = m[0 * 4 + 3];
        const m20 = m[2 * 4 + 0];
        const m21 = m[2 * 4 + 1];
        const m22 = m[2 * 4 + 2];
        const m23 = m[2 * 4 + 3];
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = c * m00 - s * m20;
        newDst[1] = c * m01 - s * m21;
        newDst[2] = c * m02 - s * m22;
        newDst[3] = c * m03 - s * m23;
        newDst[8] = c * m20 + s * m00;
        newDst[9] = c * m21 + s * m01;
        newDst[10] = c * m22 + s * m02;
        newDst[11] = c * m23 + s * m03;
        if (m !== newDst) {
            newDst[4] = m[4];
            newDst[5] = m[5];
            newDst[6] = m[6];
            newDst[7] = m[7];
            newDst[12] = m[12];
            newDst[13] = m[13];
            newDst[14] = m[14];
            newDst[15] = m[15];
        }
        return newDst;
    }
    /**
     * Creates a 4-by-4 matrix which rotates around the z-axis by the given angle.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotation matrix.
     */
    function rotationZ(angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(16));
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = c;
        newDst[1] = s;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = -s;
        newDst[5] = c;
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = 1;
        newDst[11] = 0;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = 0;
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Rotates the given 4-by-4 matrix around the z-axis by the given
     * angle.
     * @param m - The matrix.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotated matrix.
     */
    function rotateZ(m, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(16));
        const m00 = m[0 * 4 + 0];
        const m01 = m[0 * 4 + 1];
        const m02 = m[0 * 4 + 2];
        const m03 = m[0 * 4 + 3];
        const m10 = m[1 * 4 + 0];
        const m11 = m[1 * 4 + 1];
        const m12 = m[1 * 4 + 2];
        const m13 = m[1 * 4 + 3];
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        newDst[0] = c * m00 + s * m10;
        newDst[1] = c * m01 + s * m11;
        newDst[2] = c * m02 + s * m12;
        newDst[3] = c * m03 + s * m13;
        newDst[4] = c * m10 - s * m00;
        newDst[5] = c * m11 - s * m01;
        newDst[6] = c * m12 - s * m02;
        newDst[7] = c * m13 - s * m03;
        if (m !== newDst) {
            newDst[8] = m[8];
            newDst[9] = m[9];
            newDst[10] = m[10];
            newDst[11] = m[11];
            newDst[12] = m[12];
            newDst[13] = m[13];
            newDst[14] = m[14];
            newDst[15] = m[15];
        }
        return newDst;
    }
    /**
     * Creates a 4-by-4 matrix which rotates around the given axis by the given
     * angle.
     * @param axis - The axis
     *     about which to rotate.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns A matrix which rotates angle radians
     *     around the axis.
     */
    function axisRotation(axis, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(16));
        let x = axis[0];
        let y = axis[1];
        let z = axis[2];
        const n = Math.sqrt(x * x + y * y + z * z);
        x /= n;
        y /= n;
        z /= n;
        const xx = x * x;
        const yy = y * y;
        const zz = z * z;
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        const oneMinusCosine = 1 - c;
        newDst[0] = xx + (1 - xx) * c;
        newDst[1] = x * y * oneMinusCosine + z * s;
        newDst[2] = x * z * oneMinusCosine - y * s;
        newDst[3] = 0;
        newDst[4] = x * y * oneMinusCosine - z * s;
        newDst[5] = yy + (1 - yy) * c;
        newDst[6] = y * z * oneMinusCosine + x * s;
        newDst[7] = 0;
        newDst[8] = x * z * oneMinusCosine + y * s;
        newDst[9] = y * z * oneMinusCosine - x * s;
        newDst[10] = zz + (1 - zz) * c;
        newDst[11] = 0;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = 0;
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Creates a 4-by-4 matrix which rotates around the given axis by the given
     * angle. (same as axisRotation)
     * @param axis - The axis
     *     about which to rotate.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns A matrix which rotates angle radians
     *     around the axis.
     */
    const rotation = axisRotation;
    /**
     * Rotates the given 4-by-4 matrix around the given axis by the
     * given angle.
     * @param m - The matrix.
     * @param axis - The axis
     *     about which to rotate.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotated matrix.
     */
    function axisRotate(m, axis, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(16));
        let x = axis[0];
        let y = axis[1];
        let z = axis[2];
        const n = Math.sqrt(x * x + y * y + z * z);
        x /= n;
        y /= n;
        z /= n;
        const xx = x * x;
        const yy = y * y;
        const zz = z * z;
        const c = Math.cos(angleInRadians);
        const s = Math.sin(angleInRadians);
        const oneMinusCosine = 1 - c;
        const r00 = xx + (1 - xx) * c;
        const r01 = x * y * oneMinusCosine + z * s;
        const r02 = x * z * oneMinusCosine - y * s;
        const r10 = x * y * oneMinusCosine - z * s;
        const r11 = yy + (1 - yy) * c;
        const r12 = y * z * oneMinusCosine + x * s;
        const r20 = x * z * oneMinusCosine + y * s;
        const r21 = y * z * oneMinusCosine - x * s;
        const r22 = zz + (1 - zz) * c;
        const m00 = m[0];
        const m01 = m[1];
        const m02 = m[2];
        const m03 = m[3];
        const m10 = m[4];
        const m11 = m[5];
        const m12 = m[6];
        const m13 = m[7];
        const m20 = m[8];
        const m21 = m[9];
        const m22 = m[10];
        const m23 = m[11];
        newDst[0] = r00 * m00 + r01 * m10 + r02 * m20;
        newDst[1] = r00 * m01 + r01 * m11 + r02 * m21;
        newDst[2] = r00 * m02 + r01 * m12 + r02 * m22;
        newDst[3] = r00 * m03 + r01 * m13 + r02 * m23;
        newDst[4] = r10 * m00 + r11 * m10 + r12 * m20;
        newDst[5] = r10 * m01 + r11 * m11 + r12 * m21;
        newDst[6] = r10 * m02 + r11 * m12 + r12 * m22;
        newDst[7] = r10 * m03 + r11 * m13 + r12 * m23;
        newDst[8] = r20 * m00 + r21 * m10 + r22 * m20;
        newDst[9] = r20 * m01 + r21 * m11 + r22 * m21;
        newDst[10] = r20 * m02 + r21 * m12 + r22 * m22;
        newDst[11] = r20 * m03 + r21 * m13 + r22 * m23;
        if (m !== newDst) {
            newDst[12] = m[12];
            newDst[13] = m[13];
            newDst[14] = m[14];
            newDst[15] = m[15];
        }
        return newDst;
    }
    /**
     * Rotates the given 4-by-4 matrix around the given axis by the
     * given angle. (same as rotate)
     * @param m - The matrix.
     * @param axis - The axis
     *     about which to rotate.
     * @param angleInRadians - The angle by which to rotate (in radians).
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The rotated matrix.
     */
    const rotate = axisRotate;
    /**
     * Creates a 4-by-4 matrix which scales in each dimension by an amount given by
     * the corresponding entry in the given vector; assumes the vector has three
     * entries.
     * @param v - A vector of
     *     three entries specifying the factor by which to scale in each dimension.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaling matrix.
     */
    function scaling(v, dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = v[0];
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = v[1];
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = v[2];
        newDst[11] = 0;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = 0;
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Scales the given 4-by-4 matrix in each dimension by an amount
     * given by the corresponding entry in the given vector; assumes the vector has
     * three entries.
     * @param m - The matrix to be modified.
     * @param v - A vector of three entries specifying the
     *     factor by which to scale in each dimension.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaled matrix.
     */
    function scale(m, v, dst) {
        const newDst = (dst ?? new Ctor(16));
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        newDst[0] = v0 * m[0 * 4 + 0];
        newDst[1] = v0 * m[0 * 4 + 1];
        newDst[2] = v0 * m[0 * 4 + 2];
        newDst[3] = v0 * m[0 * 4 + 3];
        newDst[4] = v1 * m[1 * 4 + 0];
        newDst[5] = v1 * m[1 * 4 + 1];
        newDst[6] = v1 * m[1 * 4 + 2];
        newDst[7] = v1 * m[1 * 4 + 3];
        newDst[8] = v2 * m[2 * 4 + 0];
        newDst[9] = v2 * m[2 * 4 + 1];
        newDst[10] = v2 * m[2 * 4 + 2];
        newDst[11] = v2 * m[2 * 4 + 3];
        if (m !== newDst) {
            newDst[12] = m[12];
            newDst[13] = m[13];
            newDst[14] = m[14];
            newDst[15] = m[15];
        }
        return newDst;
    }
    /**
     * Creates a 4-by-4 matrix which scales a uniform amount in each dimension.
     * @param s - the amount to scale
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaling matrix.
     */
    function uniformScaling(s, dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = s;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        newDst[4] = 0;
        newDst[5] = s;
        newDst[6] = 0;
        newDst[7] = 0;
        newDst[8] = 0;
        newDst[9] = 0;
        newDst[10] = s;
        newDst[11] = 0;
        newDst[12] = 0;
        newDst[13] = 0;
        newDst[14] = 0;
        newDst[15] = 1;
        return newDst;
    }
    /**
     * Scales the given 4-by-4 matrix in each dimension by a uniform scale.
     * @param m - The matrix to be modified.
     * @param s - The amount to scale.
     * @param dst - matrix to hold result. If not passed a new one is created.
     * @returns The scaled matrix.
     */
    function uniformScale(m, s, dst) {
        const newDst = (dst ?? new Ctor(16));
        newDst[0] = s * m[0 * 4 + 0];
        newDst[1] = s * m[0 * 4 + 1];
        newDst[2] = s * m[0 * 4 + 2];
        newDst[3] = s * m[0 * 4 + 3];
        newDst[4] = s * m[1 * 4 + 0];
        newDst[5] = s * m[1 * 4 + 1];
        newDst[6] = s * m[1 * 4 + 2];
        newDst[7] = s * m[1 * 4 + 3];
        newDst[8] = s * m[2 * 4 + 0];
        newDst[9] = s * m[2 * 4 + 1];
        newDst[10] = s * m[2 * 4 + 2];
        newDst[11] = s * m[2 * 4 + 3];
        if (m !== newDst) {
            newDst[12] = m[12];
            newDst[13] = m[13];
            newDst[14] = m[14];
            newDst[15] = m[15];
        }
        return newDst;
    }
    return {
        create,
        set,
        fromMat3,
        fromQuat,
        negate,
        copy,
        clone,
        equalsApproximately,
        equals,
        identity,
        transpose,
        inverse,
        determinant,
        invert,
        multiply,
        mul,
        setTranslation,
        getTranslation,
        getAxis,
        setAxis,
        getScaling,
        perspective,
        perspectiveReverseZ,
        ortho,
        frustum,
        frustumReverseZ,
        aim,
        cameraAim,
        lookAt,
        translation,
        translate,
        rotationX,
        rotateX,
        rotationY,
        rotateY,
        rotationZ,
        rotateZ,
        axisRotation,
        rotation,
        axisRotate,
        rotate,
        scaling,
        scale,
        uniformScaling,
        uniformScale,
    };
}
const cache$2 = new Map();
function getAPI$2(Ctor) {
    let api = cache$2.get(Ctor);
    if (!api) {
        api = getAPIImpl$2(Ctor);
        cache$2.set(Ctor, api);
    }
    return api;
}

/*
 * Copyright 2022 Gregg Tavares
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/**
 * Generates am typed API for Qud
 * */
function getAPIImpl$1(Ctor) {
    const vec3 = getAPI$4(Ctor);
    /**
     * Creates a quat4; may be called with x, y, z to set initial values.
     * @param x - Initial x value.
     * @param y - Initial y value.
     * @param z - Initial z value.
     * @param w - Initial w value.
     * @returns the created vector
     */
    function create(x, y, z, w) {
        const newDst = new Ctor(4);
        if (x !== undefined) {
            newDst[0] = x;
            if (y !== undefined) {
                newDst[1] = y;
                if (z !== undefined) {
                    newDst[2] = z;
                    if (w !== undefined) {
                        newDst[3] = w;
                    }
                }
            }
        }
        return newDst;
    }
    /**
     * Creates a Quat; may be called with x, y, z to set initial values. (same as create)
     * @param x - Initial x value.
     * @param y - Initial y value.
     * @param z - Initial z value.
     * @param z - Initial w value.
     * @returns the created vector
     */
    const fromValues = create;
    /**
     * Sets the values of a Quat
     * Also see {@link quat.create} and {@link quat.copy}
     *
     * @param x first value
     * @param y second value
     * @param z third value
     * @param w fourth value
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector with its elements set.
     */
    function set(x, y, z, w, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = x;
        newDst[1] = y;
        newDst[2] = z;
        newDst[3] = w;
        return newDst;
    }
    /**
     * Sets a quaternion from the given angle and  axis,
     * then returns it.
     *
     * @param axis - the axis to rotate around
     * @param angleInRadians - the angle
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns The quaternion that represents the given axis and angle
     **/
    function fromAxisAngle(axis, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(4));
        const halfAngle = angleInRadians * 0.5;
        const s = Math.sin(halfAngle);
        newDst[0] = s * axis[0];
        newDst[1] = s * axis[1];
        newDst[2] = s * axis[2];
        newDst[3] = Math.cos(halfAngle);
        return newDst;
    }
    /**
     * Gets the rotation axis and angle
     * @param q - quaternion to compute from
     * @param dst - Vec3 to hold result. If not passed in a new one is created.
     * @return angle and axis
     */
    function toAxisAngle(q, dst) {
        const newDst = (dst ?? vec3.create(3));
        const angle = Math.acos(q[3]) * 2;
        const s = Math.sin(angle * 0.5);
        if (s > EPSILON) {
            newDst[0] = q[0] / s;
            newDst[1] = q[1] / s;
            newDst[2] = q[2] / s;
        }
        else {
            newDst[0] = 1;
            newDst[1] = 0;
            newDst[2] = 0;
        }
        return { angle, axis: newDst };
    }
    /**
     * Returns the angle in degrees between two rotations a and b.
     * @param a - quaternion a
     * @param b - quaternion b
     * @return angle in radians between the two quaternions
     */
    function angle(a, b) {
        const d = dot(a, b);
        return Math.acos(2 * d * d - 1);
    }
    /**
     * Multiplies two quaternions
     *
     * @param a - the first quaternion
     * @param b - the second quaternion
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is the result of a * b
     */
    function multiply(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        const ax = a[0];
        const ay = a[1];
        const az = a[2];
        const aw = a[3];
        const bx = b[0];
        const by = b[1];
        const bz = b[2];
        const bw = b[3];
        newDst[0] = ax * bw + aw * bx + ay * bz - az * by;
        newDst[1] = ay * bw + aw * by + az * bx - ax * bz;
        newDst[2] = az * bw + aw * bz + ax * by - ay * bx;
        newDst[3] = aw * bw - ax * bx - ay * by - az * bz;
        return newDst;
    }
    /**
     * Multiplies two quaternions
     *
     * @param a - the first quaternion
     * @param b - the second quaternion
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is the result of a * b
     */
    const mul = multiply;
    /**
     * Rotates the given quaternion around the X axis by the given angle.
     * @param q - quaternion to rotate
     * @param angleInRadians - The angle by which to rotate
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is the result of a * b
     */
    function rotateX(q, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(4));
        const halfAngle = angleInRadians * 0.5;
        const qx = q[0];
        const qy = q[1];
        const qz = q[2];
        const qw = q[3];
        const bx = Math.sin(halfAngle);
        const bw = Math.cos(halfAngle);
        newDst[0] = qx * bw + qw * bx;
        newDst[1] = qy * bw + qz * bx;
        newDst[2] = qz * bw - qy * bx;
        newDst[3] = qw * bw - qx * bx;
        return newDst;
    }
    /**
     * Rotates the given quaternion around the Y axis by the given angle.
     * @param q - quaternion to rotate
     * @param angleInRadians - The angle by which to rotate
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is the result of a * b
     */
    function rotateY(q, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(4));
        const halfAngle = angleInRadians * 0.5;
        const qx = q[0];
        const qy = q[1];
        const qz = q[2];
        const qw = q[3];
        const by = Math.sin(halfAngle);
        const bw = Math.cos(halfAngle);
        newDst[0] = qx * bw - qz * by;
        newDst[1] = qy * bw + qw * by;
        newDst[2] = qz * bw + qx * by;
        newDst[3] = qw * bw - qy * by;
        return newDst;
    }
    /**
     * Rotates the given quaternion around the Z axis by the given angle.
     * @param q - quaternion to rotate
     * @param angleInRadians - The angle by which to rotate
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is the result of a * b
     */
    function rotateZ(q, angleInRadians, dst) {
        const newDst = (dst ?? new Ctor(4));
        const halfAngle = angleInRadians * 0.5;
        const qx = q[0];
        const qy = q[1];
        const qz = q[2];
        const qw = q[3];
        const bz = Math.sin(halfAngle);
        const bw = Math.cos(halfAngle);
        newDst[0] = qx * bw + qy * bz;
        newDst[1] = qy * bw - qx * bz;
        newDst[2] = qz * bw + qw * bz;
        newDst[3] = qw * bw - qz * bz;
        return newDst;
    }
    /**
     * Spherically linear interpolate between two quaternions
     *
     * @param a - starting value
     * @param b - ending value
     * @param t - value where 0 = a and 1 = b
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is the result of a * b
     */
    function slerp(a, b, t, dst) {
        const newDst = (dst ?? new Ctor(4));
        const ax = a[0];
        const ay = a[1];
        const az = a[2];
        const aw = a[3];
        let bx = b[0];
        let by = b[1];
        let bz = b[2];
        let bw = b[3];
        let cosOmega = ax * bx + ay * by + az * bz + aw * bw;
        if (cosOmega < 0) {
            cosOmega = -cosOmega;
            bx = -bx;
            by = -by;
            bz = -bz;
            bw = -bw;
        }
        let scale0;
        let scale1;
        if (1.0 - cosOmega > EPSILON) {
            const omega = Math.acos(cosOmega);
            const sinOmega = Math.sin(omega);
            scale0 = Math.sin((1 - t) * omega) / sinOmega;
            scale1 = Math.sin(t * omega) / sinOmega;
        }
        else {
            scale0 = 1.0 - t;
            scale1 = t;
        }
        newDst[0] = scale0 * ax + scale1 * bx;
        newDst[1] = scale0 * ay + scale1 * by;
        newDst[2] = scale0 * az + scale1 * bz;
        newDst[3] = scale0 * aw + scale1 * bw;
        return newDst;
    }
    /**
     * Compute the inverse of a quaternion
     *
     * @param q - quaternion to compute the inverse of
     * @returns A quaternion that is the result of a * b
     */
    function inverse(q, dst) {
        const newDst = (dst ?? new Ctor(4));
        const a0 = q[0];
        const a1 = q[1];
        const a2 = q[2];
        const a3 = q[3];
        const dot = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        const invDot = dot ? 1 / dot : 0;
        newDst[0] = -a0 * invDot;
        newDst[1] = -a1 * invDot;
        newDst[2] = -a2 * invDot;
        newDst[3] = a3 * invDot;
        return newDst;
    }
    /**
     * Compute the conjugate of a quaternion
     * For quaternions with a magnitude of 1 (a unit quaternion)
     * this returns the same as the inverse but is faster to calculate.
     *
     * @param q - quaternion to compute the conjugate of.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns The conjugate of q
     */
    function conjugate(q, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = -q[0];
        newDst[1] = -q[1];
        newDst[2] = -q[2];
        newDst[3] = q[3];
        return newDst;
    }
    /**
     * Creates a quaternion from the given rotation matrix.
     *
     * The created quaternion is not normalized.
     *
     * @param m - rotation matrix
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns the result
     */
    function fromMat(m, dst) {
        const newDst = (dst ?? new Ctor(4));
        /*
        0 1 2
        3 4 5
        6 7 8
      
        0 1 2
        4 5 6
        8 9 10
         */
        // Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
        // article "Quaternion Calculus and Fast Animation".
        const trace = m[0] + m[5] + m[10];
        if (trace > 0.0) {
            // |w| > 1/2, may as well choose w > 1/2
            const root = Math.sqrt(trace + 1); // 2w
            newDst[3] = 0.5 * root;
            const invRoot = 0.5 / root; // 1/(4w)
            newDst[0] = (m[6] - m[9]) * invRoot;
            newDst[1] = (m[8] - m[2]) * invRoot;
            newDst[2] = (m[1] - m[4]) * invRoot;
        }
        else {
            // |w| <= 1/2
            let i = 0;
            if (m[5] > m[0]) {
                i = 1;
            }
            if (m[10] > m[i * 4 + i]) {
                i = 2;
            }
            const j = (i + 1) % 3;
            const k = (i + 2) % 3;
            const root = Math.sqrt(m[i * 4 + i] - m[j * 4 + j] - m[k * 4 + k] + 1.0);
            newDst[i] = 0.5 * root;
            const invRoot = 0.5 / root;
            newDst[3] = (m[j * 4 + k] - m[k * 4 + j]) * invRoot;
            newDst[j] = (m[j * 4 + i] + m[i * 4 + j]) * invRoot;
            newDst[k] = (m[k * 4 + i] + m[i * 4 + k]) * invRoot;
        }
        return newDst;
    }
    /**
     * Creates a quaternion from the given euler angle x, y, z using the provided intrinsic order for the conversion.
     *
     * @param xAngleInRadians - angle to rotate around X axis in radians.
     * @param yAngleInRadians - angle to rotate around Y axis in radians.
     * @param zAngleInRadians - angle to rotate around Z axis in radians.
     * @param order - order to apply euler angles
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion representing the same rotation as the euler angles applied in the given order
     */
    function fromEuler(xAngleInRadians, yAngleInRadians, zAngleInRadians, order, dst) {
        const newDst = (dst ?? new Ctor(4));
        const xHalfAngle = xAngleInRadians * 0.5;
        const yHalfAngle = yAngleInRadians * 0.5;
        const zHalfAngle = zAngleInRadians * 0.5;
        const sx = Math.sin(xHalfAngle);
        const cx = Math.cos(xHalfAngle);
        const sy = Math.sin(yHalfAngle);
        const cy = Math.cos(yHalfAngle);
        const sz = Math.sin(zHalfAngle);
        const cz = Math.cos(zHalfAngle);
        switch (order) {
            case 'xyz':
                newDst[0] = sx * cy * cz + cx * sy * sz;
                newDst[1] = cx * sy * cz - sx * cy * sz;
                newDst[2] = cx * cy * sz + sx * sy * cz;
                newDst[3] = cx * cy * cz - sx * sy * sz;
                break;
            case 'xzy':
                newDst[0] = sx * cy * cz - cx * sy * sz;
                newDst[1] = cx * sy * cz - sx * cy * sz;
                newDst[2] = cx * cy * sz + sx * sy * cz;
                newDst[3] = cx * cy * cz + sx * sy * sz;
                break;
            case 'yxz':
                newDst[0] = sx * cy * cz + cx * sy * sz;
                newDst[1] = cx * sy * cz - sx * cy * sz;
                newDst[2] = cx * cy * sz - sx * sy * cz;
                newDst[3] = cx * cy * cz + sx * sy * sz;
                break;
            case 'yzx':
                newDst[0] = sx * cy * cz + cx * sy * sz;
                newDst[1] = cx * sy * cz + sx * cy * sz;
                newDst[2] = cx * cy * sz - sx * sy * cz;
                newDst[3] = cx * cy * cz - sx * sy * sz;
                break;
            case 'zxy':
                newDst[0] = sx * cy * cz - cx * sy * sz;
                newDst[1] = cx * sy * cz + sx * cy * sz;
                newDst[2] = cx * cy * sz + sx * sy * cz;
                newDst[3] = cx * cy * cz - sx * sy * sz;
                break;
            case 'zyx':
                newDst[0] = sx * cy * cz - cx * sy * sz;
                newDst[1] = cx * sy * cz + sx * cy * sz;
                newDst[2] = cx * cy * sz - sx * sy * cz;
                newDst[3] = cx * cy * cz + sx * sy * sz;
                break;
            default:
                throw new Error(`Unknown rotation order: ${order}`);
        }
        return newDst;
    }
    /**
     * Copies a quaternion. (same as {@link quat.clone})
     * Also see {@link quat.create} and {@link quat.set}
     * @param q - The quaternion.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is a copy of q
     */
    function copy(q, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = q[0];
        newDst[1] = q[1];
        newDst[2] = q[2];
        newDst[3] = q[3];
        return newDst;
    }
    /**
     * Clones a quaternion. (same as {@link quat.copy})
     * Also see {@link quat.create} and {@link quat.set}
     * @param q - The quaternion.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A copy of q.
     */
    const clone = copy;
    /**
     * Adds two quaternions; assumes a and b have the same dimension.
     * @param a - Operand quaternion.
     * @param b - Operand quaternion.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is the sum of a and b.
     */
    function add(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] + b[0];
        newDst[1] = a[1] + b[1];
        newDst[2] = a[2] + b[2];
        newDst[3] = a[3] + b[3];
        return newDst;
    }
    /**
     * Subtracts two quaternions.
     * @param a - Operand quaternion.
     * @param b - Operand quaternion.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is the difference of a and b.
     */
    function subtract(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] - b[0];
        newDst[1] = a[1] - b[1];
        newDst[2] = a[2] - b[2];
        newDst[3] = a[3] - b[3];
        return newDst;
    }
    /**
     * Subtracts two quaternions.
     * @param a - Operand quaternion.
     * @param b - Operand quaternion.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns A quaternion that is the difference of a and b.
     */
    const sub = subtract;
    /**
     * Multiplies a quaternion by a scalar.
     * @param v - The quaternion.
     * @param k - The scalar.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns The scaled quaternion.
     */
    function mulScalar(v, k, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = v[0] * k;
        newDst[1] = v[1] * k;
        newDst[2] = v[2] * k;
        newDst[3] = v[3] * k;
        return newDst;
    }
    /**
     * Multiplies a quaternion by a scalar. (same as mulScalar)
     * @param v - The quaternion.
     * @param k - The scalar.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns The scaled quaternion.
     */
    const scale = mulScalar;
    /**
     * Divides a vector by a scalar.
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns The scaled quaternion.
     */
    function divScalar(v, k, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = v[0] / k;
        newDst[1] = v[1] / k;
        newDst[2] = v[2] / k;
        newDst[3] = v[3] / k;
        return newDst;
    }
    /**
     * Computes the dot product of two quaternions
     * @param a - Operand quaternion.
     * @param b - Operand quaternion.
     * @returns dot product
     */
    function dot(a, b) {
        return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]) + (a[3] * b[3]);
    }
    /**
     * Performs linear interpolation on two quaternions.
     * Given quaternions a and b and interpolation coefficient t, returns
     * a + t * (b - a).
     * @param a - Operand quaternion.
     * @param b - Operand quaternion.
     * @param t - Interpolation coefficient.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns The linear interpolated result.
     */
    function lerp(a, b, t, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] + t * (b[0] - a[0]);
        newDst[1] = a[1] + t * (b[1] - a[1]);
        newDst[2] = a[2] + t * (b[2] - a[2]);
        newDst[3] = a[3] + t * (b[3] - a[3]);
        return newDst;
    }
    /**
     * Computes the length of quaternion
     * @param v - quaternion.
     * @returns length of quaternion.
     */
    function length(v) {
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        const v3 = v[3];
        return Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
    }
    /**
     * Computes the length of quaternion (same as length)
     * @param v - quaternion.
     * @returns length of quaternion.
     */
    const len = length;
    /**
     * Computes the square of the length of quaternion
     * @param v - quaternion.
     * @returns square of the length of quaternion.
     */
    function lengthSq(v) {
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        const v3 = v[3];
        return v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    /**
     * Computes the square of the length of quaternion (same as lengthSq)
     * @param v - quaternion.
     * @returns square of the length of quaternion.
     */
    const lenSq = lengthSq;
    /**
     * Divides a quaternion by its Euclidean length and returns the quotient.
     * @param v - The quaternion.
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns The normalized quaternion.
     */
    function normalize(v, dst) {
        const newDst = (dst ?? new Ctor(4));
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        const v3 = v[3];
        const len = Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
        if (len > 0.00001) {
            newDst[0] = v0 / len;
            newDst[1] = v1 / len;
            newDst[2] = v2 / len;
            newDst[3] = v3 / len;
        }
        else {
            newDst[0] = 0;
            newDst[1] = 0;
            newDst[2] = 0;
            newDst[3] = 1;
        }
        return newDst;
    }
    /**
     * Check if 2 quaternions are approximately equal
     * @param a - Operand quaternion.
     * @param b - Operand quaternion.
     * @returns true if quaternions are approximately equal
     */
    function equalsApproximately(a, b) {
        return Math.abs(a[0] - b[0]) < EPSILON &&
            Math.abs(a[1] - b[1]) < EPSILON &&
            Math.abs(a[2] - b[2]) < EPSILON &&
            Math.abs(a[3] - b[3]) < EPSILON;
    }
    /**
     * Check if 2 quaternions are exactly equal
     * @param a - Operand quaternion.
     * @param b - Operand quaternion.
     * @returns true if quaternions are exactly equal
     */
    function equals(a, b) {
        return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
    }
    /**
     * Creates an identity quaternion
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns an identity quaternion
     */
    function identity(dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = 0;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 1;
        return newDst;
    }
    const tempVec3 = vec3.create();
    const xUnitVec3 = vec3.create();
    const yUnitVec3 = vec3.create();
    /**
     * Computes a quaternion to represent the shortest rotation from one vector to another.
     *
     * @param aUnit - the start vector
     * @param bUnit - the end vector
     * @param dst - quaternion to hold result. If not passed in a new one is created.
     * @returns the result
     */
    function rotationTo(aUnit, bUnit, dst) {
        const newDst = (dst ?? new Ctor(4));
        const dot = vec3.dot(aUnit, bUnit);
        if (dot < -0.999999) {
            vec3.cross(xUnitVec3, aUnit, tempVec3);
            if (vec3.len(tempVec3) < 0.000001) {
                vec3.cross(yUnitVec3, aUnit, tempVec3);
            }
            vec3.normalize(tempVec3, tempVec3);
            fromAxisAngle(tempVec3, Math.PI, newDst);
            return newDst;
        }
        else if (dot > 0.999999) {
            newDst[0] = 0;
            newDst[1] = 0;
            newDst[2] = 0;
            newDst[3] = 1;
            return newDst;
        }
        else {
            vec3.cross(aUnit, bUnit, tempVec3);
            newDst[0] = tempVec3[0];
            newDst[1] = tempVec3[1];
            newDst[2] = tempVec3[2];
            newDst[3] = 1 + dot;
            return normalize(newDst, newDst);
        }
    }
    const tempQuat1 = new Ctor(4);
    const tempQuat2 = new Ctor(4);
    /**
     * Performs a spherical linear interpolation with two control points
     *
     * @param a - the first quaternion
     * @param b - the second quaternion
     * @param c - the third quaternion
     * @param d - the fourth quaternion
     * @param t - Interpolation coefficient 0 to 1
     * @returns result
     */
    function sqlerp(a, b, c, d, t, dst) {
        const newDst = (dst ?? new Ctor(4));
        slerp(a, d, t, tempQuat1);
        slerp(b, c, t, tempQuat2);
        slerp(tempQuat1, tempQuat2, 2 * t * (1 - t), newDst);
        return newDst;
    }
    return {
        create,
        fromValues,
        set,
        fromAxisAngle,
        toAxisAngle,
        angle,
        multiply,
        mul,
        rotateX,
        rotateY,
        rotateZ,
        slerp,
        inverse,
        conjugate,
        fromMat,
        fromEuler,
        copy,
        clone,
        add,
        subtract,
        sub,
        mulScalar,
        scale,
        divScalar,
        dot,
        lerp,
        length,
        len,
        lengthSq,
        lenSq,
        normalize,
        equalsApproximately,
        equals,
        identity,
        rotationTo,
        sqlerp,
    };
}
const cache$1 = new Map();
/**
 *
 * Quat4 math functions.
 *
 * Almost all functions take an optional `newDst` argument. If it is not passed in the
 * functions will create a new `Quat4`. In other words you can do this
 *
 *     const v = quat4.cross(v1, v2);  // Creates a new Quat4 with the cross product of v1 x v2.
 *
 * or
 *
 *     const v = quat4.create();
 *     quat4.cross(v1, v2, v);  // Puts the cross product of v1 x v2 in v
 *
 * The first style is often easier but depending on where it's used it generates garbage where
 * as there is almost never allocation with the second style.
 *
 * It is always safe to pass any vector as the destination. So for example
 *
 *     quat4.cross(v1, v2, v1);  // Puts the cross product of v1 x v2 in v1
 *
 */
function getAPI$1(Ctor) {
    let api = cache$1.get(Ctor);
    if (!api) {
        api = getAPIImpl$1(Ctor);
        cache$1.set(Ctor, api);
    }
    return api;
}

/*
 * Copyright 2022 Gregg Tavares
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/**
 * Generates am typed API for Vec4
 * */
function getAPIImpl(Ctor) {
    /**
     * Creates a vec4; may be called with x, y, z to set initial values.
     * @param x - Initial x value.
     * @param y - Initial y value.
     * @param z - Initial z value.
     * @param w - Initial w value.
     * @returns the created vector
     */
    function create(x, y, z, w) {
        const newDst = new Ctor(4);
        if (x !== undefined) {
            newDst[0] = x;
            if (y !== undefined) {
                newDst[1] = y;
                if (z !== undefined) {
                    newDst[2] = z;
                    if (w !== undefined) {
                        newDst[3] = w;
                    }
                }
            }
        }
        return newDst;
    }
    /**
     * Creates a vec4; may be called with x, y, z to set initial values. (same as create)
     * @param x - Initial x value.
     * @param y - Initial y value.
     * @param z - Initial z value.
     * @param z - Initial w value.
     * @returns the created vector
     */
    const fromValues = create;
    /**
     * Sets the values of a Vec4
     * Also see {@link vec4.create} and {@link vec4.copy}
     *
     * @param x first value
     * @param y second value
     * @param z third value
     * @param w fourth value
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector with its elements set.
     */
    function set(x, y, z, w, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = x;
        newDst[1] = y;
        newDst[2] = z;
        newDst[3] = w;
        return newDst;
    }
    /**
     * Applies Math.ceil to each element of vector
     * @param v - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the ceil of each element of v.
     */
    function ceil(v, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = Math.ceil(v[0]);
        newDst[1] = Math.ceil(v[1]);
        newDst[2] = Math.ceil(v[2]);
        newDst[3] = Math.ceil(v[3]);
        return newDst;
    }
    /**
     * Applies Math.floor to each element of vector
     * @param v - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the floor of each element of v.
     */
    function floor(v, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = Math.floor(v[0]);
        newDst[1] = Math.floor(v[1]);
        newDst[2] = Math.floor(v[2]);
        newDst[3] = Math.floor(v[3]);
        return newDst;
    }
    /**
     * Applies Math.round to each element of vector
     * @param v - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the round of each element of v.
     */
    function round(v, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = Math.round(v[0]);
        newDst[1] = Math.round(v[1]);
        newDst[2] = Math.round(v[2]);
        newDst[3] = Math.round(v[3]);
        return newDst;
    }
    /**
     * Clamp each element of vector between min and max
     * @param v - Operand vector.
     * @param max - Min value, default 0
     * @param min - Max value, default 1
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that the clamped value of each element of v.
     */
    function clamp(v, min = 0, max = 1, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = Math.min(max, Math.max(min, v[0]));
        newDst[1] = Math.min(max, Math.max(min, v[1]));
        newDst[2] = Math.min(max, Math.max(min, v[2]));
        newDst[3] = Math.min(max, Math.max(min, v[3]));
        return newDst;
    }
    /**
     * Adds two vectors; assumes a and b have the same dimension.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the sum of a and b.
     */
    function add(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] + b[0];
        newDst[1] = a[1] + b[1];
        newDst[2] = a[2] + b[2];
        newDst[3] = a[3] + b[3];
        return newDst;
    }
    /**
     * Adds two vectors, scaling the 2nd; assumes a and b have the same dimension.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param scale - Amount to scale b
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the sum of a + b * scale.
     */
    function addScaled(a, b, scale, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] + b[0] * scale;
        newDst[1] = a[1] + b[1] * scale;
        newDst[2] = a[2] + b[2] * scale;
        newDst[3] = a[3] + b[3] * scale;
        return newDst;
    }
    /**
     * Subtracts two vectors.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the difference of a and b.
     */
    function subtract(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] - b[0];
        newDst[1] = a[1] - b[1];
        newDst[2] = a[2] - b[2];
        newDst[3] = a[3] - b[3];
        return newDst;
    }
    /**
     * Subtracts two vectors.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A vector that is the difference of a and b.
     */
    const sub = subtract;
    /**
     * Check if 2 vectors are approximately equal
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns true if vectors are approximately equal
     */
    function equalsApproximately(a, b) {
        return Math.abs(a[0] - b[0]) < EPSILON &&
            Math.abs(a[1] - b[1]) < EPSILON &&
            Math.abs(a[2] - b[2]) < EPSILON &&
            Math.abs(a[3] - b[3]) < EPSILON;
    }
    /**
     * Check if 2 vectors are exactly equal
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns true if vectors are exactly equal
     */
    function equals(a, b) {
        return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
    }
    /**
     * Performs linear interpolation on two vectors.
     * Given vectors a and b and interpolation coefficient t, returns
     * a + t * (b - a).
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param t - Interpolation coefficient.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The linear interpolated result.
     */
    function lerp(a, b, t, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] + t * (b[0] - a[0]);
        newDst[1] = a[1] + t * (b[1] - a[1]);
        newDst[2] = a[2] + t * (b[2] - a[2]);
        newDst[3] = a[3] + t * (b[3] - a[3]);
        return newDst;
    }
    /**
     * Performs linear interpolation on two vectors.
     * Given vectors a and b and interpolation coefficient vector t, returns
     * a + t * (b - a).
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param t - Interpolation coefficients vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns the linear interpolated result.
     */
    function lerpV(a, b, t, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] + t[0] * (b[0] - a[0]);
        newDst[1] = a[1] + t[1] * (b[1] - a[1]);
        newDst[2] = a[2] + t[2] * (b[2] - a[2]);
        newDst[3] = a[3] + t[3] * (b[3] - a[3]);
        return newDst;
    }
    /**
     * Return max values of two vectors.
     * Given vectors a and b returns
     * [max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2])].
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The max components vector.
     */
    function max(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = Math.max(a[0], b[0]);
        newDst[1] = Math.max(a[1], b[1]);
        newDst[2] = Math.max(a[2], b[2]);
        newDst[3] = Math.max(a[3], b[3]);
        return newDst;
    }
    /**
     * Return min values of two vectors.
     * Given vectors a and b returns
     * [min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2])].
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The min components vector.
     */
    function min(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = Math.min(a[0], b[0]);
        newDst[1] = Math.min(a[1], b[1]);
        newDst[2] = Math.min(a[2], b[2]);
        newDst[3] = Math.min(a[3], b[3]);
        return newDst;
    }
    /**
     * Multiplies a vector by a scalar.
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The scaled vector.
     */
    function mulScalar(v, k, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = v[0] * k;
        newDst[1] = v[1] * k;
        newDst[2] = v[2] * k;
        newDst[3] = v[3] * k;
        return newDst;
    }
    /**
     * Multiplies a vector by a scalar. (same as mulScalar)
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The scaled vector.
     */
    const scale = mulScalar;
    /**
     * Divides a vector by a scalar.
     * @param v - The vector.
     * @param k - The scalar.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The scaled vector.
     */
    function divScalar(v, k, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = v[0] / k;
        newDst[1] = v[1] / k;
        newDst[2] = v[2] / k;
        newDst[3] = v[3] / k;
        return newDst;
    }
    /**
     * Inverse a vector.
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The inverted vector.
     */
    function inverse(v, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = 1 / v[0];
        newDst[1] = 1 / v[1];
        newDst[2] = 1 / v[2];
        newDst[3] = 1 / v[3];
        return newDst;
    }
    /**
     * Invert a vector. (same as inverse)
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The inverted vector.
     */
    const invert = inverse;
    /**
     * Computes the dot product of two vectors
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @returns dot product
     */
    function dot(a, b) {
        return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]) + (a[3] * b[3]);
    }
    /**
     * Computes the length of vector
     * @param v - vector.
     * @returns length of vector.
     */
    function length(v) {
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        const v3 = v[3];
        return Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
    }
    /**
     * Computes the length of vector (same as length)
     * @param v - vector.
     * @returns length of vector.
     */
    const len = length;
    /**
     * Computes the square of the length of vector
     * @param v - vector.
     * @returns square of the length of vector.
     */
    function lengthSq(v) {
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        const v3 = v[3];
        return v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    /**
     * Computes the square of the length of vector (same as lengthSq)
     * @param v - vector.
     * @returns square of the length of vector.
     */
    const lenSq = lengthSq;
    /**
     * Computes the distance between 2 points
     * @param a - vector.
     * @param b - vector.
     * @returns distance between a and b
     */
    function distance(a, b) {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        const dz = a[2] - b[2];
        const dw = a[3] - b[3];
        return Math.sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
    }
    /**
     * Computes the distance between 2 points (same as distance)
     * @param a - vector.
     * @param b - vector.
     * @returns distance between a and b
     */
    const dist = distance;
    /**
     * Computes the square of the distance between 2 points
     * @param a - vector.
     * @param b - vector.
     * @returns square of the distance between a and b
     */
    function distanceSq(a, b) {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        const dz = a[2] - b[2];
        const dw = a[3] - b[3];
        return dx * dx + dy * dy + dz * dz + dw * dw;
    }
    /**
     * Computes the square of the distance between 2 points (same as distanceSq)
     * @param a - vector.
     * @param b - vector.
     * @returns square of the distance between a and b
     */
    const distSq = distanceSq;
    /**
     * Divides a vector by its Euclidean length and returns the quotient.
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The normalized vector.
     */
    function normalize(v, dst) {
        const newDst = (dst ?? new Ctor(4));
        const v0 = v[0];
        const v1 = v[1];
        const v2 = v[2];
        const v3 = v[3];
        const len = Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
        if (len > 0.00001) {
            newDst[0] = v0 / len;
            newDst[1] = v1 / len;
            newDst[2] = v2 / len;
            newDst[3] = v3 / len;
        }
        else {
            newDst[0] = 0;
            newDst[1] = 0;
            newDst[2] = 0;
            newDst[3] = 0;
        }
        return newDst;
    }
    /**
     * Negates a vector.
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns -v.
     */
    function negate(v, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = -v[0];
        newDst[1] = -v[1];
        newDst[2] = -v[2];
        newDst[3] = -v[3];
        return newDst;
    }
    /**
     * Copies a vector. (same as {@link vec4.clone})
     * Also see {@link vec4.create} and {@link vec4.set}
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A copy of v.
     */
    function copy(v, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = v[0];
        newDst[1] = v[1];
        newDst[2] = v[2];
        newDst[3] = v[3];
        return newDst;
    }
    /**
     * Clones a vector. (same as {@link vec4.copy})
     * Also see {@link vec4.create} and {@link vec4.set}
     * @param v - The vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns A copy of v.
     */
    const clone = copy;
    /**
     * Multiplies a vector by another vector (component-wise); assumes a and
     * b have the same length.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of products of entries of a and b.
     */
    function multiply(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] * b[0];
        newDst[1] = a[1] * b[1];
        newDst[2] = a[2] * b[2];
        newDst[3] = a[3] * b[3];
        return newDst;
    }
    /**
     * Multiplies a vector by another vector (component-wise); assumes a and
     * b have the same length. (same as mul)
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of products of entries of a and b.
     */
    const mul = multiply;
    /**
     * Divides a vector by another vector (component-wise); assumes a and
     * b have the same length.
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of quotients of entries of a and b.
     */
    function divide(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = a[0] / b[0];
        newDst[1] = a[1] / b[1];
        newDst[2] = a[2] / b[2];
        newDst[3] = a[3] / b[3];
        return newDst;
    }
    /**
     * Divides a vector by another vector (component-wise); assumes a and
     * b have the same length. (same as divide)
     * @param a - Operand vector.
     * @param b - Operand vector.
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The vector of quotients of entries of a and b.
     */
    const div = divide;
    /**
     * Zero's a vector
     * @param dst - vector to hold result. If not passed in a new one is created.
     * @returns The zeroed vector.
     */
    function zero(dst) {
        const newDst = (dst ?? new Ctor(4));
        newDst[0] = 0;
        newDst[1] = 0;
        newDst[2] = 0;
        newDst[3] = 0;
        return newDst;
    }
    /**
     * transform vec4 by 4x4 matrix
     * @param v - the vector
     * @param m - The matrix.
     * @param dst - optional vec4 to store result. If not passed a new one is created.
     * @returns the transformed vector
     */
    function transformMat4(v, m, dst) {
        const newDst = (dst ?? new Ctor(4));
        const x = v[0];
        const y = v[1];
        const z = v[2];
        const w = v[3];
        newDst[0] = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
        newDst[1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
        newDst[2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
        newDst[3] = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
        return newDst;
    }
    /**
     * Treat a 4D vector as a direction and set it's length
     *
     * @param a The vec4 to lengthen
     * @param len The length of the resulting vector
     * @returns The lengthened vector
     */
    function setLength(a, len, dst) {
        const newDst = (dst ?? new Ctor(4));
        normalize(a, newDst);
        return mulScalar(newDst, len, newDst);
    }
    /**
     * Ensure a vector is not longer than a max length
     *
     * @param a The vec4 to limit
     * @param maxLen The longest length of the resulting vector
     * @returns The vector, shortened to maxLen if it's too long
     */
    function truncate(a, maxLen, dst) {
        const newDst = (dst ?? new Ctor(4));
        if (length(a) > maxLen) {
            return setLength(a, maxLen, newDst);
        }
        return copy(a, newDst);
    }
    /**
     * Return the vector exactly between 2 endpoint vectors
     *
     * @param a Endpoint 1
     * @param b Endpoint 2
     * @returns The vector exactly residing between endpoints 1 and 2
     */
    function midpoint(a, b, dst) {
        const newDst = (dst ?? new Ctor(4));
        return lerp(a, b, 0.5, newDst);
    }
    return {
        create,
        fromValues,
        set,
        ceil,
        floor,
        round,
        clamp,
        add,
        addScaled,
        subtract,
        sub,
        equalsApproximately,
        equals,
        lerp,
        lerpV,
        max,
        min,
        mulScalar,
        scale,
        divScalar,
        inverse,
        invert,
        dot,
        length,
        len,
        lengthSq,
        lenSq,
        distance,
        dist,
        distanceSq,
        distSq,
        normalize,
        negate,
        copy,
        clone,
        multiply,
        mul,
        divide,
        div,
        zero,
        transformMat4,
        setLength,
        truncate,
        midpoint,
    };
}
const cache = new Map();
/**
 *
 * Vec4 math functions.
 *
 * Almost all functions take an optional `newDst` argument. If it is not passed in the
 * functions will create a new `Vec4`. In other words you can do this
 *
 *     const v = vec4.cross(v1, v2);  // Creates a new Vec4 with the cross product of v1 x v2.
 *
 * or
 *
 *     const v = vec4.create();
 *     vec4.cross(v1, v2, v);  // Puts the cross product of v1 x v2 in v
 *
 * The first style is often easier but depending on where it's used it generates garbage where
 * as there is almost never allocation with the second style.
 *
 * It is always safe to pass any vector as the destination. So for example
 *
 *     vec4.cross(v1, v2, v1);  // Puts the cross product of v1 x v2 in v1
 *
 */
function getAPI(Ctor) {
    let api = cache.get(Ctor);
    if (!api) {
        api = getAPIImpl(Ctor);
        cache.set(Ctor, api);
    }
    return api;
}

/**
 * Some docs
 * @namespace wgpu-matrix
 */
/**
 * Generate wgpu-matrix API for type
 */
function wgpuMatrixAPI(Mat3Ctor, Mat4Ctor, QuatCtor, Vec2Ctor, Vec3Ctor, Vec4Ctor) {
    return {
        /** @namespace mat3 */
        mat3: getAPI$3(Mat3Ctor),
        /** @namespace mat4 */
        mat4: getAPI$2(Mat4Ctor),
        /** @namespace quat */
        quat: getAPI$1(QuatCtor),
        /** @namespace vec2 */
        vec2: getAPI$5(Vec2Ctor),
        /** @namespace vec3 */
        vec3: getAPI$4(Vec3Ctor),
        /** @namespace vec4 */
        vec4: getAPI(Vec4Ctor),
    };
}
const { 
/**
 * 3x3 Matrix functions that default to returning `Float32Array`
 * @namespace
 */
mat3, 
/**
 * 4x4 Matrix functions that default to returning `Float32Array`
 * @namespace
 */
mat4, 
/**
 * Quaternion functions that default to returning `Float32Array`
 * @namespace
 */
quat, 
/**
 * Vec2 functions that default to returning `Float32Array`
 * @namespace
 */
vec2, 
/**
 * Vec3 functions that default to returning `Float32Array`
 * @namespace
 */
vec3, 
/**
 * Vec3 functions that default to returning `Float32Array`
 * @namespace
 */
vec4, } = wgpuMatrixAPI(Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array);
const { 
/**
 * 3x3 Matrix functions that default to returning `Float64Array`
 * @namespace
 */
mat3: mat3d, 
/**
 * 4x4 Matrix functions that default to returning `Float64Array`
 * @namespace
 */
mat4: mat4d, 
/**
 * Quaternion functions that default to returning `Float64Array`
 * @namespace
 */
quat: quatd, 
/**
 * Vec2 functions that default to returning `Float64Array`
 * @namespace
 */
vec2: vec2d, 
/**
 * Vec3 functions that default to returning `Float64Array`
 * @namespace
 */
vec3: vec3d, 
/**
 * Vec3 functions that default to returning `Float64Array`
 * @namespace
 */
vec4: vec4d, } = wgpuMatrixAPI(Float64Array, Float64Array, Float64Array, Float64Array, Float64Array, Float64Array);
const { 
/**
 * 3x3 Matrix functions that default to returning `number[]`
 * @namespace
 */
mat3: mat3n, 
/**
 * 4x4 Matrix functions that default to returning `number[]`
 * @namespace
 */
mat4: mat4n, 
/**
 * Quaternion functions that default to returning `number[]`
 * @namespace
 */
quat: quatn, 
/**
 * Vec2 functions that default to returning `number[]`
 * @namespace
 */
vec2: vec2n, 
/**
 * Vec3 functions that default to returning `number[]`
 * @namespace
 */
vec3: vec3n, 
/**
 * Vec3 functions that default to returning `number[]`
 * @namespace
 */
vec4: vec4n, } = wgpuMatrixAPI(ZeroArray, Array, Array, Array, Array, Array);


//# sourceMappingURL=wgpu-matrix.module.js.map


/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/define property getters */
/******/ 	(() => {
/******/ 		// define getter functions for harmony exports
/******/ 		__webpack_require__.d = (exports, definition) => {
/******/ 			for(var key in definition) {
/******/ 				if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 					Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 				}
/******/ 			}
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/hasOwnProperty shorthand */
/******/ 	(() => {
/******/ 		__webpack_require__.o = (obj, prop) => (Object.prototype.hasOwnProperty.call(obj, prop))
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	(() => {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = (exports) => {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	})();
/******/ 	
/************************************************************************/
var __webpack_exports__ = {};
/*!*********************!*\
  !*** ./src/main.ts ***!
  \*********************/
__webpack_require__.r(__webpack_exports__);
/* harmony import */ var _physics__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./physics */ "./src/physics.ts");
/* harmony import */ var _renderer__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./renderer */ "./src/renderer.ts");
/* harmony import */ var wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! wgpu-matrix */ "./node_modules/wgpu-matrix/dist/3.x/wgpu-matrix.module.js");



const objects = [];
function createCircle(x, y, radius, color) {
    const object = {
        mass: 1,
        position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(x, y),
        velocity: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.zero(),
        forces: [],
        constraints: [],
        shape: {
            vertices: [
                {
                    position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(-radius / 2, -radius / 2, 0, 1),
                    color: color,
                    shape: _physics__WEBPACK_IMPORTED_MODULE_0__.Shapes.circle,
                    uv: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(0, 1)
                },
                {
                    position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(radius / 2, -radius / 2, 0, 1),
                    color: color,
                    shape: _physics__WEBPACK_IMPORTED_MODULE_0__.Shapes.circle,
                    uv: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(1, 1)
                },
                {
                    position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(-radius / 2, radius / 2, 0, 1),
                    color: color,
                    shape: _physics__WEBPACK_IMPORTED_MODULE_0__.Shapes.circle,
                    uv: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(0, 0)
                },
                {
                    position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(radius / 2, radius / 2, 0, 1),
                    color: color,
                    shape: _physics__WEBPACK_IMPORTED_MODULE_0__.Shapes.circle,
                    uv: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(1, 0)
                },
            ],
            indices: [
                0, 2, 1, 1, 2, 3
            ]
        }
    };
    objects.push(object);
    return object;
}
function createRect(x, y, width, height, color) {
    const object = {
        mass: 2,
        position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(x, y),
        velocity: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.zero(),
        forces: [],
        constraints: [],
        shape: {
            vertices: [
                {
                    position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(-width / 2, -height / 2, 0, 1),
                    color: color,
                    shape: _physics__WEBPACK_IMPORTED_MODULE_0__.Shapes.rect,
                    uv: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(0, 1)
                },
                {
                    position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(width / 2, -height / 2, 0, 1),
                    color: color,
                    shape: _physics__WEBPACK_IMPORTED_MODULE_0__.Shapes.rect,
                    uv: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(1, 1)
                },
                {
                    position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(-width / 2, height / 2, 0, 1),
                    color: color,
                    shape: _physics__WEBPACK_IMPORTED_MODULE_0__.Shapes.rect,
                    uv: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(0, 0)
                },
                {
                    position: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(width / 2, height / 2, 0, 1),
                    color: color,
                    shape: _physics__WEBPACK_IMPORTED_MODULE_0__.Shapes.rect,
                    uv: wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec2.fromValues(1, 0)
                },
            ],
            indices: [
                0, 1, 2, 1, 2, 3
            ]
        }
    };
    objects.push(object);
    return object;
}
const main_viewMatrix = wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.identity();
let height = 0;
let width = 0;
function absoluteToNormalized(x, y) {
    const xNorm = (x / (width) - .5) * 2;
    const yNorm = (-y / (height) + .5) * 2;
    const vCanvas = wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(xNorm, yNorm, 0, 1);
    const vWorld = wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.transformMat4(vCanvas, wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.inverse(main_viewMatrix));
    return [vWorld[0], vWorld[1]];
}
function deltaToNormalized(x, y) {
    const ret = [x / width * (2 / main_viewMatrix[0]), -y / height * (2 / main_viewMatrix[5])];
    return ret;
}
async function start() {
    const canvas = document.querySelector("canvas");
    width = canvas.width;
    height = canvas.height;
    canvas.addEventListener("mousewheel", (ev) => {
        ev.preventDefault();
        if (ev.ctrlKey) {
            if (!(ev.deltaY > 0 && main_viewMatrix[0] <= .8) && !(ev.deltaY < 0 && main_viewMatrix[0] >= 4)) {
                let [x, y] = absoluteToNormalized(ev.offsetX, ev.offsetY);
                const mouseToOrigin = wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.translation(wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(x, y, 0, 0));
                const deltaZoom = ev.deltaY / 60;
                const zoom = wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.uniformScale(wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.identity(), 1 - deltaZoom);
                [x, y] = absoluteToNormalized(ev.offsetX, ev.offsetY);
                const undoMouseToOrigin = wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.translation(wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(-x, -y, 0, 0));
                wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.multiply(main_viewMatrix, mouseToOrigin, main_viewMatrix);
                wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.multiply(main_viewMatrix, zoom, main_viewMatrix);
                wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.multiply(main_viewMatrix, undoMouseToOrigin, main_viewMatrix);
            }
        }
    });
    const mousePos = document.getElementById("mousePos");
    canvas.addEventListener("mousemove", (ev) => {
        const [x, y] = absoluteToNormalized(ev.offsetX, ev.offsetY);
        mousePos.innerText = `Screen: ${ev.offsetX}, ${ev.offsetY}, Camera: ${x}, ${y}`;
        if (ev.buttons === 1) {
            let [x, y] = deltaToNormalized(ev.movementX, ev.movementY);
            const translation = wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.translation(wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.vec4.fromValues(x, y, 0, 0));
            wgpu_matrix__WEBPACK_IMPORTED_MODULE_2__.mat4.multiply(main_viewMatrix, translation, main_viewMatrix);
        }
    });
    const environmentBmp = await fetch("/environment.bmp").then(resp => resp.blob()).then(blob => createImageBitmap(blob));
    const dirtBmp = await fetch("/dirt_texture.jpg").then(resp => resp.blob()).then(blob => createImageBitmap(blob));
    const renderer = await (0,_renderer__WEBPACK_IMPORTED_MODULE_1__.requestRenderer)(canvas, [environmentBmp, dirtBmp]);
    createRect(0, 0, 2, 2, _physics__WEBPACK_IMPORTED_MODULE_0__.red);
    renderer.addVertexBuffer(objects);
    renderer.viewMatrix = main_viewMatrix;
    renderer.render();
    let paused = true;
    let frameHandle = 0;
    let prevTime = performance.now();
    const frame = (time) => {
        const deltaTime = (time - prevTime) / 1000;
        prevTime = time;
        (0,_physics__WEBPACK_IMPORTED_MODULE_0__.simulate)(objects, deltaTime);
        renderer.addVertexBuffer(objects);
        renderer.viewMatrix = main_viewMatrix;
        renderer.render();
        if (!paused) {
            frameHandle = requestAnimationFrame(frame);
        }
    };
    const pauseButton = document.getElementById("pause");
    pauseButton.addEventListener("click", () => {
        if (paused) {
            prevTime = performance.now();
            frameHandle = requestAnimationFrame(frame);
            pauseButton.innerText = "Pause";
            paused = false;
        }
        else {
            cancelAnimationFrame(frameHandle);
            pauseButton.innerText = "Play";
            paused = true;
        }
    });
}
start();

/******/ })()
;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,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Q0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0EscUJBQXFCLGlCQUFpQjtBQUN0QztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsdUJBQXVCO0FBQ3ZCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsdUJBQXVCO0FBQ3ZCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCLG1CQUFtQixLQUFLO0FBQ3pDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHlCQUF5QjtBQUN6QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwwQ0FBMEM7QUFDMUM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGtEQUFrRDtBQUNsRDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxnREFBZ0Q7QUFDaEQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGtDQUFrQyxpQkFBaUI7QUFDbkQsaUJBQWlCLG1CQUFtQixLQUFLO0FBQ3pDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxrQ0FBa0MsZ0JBQWdCO0FBQ2xELGlCQUFpQixtQkFBbUIsS0FBSztBQUN6QztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrREFBK0Q7QUFDL0Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrREFBK0Q7QUFDL0Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDREQUE0RDtBQUM1RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDREQUE0RDtBQUM1RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QjtBQUN2QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QjtBQUN2QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCLG1CQUFtQixLQUFLO0FBQ3pDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHlCQUF5QjtBQUN6QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDBDQUEwQztBQUMxQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esa0RBQWtEO0FBQ2xEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsZ0RBQWdEO0FBQ2hEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esa0NBQWtDLGlCQUFpQjtBQUNuRCxpQkFBaUIsbUJBQW1CLEtBQUs7QUFDekM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esa0NBQWtDLGdCQUFnQjtBQUNsRCxpQkFBaUIsbUJBQW1CLEtBQUs7QUFDekM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsK0RBQStEO0FBQy9EO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrREFBK0Q7QUFDL0Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDREQUE0RDtBQUM1RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsNERBQTREO0FBQzVEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxlQUFlLGNBQWM7QUFDN0IsZUFBZSxjQUFjO0FBQzdCLGVBQWUsUUFBUTtBQUN2QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxlQUFlLGNBQWM7QUFDN0IsZUFBZSxjQUFjO0FBQzdCLGVBQWUsUUFBUTtBQUN2QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxlQUFlLGNBQWM7QUFDN0IsZUFBZSxjQUFjO0FBQzdCLGVBQWUsUUFBUTtBQUN2QjtBQUNBLGlCQUFpQixNQUFNO0FBQ3ZCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQixtQkFBbUIsS0FBSztBQUN6QztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGtDQUFrQyxpQkFBaUI7QUFDbkQsaUJBQWlCLG1CQUFtQixLQUFLO0FBQ3pDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlDQUFpQyxnQkFBZ0I7QUFDakQsaUJBQWlCLG1CQUFtQixLQUFLO0FBQ3pDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLG9EQUFvRDtBQUNwRDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSw2REFBNkQ7QUFDN0Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLG9EQUFvRDtBQUNwRDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSw2REFBNkQ7QUFDN0Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHFEQUFxRDtBQUNyRDtBQUNBO0FBQ0E7QUFDQTtBQUNBLDhDQUE4QztBQUM5QztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsNENBQTRDO0FBQzVDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxpQkFBaUIsbUJBQW1CLEtBQUs7QUFDekM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esa0NBQWtDLGlCQUFpQjtBQUNuRCxpQkFBaUIsbUJBQW1CLEtBQUs7QUFDekM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxpQ0FBaUMsZ0JBQWdCO0FBQ2pELGlCQUFpQixtQkFBbUIsS0FBSztBQUN6QztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLG9EQUFvRDtBQUNwRDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDZEQUE2RDtBQUM3RDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esd0JBQXdCO0FBQ3hCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSx1QkFBdUI7QUFDdkI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCLG1CQUFtQixLQUFLO0FBQ3pDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsaUJBQWlCO0FBQ2pCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsK0NBQStDO0FBQy9DO0FBQ0Esd0NBQXdDO0FBQ3hDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwyREFBMkQsTUFBTTtBQUNqRTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHNDQUFzQyxpQkFBaUI7QUFDdkQsaUJBQWlCLG1CQUFtQixLQUFLO0FBQ3pDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esc0NBQXNDLGdCQUFnQjtBQUN0RCxpQkFBaUIsbUJBQW1CLEtBQUs7QUFDekM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsNkJBQTZCO0FBQzdCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVDQUF1QztBQUN2QztBQUNBO0FBQ0E7QUFDQTtBQUNBLGdDQUFnQztBQUNoQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxpQ0FBaUM7QUFDakM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLHVCQUF1QjtBQUN2QjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsdUJBQXVCO0FBQ3ZCO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLGlCQUFpQixtQkFBbUIsS0FBSztBQUN6QztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSx5QkFBeUI7QUFDekI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLDBDQUEwQztBQUMxQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0Esa0NBQWtDLGlCQUFpQjtBQUNuRCxpQkFBaUIsbUJBQW1CLEtBQUs7QUFDekM7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxrQ0FBa0MsZ0JBQWdCO0FBQ2xELGlCQUFpQixtQkFBbUIsS0FBSztBQUN6QztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrREFBK0Q7QUFDL0Q7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsK0RBQStEO0FBQy9EO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSw0REFBNEQ7QUFDNUQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsNERBQTREO0FBQzVEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxzQ0FBc0M7QUFDdEM7QUFDQTtBQUNBO0FBQ0E7QUFDQSwrQkFBK0I7QUFDL0I7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsZ0NBQWdDO0FBQ2hDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FB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QUksQ0FBQyxNQUFNLEVBQUUsQ0FBQztZQUNWLFdBQVcsR0FBRyxxQkFBcUIsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUMvQyxDQUFDO0lBQ0wsQ0FBQztJQUVELE1BQU0sV0FBVyxHQUFHLFFBQVEsQ0FBQyxjQUFjLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDckQsV0FBVyxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRSxHQUFHLEVBQUU7UUFDdkMsSUFBSSxNQUFNLEVBQUUsQ0FBQztZQUNULFFBQVEsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7WUFDN0IsV0FBVyxHQUFHLHFCQUFxQixDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQzNDLFdBQVcsQ0FBQyxTQUFTLEdBQUcsT0FBTyxDQUFDO1lBQ2hDLE1BQU0sR0FBRyxLQUFLLENBQUM7UUFDbkIsQ0FBQzthQUFNLENBQUM7WUFDSixvQkFBb0IsQ0FBQyxXQUFXLENBQUMsQ0FBQztZQUNsQyxXQUFXLENBQUMsU0FBUyxHQUFHLE1BQU07WUFDOUIsTUFBTSxHQUFHLElBQUksQ0FBQztRQUNsQixDQUFDO0lBQ0wsQ0FBQyxDQUFDLENBQUM7QUFDUCxDQUFDO0FBRUQsS0FBSyxFQUFFLENBQUMiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8vLi9zcmMvcmVuZGVyLndnc2wiLCJ3ZWJwYWNrOi8vLy4vc3JjL3BoeXNpY3MudHMiLCJ3ZWJwYWNrOi8vLy4vc3JjL3JlbmRlcmVyLnRzIiwid2VicGFjazovLy8uL25vZGVfbW9kdWxlcy93Z3B1LW1hdHJpeC9kaXN0LzMueC93Z3B1LW1hdHJpeC5tb2R1bGUuanMiLCJ3ZWJwYWNrOi8vL3dlYnBhY2svYm9vdHN0cmFwIiwid2VicGFjazovLy93ZWJwYWNrL3J1bnRpbWUvZGVmaW5lIHByb3BlcnR5IGdldHRlcnMiLCJ3ZWJwYWNrOi8vL3dlYnBhY2svcnVudGltZS9oYXNPd25Qcm9wZXJ0eSBzaG9ydGhhbmQiLCJ3ZWJwYWNrOi8vL3dlYnBhY2svcnVudGltZS9tYWtlIG5hbWVzcGFjZSBvYmplY3QiLCJ3ZWJwYWNrOi8vLy4vc3JjL21haW4udHMiXSwic291cmNlc0NvbnRlbnQiOlsiZXhwb3J0IGRlZmF1bHQgXCJzdHJ1Y3QgVW5pZm9ybXMge1xcclxcbiAgdmlld01hdHJpeCA6IG1hdDR4NGYsXFxyXFxufVxcclxcblxcclxcbkBncm91cCgwKSBAYmluZGluZygwKSB2YXI8dW5pZm9ybT4gdW5pZm9ybXMgOiBVbmlmb3JtcztcXHJcXG5cXHJcXG5zdHJ1Y3QgVmVydGV4T3V0IHtcXHJcXG4gIEBidWlsdGluKHBvc2l0aW9uKSBwb3NpdGlvbiA6IHZlYzRmLFxcclxcbiAgQGxvY2F0aW9uKDApIGNvbG9yIDogdmVjNGYsXFxyXFxuICBAbG9jYXRpb24oMSkgdXYgOiB2ZWMyZixcXHJcXG4gIEBsb2NhdGlvbigyKSBzaGFwZTogZjMyLFxcclxcbn1cXHJcXG5cXHJcXG5AdmVydGV4XFxyXFxuZm4gdmVydGV4X21haW4oQGxvY2F0aW9uKDApIHBvc2l0aW9uOiB2ZWMyZixcXHJcXG4gICAgICAgICAgICAgICBAbG9jYXRpb24oMSkgY29sb3I6IHZlYzRmLFxcclxcbiAgICAgICAgICAgICAgICBAbG9jYXRpb24oMikgdXYgOiB2ZWMyZixcXHJcXG4gICAgICAgICAgICAgICAgQGxvY2F0aW9uKDMpIHNoYXBlOiBmMzIsXFxyXFxuXFxyXFxuKSAtPiBWZXJ0ZXhPdXRcXHJcXG57XFxyXFxuICB2YXIgb3V0cHV0IDogVmVydGV4T3V0O1xcclxcbiAgb3V0cHV0LnBvc2l0aW9uID0gdW5pZm9ybXMudmlld01hdHJpeCAqIHZlYzQocG9zaXRpb24sIDAuLCAxLik7XFxyXFxuICBvdXRwdXQuY29sb3IgPSBjb2xvcjtcXHJcXG4gIG91dHB1dC51diA9IHV2O1xcclxcbiAgb3V0cHV0LnNoYXBlID0gc2hhcGU7XFxyXFxuICByZXR1cm4gb3V0cHV0O1xcclxcbn1cXHJcXG5cXHJcXG5AZ3JvdXAoMSkgQGJpbmRpbmcoMCkgdmFyIG15U2FtcGxlcjogc2FtcGxlcjtcXHJcXG5AZ3JvdXAoMSkgQGJpbmRpbmcoMSkgdmFyIGVudmlyb25tZW50VGV4dHVyZTogdGV4dHVyZV8yZDxmMzI+O1xcclxcbkBncm91cCgxKSBAYmluZGluZygyKSB2YXIgZGlydFRleHR1cmU6IHRleHR1cmVfMmQ8ZjMyPjtcXHJcXG5cXHJcXG5AZnJhZ21lbnRcXHJcXG5mbiBmcmFnbWVudF9tYWluKGZyYWdEYXRhOiBWZXJ0ZXhPdXQpIC0+IEBsb2NhdGlvbigwKSB2ZWM0ZlxcclxcbntcXHJcXG4gIGxldCBlbnZpcm9ubWVudFNhbXBsZSA9IHRleHR1cmVTYW1wbGUoZW52aXJvbm1lbnRUZXh0dXJlLCBteVNhbXBsZXIsIGZyYWdEYXRhLnV2KTtcXHJcXG4gIGxldCBzYW1wbGUgPSB0ZXh0dXJlU2FtcGxlKGRpcnRUZXh0dXJlLCBteVNhbXBsZXIsIGZyYWdEYXRhLnV2KTtcXHJcXG4gIC8vIEhhbmRsZSBkZWJ1ZyBjaXJjbGVzXFxyXFxuICBpZiAoZnJhZ0RhdGEuc2hhcGUgPT0gMSkge1xcclxcbiAgICBsZXQgb3V0T2ZDaXJjbGUgPSBmcmFnRGF0YS5zaGFwZSA9PSAxLiAmJiBsZW5ndGgoZnJhZ0RhdGEudXYgLSB2ZWMyKC41KSkgPj0gLjU7XFxyXFxuICAgIGlmIChvdXRPZkNpcmNsZSkge1xcclxcbiAgICAgIGRpc2NhcmQ7XFxyXFxuICAgIH1cXHJcXG4gICAgcmV0dXJuIGZyYWdEYXRhLmNvbG9yO1xcclxcbiAgfVxcclxcbiAgLy8gT3RoZXJ3aXNlLCByZWN0YW5nbGUgdGV4dHVyZVxcclxcbiAgaWYgKGVudmlyb25tZW50U2FtcGxlLnggPiAwLikge1xcclxcbiAgICByZXR1cm4gc2FtcGxlO1xcclxcbiAgfSBlbHNlIGlmIChlbnZpcm9ubWVudFNhbXBsZS55ID4gMCkge1xcclxcbiAgICByZXR1cm4gdmVjNCgwLixlbnZpcm9ubWVudFNhbXBsZS55LDAuLDEuKTtcXHJcXG4gIH0gZWxzZSB7XFxyXFxuICAgIGRpc2NhcmQ7XFxyXFxuICB9XFxyXFxuICByZXR1cm4gdmVjNCgxLik7XFxyXFxufVwiOyIsImltcG9ydCB7IHZlYzQsIFZlYzQsIFZlYzIsIHZlYzIgfSBmcm9tIFwid2dwdS1tYXRyaXhcIjtcclxuXHJcbmV4cG9ydCBjb25zdCB3aGl0ZSA9IHZlYzQuZnJvbVZhbHVlcygxLCAxLCAxLCAxKTtcclxuZXhwb3J0IGNvbnN0IHJlZCA9IHZlYzQuZnJvbVZhbHVlcygxLCAwLCAwLCAxKTtcclxuZXhwb3J0IGNvbnN0IGJsdWUgPSB2ZWM0LmZyb21WYWx1ZXMoMCwgMCwgMSwgMSk7XHJcbmV4cG9ydCBjb25zdCBncmVlbiA9IHZlYzQuZnJvbVZhbHVlcygwLCAxLCAwLCAxKTtcclxuZXhwb3J0IGNvbnN0IGJsYWNrID0gdmVjNC5mcm9tVmFsdWVzKDAsIDAsIDAsIDEpO1xyXG5cclxuZXhwb3J0IGludGVyZmFjZSBWZXJ0ZXgge1xyXG4gICAgcG9zaXRpb246IFZlYzIsXHJcbiAgICBjb2xvcjogVmVjNCxcclxuICAgIHV2OiBWZWMyLFxyXG4gICAgc2hhcGU6IFNoYXBlc1xyXG59XHJcblxyXG5leHBvcnQgZW51bSBTaGFwZXMge1xyXG4gICAgcmVjdCA9IDAsXHJcbiAgICBjaXJjbGUgPSAxLFxyXG59XHJcblxyXG5leHBvcnQgaW50ZXJmYWNlIFNoYXBlIHtcclxuICAgIHZlcnRpY2VzOiBWZXJ0ZXhbXSxcclxuICAgIGluZGljZXM6IG51bWJlcltdXHJcbn1cclxuXHJcbmV4cG9ydCBpbnRlcmZhY2UgUGh5c09iamVjdCB7XHJcbiAgICBwb3NpdGlvbjogVmVjMixcclxuICAgIHZlbG9jaXR5OiBWZWMyLFxyXG4gICAgc2hhcGU6IFNoYXBlLFxyXG4gICAgbWFzczogbnVtYmVyLFxyXG4gICAgZm9yY2VzOiBGb3JjZVtdLFxyXG4gICAgY29uc3RyYWludHM6IENvbnN0cmFpbnRbXSxcclxuICAgIHN1bUFjY2VsZXJhdGlvbj86IFZlYzIsXHJcbn1cclxuXHJcbmV4cG9ydCBpbnRlcmZhY2UgQ29uc3RyYWludCB7XHJcbiAgICBhcHBseUNvbnN0cmFpbnQoYWNjZWxlcmF0aW9uOiBWZWMyKTogVmVjMixcclxufVxyXG5cclxuZXhwb3J0IGludGVyZmFjZSBGb3JjZSB7XHJcbiAgICBnZXRBY2NlbGVyYXRpb246IChvYmplY3Q6IFBoeXNPYmplY3QpID0+IFZlYzIsXHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBzaW11bGF0ZShvYmplY3RzOiBQaHlzT2JqZWN0W10sIGRlbHRhVGltZTogbnVtYmVyKSB7XHJcbiAgICBvYmplY3RzLmZvckVhY2gob2JqID0+IHtcclxuICAgICAgICBsZXQgYWNjZWxlcmF0aW9uID0gdmVjMi56ZXJvKCk7XHJcbiAgICAgICAgb2JqLmZvcmNlcy5mb3JFYWNoKGZvcmNlID0+IHtcclxuICAgICAgICAgICAgY29uc3QgYSA9IGZvcmNlLmdldEFjY2VsZXJhdGlvbihvYmopO1xyXG4gICAgICAgICAgICB2ZWMyLmFkZChhY2NlbGVyYXRpb24sIGEsIGFjY2VsZXJhdGlvbik7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgb2JqLmNvbnN0cmFpbnRzLmZvckVhY2goY29uc3RyYWludCA9PiB7XHJcbiAgICAgICAgICAgIGFjY2VsZXJhdGlvbiA9IGNvbnN0cmFpbnQuYXBwbHlDb25zdHJhaW50KGFjY2VsZXJhdGlvbik7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgaWYgKG9iai5jb25zdHJhaW50cy5sZW5ndGggPiAwKSB7XHJcbiAgICAgICAgICAgIC8vIGNvbnNvbGUubG9nKFwibmV0IGFjY2VsXCIsIGFjY2VsZXJhdGlvblswXSwgYWNjZWxlcmF0aW9uWzFdLCBcInZlbFwiLCBvYmoudmVsb2NpdHlbMF0sIG9iai52ZWxvY2l0eVsxXSk7XHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICBvYmoudmVsb2NpdHkgPSB2ZWMyLmFkZFNjYWxlZChvYmoudmVsb2NpdHksIGFjY2VsZXJhdGlvbiwgZGVsdGFUaW1lKTtcclxuICAgICAgICBvYmoucG9zaXRpb24gPSB2ZWMyLmFkZFNjYWxlZChvYmoucG9zaXRpb24sIG9iai52ZWxvY2l0eSwgZGVsdGFUaW1lKTtcclxuICAgIH0pO1xyXG59XHJcblxyXG5leHBvcnQgY2xhc3MgR2xvYmFsR3Jhdml0eSBpbXBsZW1lbnRzIEZvcmNlIHtcclxuICAgIGFjY2VsZXJhdGlvbjogbnVtYmVyXHJcbiAgICBjb25zdHJ1Y3RvcihhY2NlbGVyYXRpb246IG51bWJlcikge1xyXG4gICAgICAgIHRoaXMuYWNjZWxlcmF0aW9uID0gYWNjZWxlcmF0aW9uO1xyXG4gICAgfVxyXG4gICAgZ2V0QWNjZWxlcmF0aW9uKCk6IFZlYzIge1xyXG4gICAgICAgIHJldHVybiB2ZWMyLmZyb21WYWx1ZXMoMCwgdGhpcy5hY2NlbGVyYXRpb24pO1xyXG4gICAgfVxyXG59XHJcblxyXG5lbnVtIElucHV0RGlyZWN0aW9uIHtcclxuICAgIGxlZnQgPSAtMSxcclxuICAgIHplcm8gPSAwLFxyXG4gICAgcmlnaHQgPSAxLFxyXG59XHJcblxyXG5leHBvcnQgY2xhc3MgQ2FydENvbnRyb2wgaW1wbGVtZW50cyBGb3JjZSB7XHJcbiAgICBmb3JjZTogbnVtYmVyO1xyXG4gICAgZGlyZWN0aW9uOiBudW1iZXJcclxuICAgIGNvbnN0cnVjdG9yKGZvcmNlOiBudW1iZXIpIHtcclxuICAgICAgICB0aGlzLmZvcmNlID0gZm9yY2U7XHJcbiAgICAgICAgdGhpcy5kaXJlY3Rpb24gPSBJbnB1dERpcmVjdGlvbi56ZXJvO1xyXG4gICAgICAgIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoXCJrZXlwcmVzc1wiLChldmVudCkgPT4ge1xyXG4gICAgICAgICAgICBpZiAoZXZlbnQua2V5ID09IFwiYVwiICYmIHRoaXMuZGlyZWN0aW9uID4gLTEpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMuZGlyZWN0aW9uICs9IElucHV0RGlyZWN0aW9uLmxlZnQ7XHJcbiAgICAgICAgICAgIH0gZWxzZSBpZiAoZXZlbnQua2V5ID09IFwiZFwiICYmIHRoaXMuZGlyZWN0aW9uIDwgMSkge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5kaXJlY3Rpb24gKz0gSW5wdXREaXJlY3Rpb24ucmlnaHQ7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9KTtcclxuICAgICAgICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKFwia2V5dXBcIiwoZXZlbnQpID0+IHtcclxuICAgICAgICAgICAgaWYgKGV2ZW50LmtleSA9PSBcImFcIikge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5kaXJlY3Rpb24gLT0gSW5wdXREaXJlY3Rpb24ubGVmdDtcclxuICAgICAgICAgICAgfSBlbHNlIGlmIChldmVudC5rZXkgPT0gXCJkXCIpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMuZGlyZWN0aW9uIC09IElucHV0RGlyZWN0aW9uLnJpZ2h0O1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIC8vIGNvbnNvbGUubG9nKHRoaXMuZGlyZWN0aW9uKTtcclxuICAgICAgICB9KTtcclxuICAgIH1cclxuICAgIGdldEFjY2VsZXJhdGlvbihvYmplY3Q6IFBoeXNPYmplY3QpOiBWZWMyIHtcclxuICAgICAgICByZXR1cm4gdmVjMi5mcm9tVmFsdWVzKHRoaXMuZGlyZWN0aW9uICogKHRoaXMuZm9yY2UgLyBvYmplY3QubWFzcyksIDApO1xyXG4gICAgfVxyXG59XHJcblxyXG5leHBvcnQgY2xhc3MgRHJhZyBpbXBsZW1lbnRzIEZvcmNlIHtcclxuICAgIGNvZWZmaWNpZW50OiBudW1iZXJcclxuICAgIGNvbnN0cnVjdG9yKGNvZWZmaWNpZW50OiBudW1iZXIpIHtcclxuICAgICAgICB0aGlzLmNvZWZmaWNpZW50ID0gY29lZmZpY2llbnQ7XHJcbiAgICB9XHJcbiAgICBnZXRBY2NlbGVyYXRpb24ob2JqZWN0OiBQaHlzT2JqZWN0KTogVmVjMiB7XHJcbiAgICAgICAgcmV0dXJuIHZlYzIuc2NhbGUodmVjMi5uZWdhdGUob2JqZWN0LnZlbG9jaXR5KSwgdGhpcy5jb2VmZmljaWVudCAvIG9iamVjdC5tYXNzKTtcclxuICAgIH1cclxufVxyXG5cclxuZXhwb3J0IGNsYXNzIFBpdm90IGltcGxlbWVudHMgQ29uc3RyYWludCB7XHJcbiAgICBvYmplY3Q6IFBoeXNPYmplY3Q7XHJcbiAgICBwb3NpdGlvbjogVmVjMjtcclxuICAgIGxlbmd0aDogbnVtYmVyO1xyXG4gICAgY29uc3RydWN0b3Iob2JqZWN0OiBQaHlzT2JqZWN0LCB4OiBudW1iZXIsIHk6IG51bWJlcikge1xyXG4gICAgICAgIHRoaXMucG9zaXRpb24gPSB2ZWMyLmZyb21WYWx1ZXMoeCx5KTtcclxuICAgICAgICB0aGlzLm9iamVjdCA9IG9iamVjdDtcclxuICAgICAgICB0aGlzLmxlbmd0aCA9IHZlYzIubGVuKHZlYzIuc3ViKHRoaXMucG9zaXRpb24sIHRoaXMub2JqZWN0LnBvc2l0aW9uKSk7XHJcbiAgICB9XHJcbiAgICBhcHBseUNvbnN0cmFpbnQoYWNjZWxlcmF0aW9uOiBWZWMyKTogVmVjMiB7XHJcbiAgICAgICAgLy8gY29uc29sZS5sb2coYWNjZWxlcmF0aW9uKTtcclxuICAgICAgICBjb25zdCB0ZW5zaW9uRGlyZWN0aW9uID0gdmVjMi5ub3JtYWxpemUodmVjMi5zdWIodGhpcy5wb3NpdGlvbiwgdGhpcy5vYmplY3QucG9zaXRpb24pKTtcclxuICAgICAgICBjb25zdCB0ZW5zaW9uID0gdmVjMi5zY2FsZSh0ZW5zaW9uRGlyZWN0aW9uLCBcclxuICAgICAgICAgICAgLSh2ZWMyLmRvdChhY2NlbGVyYXRpb24sIHRlbnNpb25EaXJlY3Rpb24pICsgKGxlbmd0aCAqIHZlYzIubGVuU3EodGhpcy5vYmplY3QudmVsb2NpdHkpKSkpO1xyXG4gICAgICAgIC8vIGNvbnNvbGUubG9nKHRlbnNpb24pO1xyXG4gICAgICAgIHZlYzIuYWRkKHRoaXMucG9zaXRpb24sIHZlYzIuc2NhbGUodGVuc2lvbkRpcmVjdGlvbiwgLXRoaXMubGVuZ3RoKSwgdGhpcy5vYmplY3QucG9zaXRpb24pO1xyXG4gICAgICAgIHJldHVybiB2ZWMyLmFkZChhY2NlbGVyYXRpb24sIHRlbnNpb24pO1xyXG4gICAgfVxyXG59IiwiaW1wb3J0IHsgbWF0NCwgTWF0NCwgdmVjMiwgdmVjNCB9IGZyb20gXCJ3Z3B1LW1hdHJpeFwiO1xyXG5pbXBvcnQgeyBQaHlzT2JqZWN0IH0gZnJvbSBcIi4vcGh5c2ljc1wiO1xyXG5pbXBvcnQgc2hhZGVyQ29kZSBmcm9tIFwiLi9yZW5kZXIud2dzbFwiO1xyXG5cclxuY29uc3QgVkVSVEVYX1NUUklERSA9IDk7XHJcblxyXG50eXBlIFRleHR1cmVIYW5kbGUgPSBudW1iZXI7XHJcblxyXG5leHBvcnQgY2xhc3MgUmVuZGVyZXIge1xyXG4gICAgZGV2aWNlOiBHUFVEZXZpY2U7XHJcbiAgICBjYW52YXNDb250ZXh0OiBHUFVDYW52YXNDb250ZXh0O1xyXG4gICAgdmVydGV4QnVmZmVyPzogR1BVQnVmZmVyO1xyXG4gICAgaW5kZXhCdWZmZXI/OiBHUFVCdWZmZXI7XHJcbiAgICBpbmRleENvdW50OiBudW1iZXI7XHJcbiAgICB2ZXJ0ZXhCdWZmZXJEZXNjcmlwdG9yOiBHUFVWZXJ0ZXhCdWZmZXJMYXlvdXRbXTtcclxuICAgIHBpcGVsaW5lRGVzY3JpcHRvcjogR1BVUmVuZGVyUGlwZWxpbmVEZXNjcmlwdG9yO1xyXG4gICAgdW5pZm9ybUJ1ZmZlcjogR1BVQnVmZmVyO1xyXG4gICAgdW5pZm9ybUJpbmRHcm91cDogR1BVQmluZEdyb3VwO1xyXG4gICAgdGV4dHVyZXNCaW5kR3JvdXA6IEdQVUJpbmRHcm91cDtcclxuICAgIHZpZXdNYXRyaXg6IE1hdDQ7XHJcbiAgICB0ZXh0dXJlczogR1BVVGV4dHVyZVtdID0gW107XHJcbiAgICBjb25zdHJ1Y3RvcihkZXZpY2U6IEdQVURldmljZSwgY2FudmFzQ29udGV4dDogR1BVQ2FudmFzQ29udGV4dCwgdGV4dHVyZXM6IEltYWdlQml0bWFwW10pIHtcclxuICAgICAgICB0aGlzLmRldmljZSA9IGRldmljZTtcclxuICAgICAgICB0aGlzLmNhbnZhc0NvbnRleHQgPSBjYW52YXNDb250ZXh0O1xyXG5cclxuICAgICAgICB0aGlzLmFkZFRleHR1cmUodGV4dHVyZXNbMF0pOyAvLyBlbnZpcm9ubWVudFxyXG4gICAgICAgIHRoaXMuYWRkVGV4dHVyZSh0ZXh0dXJlc1sxXSk7IC8vIGRpcnRcclxuXHJcbiAgICAgICAgdGhpcy52aWV3TWF0cml4ID0gbWF0NC5pZGVudGl0eSgpO1xyXG4gICAgICAgIHRoaXMudW5pZm9ybUJ1ZmZlciA9IGRldmljZS5jcmVhdGVCdWZmZXIoe1xyXG4gICAgICAgICAgc2l6ZTogNCAqIDE2LFxyXG4gICAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlVOSUZPUk0gfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVFxyXG4gICAgICAgIH0pXHJcblxyXG4gICAgICAgIGNvbnN0IHVuaWZvcm1CaW5kR3JvdXBMYXlvdXQgPSBkZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcclxuICAgICAgICAgIGVudHJpZXM6IFtcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIGJpbmRpbmc6IDAsXHJcbiAgICAgICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuVkVSVEVYLFxyXG4gICAgICAgICAgICAgIGJ1ZmZlcjoge1xyXG4gICAgICAgICAgICAgICAgdHlwZTogXCJ1bmlmb3JtXCIsXHJcbiAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgXVxyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIHRoaXMudW5pZm9ybUJpbmRHcm91cCA9IGRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xyXG4gICAgICAgICAgbGF5b3V0OiB1bmlmb3JtQmluZEdyb3VwTGF5b3V0LFxyXG4gICAgICAgICAgZW50cmllczogW1xyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgYmluZGluZzogMCxcclxuICAgICAgICAgICAgICByZXNvdXJjZToge1xyXG4gICAgICAgICAgICAgICAgYnVmZmVyOiB0aGlzLnVuaWZvcm1CdWZmZXIsXHJcbiAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgXVxyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIFxyXG4gICAgICAgIGNvbnN0IHNhbXBsZXIgPSBkZXZpY2UuY3JlYXRlU2FtcGxlcih7XHJcbiAgICAgICAgICBtYWdGaWx0ZXI6IFwibmVhcmVzdFwiLFxyXG4gICAgICAgICAgbWluRmlsdGVyOiBcIm5lYXJlc3RcIixcclxuICAgICAgICB9KTtcclxuICAgICAgICBjb25zdCB0ZXh0dXJlc0JpbmRHcm91cExheW91dCA9IGRldmljZS5jcmVhdGVCaW5kR3JvdXBMYXlvdXQoe1xyXG4gICAgICAgICAgZW50cmllczogW1xyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgYmluZGluZzogMCxcclxuICAgICAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5GUkFHTUVOVCxcclxuICAgICAgICAgICAgICBzYW1wbGVyOiB7fVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgYmluZGluZzogMSxcclxuICAgICAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5GUkFHTUVOVCxcclxuICAgICAgICAgICAgICB0ZXh0dXJlOiB7fSxcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIGJpbmRpbmc6IDIsXHJcbiAgICAgICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuRlJBR01FTlQsXHJcbiAgICAgICAgICAgICAgdGV4dHVyZToge30sXHJcbiAgICAgICAgICAgIH0sXHJcblxyXG4gICAgICAgICAgXVxyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIHRoaXMudGV4dHVyZXNCaW5kR3JvdXAgPSBkZXZpY2UuY3JlYXRlQmluZEdyb3VwKHtcclxuICAgICAgICAgIGVudHJpZXM6IFtcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIGJpbmRpbmc6IDAsXHJcbiAgICAgICAgICAgICAgcmVzb3VyY2U6IHNhbXBsZXIsXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICBiaW5kaW5nOiAxLFxyXG4gICAgICAgICAgICAgIHJlc291cmNlOiB0aGlzLnRleHR1cmVzWzBdLmNyZWF0ZVZpZXcoKSxcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIGJpbmRpbmc6IDIsXHJcbiAgICAgICAgICAgICAgcmVzb3VyY2U6IHRoaXMudGV4dHVyZXNbMV0uY3JlYXRlVmlldygpLFxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgXSxcclxuICAgICAgICAgIGxheW91dDogdGV4dHVyZXNCaW5kR3JvdXBMYXlvdXQsXHJcbiAgICAgICAgfSlcclxuXHJcbiAgICAgICAgdGhpcy52ZXJ0ZXhCdWZmZXJEZXNjcmlwdG9yID0gW1xyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgYXR0cmlidXRlczogW1xyXG4gICAgICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgICAgICBzaGFkZXJMb2NhdGlvbjogMCwgLy8gcG9zaXRpb25cclxuICAgICAgICAgICAgICAgICAgb2Zmc2V0OiAwLFxyXG4gICAgICAgICAgICAgICAgICBmb3JtYXQ6IFwiZmxvYXQzMngyXCIsXHJcbiAgICAgICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgICAgICBzaGFkZXJMb2NhdGlvbjogMSwgLy8gY29sb3JcclxuICAgICAgICAgICAgICAgICAgb2Zmc2V0OiA4LFxyXG4gICAgICAgICAgICAgICAgICBmb3JtYXQ6IFwiZmxvYXQzMng0XCIsXHJcbiAgICAgICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgICAgICBzaGFkZXJMb2NhdGlvbjogMiwgLy8gdXZcclxuICAgICAgICAgICAgICAgICAgb2Zmc2V0OiAyNCxcclxuICAgICAgICAgICAgICAgICAgZm9ybWF0OiBcImZsb2F0MzJ4MlwiLFxyXG4gICAgICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICAgICAgc2hhZGVyTG9jYXRpb246IDMsIC8vIHNoYXBlXHJcbiAgICAgICAgICAgICAgICAgIG9mZnNldDogMzIsXHJcbiAgICAgICAgICAgICAgICAgIGZvcm1hdDogXCJmbG9hdDMyXCIsXHJcbiAgICAgICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICAgIF0sXHJcbiAgICAgICAgICAgICAgYXJyYXlTdHJpZGU6IFZFUlRFWF9TVFJJREUgKiA0LFxyXG4gICAgICAgICAgICAgIHN0ZXBNb2RlOiBcInZlcnRleFwiLFxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgXTtcclxuXHJcbiAgICAgICAgICBjb25zdCBzaGFkZXJNb2R1bGUgPSBkZXZpY2UuY3JlYXRlU2hhZGVyTW9kdWxlKHtcclxuICAgICAgICAgICAgY29kZTogc2hhZGVyQ29kZVxyXG4gICAgICAgICAgfSlcclxuXHJcbiAgICAgICAgICB0aGlzLnBpcGVsaW5lRGVzY3JpcHRvciA9IHtcclxuICAgICAgICAgICAgdmVydGV4OiB7XHJcbiAgICAgICAgICAgICAgbW9kdWxlOiBzaGFkZXJNb2R1bGUsXHJcbiAgICAgICAgICAgICAgZW50cnlQb2ludDogXCJ2ZXJ0ZXhfbWFpblwiLFxyXG4gICAgICAgICAgICAgIGJ1ZmZlcnM6IHRoaXMudmVydGV4QnVmZmVyRGVzY3JpcHRvcixcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgZnJhZ21lbnQ6IHtcclxuICAgICAgICAgICAgICBtb2R1bGU6IHNoYWRlck1vZHVsZSxcclxuICAgICAgICAgICAgICBlbnRyeVBvaW50OiBcImZyYWdtZW50X21haW5cIixcclxuICAgICAgICAgICAgICB0YXJnZXRzOiBbXHJcbiAgICAgICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgICAgIGZvcm1hdDogbmF2aWdhdG9yLmdwdS5nZXRQcmVmZXJyZWRDYW52YXNGb3JtYXQoKSxcclxuICAgICAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgICAgXSxcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgcHJpbWl0aXZlOiB7XHJcbiAgICAgICAgICAgICAgdG9wb2xvZ3k6IFwidHJpYW5nbGUtbGlzdFwiLFxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBsYXlvdXQ6IGRldmljZS5jcmVhdGVQaXBlbGluZUxheW91dCh7XHJcbiAgICAgICAgICAgICAgYmluZEdyb3VwTGF5b3V0czogW3VuaWZvcm1CaW5kR3JvdXBMYXlvdXQsIHRleHR1cmVzQmluZEdyb3VwTGF5b3V0XVxyXG4gICAgICAgICAgICB9KSxcclxuICAgICAgICAgIH07XHJcbiAgICB9XHJcblxyXG4gICAgYWRkVmVydGV4QnVmZmVyID0gKG9iamVjdExpc3Q6IFBoeXNPYmplY3RbXSkgPT4ge1xyXG4gICAgICAgIGxldCB2ZXJ0ZXhMaXN0OiBudW1iZXJbXSA9IFtdO1xyXG4gICAgICAgIGxldCBpbmRleExpc3Q6IG51bWJlcltdID0gW107XHJcbiAgICAgICAgb2JqZWN0TGlzdC5mb3JFYWNoKG9iaiA9PiB7XHJcbiAgICAgICAgICAgIGxldCBzdGFydEluZGV4ID0gdmVydGV4TGlzdC5sZW5ndGggLyBWRVJURVhfU1RSSURFO1xyXG4gICAgICAgICAgICB2ZXJ0ZXhMaXN0LnB1c2goLi4ub2JqLnNoYXBlLnZlcnRpY2VzLmZsYXRNYXAodmVydGV4ID0+IHtcclxuICAgICAgICAgICAgICAgIHJldHVybiBbLi4udmVjMi5hZGQodmVydGV4LnBvc2l0aW9uLCBvYmoucG9zaXRpb24pLCAuLi52ZXJ0ZXguY29sb3IsIC4uLnZlcnRleC51diwgdmVydGV4LnNoYXBlXTtcclxuICAgICAgICAgICAgfSkpO1xyXG4gICAgICAgICAgICBpbmRleExpc3QucHVzaCguLi5vYmouc2hhcGUuaW5kaWNlcy5tYXAoaW5kZXggPT4gaW5kZXggKyBzdGFydEluZGV4KSk7XHJcbiAgICAgICAgfSk7XHJcblxyXG4gICAgICAgIGNvbnN0IHZlcnRpY2VzID0gbmV3IEZsb2F0MzJBcnJheSh2ZXJ0ZXhMaXN0KTtcclxuICAgICAgICB0aGlzLnZlcnRleEJ1ZmZlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XHJcbiAgICAgICAgICAgIHNpemU6IHZlcnRpY2VzLmJ5dGVMZW5ndGgsXHJcbiAgICAgICAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5WRVJURVggfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCxcclxuICAgICAgICB9KTtcclxuICAgICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcih0aGlzLnZlcnRleEJ1ZmZlciwgMCwgdmVydGljZXMsIDAsIHZlcnRpY2VzLmxlbmd0aCk7XHJcblxyXG4gICAgICAgIGNvbnN0IGluZGljZXMgPSBuZXcgVWludDMyQXJyYXkoaW5kZXhMaXN0KTtcclxuICAgICAgICB0aGlzLmluZGV4Q291bnQgPSBpbmRleExpc3QubGVuZ3RoO1xyXG4gICAgICAgIHRoaXMuaW5kZXhCdWZmZXIgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xyXG4gICAgICAgICAgc2l6ZTogaW5kaWNlcy5ieXRlTGVuZ3RoLFxyXG4gICAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLklOREVYIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXHJcbiAgICAgICAgfSlcclxuICAgICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcih0aGlzLmluZGV4QnVmZmVyLCAwLCBpbmRpY2VzLCAwLCBpbmRpY2VzLmxlbmd0aCk7XHJcbiAgICB9O1xyXG5cclxuICAgIHJlbmRlciA9ICgpID0+IHtcclxuICAgICAgICBjb25zdCByZW5kZXJQaXBlbGluZSA9IHRoaXMuZGV2aWNlLmNyZWF0ZVJlbmRlclBpcGVsaW5lKHRoaXMucGlwZWxpbmVEZXNjcmlwdG9yKTtcclxuICAgICAgICBjb25zdCBjb21tYW5kRW5jb2RlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbW1hbmRFbmNvZGVyKCk7XHJcbiAgICAgICAgY29uc3QgY2xlYXJDb2xvciA9IHsgcjogMC4wLCBnOiAwLjUsIGI6IDEuMCwgYTogMS4wIH07XHJcblxyXG4gICAgICAgIGNvbnN0IHJlbmRlclBhc3NEZXNjcmlwdG9yID0ge1xyXG4gICAgICAgIGNvbG9yQXR0YWNobWVudHM6IFtcclxuICAgICAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICAgICAgICBjbGVhclZhbHVlOiBjbGVhckNvbG9yLFxyXG4gICAgICAgICAgICAgICAgICAgIGxvYWRPcDogXCJjbGVhclwiLFxyXG4gICAgICAgICAgICAgICAgICAgIHN0b3JlT3A6IFwic3RvcmVcIixcclxuICAgICAgICAgICAgICAgICAgICB2aWV3OiB0aGlzLmNhbnZhc0NvbnRleHQuZ2V0Q3VycmVudFRleHR1cmUoKS5jcmVhdGVWaWV3KCksXHJcbiAgICAgICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICBdIGFzIEdQVVJlbmRlclBhc3NDb2xvckF0dGFjaG1lbnRbXSxcclxuICAgICAgICB9O1xyXG5cclxuICAgICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcclxuICAgICAgICAgIHRoaXMudW5pZm9ybUJ1ZmZlcixcclxuICAgICAgICAgIDAsXHJcbiAgICAgICAgICB0aGlzLnZpZXdNYXRyaXguYnVmZmVyLFxyXG4gICAgICAgICAgdGhpcy52aWV3TWF0cml4LmJ5dGVPZmZzZXQsXHJcbiAgICAgICAgICB0aGlzLnZpZXdNYXRyaXguYnl0ZUxlbmd0aCxcclxuICAgICAgICApO1xyXG5cclxuICAgICAgICBjb25zdCBwYXNzRW5jb2RlciA9IGNvbW1hbmRFbmNvZGVyLmJlZ2luUmVuZGVyUGFzcyhyZW5kZXJQYXNzRGVzY3JpcHRvcik7XHJcbiAgICAgICAgcGFzc0VuY29kZXIuc2V0UGlwZWxpbmUocmVuZGVyUGlwZWxpbmUpO1xyXG4gICAgICAgIHBhc3NFbmNvZGVyLnNldEJpbmRHcm91cCgwLCB0aGlzLnVuaWZvcm1CaW5kR3JvdXApO1xyXG4gICAgICAgIHBhc3NFbmNvZGVyLnNldEJpbmRHcm91cCgxLCB0aGlzLnRleHR1cmVzQmluZEdyb3VwKTtcclxuICAgICAgICBwYXNzRW5jb2Rlci5zZXRWZXJ0ZXhCdWZmZXIoMCwgdGhpcy52ZXJ0ZXhCdWZmZXIpO1xyXG4gICAgICAgIHBhc3NFbmNvZGVyLnNldEluZGV4QnVmZmVyKHRoaXMuaW5kZXhCdWZmZXIsIFwidWludDMyXCIpO1xyXG4gICAgICAgIHBhc3NFbmNvZGVyLmRyYXdJbmRleGVkKHRoaXMuaW5kZXhDb3VudCk7XHJcbiAgICAgICAgcGFzc0VuY29kZXIuZW5kKCk7XHJcbiAgICAgICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xyXG4gICAgICAgIHRoaXMudmVydGV4QnVmZmVyLmRlc3Ryb3koKTtcclxuICAgICAgICB0aGlzLmluZGV4QnVmZmVyLmRlc3Ryb3koKTtcclxuICAgIH1cclxuXHJcbiAgICBwcml2YXRlIGFkZFRleHR1cmUoYml0bWFwOiBJbWFnZUJpdG1hcCk6IFRleHR1cmVIYW5kbGUge1xyXG4gICAgICBjb25zdCB0ZXh0dXJlID0gdGhpcy5kZXZpY2UuY3JlYXRlVGV4dHVyZSh7XHJcbiAgICAgICAgICBzaXplOiBbYml0bWFwLndpZHRoLCBiaXRtYXAuaGVpZ2h0XSxcclxuICAgICAgICAgIGZvcm1hdDogICdyZ2JhOHVub3JtJyxcclxuICAgICAgICAgIHVzYWdlOiBcclxuICAgICAgICAgICAgICBHUFVUZXh0dXJlVXNhZ2UuVEVYVFVSRV9CSU5ESU5HIHxcclxuICAgICAgICAgICAgICBHUFVUZXh0dXJlVXNhZ2UuU1RPUkFHRV9CSU5ESU5HIHxcclxuICAgICAgICAgICAgICBHUFVUZXh0dXJlVXNhZ2UuQ09QWV9EU1QgfFxyXG4gICAgICAgICAgICAgIEdQVVRleHR1cmVVc2FnZS5SRU5ERVJfQVRUQUNITUVOVFxyXG4gICAgICB9KTtcclxuICBcclxuICAgICAgdGhpcy5kZXZpY2UucXVldWUuY29weUV4dGVybmFsSW1hZ2VUb1RleHR1cmUoXHJcbiAgICAgICAgICB7c291cmNlOiBiaXRtYXB9LFxyXG4gICAgICAgICAge3RleHR1cmU6IHRleHR1cmV9LFxyXG4gICAgICAgICAgW2JpdG1hcC53aWR0aCwgYml0bWFwLmhlaWdodF1cclxuICAgICAgKTtcclxuICAgICAgcmV0dXJuIHRoaXMudGV4dHVyZXMucHVzaCh0ZXh0dXJlKSAtIDEgYXMgVGV4dHVyZUhhbmRsZTtcclxuICB9XHJcbn1cclxuXHJcbmV4cG9ydCBhc3luYyBmdW5jdGlvbiByZXF1ZXN0UmVuZGVyZXIoY2FudmFzOiBIVE1MQ2FudmFzRWxlbWVudCwgdGV4dHVyZXM6IEltYWdlQml0bWFwW10pOiBQcm9taXNlPFJlbmRlcmVyPiB7XHJcbiAgICBjb25zdCBkZXZpY2UgPSBhd2FpdCBuYXZpZ2F0b3IuZ3B1LnJlcXVlc3RBZGFwdGVyKClcclxuICAgICAgICAudGhlbihhZGFwdGVyID0+IGFkYXB0ZXIucmVxdWVzdERldmljZSgpKTtcclxuXHJcbiAgICBjb25zdCBjYW52YXNDb250ZXh0ID0gY2FudmFzLmdldENvbnRleHQoXCJ3ZWJncHVcIikgYXMgdW5rbm93biBhcyBHUFVDYW52YXNDb250ZXh0O1xyXG4gICAgY2FudmFzQ29udGV4dC5jb25maWd1cmUoe1xyXG4gICAgICAgIGRldmljZTogZGV2aWNlLFxyXG4gICAgICAgIGZvcm1hdDogbmF2aWdhdG9yLmdwdS5nZXRQcmVmZXJyZWRDYW52YXNGb3JtYXQoKSxcclxuICAgICAgICBhbHBoYU1vZGU6IFwicHJlbXVsdGlwbGllZFwiLFxyXG4gICAgfSk7XHJcbiAgICByZXR1cm4gbmV3IFJlbmRlcmVyKGRldmljZSwgY2FudmFzQ29udGV4dCwgdGV4dHVyZXMpO1xyXG59IiwiLyogd2dwdS1tYXRyaXhAMy4yLjAsIGxpY2Vuc2UgTUlUICovXG5mdW5jdGlvbiB3cmFwQ29uc3RydWN0b3IoT3JpZ2luYWxDb25zdHJ1Y3RvciwgbW9kaWZpZXIpIHtcbiAgICByZXR1cm4gY2xhc3MgZXh0ZW5kcyBPcmlnaW5hbENvbnN0cnVjdG9yIHtcbiAgICAgICAgY29uc3RydWN0b3IoLi4uYXJncykge1xuICAgICAgICAgICAgc3VwZXIoLi4uYXJncyk7XG4gICAgICAgICAgICBtb2RpZmllcih0aGlzKTtcbiAgICAgICAgfVxuICAgIH07IC8vIFR5cGUgYXNzZXJ0aW9uIGlzIG5lY2Vzc2FyeSBoZXJlXG59XG5jb25zdCBaZXJvQXJyYXkgPSB3cmFwQ29uc3RydWN0b3IoKEFycmF5KSwgYSA9PiBhLmZpbGwoMCkpO1xuXG4vKlxuICogQ29weXJpZ2h0IDIwMjIgR3JlZ2cgVGF2YXJlc1xuICpcbiAqIFBlcm1pc3Npb24gaXMgaGVyZWJ5IGdyYW50ZWQsIGZyZWUgb2YgY2hhcmdlLCB0byBhbnkgcGVyc29uIG9idGFpbmluZyBhXG4gKiBjb3B5IG9mIHRoaXMgc29mdHdhcmUgYW5kIGFzc29jaWF0ZWQgZG9jdW1lbnRhdGlvbiBmaWxlcyAodGhlIFwiU29mdHdhcmVcIiksXG4gKiB0byBkZWFsIGluIHRoZSBTb2Z0d2FyZSB3aXRob3V0IHJlc3RyaWN0aW9uLCBpbmNsdWRpbmcgd2l0aG91dCBsaW1pdGF0aW9uXG4gKiB0aGUgcmlnaHRzIHRvIHVzZSwgY29weSwgbW9kaWZ5LCBtZXJnZSwgcHVibGlzaCwgZGlzdHJpYnV0ZSwgc3VibGljZW5zZSxcbiAqIGFuZC9vciBzZWxsIGNvcGllcyBvZiB0aGUgU29mdHdhcmUsIGFuZCB0byBwZXJtaXQgcGVyc29ucyB0byB3aG9tIHRoZVxuICogU29mdHdhcmUgaXMgZnVybmlzaGVkIHRvIGRvIHNvLCBzdWJqZWN0IHRvIHRoZSBmb2xsb3dpbmcgY29uZGl0aW9uczpcbiAqXG4gKiBUaGUgYWJvdmUgY29weXJpZ2h0IG5vdGljZSBhbmQgdGhpcyBwZXJtaXNzaW9uIG5vdGljZSBzaGFsbCBiZSBpbmNsdWRlZCBpblxuICogYWxsIGNvcGllcyBvciBzdWJzdGFudGlhbCBwb3J0aW9ucyBvZiB0aGUgU29mdHdhcmUuXG4gKlxuICogVEhFIFNPRlRXQVJFIElTIFBST1ZJREVEIFwiQVMgSVNcIiwgV0lUSE9VVCBXQVJSQU5UWSBPRiBBTlkgS0lORCwgRVhQUkVTUyBPUlxuICogSU1QTElFRCwgSU5DTFVESU5HIEJVVCBOT1QgTElNSVRFRCBUTyBUSEUgV0FSUkFOVElFUyBPRiBNRVJDSEFOVEFCSUxJVFksXG4gKiBGSVRORVNTIEZPUiBBIFBBUlRJQ1VMQVIgUFVSUE9TRSBBTkQgTk9OSU5GUklOR0VNRU5ULiAgSU4gTk8gRVZFTlQgU0hBTExcbiAqIFRIRSBBVVRIT1JTIE9SIENPUFlSSUdIVCBIT0xERVJTIEJFIExJQUJMRSBGT1IgQU5ZIENMQUlNLCBEQU1BR0VTIE9SIE9USEVSXG4gKiBMSUFCSUxJVFksIFdIRVRIRVIgSU4gQU4gQUNUSU9OIE9GIENPTlRSQUNULCBUT1JUIE9SIE9USEVSV0lTRSwgQVJJU0lOR1xuICogRlJPTSwgT1VUIE9GIE9SIElOIENPTk5FQ1RJT04gV0lUSCBUSEUgU09GVFdBUkUgT1IgVEhFIFVTRSBPUiBPVEhFUlxuICogREVBTElOR1MgSU4gVEhFIFNPRlRXQVJFLlxuICovXG5sZXQgRVBTSUxPTiA9IDAuMDAwMDAxO1xuLyoqXG4gKiBTZXQgdGhlIHZhbHVlIGZvciBFUFNJTE9OIGZvciB2YXJpb3VzIGNoZWNrc1xuICogQHBhcmFtIHYgLSBWYWx1ZSB0byB1c2UgZm9yIEVQU0lMT04uXG4gKiBAcmV0dXJucyBwcmV2aW91cyB2YWx1ZSBvZiBFUFNJTE9OO1xuICovXG5mdW5jdGlvbiBzZXRFcHNpbG9uKHYpIHtcbiAgICBjb25zdCBvbGQgPSBFUFNJTE9OO1xuICAgIEVQU0lMT04gPSB2O1xuICAgIHJldHVybiBvbGQ7XG59XG4vKipcbiAqIENvbnZlcnQgZGVncmVlcyB0byByYWRpYW5zXG4gKiBAcGFyYW0gZGVncmVlcyAtIEFuZ2xlIGluIGRlZ3JlZXNcbiAqIEByZXR1cm5zIGFuZ2xlIGNvbnZlcnRlZCB0byByYWRpYW5zXG4gKi9cbmZ1bmN0aW9uIGRlZ1RvUmFkKGRlZ3JlZXMpIHtcbiAgICByZXR1cm4gZGVncmVlcyAqIE1hdGguUEkgLyAxODA7XG59XG4vKipcbiAqIENvbnZlcnQgcmFkaWFucyB0byBkZWdyZWVzXG4gKiBAcGFyYW0gcmFkaWFucyAtIEFuZ2xlIGluIHJhZGlhbnNcbiAqIEByZXR1cm5zIGFuZ2xlIGNvbnZlcnRlZCB0byBkZWdyZWVzXG4gKi9cbmZ1bmN0aW9uIHJhZFRvRGVnKHJhZGlhbnMpIHtcbiAgICByZXR1cm4gcmFkaWFucyAqIDE4MCAvIE1hdGguUEk7XG59XG4vKipcbiAqIExlcnBzIGJldHdlZW4gYSBhbmQgYiB2aWEgdFxuICogQHBhcmFtIGEgLSBzdGFydGluZyB2YWx1ZVxuICogQHBhcmFtIGIgLSBlbmRpbmcgdmFsdWVcbiAqIEBwYXJhbSB0IC0gdmFsdWUgd2hlcmUgMCA9IGEgYW5kIDEgPSBiXG4gKiBAcmV0dXJucyBhICsgKGIgLSBhKSAqIHRcbiAqL1xuZnVuY3Rpb24gbGVycChhLCBiLCB0KSB7XG4gICAgcmV0dXJuIGEgKyAoYiAtIGEpICogdDtcbn1cbi8qKlxuICogQ29tcHV0ZSB0aGUgb3Bwb3NpdGUgb2YgbGVycC4gR2l2ZW4gYSBhbmQgYiBhbmQgYSB2YWx1ZSBiZXR3ZWVuXG4gKiBhIGFuZCBiIHJldHVybnMgYSB2YWx1ZSBiZXR3ZWVuIDAgYW5kIDEuIDAgaWYgYSwgMSBpZiBiLlxuICogTm90ZTogbm8gY2xhbXBpbmcgaXMgZG9uZS5cbiAqIEBwYXJhbSBhIC0gc3RhcnQgdmFsdWVcbiAqIEBwYXJhbSBiIC0gZW5kIHZhbHVlXG4gKiBAcGFyYW0gdiAtIHZhbHVlIGJldHdlZW4gYSBhbmQgYlxuICogQHJldHVybnMgKHYgLSBhKSAvIChiIC0gYSlcbiAqL1xuZnVuY3Rpb24gaW52ZXJzZUxlcnAoYSwgYiwgdikge1xuICAgIGNvbnN0IGQgPSBiIC0gYTtcbiAgICByZXR1cm4gKE1hdGguYWJzKGIgLSBhKSA8IEVQU0lMT04pXG4gICAgICAgID8gYVxuICAgICAgICA6ICh2IC0gYSkgLyBkO1xufVxuLyoqXG4gKiBDb21wdXRlIHRoZSBldWNsaWRlYW4gbW9kdWxvXG4gKlxuICogYGBgXG4gKiAvLyB0YWJsZSBmb3IgbiAvIDNcbiAqIC01LCAtNCwgLTMsIC0yLCAtMSwgIDAsICAxLCAgMiwgIDMsICA0LCAgNSAgIDwtIG5cbiAqIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogLTIgIC0xICAtMCAgLTIgIC0xICAgMCwgIDEsICAyLCAgMCwgIDEsICAyICAgPC0gbiAlIDNcbiAqICAxICAgMiAgIDAgICAxICAgMiAgIDAsICAxLCAgMiwgIDAsICAxLCAgMiAgIDwtIGV1Y2xpZGVhbk1vZHVsZShuLCAzKVxuICogYGBgXG4gKlxuICogQHBhcmFtIG4gLSBkaXZpZGVuZFxuICogQHBhcmFtIG0gLSBkaXZpc29yXG4gKiBAcmV0dXJucyB0aGUgZXVjbGlkZWFuIG1vZHVsbyBvZiBuIC8gbVxuICovXG5mdW5jdGlvbiBldWNsaWRlYW5Nb2R1bG8obiwgbSkge1xuICAgIHJldHVybiAoKG4gJSBtKSArIG0pICUgbTtcbn1cblxudmFyIHV0aWxzID0ge1xuICAgIF9fcHJvdG9fXzogbnVsbCxcbiAgICBnZXQgRVBTSUxPTiAoKSB7IHJldHVybiBFUFNJTE9OOyB9LFxuICAgIGRlZ1RvUmFkOiBkZWdUb1JhZCxcbiAgICBldWNsaWRlYW5Nb2R1bG86IGV1Y2xpZGVhbk1vZHVsbyxcbiAgICBpbnZlcnNlTGVycDogaW52ZXJzZUxlcnAsXG4gICAgbGVycDogbGVycCxcbiAgICByYWRUb0RlZzogcmFkVG9EZWcsXG4gICAgc2V0RXBzaWxvbjogc2V0RXBzaWxvblxufTtcblxuLypcbiAqIENvcHlyaWdodCAyMDIyIEdyZWdnIFRhdmFyZXNcbiAqXG4gKiBQZXJtaXNzaW9uIGlzIGhlcmVieSBncmFudGVkLCBmcmVlIG9mIGNoYXJnZSwgdG8gYW55IHBlcnNvbiBvYnRhaW5pbmcgYVxuICogY29weSBvZiB0aGlzIHNvZnR3YXJlIGFuZCBhc3NvY2lhdGVkIGRvY3VtZW50YXRpb24gZmlsZXMgKHRoZSBcIlNvZnR3YXJlXCIpLFxuICogdG8gZGVhbCBpbiB0aGUgU29mdHdhcmUgd2l0aG91dCByZXN0cmljdGlvbiwgaW5jbHVkaW5nIHdpdGhvdXQgbGltaXRhdGlvblxuICogdGhlIHJpZ2h0cyB0byB1c2UsIGNvcHksIG1vZGlmeSwgbWVyZ2UsIHB1Ymxpc2gsIGRpc3RyaWJ1dGUsIHN1YmxpY2Vuc2UsXG4gKiBhbmQvb3Igc2VsbCBjb3BpZXMgb2YgdGhlIFNvZnR3YXJlLCBhbmQgdG8gcGVybWl0IHBlcnNvbnMgdG8gd2hvbSB0aGVcbiAqIFNvZnR3YXJlIGlzIGZ1cm5pc2hlZCB0byBkbyBzbywgc3ViamVjdCB0byB0aGUgZm9sbG93aW5nIGNvbmRpdGlvbnM6XG4gKlxuICogVGhlIGFib3ZlIGNvcHlyaWdodCBub3RpY2UgYW5kIHRoaXMgcGVybWlzc2lvbiBub3RpY2Ugc2hhbGwgYmUgaW5jbHVkZWQgaW5cbiAqIGFsbCBjb3BpZXMgb3Igc3Vic3RhbnRpYWwgcG9ydGlvbnMgb2YgdGhlIFNvZnR3YXJlLlxuICpcbiAqIFRIRSBTT0ZUV0FSRSBJUyBQUk9WSURFRCBcIkFTIElTXCIsIFdJVEhPVVQgV0FSUkFOVFkgT0YgQU5ZIEtJTkQsIEVYUFJFU1MgT1JcbiAqIElNUExJRUQsIElOQ0xVRElORyBCVVQgTk9UIExJTUlURUQgVE8gVEhFIFdBUlJBTlRJRVMgT0YgTUVSQ0hBTlRBQklMSVRZLFxuICogRklUTkVTUyBGT1IgQSBQQVJUSUNVTEFSIFBVUlBPU0UgQU5EIE5PTklORlJJTkdFTUVOVC4gIElOIE5PIEVWRU5UIFNIQUxMXG4gKiBUSEUgQVVUSE9SUyBPUiBDT1BZUklHSFQgSE9MREVSUyBCRSBMSUFCTEUgRk9SIEFOWSBDTEFJTSwgREFNQUdFUyBPUiBPVEhFUlxuICogTElBQklMSVRZLCBXSEVUSEVSIElOIEFOIEFDVElPTiBPRiBDT05UUkFDVCwgVE9SVCBPUiBPVEhFUldJU0UsIEFSSVNJTkdcbiAqIEZST00sIE9VVCBPRiBPUiBJTiBDT05ORUNUSU9OIFdJVEggVEhFIFNPRlRXQVJFIE9SIFRIRSBVU0UgT1IgT1RIRVJcbiAqIERFQUxJTkdTIElOIFRIRSBTT0ZUV0FSRS5cbiAqL1xuLyoqXG4gKiBHZW5lcmF0ZXMgYW0gdHlwZWQgQVBJIGZvciBWZWMzXG4gKi9cbmZ1bmN0aW9uIGdldEFQSUltcGwkNShDdG9yKSB7XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIFZlYzI7IG1heSBiZSBjYWxsZWQgd2l0aCB4LCB5LCB6IHRvIHNldCBpbml0aWFsIHZhbHVlcy5cbiAgICAgKlxuICAgICAqIE5vdGU6IFNpbmNlIHBhc3NpbmcgaW4gYSByYXcgSmF2YVNjcmlwdCBhcnJheVxuICAgICAqIGlzIHZhbGlkIGluIGFsbCBjaXJjdW1zdGFuY2VzLCBpZiB5b3Ugd2FudCB0b1xuICAgICAqIGZvcmNlIGEgSmF2YVNjcmlwdCBhcnJheSBpbnRvIGEgVmVjMidzIHNwZWNpZmllZCB0eXBlXG4gICAgICogaXQgd291bGQgYmUgZmFzdGVyIHRvIHVzZVxuICAgICAqXG4gICAgICogYGBgXG4gICAgICogY29uc3QgdiA9IHZlYzIuY2xvbmUoc29tZUpTQXJyYXkpO1xuICAgICAqIGBgYFxuICAgICAqXG4gICAgICogQHBhcmFtIHggLSBJbml0aWFsIHggdmFsdWUuXG4gICAgICogQHBhcmFtIHkgLSBJbml0aWFsIHkgdmFsdWUuXG4gICAgICogQHJldHVybnMgdGhlIGNyZWF0ZWQgdmVjdG9yXG4gICAgICovXG4gICAgZnVuY3Rpb24gY3JlYXRlKHggPSAwLCB5ID0gMCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSBuZXcgQ3RvcigyKTtcbiAgICAgICAgaWYgKHggIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0geDtcbiAgICAgICAgICAgIGlmICh5ICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICBuZXdEc3RbMV0gPSB5O1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENyZWF0ZXMgYSBWZWMyOyBtYXkgYmUgY2FsbGVkIHdpdGggeCwgeSwgeiB0byBzZXQgaW5pdGlhbCB2YWx1ZXMuIChzYW1lIGFzIGNyZWF0ZSlcbiAgICAgKiBAcGFyYW0geCAtIEluaXRpYWwgeCB2YWx1ZS5cbiAgICAgKiBAcGFyYW0geSAtIEluaXRpYWwgeSB2YWx1ZS5cbiAgICAgKiBAcmV0dXJucyB0aGUgY3JlYXRlZCB2ZWN0b3JcbiAgICAgKi9cbiAgICBjb25zdCBmcm9tVmFsdWVzID0gY3JlYXRlO1xuICAgIC8qKlxuICAgICAqIFNldHMgdGhlIHZhbHVlcyBvZiBhIFZlYzJcbiAgICAgKiBBbHNvIHNlZSB7QGxpbmsgdmVjMi5jcmVhdGV9IGFuZCB7QGxpbmsgdmVjMi5jb3B5fVxuICAgICAqXG4gICAgICogQHBhcmFtIHggZmlyc3QgdmFsdWVcbiAgICAgKiBAcGFyYW0geSBzZWNvbmQgdmFsdWVcbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHdpdGggaXRzIGVsZW1lbnRzIHNldC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzZXQoeCwgeSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSB4O1xuICAgICAgICBuZXdEc3RbMV0gPSB5O1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBBcHBsaWVzIE1hdGguY2VpbCB0byBlYWNoIGVsZW1lbnQgb2YgdmVjdG9yXG4gICAgICogQHBhcmFtIHYgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHRoYXQgaXMgdGhlIGNlaWwgb2YgZWFjaCBlbGVtZW50IG9mIHYuXG4gICAgICovXG4gICAgZnVuY3Rpb24gY2VpbCh2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGguY2VpbCh2WzBdKTtcbiAgICAgICAgbmV3RHN0WzFdID0gTWF0aC5jZWlsKHZbMV0pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBBcHBsaWVzIE1hdGguZmxvb3IgdG8gZWFjaCBlbGVtZW50IG9mIHZlY3RvclxuICAgICAqIEBwYXJhbSB2IC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB0aGF0IGlzIHRoZSBmbG9vciBvZiBlYWNoIGVsZW1lbnQgb2Ygdi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBmbG9vcih2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGguZmxvb3IodlswXSk7XG4gICAgICAgIG5ld0RzdFsxXSA9IE1hdGguZmxvb3IodlsxXSk7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEFwcGxpZXMgTWF0aC5yb3VuZCB0byBlYWNoIGVsZW1lbnQgb2YgdmVjdG9yXG4gICAgICogQHBhcmFtIHYgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHRoYXQgaXMgdGhlIHJvdW5kIG9mIGVhY2ggZWxlbWVudCBvZiB2LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdW5kKHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDIpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gTWF0aC5yb3VuZCh2WzBdKTtcbiAgICAgICAgbmV3RHN0WzFdID0gTWF0aC5yb3VuZCh2WzFdKTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ2xhbXAgZWFjaCBlbGVtZW50IG9mIHZlY3RvciBiZXR3ZWVuIG1pbiBhbmQgbWF4XG4gICAgICogQHBhcmFtIHYgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gbWF4IC0gTWluIHZhbHVlLCBkZWZhdWx0IDBcbiAgICAgKiBAcGFyYW0gbWluIC0gTWF4IHZhbHVlLCBkZWZhdWx0IDFcbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHRoYXQgdGhlIGNsYW1wZWQgdmFsdWUgb2YgZWFjaCBlbGVtZW50IG9mIHYuXG4gICAgICovXG4gICAgZnVuY3Rpb24gY2xhbXAodiwgbWluID0gMCwgbWF4ID0gMSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSBNYXRoLm1pbihtYXgsIE1hdGgubWF4KG1pbiwgdlswXSkpO1xuICAgICAgICBuZXdEc3RbMV0gPSBNYXRoLm1pbihtYXgsIE1hdGgubWF4KG1pbiwgdlsxXSkpO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBBZGRzIHR3byB2ZWN0b3JzOyBhc3N1bWVzIGEgYW5kIGIgaGF2ZSB0aGUgc2FtZSBkaW1lbnNpb24uXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSB2ZWN0b3IgdGhhdCBpcyB0aGUgc3VtIG9mIGEgYW5kIGIuXG4gICAgICovXG4gICAgZnVuY3Rpb24gYWRkKGEsIGIsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDIpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSArIGJbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IGFbMV0gKyBiWzFdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBBZGRzIHR3byB2ZWN0b3JzLCBzY2FsaW5nIHRoZSAybmQ7IGFzc3VtZXMgYSBhbmQgYiBoYXZlIHRoZSBzYW1lIGRpbWVuc2lvbi5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIHNjYWxlIC0gQW1vdW50IHRvIHNjYWxlIGJcbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHRoYXQgaXMgdGhlIHN1bSBvZiBhICsgYiAqIHNjYWxlLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGFkZFNjYWxlZChhLCBiLCBzY2FsZSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSBhWzBdICsgYlswXSAqIHNjYWxlO1xuICAgICAgICBuZXdEc3RbMV0gPSBhWzFdICsgYlsxXSAqIHNjYWxlO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSZXR1cm5zIHRoZSBhbmdsZSBpbiByYWRpYW5zIGJldHdlZW4gdHdvIHZlY3RvcnMuXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIFRoZSBhbmdsZSBpbiByYWRpYW5zIGJldHdlZW4gdGhlIDIgdmVjdG9ycy5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBhbmdsZShhLCBiKSB7XG4gICAgICAgIGNvbnN0IGF4ID0gYVswXTtcbiAgICAgICAgY29uc3QgYXkgPSBhWzFdO1xuICAgICAgICBjb25zdCBieCA9IGJbMF07XG4gICAgICAgIGNvbnN0IGJ5ID0gYlsxXTtcbiAgICAgICAgY29uc3QgbWFnMSA9IE1hdGguc3FydChheCAqIGF4ICsgYXkgKiBheSk7XG4gICAgICAgIGNvbnN0IG1hZzIgPSBNYXRoLnNxcnQoYnggKiBieCArIGJ5ICogYnkpO1xuICAgICAgICBjb25zdCBtYWcgPSBtYWcxICogbWFnMjtcbiAgICAgICAgY29uc3QgY29zaW5lID0gbWFnICYmIGRvdChhLCBiKSAvIG1hZztcbiAgICAgICAgcmV0dXJuIE1hdGguYWNvcyhjb3NpbmUpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBTdWJ0cmFjdHMgdHdvIHZlY3RvcnMuXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSB2ZWN0b3IgdGhhdCBpcyB0aGUgZGlmZmVyZW5jZSBvZiBhIGFuZCBiLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHN1YnRyYWN0KGEsIGIsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDIpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSAtIGJbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IGFbMV0gLSBiWzFdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBTdWJ0cmFjdHMgdHdvIHZlY3RvcnMuXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSB2ZWN0b3IgdGhhdCBpcyB0aGUgZGlmZmVyZW5jZSBvZiBhIGFuZCBiLlxuICAgICAqL1xuICAgIGNvbnN0IHN1YiA9IHN1YnRyYWN0O1xuICAgIC8qKlxuICAgICAqIENoZWNrIGlmIDIgdmVjdG9ycyBhcmUgYXBwcm94aW1hdGVseSBlcXVhbFxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyB0cnVlIGlmIHZlY3RvcnMgYXJlIGFwcHJveGltYXRlbHkgZXF1YWxcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBlcXVhbHNBcHByb3hpbWF0ZWx5KGEsIGIpIHtcbiAgICAgICAgcmV0dXJuIE1hdGguYWJzKGFbMF0gLSBiWzBdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbMV0gLSBiWzFdKSA8IEVQU0lMT047XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENoZWNrIGlmIDIgdmVjdG9ycyBhcmUgZXhhY3RseSBlcXVhbFxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyB0cnVlIGlmIHZlY3RvcnMgYXJlIGV4YWN0bHkgZXF1YWxcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBlcXVhbHMoYSwgYikge1xuICAgICAgICByZXR1cm4gYVswXSA9PT0gYlswXSAmJiBhWzFdID09PSBiWzFdO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBQZXJmb3JtcyBsaW5lYXIgaW50ZXJwb2xhdGlvbiBvbiB0d28gdmVjdG9ycy5cbiAgICAgKiBHaXZlbiB2ZWN0b3JzIGEgYW5kIGIgYW5kIGludGVycG9sYXRpb24gY29lZmZpY2llbnQgdCwgcmV0dXJuc1xuICAgICAqIGEgKyB0ICogKGIgLSBhKS5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIHQgLSBJbnRlcnBvbGF0aW9uIGNvZWZmaWNpZW50LlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIGxpbmVhciBpbnRlcnBvbGF0ZWQgcmVzdWx0LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGxlcnAoYSwgYiwgdCwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSBhWzBdICsgdCAqIChiWzBdIC0gYVswXSk7XG4gICAgICAgIG5ld0RzdFsxXSA9IGFbMV0gKyB0ICogKGJbMV0gLSBhWzFdKTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUGVyZm9ybXMgbGluZWFyIGludGVycG9sYXRpb24gb24gdHdvIHZlY3RvcnMuXG4gICAgICogR2l2ZW4gdmVjdG9ycyBhIGFuZCBiIGFuZCBpbnRlcnBvbGF0aW9uIGNvZWZmaWNpZW50IHZlY3RvciB0LCByZXR1cm5zXG4gICAgICogYSArIHQgKiAoYiAtIGEpLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gdCAtIEludGVycG9sYXRpb24gY29lZmZpY2llbnRzIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIHRoZSBsaW5lYXIgaW50ZXJwb2xhdGVkIHJlc3VsdC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBsZXJwVihhLCBiLCB0LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IGFbMF0gKyB0WzBdICogKGJbMF0gLSBhWzBdKTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSArIHRbMV0gKiAoYlsxXSAtIGFbMV0pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSZXR1cm4gbWF4IHZhbHVlcyBvZiB0d28gdmVjdG9ycy5cbiAgICAgKiBHaXZlbiB2ZWN0b3JzIGEgYW5kIGIgcmV0dXJuc1xuICAgICAqIFttYXgoYVswXSwgYlswXSksIG1heChhWzFdLCBiWzFdKSwgbWF4KGFbMl0sIGJbMl0pXS5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgbWF4IGNvbXBvbmVudHMgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG1heChhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGgubWF4KGFbMF0sIGJbMF0pO1xuICAgICAgICBuZXdEc3RbMV0gPSBNYXRoLm1heChhWzFdLCBiWzFdKTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUmV0dXJuIG1pbiB2YWx1ZXMgb2YgdHdvIHZlY3RvcnMuXG4gICAgICogR2l2ZW4gdmVjdG9ycyBhIGFuZCBiIHJldHVybnNcbiAgICAgKiBbbWluKGFbMF0sIGJbMF0pLCBtaW4oYVsxXSwgYlsxXSksIG1pbihhWzJdLCBiWzJdKV0uXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIG1pbiBjb21wb25lbnRzIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBtaW4oYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSBNYXRoLm1pbihhWzBdLCBiWzBdKTtcbiAgICAgICAgbmV3RHN0WzFdID0gTWF0aC5taW4oYVsxXSwgYlsxXSk7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIE11bHRpcGxpZXMgYSB2ZWN0b3IgYnkgYSBzY2FsYXIuXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBrIC0gVGhlIHNjYWxhci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBzY2FsZWQgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG11bFNjYWxhcih2LCBrLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IHZbMF0gKiBrO1xuICAgICAgICBuZXdEc3RbMV0gPSB2WzFdICogaztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogTXVsdGlwbGllcyBhIHZlY3RvciBieSBhIHNjYWxhci4gKHNhbWUgYXMgbXVsU2NhbGFyKVxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gayAtIFRoZSBzY2FsYXIuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBjb25zdCBzY2FsZSA9IG11bFNjYWxhcjtcbiAgICAvKipcbiAgICAgKiBEaXZpZGVzIGEgdmVjdG9yIGJ5IGEgc2NhbGFyLlxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gayAtIFRoZSBzY2FsYXIuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBkaXZTY2FsYXIodiwgaywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2WzBdIC8gaztcbiAgICAgICAgbmV3RHN0WzFdID0gdlsxXSAvIGs7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEludmVyc2UgYSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIGludmVydGVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBpbnZlcnNlKHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDIpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gMSAvIHZbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IDEgLyB2WzFdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBJbnZlcnQgYSB2ZWN0b3IuIChzYW1lIGFzIGludmVyc2UpXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIGludmVydGVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBjb25zdCBpbnZlcnQgPSBpbnZlcnNlO1xuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBjcm9zcyBwcm9kdWN0IG9mIHR3byB2ZWN0b3JzOyBhc3N1bWVzIGJvdGggdmVjdG9ycyBoYXZlXG4gICAgICogdGhyZWUgZW50cmllcy5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgdmVjdG9yIG9mIGEgY3Jvc3MgYi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBjcm9zcyhhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIGNvbnN0IHogPSBhWzBdICogYlsxXSAtIGFbMV0gKiBiWzBdO1xuICAgICAgICBuZXdEc3RbMF0gPSAwO1xuICAgICAgICBuZXdEc3RbMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMl0gPSB6O1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgdHdvIHZlY3RvcnM7IGFzc3VtZXMgYm90aCB2ZWN0b3JzIGhhdmVcbiAgICAgKiB0aHJlZSBlbnRyaWVzLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBkb3QgcHJvZHVjdFxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGRvdChhLCBiKSB7XG4gICAgICAgIHJldHVybiBhWzBdICogYlswXSArIGFbMV0gKiBiWzFdO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgbGVuZ3RoIG9mIHZlY3RvclxuICAgICAqIEBwYXJhbSB2IC0gdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIGxlbmd0aCBvZiB2ZWN0b3IuXG4gICAgICovXG4gICAgZnVuY3Rpb24gbGVuZ3RoKHYpIHtcbiAgICAgICAgY29uc3QgdjAgPSB2WzBdO1xuICAgICAgICBjb25zdCB2MSA9IHZbMV07XG4gICAgICAgIHJldHVybiBNYXRoLnNxcnQodjAgKiB2MCArIHYxICogdjEpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgbGVuZ3RoIG9mIHZlY3RvciAoc2FtZSBhcyBsZW5ndGgpXG4gICAgICogQHBhcmFtIHYgLSB2ZWN0b3IuXG4gICAgICogQHJldHVybnMgbGVuZ3RoIG9mIHZlY3Rvci5cbiAgICAgKi9cbiAgICBjb25zdCBsZW4gPSBsZW5ndGg7XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIHNxdWFyZSBvZiB0aGUgbGVuZ3RoIG9mIHZlY3RvclxuICAgICAqIEBwYXJhbSB2IC0gdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIHNxdWFyZSBvZiB0aGUgbGVuZ3RoIG9mIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBsZW5ndGhTcSh2KSB7XG4gICAgICAgIGNvbnN0IHYwID0gdlswXTtcbiAgICAgICAgY29uc3QgdjEgPSB2WzFdO1xuICAgICAgICByZXR1cm4gdjAgKiB2MCArIHYxICogdjE7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBzcXVhcmUgb2YgdGhlIGxlbmd0aCBvZiB2ZWN0b3IgKHNhbWUgYXMgbGVuZ3RoU3EpXG4gICAgICogQHBhcmFtIHYgLSB2ZWN0b3IuXG4gICAgICogQHJldHVybnMgc3F1YXJlIG9mIHRoZSBsZW5ndGggb2YgdmVjdG9yLlxuICAgICAqL1xuICAgIGNvbnN0IGxlblNxID0gbGVuZ3RoU3E7XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIGRpc3RhbmNlIGJldHdlZW4gMiBwb2ludHNcbiAgICAgKiBAcGFyYW0gYSAtIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBkaXN0YW5jZSBiZXR3ZWVuIGEgYW5kIGJcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBkaXN0YW5jZShhLCBiKSB7XG4gICAgICAgIGNvbnN0IGR4ID0gYVswXSAtIGJbMF07XG4gICAgICAgIGNvbnN0IGR5ID0gYVsxXSAtIGJbMV07XG4gICAgICAgIHJldHVybiBNYXRoLnNxcnQoZHggKiBkeCArIGR5ICogZHkpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgZGlzdGFuY2UgYmV0d2VlbiAyIHBvaW50cyAoc2FtZSBhcyBkaXN0YW5jZSlcbiAgICAgKiBAcGFyYW0gYSAtIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBkaXN0YW5jZSBiZXR3ZWVuIGEgYW5kIGJcbiAgICAgKi9cbiAgICBjb25zdCBkaXN0ID0gZGlzdGFuY2U7XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIHNxdWFyZSBvZiB0aGUgZGlzdGFuY2UgYmV0d2VlbiAyIHBvaW50c1xuICAgICAqIEBwYXJhbSBhIC0gdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIHNxdWFyZSBvZiB0aGUgZGlzdGFuY2UgYmV0d2VlbiBhIGFuZCBiXG4gICAgICovXG4gICAgZnVuY3Rpb24gZGlzdGFuY2VTcShhLCBiKSB7XG4gICAgICAgIGNvbnN0IGR4ID0gYVswXSAtIGJbMF07XG4gICAgICAgIGNvbnN0IGR5ID0gYVsxXSAtIGJbMV07XG4gICAgICAgIHJldHVybiBkeCAqIGR4ICsgZHkgKiBkeTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIHNxdWFyZSBvZiB0aGUgZGlzdGFuY2UgYmV0d2VlbiAyIHBvaW50cyAoc2FtZSBhcyBkaXN0YW5jZVNxKVxuICAgICAqIEBwYXJhbSBhIC0gdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIHNxdWFyZSBvZiB0aGUgZGlzdGFuY2UgYmV0d2VlbiBhIGFuZCBiXG4gICAgICovXG4gICAgY29uc3QgZGlzdFNxID0gZGlzdGFuY2VTcTtcbiAgICAvKipcbiAgICAgKiBEaXZpZGVzIGEgdmVjdG9yIGJ5IGl0cyBFdWNsaWRlYW4gbGVuZ3RoIGFuZCByZXR1cm5zIHRoZSBxdW90aWVudC5cbiAgICAgKiBAcGFyYW0gdiAtIFRoZSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgbm9ybWFsaXplZCB2ZWN0b3IuXG4gICAgICovXG4gICAgZnVuY3Rpb24gbm9ybWFsaXplKHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDIpKTtcbiAgICAgICAgY29uc3QgdjAgPSB2WzBdO1xuICAgICAgICBjb25zdCB2MSA9IHZbMV07XG4gICAgICAgIGNvbnN0IGxlbiA9IE1hdGguc3FydCh2MCAqIHYwICsgdjEgKiB2MSk7XG4gICAgICAgIGlmIChsZW4gPiAwLjAwMDAxKSB7XG4gICAgICAgICAgICBuZXdEc3RbMF0gPSB2MCAvIGxlbjtcbiAgICAgICAgICAgIG5ld0RzdFsxXSA9IHYxIC8gbGVuO1xuICAgICAgICB9XG4gICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0gMDtcbiAgICAgICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogTmVnYXRlcyBhIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gdiAtIFRoZSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyAtdi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBuZWdhdGUodiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSAtdlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gLXZbMV07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvcGllcyBhIHZlY3Rvci4gKHNhbWUgYXMge0BsaW5rIHZlYzIuY2xvbmV9KVxuICAgICAqIEFsc28gc2VlIHtAbGluayB2ZWMyLmNyZWF0ZX0gYW5kIHtAbGluayB2ZWMyLnNldH1cbiAgICAgKiBAcGFyYW0gdiAtIFRoZSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIGNvcHkgb2Ygdi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBjb3B5KHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDIpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gdlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gdlsxXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ2xvbmVzIGEgdmVjdG9yLiAoc2FtZSBhcyB7QGxpbmsgdmVjMi5jb3B5fSlcbiAgICAgKiBBbHNvIHNlZSB7QGxpbmsgdmVjMi5jcmVhdGV9IGFuZCB7QGxpbmsgdmVjMi5zZXR9XG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSBjb3B5IG9mIHYuXG4gICAgICovXG4gICAgY29uc3QgY2xvbmUgPSBjb3B5O1xuICAgIC8qKlxuICAgICAqIE11bHRpcGxpZXMgYSB2ZWN0b3IgYnkgYW5vdGhlciB2ZWN0b3IgKGNvbXBvbmVudC13aXNlKTsgYXNzdW1lcyBhIGFuZFxuICAgICAqIGIgaGF2ZSB0aGUgc2FtZSBsZW5ndGguXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHZlY3RvciBvZiBwcm9kdWN0cyBvZiBlbnRyaWVzIG9mIGEgYW5kIGIuXG4gICAgICovXG4gICAgZnVuY3Rpb24gbXVsdGlwbHkoYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSBhWzBdICogYlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSAqIGJbMV07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIE11bHRpcGxpZXMgYSB2ZWN0b3IgYnkgYW5vdGhlciB2ZWN0b3IgKGNvbXBvbmVudC13aXNlKTsgYXNzdW1lcyBhIGFuZFxuICAgICAqIGIgaGF2ZSB0aGUgc2FtZSBsZW5ndGguIChzYW1lIGFzIG11bClcbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgdmVjdG9yIG9mIHByb2R1Y3RzIG9mIGVudHJpZXMgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBjb25zdCBtdWwgPSBtdWx0aXBseTtcbiAgICAvKipcbiAgICAgKiBEaXZpZGVzIGEgdmVjdG9yIGJ5IGFub3RoZXIgdmVjdG9yIChjb21wb25lbnQtd2lzZSk7IGFzc3VtZXMgYSBhbmRcbiAgICAgKiBiIGhhdmUgdGhlIHNhbWUgbGVuZ3RoLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSB2ZWN0b3Igb2YgcXVvdGllbnRzIG9mIGVudHJpZXMgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBkaXZpZGUoYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSBhWzBdIC8gYlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSAvIGJbMV07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIERpdmlkZXMgYSB2ZWN0b3IgYnkgYW5vdGhlciB2ZWN0b3IgKGNvbXBvbmVudC13aXNlKTsgYXNzdW1lcyBhIGFuZFxuICAgICAqIGIgaGF2ZSB0aGUgc2FtZSBsZW5ndGguIChzYW1lIGFzIGRpdmlkZSlcbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgdmVjdG9yIG9mIHF1b3RpZW50cyBvZiBlbnRyaWVzIG9mIGEgYW5kIGIuXG4gICAgICovXG4gICAgY29uc3QgZGl2ID0gZGl2aWRlO1xuICAgIC8qKlxuICAgICAqIENyZWF0ZXMgYSByYW5kb20gdW5pdCB2ZWN0b3IgKiBzY2FsZVxuICAgICAqIEBwYXJhbSBzY2FsZSAtIERlZmF1bHQgMVxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHJhbmRvbSB2ZWN0b3IuXG4gICAgICovXG4gICAgZnVuY3Rpb24gcmFuZG9tKHNjYWxlID0gMSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBjb25zdCBhbmdsZSA9IE1hdGgucmFuZG9tKCkgKiAyICogTWF0aC5QSTtcbiAgICAgICAgbmV3RHN0WzBdID0gTWF0aC5jb3MoYW5nbGUpICogc2NhbGU7XG4gICAgICAgIG5ld0RzdFsxXSA9IE1hdGguc2luKGFuZ2xlKSAqIHNjYWxlO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBaZXJvJ3MgYSB2ZWN0b3JcbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSB6ZXJvZWQgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHplcm8oZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSAwO1xuICAgICAgICBuZXdEc3RbMV0gPSAwO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiB0cmFuc2Zvcm0gVmVjMiBieSA0eDQgbWF0cml4XG4gICAgICogQHBhcmFtIHYgLSB0aGUgdmVjdG9yXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBkc3QgLSBvcHRpb25hbCBWZWMyIHRvIHN0b3JlIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyB0aGUgdHJhbnNmb3JtZWQgdmVjdG9yXG4gICAgICovXG4gICAgZnVuY3Rpb24gdHJhbnNmb3JtTWF0NCh2LCBtLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigyKSk7XG4gICAgICAgIGNvbnN0IHggPSB2WzBdO1xuICAgICAgICBjb25zdCB5ID0gdlsxXTtcbiAgICAgICAgbmV3RHN0WzBdID0geCAqIG1bMF0gKyB5ICogbVs0XSArIG1bMTJdO1xuICAgICAgICBuZXdEc3RbMV0gPSB4ICogbVsxXSArIHkgKiBtWzVdICsgbVsxM107XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFRyYW5zZm9ybXMgdmVjNCBieSAzeDMgbWF0cml4XG4gICAgICpcbiAgICAgKiBAcGFyYW0gdiAtIHRoZSB2ZWN0b3JcbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGRzdCAtIG9wdGlvbmFsIFZlYzIgdG8gc3RvcmUgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIHRoZSB0cmFuc2Zvcm1lZCB2ZWN0b3JcbiAgICAgKi9cbiAgICBmdW5jdGlvbiB0cmFuc2Zvcm1NYXQzKHYsIG0sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDIpKTtcbiAgICAgICAgY29uc3QgeCA9IHZbMF07XG4gICAgICAgIGNvbnN0IHkgPSB2WzFdO1xuICAgICAgICBuZXdEc3RbMF0gPSBtWzBdICogeCArIG1bNF0gKiB5ICsgbVs4XTtcbiAgICAgICAgbmV3RHN0WzFdID0gbVsxXSAqIHggKyBtWzVdICogeSArIG1bOV07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJvdGF0ZSBhIDJEIHZlY3RvclxuICAgICAqXG4gICAgICogQHBhcmFtIGEgVGhlIHZlYzIgcG9pbnQgdG8gcm90YXRlXG4gICAgICogQHBhcmFtIGIgVGhlIG9yaWdpbiBvZiB0aGUgcm90YXRpb25cbiAgICAgKiBAcGFyYW0gcmFkIFRoZSBhbmdsZSBvZiByb3RhdGlvbiBpbiByYWRpYW5zXG4gICAgICogQHJldHVybnMgdGhlIHJvdGF0ZWQgdmVjdG9yXG4gICAgICovXG4gICAgZnVuY3Rpb24gcm90YXRlKGEsIGIsIHJhZCwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICAvLyBUcmFuc2xhdGUgcG9pbnQgdG8gdGhlIG9yaWdpblxuICAgICAgICBjb25zdCBwMCA9IGFbMF0gLSBiWzBdO1xuICAgICAgICBjb25zdCBwMSA9IGFbMV0gLSBiWzFdO1xuICAgICAgICBjb25zdCBzaW5DID0gTWF0aC5zaW4ocmFkKTtcbiAgICAgICAgY29uc3QgY29zQyA9IE1hdGguY29zKHJhZCk7XG4gICAgICAgIC8vcGVyZm9ybSByb3RhdGlvbiBhbmQgdHJhbnNsYXRlIHRvIGNvcnJlY3QgcG9zaXRpb25cbiAgICAgICAgbmV3RHN0WzBdID0gcDAgKiBjb3NDIC0gcDEgKiBzaW5DICsgYlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gcDAgKiBzaW5DICsgcDEgKiBjb3NDICsgYlsxXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogVHJlYXQgYSAyRCB2ZWN0b3IgYXMgYSBkaXJlY3Rpb24gYW5kIHNldCBpdCdzIGxlbmd0aFxuICAgICAqXG4gICAgICogQHBhcmFtIGEgVGhlIHZlYzIgdG8gbGVuZ3RoZW5cbiAgICAgKiBAcGFyYW0gbGVuIFRoZSBsZW5ndGggb2YgdGhlIHJlc3VsdGluZyB2ZWN0b3JcbiAgICAgKiBAcmV0dXJucyBUaGUgbGVuZ3RoZW5lZCB2ZWN0b3JcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzZXRMZW5ndGgoYSwgbGVuLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigyKSk7XG4gICAgICAgIG5vcm1hbGl6ZShhLCBuZXdEc3QpO1xuICAgICAgICByZXR1cm4gbXVsU2NhbGFyKG5ld0RzdCwgbGVuLCBuZXdEc3QpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBFbnN1cmUgYSB2ZWN0b3IgaXMgbm90IGxvbmdlciB0aGFuIGEgbWF4IGxlbmd0aFxuICAgICAqXG4gICAgICogQHBhcmFtIGEgVGhlIHZlYzIgdG8gbGltaXRcbiAgICAgKiBAcGFyYW0gbWF4TGVuIFRoZSBsb25nZXN0IGxlbmd0aCBvZiB0aGUgcmVzdWx0aW5nIHZlY3RvclxuICAgICAqIEByZXR1cm5zIFRoZSB2ZWN0b3IsIHNob3J0ZW5lZCB0byBtYXhMZW4gaWYgaXQncyB0b28gbG9uZ1xuICAgICAqL1xuICAgIGZ1bmN0aW9uIHRydW5jYXRlKGEsIG1heExlbiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICBpZiAobGVuZ3RoKGEpID4gbWF4TGVuKSB7XG4gICAgICAgICAgICByZXR1cm4gc2V0TGVuZ3RoKGEsIG1heExlbiwgbmV3RHN0KTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gY29weShhLCBuZXdEc3QpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSZXR1cm4gdGhlIHZlY3RvciBleGFjdGx5IGJldHdlZW4gMiBlbmRwb2ludCB2ZWN0b3JzXG4gICAgICpcbiAgICAgKiBAcGFyYW0gYSBFbmRwb2ludCAxXG4gICAgICogQHBhcmFtIGIgRW5kcG9pbnQgMlxuICAgICAqIEByZXR1cm5zIFRoZSB2ZWN0b3IgZXhhY3RseSByZXNpZGluZyBiZXR3ZWVuIGVuZHBvaW50cyAxIGFuZCAyXG4gICAgICovXG4gICAgZnVuY3Rpb24gbWlkcG9pbnQoYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMikpO1xuICAgICAgICByZXR1cm4gbGVycChhLCBiLCAwLjUsIG5ld0RzdCk7XG4gICAgfVxuICAgIHJldHVybiB7XG4gICAgICAgIGNyZWF0ZSxcbiAgICAgICAgZnJvbVZhbHVlcyxcbiAgICAgICAgc2V0LFxuICAgICAgICBjZWlsLFxuICAgICAgICBmbG9vcixcbiAgICAgICAgcm91bmQsXG4gICAgICAgIGNsYW1wLFxuICAgICAgICBhZGQsXG4gICAgICAgIGFkZFNjYWxlZCxcbiAgICAgICAgYW5nbGUsXG4gICAgICAgIHN1YnRyYWN0LFxuICAgICAgICBzdWIsXG4gICAgICAgIGVxdWFsc0FwcHJveGltYXRlbHksXG4gICAgICAgIGVxdWFscyxcbiAgICAgICAgbGVycCxcbiAgICAgICAgbGVycFYsXG4gICAgICAgIG1heCxcbiAgICAgICAgbWluLFxuICAgICAgICBtdWxTY2FsYXIsXG4gICAgICAgIHNjYWxlLFxuICAgICAgICBkaXZTY2FsYXIsXG4gICAgICAgIGludmVyc2UsXG4gICAgICAgIGludmVydCxcbiAgICAgICAgY3Jvc3MsXG4gICAgICAgIGRvdCxcbiAgICAgICAgbGVuZ3RoLFxuICAgICAgICBsZW4sXG4gICAgICAgIGxlbmd0aFNxLFxuICAgICAgICBsZW5TcSxcbiAgICAgICAgZGlzdGFuY2UsXG4gICAgICAgIGRpc3QsXG4gICAgICAgIGRpc3RhbmNlU3EsXG4gICAgICAgIGRpc3RTcSxcbiAgICAgICAgbm9ybWFsaXplLFxuICAgICAgICBuZWdhdGUsXG4gICAgICAgIGNvcHksXG4gICAgICAgIGNsb25lLFxuICAgICAgICBtdWx0aXBseSxcbiAgICAgICAgbXVsLFxuICAgICAgICBkaXZpZGUsXG4gICAgICAgIGRpdixcbiAgICAgICAgcmFuZG9tLFxuICAgICAgICB6ZXJvLFxuICAgICAgICB0cmFuc2Zvcm1NYXQ0LFxuICAgICAgICB0cmFuc2Zvcm1NYXQzLFxuICAgICAgICByb3RhdGUsXG4gICAgICAgIHNldExlbmd0aCxcbiAgICAgICAgdHJ1bmNhdGUsXG4gICAgICAgIG1pZHBvaW50LFxuICAgIH07XG59XG5jb25zdCBjYWNoZSQ1ID0gbmV3IE1hcCgpO1xuZnVuY3Rpb24gZ2V0QVBJJDUoQ3Rvcikge1xuICAgIGxldCBhcGkgPSBjYWNoZSQ1LmdldChDdG9yKTtcbiAgICBpZiAoIWFwaSkge1xuICAgICAgICBhcGkgPSBnZXRBUElJbXBsJDUoQ3Rvcik7XG4gICAgICAgIGNhY2hlJDUuc2V0KEN0b3IsIGFwaSk7XG4gICAgfVxuICAgIHJldHVybiBhcGk7XG59XG5cbi8qXG4gKiBDb3B5cmlnaHQgMjAyMiBHcmVnZyBUYXZhcmVzXG4gKlxuICogUGVybWlzc2lvbiBpcyBoZXJlYnkgZ3JhbnRlZCwgZnJlZSBvZiBjaGFyZ2UsIHRvIGFueSBwZXJzb24gb2J0YWluaW5nIGFcbiAqIGNvcHkgb2YgdGhpcyBzb2Z0d2FyZSBhbmQgYXNzb2NpYXRlZCBkb2N1bWVudGF0aW9uIGZpbGVzICh0aGUgXCJTb2Z0d2FyZVwiKSxcbiAqIHRvIGRlYWwgaW4gdGhlIFNvZnR3YXJlIHdpdGhvdXQgcmVzdHJpY3Rpb24sIGluY2x1ZGluZyB3aXRob3V0IGxpbWl0YXRpb25cbiAqIHRoZSByaWdodHMgdG8gdXNlLCBjb3B5LCBtb2RpZnksIG1lcmdlLCBwdWJsaXNoLCBkaXN0cmlidXRlLCBzdWJsaWNlbnNlLFxuICogYW5kL29yIHNlbGwgY29waWVzIG9mIHRoZSBTb2Z0d2FyZSwgYW5kIHRvIHBlcm1pdCBwZXJzb25zIHRvIHdob20gdGhlXG4gKiBTb2Z0d2FyZSBpcyBmdXJuaXNoZWQgdG8gZG8gc28sIHN1YmplY3QgdG8gdGhlIGZvbGxvd2luZyBjb25kaXRpb25zOlxuICpcbiAqIFRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluXG4gKiBhbGwgY29waWVzIG9yIHN1YnN0YW50aWFsIHBvcnRpb25zIG9mIHRoZSBTb2Z0d2FyZS5cbiAqXG4gKiBUSEUgU09GVFdBUkUgSVMgUFJPVklERUQgXCJBUyBJU1wiLCBXSVRIT1VUIFdBUlJBTlRZIE9GIEFOWSBLSU5ELCBFWFBSRVNTIE9SXG4gKiBJTVBMSUVELCBJTkNMVURJTkcgQlVUIE5PVCBMSU1JVEVEIFRPIFRIRSBXQVJSQU5USUVTIE9GIE1FUkNIQU5UQUJJTElUWSxcbiAqIEZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFIEFORCBOT05JTkZSSU5HRU1FTlQuICBJTiBOTyBFVkVOVCBTSEFMTFxuICogVEhFIEFVVEhPUlMgT1IgQ09QWVJJR0hUIEhPTERFUlMgQkUgTElBQkxFIEZPUiBBTlkgQ0xBSU0sIERBTUFHRVMgT1IgT1RIRVJcbiAqIExJQUJJTElUWSwgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HXG4gKiBGUk9NLCBPVVQgT0YgT1IgSU4gQ09OTkVDVElPTiBXSVRIIFRIRSBTT0ZUV0FSRSBPUiBUSEUgVVNFIE9SIE9USEVSXG4gKiBERUFMSU5HUyBJTiBUSEUgU09GVFdBUkUuXG4gKi9cbi8qKlxuICogR2VuZXJhdGVzIGFtIHR5cGVkIEFQSSBmb3IgVmVjM1xuICogKi9cbmZ1bmN0aW9uIGdldEFQSUltcGwkNChDdG9yKSB7XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIHZlYzM7IG1heSBiZSBjYWxsZWQgd2l0aCB4LCB5LCB6IHRvIHNldCBpbml0aWFsIHZhbHVlcy5cbiAgICAgKiBAcGFyYW0geCAtIEluaXRpYWwgeCB2YWx1ZS5cbiAgICAgKiBAcGFyYW0geSAtIEluaXRpYWwgeSB2YWx1ZS5cbiAgICAgKiBAcGFyYW0geiAtIEluaXRpYWwgeiB2YWx1ZS5cbiAgICAgKiBAcmV0dXJucyB0aGUgY3JlYXRlZCB2ZWN0b3JcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBjcmVhdGUoeCwgeSwgeikge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSBuZXcgQ3RvcigzKTtcbiAgICAgICAgaWYgKHggIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0geDtcbiAgICAgICAgICAgIGlmICh5ICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICBuZXdEc3RbMV0gPSB5O1xuICAgICAgICAgICAgICAgIGlmICh6ICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgbmV3RHN0WzJdID0gejtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIHZlYzM7IG1heSBiZSBjYWxsZWQgd2l0aCB4LCB5LCB6IHRvIHNldCBpbml0aWFsIHZhbHVlcy4gKHNhbWUgYXMgY3JlYXRlKVxuICAgICAqIEBwYXJhbSB4IC0gSW5pdGlhbCB4IHZhbHVlLlxuICAgICAqIEBwYXJhbSB5IC0gSW5pdGlhbCB5IHZhbHVlLlxuICAgICAqIEBwYXJhbSB6IC0gSW5pdGlhbCB6IHZhbHVlLlxuICAgICAqIEByZXR1cm5zIHRoZSBjcmVhdGVkIHZlY3RvclxuICAgICAqL1xuICAgIGNvbnN0IGZyb21WYWx1ZXMgPSBjcmVhdGU7XG4gICAgLyoqXG4gICAgICogU2V0cyB0aGUgdmFsdWVzIG9mIGEgVmVjM1xuICAgICAqIEFsc28gc2VlIHtAbGluayB2ZWMzLmNyZWF0ZX0gYW5kIHtAbGluayB2ZWMzLmNvcHl9XG4gICAgICpcbiAgICAgKiBAcGFyYW0geCBmaXJzdCB2YWx1ZVxuICAgICAqIEBwYXJhbSB5IHNlY29uZCB2YWx1ZVxuICAgICAqIEBwYXJhbSB6IHRoaXJkIHZhbHVlXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB3aXRoIGl0cyBlbGVtZW50cyBzZXQuXG4gICAgICovXG4gICAgZnVuY3Rpb24gc2V0KHgsIHksIHosIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbmV3RHN0WzBdID0geDtcbiAgICAgICAgbmV3RHN0WzFdID0geTtcbiAgICAgICAgbmV3RHN0WzJdID0gejtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQXBwbGllcyBNYXRoLmNlaWwgdG8gZWFjaCBlbGVtZW50IG9mIHZlY3RvclxuICAgICAqIEBwYXJhbSB2IC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB0aGF0IGlzIHRoZSBjZWlsIG9mIGVhY2ggZWxlbWVudCBvZiB2LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGNlaWwodiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBuZXdEc3RbMF0gPSBNYXRoLmNlaWwodlswXSk7XG4gICAgICAgIG5ld0RzdFsxXSA9IE1hdGguY2VpbCh2WzFdKTtcbiAgICAgICAgbmV3RHN0WzJdID0gTWF0aC5jZWlsKHZbMl0pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBBcHBsaWVzIE1hdGguZmxvb3IgdG8gZWFjaCBlbGVtZW50IG9mIHZlY3RvclxuICAgICAqIEBwYXJhbSB2IC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB0aGF0IGlzIHRoZSBmbG9vciBvZiBlYWNoIGVsZW1lbnQgb2Ygdi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBmbG9vcih2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGguZmxvb3IodlswXSk7XG4gICAgICAgIG5ld0RzdFsxXSA9IE1hdGguZmxvb3IodlsxXSk7XG4gICAgICAgIG5ld0RzdFsyXSA9IE1hdGguZmxvb3IodlsyXSk7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEFwcGxpZXMgTWF0aC5yb3VuZCB0byBlYWNoIGVsZW1lbnQgb2YgdmVjdG9yXG4gICAgICogQHBhcmFtIHYgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHRoYXQgaXMgdGhlIHJvdW5kIG9mIGVhY2ggZWxlbWVudCBvZiB2LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdW5kKHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gTWF0aC5yb3VuZCh2WzBdKTtcbiAgICAgICAgbmV3RHN0WzFdID0gTWF0aC5yb3VuZCh2WzFdKTtcbiAgICAgICAgbmV3RHN0WzJdID0gTWF0aC5yb3VuZCh2WzJdKTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ2xhbXAgZWFjaCBlbGVtZW50IG9mIHZlY3RvciBiZXR3ZWVuIG1pbiBhbmQgbWF4XG4gICAgICogQHBhcmFtIHYgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gbWF4IC0gTWluIHZhbHVlLCBkZWZhdWx0IDBcbiAgICAgKiBAcGFyYW0gbWluIC0gTWF4IHZhbHVlLCBkZWZhdWx0IDFcbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHRoYXQgdGhlIGNsYW1wZWQgdmFsdWUgb2YgZWFjaCBlbGVtZW50IG9mIHYuXG4gICAgICovXG4gICAgZnVuY3Rpb24gY2xhbXAodiwgbWluID0gMCwgbWF4ID0gMSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBuZXdEc3RbMF0gPSBNYXRoLm1pbihtYXgsIE1hdGgubWF4KG1pbiwgdlswXSkpO1xuICAgICAgICBuZXdEc3RbMV0gPSBNYXRoLm1pbihtYXgsIE1hdGgubWF4KG1pbiwgdlsxXSkpO1xuICAgICAgICBuZXdEc3RbMl0gPSBNYXRoLm1pbihtYXgsIE1hdGgubWF4KG1pbiwgdlsyXSkpO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBBZGRzIHR3byB2ZWN0b3JzOyBhc3N1bWVzIGEgYW5kIGIgaGF2ZSB0aGUgc2FtZSBkaW1lbnNpb24uXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSB2ZWN0b3IgdGhhdCBpcyB0aGUgc3VtIG9mIGEgYW5kIGIuXG4gICAgICovXG4gICAgZnVuY3Rpb24gYWRkKGEsIGIsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSArIGJbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IGFbMV0gKyBiWzFdO1xuICAgICAgICBuZXdEc3RbMl0gPSBhWzJdICsgYlsyXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQWRkcyB0d28gdmVjdG9ycywgc2NhbGluZyB0aGUgMm5kOyBhc3N1bWVzIGEgYW5kIGIgaGF2ZSB0aGUgc2FtZSBkaW1lbnNpb24uXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBzY2FsZSAtIEFtb3VudCB0byBzY2FsZSBiXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB0aGF0IGlzIHRoZSBzdW0gb2YgYSArIGIgKiBzY2FsZS5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBhZGRTY2FsZWQoYSwgYiwgc2NhbGUsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSArIGJbMF0gKiBzY2FsZTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSArIGJbMV0gKiBzY2FsZTtcbiAgICAgICAgbmV3RHN0WzJdID0gYVsyXSArIGJbMl0gKiBzY2FsZTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUmV0dXJucyB0aGUgYW5nbGUgaW4gcmFkaWFucyBiZXR3ZWVuIHR3byB2ZWN0b3JzLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBUaGUgYW5nbGUgaW4gcmFkaWFucyBiZXR3ZWVuIHRoZSAyIHZlY3RvcnMuXG4gICAgICovXG4gICAgZnVuY3Rpb24gYW5nbGUoYSwgYikge1xuICAgICAgICBjb25zdCBheCA9IGFbMF07XG4gICAgICAgIGNvbnN0IGF5ID0gYVsxXTtcbiAgICAgICAgY29uc3QgYXogPSBhWzJdO1xuICAgICAgICBjb25zdCBieCA9IGJbMF07XG4gICAgICAgIGNvbnN0IGJ5ID0gYlsxXTtcbiAgICAgICAgY29uc3QgYnogPSBiWzJdO1xuICAgICAgICBjb25zdCBtYWcxID0gTWF0aC5zcXJ0KGF4ICogYXggKyBheSAqIGF5ICsgYXogKiBheik7XG4gICAgICAgIGNvbnN0IG1hZzIgPSBNYXRoLnNxcnQoYnggKiBieCArIGJ5ICogYnkgKyBieiAqIGJ6KTtcbiAgICAgICAgY29uc3QgbWFnID0gbWFnMSAqIG1hZzI7XG4gICAgICAgIGNvbnN0IGNvc2luZSA9IG1hZyAmJiBkb3QoYSwgYikgLyBtYWc7XG4gICAgICAgIHJldHVybiBNYXRoLmFjb3MoY29zaW5lKTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogU3VidHJhY3RzIHR3byB2ZWN0b3JzLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHRoYXQgaXMgdGhlIGRpZmZlcmVuY2Ugb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzdWJ0cmFjdChhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IGFbMF0gLSBiWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSBhWzFdIC0gYlsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gYVsyXSAtIGJbMl07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFN1YnRyYWN0cyB0d28gdmVjdG9ycy5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB0aGF0IGlzIHRoZSBkaWZmZXJlbmNlIG9mIGEgYW5kIGIuXG4gICAgICovXG4gICAgY29uc3Qgc3ViID0gc3VidHJhY3Q7XG4gICAgLyoqXG4gICAgICogQ2hlY2sgaWYgMiB2ZWN0b3JzIGFyZSBhcHByb3hpbWF0ZWx5IGVxdWFsXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIHRydWUgaWYgdmVjdG9ycyBhcmUgYXBwcm94aW1hdGVseSBlcXVhbFxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGVxdWFsc0FwcHJveGltYXRlbHkoYSwgYikge1xuICAgICAgICByZXR1cm4gTWF0aC5hYnMoYVswXSAtIGJbMF0pIDwgRVBTSUxPTiAmJlxuICAgICAgICAgICAgTWF0aC5hYnMoYVsxXSAtIGJbMV0pIDwgRVBTSUxPTiAmJlxuICAgICAgICAgICAgTWF0aC5hYnMoYVsyXSAtIGJbMl0pIDwgRVBTSUxPTjtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ2hlY2sgaWYgMiB2ZWN0b3JzIGFyZSBleGFjdGx5IGVxdWFsXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIHRydWUgaWYgdmVjdG9ycyBhcmUgZXhhY3RseSBlcXVhbFxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGVxdWFscyhhLCBiKSB7XG4gICAgICAgIHJldHVybiBhWzBdID09PSBiWzBdICYmIGFbMV0gPT09IGJbMV0gJiYgYVsyXSA9PT0gYlsyXTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUGVyZm9ybXMgbGluZWFyIGludGVycG9sYXRpb24gb24gdHdvIHZlY3RvcnMuXG4gICAgICogR2l2ZW4gdmVjdG9ycyBhIGFuZCBiIGFuZCBpbnRlcnBvbGF0aW9uIGNvZWZmaWNpZW50IHQsIHJldHVybnNcbiAgICAgKiBhICsgdCAqIChiIC0gYSkuXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSB0IC0gSW50ZXJwb2xhdGlvbiBjb2VmZmljaWVudC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBsaW5lYXIgaW50ZXJwb2xhdGVkIHJlc3VsdC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBsZXJwKGEsIGIsIHQsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSArIHQgKiAoYlswXSAtIGFbMF0pO1xuICAgICAgICBuZXdEc3RbMV0gPSBhWzFdICsgdCAqIChiWzFdIC0gYVsxXSk7XG4gICAgICAgIG5ld0RzdFsyXSA9IGFbMl0gKyB0ICogKGJbMl0gLSBhWzJdKTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUGVyZm9ybXMgbGluZWFyIGludGVycG9sYXRpb24gb24gdHdvIHZlY3RvcnMuXG4gICAgICogR2l2ZW4gdmVjdG9ycyBhIGFuZCBiIGFuZCBpbnRlcnBvbGF0aW9uIGNvZWZmaWNpZW50IHZlY3RvciB0LCByZXR1cm5zXG4gICAgICogYSArIHQgKiAoYiAtIGEpLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gdCAtIEludGVycG9sYXRpb24gY29lZmZpY2llbnRzIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIHRoZSBsaW5lYXIgaW50ZXJwb2xhdGVkIHJlc3VsdC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBsZXJwVihhLCBiLCB0LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IGFbMF0gKyB0WzBdICogKGJbMF0gLSBhWzBdKTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSArIHRbMV0gKiAoYlsxXSAtIGFbMV0pO1xuICAgICAgICBuZXdEc3RbMl0gPSBhWzJdICsgdFsyXSAqIChiWzJdIC0gYVsyXSk7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJldHVybiBtYXggdmFsdWVzIG9mIHR3byB2ZWN0b3JzLlxuICAgICAqIEdpdmVuIHZlY3RvcnMgYSBhbmQgYiByZXR1cm5zXG4gICAgICogW21heChhWzBdLCBiWzBdKSwgbWF4KGFbMV0sIGJbMV0pLCBtYXgoYVsyXSwgYlsyXSldLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBtYXggY29tcG9uZW50cyB2ZWN0b3IuXG4gICAgICovXG4gICAgZnVuY3Rpb24gbWF4KGEsIGIsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gTWF0aC5tYXgoYVswXSwgYlswXSk7XG4gICAgICAgIG5ld0RzdFsxXSA9IE1hdGgubWF4KGFbMV0sIGJbMV0pO1xuICAgICAgICBuZXdEc3RbMl0gPSBNYXRoLm1heChhWzJdLCBiWzJdKTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUmV0dXJuIG1pbiB2YWx1ZXMgb2YgdHdvIHZlY3RvcnMuXG4gICAgICogR2l2ZW4gdmVjdG9ycyBhIGFuZCBiIHJldHVybnNcbiAgICAgKiBbbWluKGFbMF0sIGJbMF0pLCBtaW4oYVsxXSwgYlsxXSksIG1pbihhWzJdLCBiWzJdKV0uXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIG1pbiBjb21wb25lbnRzIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBtaW4oYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBuZXdEc3RbMF0gPSBNYXRoLm1pbihhWzBdLCBiWzBdKTtcbiAgICAgICAgbmV3RHN0WzFdID0gTWF0aC5taW4oYVsxXSwgYlsxXSk7XG4gICAgICAgIG5ld0RzdFsyXSA9IE1hdGgubWluKGFbMl0sIGJbMl0pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBNdWx0aXBsaWVzIGEgdmVjdG9yIGJ5IGEgc2NhbGFyLlxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gayAtIFRoZSBzY2FsYXIuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBtdWxTY2FsYXIodiwgaywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2WzBdICogaztcbiAgICAgICAgbmV3RHN0WzFdID0gdlsxXSAqIGs7XG4gICAgICAgIG5ld0RzdFsyXSA9IHZbMl0gKiBrO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBNdWx0aXBsaWVzIGEgdmVjdG9yIGJ5IGEgc2NhbGFyLiAoc2FtZSBhcyBtdWxTY2FsYXIpXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBrIC0gVGhlIHNjYWxhci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBzY2FsZWQgdmVjdG9yLlxuICAgICAqL1xuICAgIGNvbnN0IHNjYWxlID0gbXVsU2NhbGFyO1xuICAgIC8qKlxuICAgICAqIERpdmlkZXMgYSB2ZWN0b3IgYnkgYSBzY2FsYXIuXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBrIC0gVGhlIHNjYWxhci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBzY2FsZWQgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGRpdlNjYWxhcih2LCBrLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IHZbMF0gLyBrO1xuICAgICAgICBuZXdEc3RbMV0gPSB2WzFdIC8gaztcbiAgICAgICAgbmV3RHN0WzJdID0gdlsyXSAvIGs7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEludmVyc2UgYSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIGludmVydGVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBpbnZlcnNlKHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gMSAvIHZbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IDEgLyB2WzFdO1xuICAgICAgICBuZXdEc3RbMl0gPSAxIC8gdlsyXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogSW52ZXJ0IGEgdmVjdG9yLiAoc2FtZSBhcyBpbnZlcnNlKVxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBpbnZlcnRlZCB2ZWN0b3IuXG4gICAgICovXG4gICAgY29uc3QgaW52ZXJ0ID0gaW52ZXJzZTtcbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgY3Jvc3MgcHJvZHVjdCBvZiB0d28gdmVjdG9yczsgYXNzdW1lcyBib3RoIHZlY3RvcnMgaGF2ZVxuICAgICAqIHRocmVlIGVudHJpZXMuXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHZlY3RvciBvZiBhIGNyb3NzIGIuXG4gICAgICovXG4gICAgZnVuY3Rpb24gY3Jvc3MoYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBjb25zdCB0MSA9IGFbMl0gKiBiWzBdIC0gYVswXSAqIGJbMl07XG4gICAgICAgIGNvbnN0IHQyID0gYVswXSAqIGJbMV0gLSBhWzFdICogYlswXTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVsxXSAqIGJbMl0gLSBhWzJdICogYlsxXTtcbiAgICAgICAgbmV3RHN0WzFdID0gdDE7XG4gICAgICAgIG5ld0RzdFsyXSA9IHQyO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgdHdvIHZlY3RvcnM7IGFzc3VtZXMgYm90aCB2ZWN0b3JzIGhhdmVcbiAgICAgKiB0aHJlZSBlbnRyaWVzLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBkb3QgcHJvZHVjdFxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGRvdChhLCBiKSB7XG4gICAgICAgIHJldHVybiAoYVswXSAqIGJbMF0pICsgKGFbMV0gKiBiWzFdKSArIChhWzJdICogYlsyXSk7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBsZW5ndGggb2YgdmVjdG9yXG4gICAgICogQHBhcmFtIHYgLSB2ZWN0b3IuXG4gICAgICogQHJldHVybnMgbGVuZ3RoIG9mIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBsZW5ndGgodikge1xuICAgICAgICBjb25zdCB2MCA9IHZbMF07XG4gICAgICAgIGNvbnN0IHYxID0gdlsxXTtcbiAgICAgICAgY29uc3QgdjIgPSB2WzJdO1xuICAgICAgICByZXR1cm4gTWF0aC5zcXJ0KHYwICogdjAgKyB2MSAqIHYxICsgdjIgKiB2Mik7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBsZW5ndGggb2YgdmVjdG9yIChzYW1lIGFzIGxlbmd0aClcbiAgICAgKiBAcGFyYW0gdiAtIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBsZW5ndGggb2YgdmVjdG9yLlxuICAgICAqL1xuICAgIGNvbnN0IGxlbiA9IGxlbmd0aDtcbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgc3F1YXJlIG9mIHRoZSBsZW5ndGggb2YgdmVjdG9yXG4gICAgICogQHBhcmFtIHYgLSB2ZWN0b3IuXG4gICAgICogQHJldHVybnMgc3F1YXJlIG9mIHRoZSBsZW5ndGggb2YgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGxlbmd0aFNxKHYpIHtcbiAgICAgICAgY29uc3QgdjAgPSB2WzBdO1xuICAgICAgICBjb25zdCB2MSA9IHZbMV07XG4gICAgICAgIGNvbnN0IHYyID0gdlsyXTtcbiAgICAgICAgcmV0dXJuIHYwICogdjAgKyB2MSAqIHYxICsgdjIgKiB2MjtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIHNxdWFyZSBvZiB0aGUgbGVuZ3RoIG9mIHZlY3RvciAoc2FtZSBhcyBsZW5ndGhTcSlcbiAgICAgKiBAcGFyYW0gdiAtIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBzcXVhcmUgb2YgdGhlIGxlbmd0aCBvZiB2ZWN0b3IuXG4gICAgICovXG4gICAgY29uc3QgbGVuU3EgPSBsZW5ndGhTcTtcbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgZGlzdGFuY2UgYmV0d2VlbiAyIHBvaW50c1xuICAgICAqIEBwYXJhbSBhIC0gdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIGRpc3RhbmNlIGJldHdlZW4gYSBhbmQgYlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGRpc3RhbmNlKGEsIGIpIHtcbiAgICAgICAgY29uc3QgZHggPSBhWzBdIC0gYlswXTtcbiAgICAgICAgY29uc3QgZHkgPSBhWzFdIC0gYlsxXTtcbiAgICAgICAgY29uc3QgZHogPSBhWzJdIC0gYlsyXTtcbiAgICAgICAgcmV0dXJuIE1hdGguc3FydChkeCAqIGR4ICsgZHkgKiBkeSArIGR6ICogZHopO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgZGlzdGFuY2UgYmV0d2VlbiAyIHBvaW50cyAoc2FtZSBhcyBkaXN0YW5jZSlcbiAgICAgKiBAcGFyYW0gYSAtIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBkaXN0YW5jZSBiZXR3ZWVuIGEgYW5kIGJcbiAgICAgKi9cbiAgICBjb25zdCBkaXN0ID0gZGlzdGFuY2U7XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIHNxdWFyZSBvZiB0aGUgZGlzdGFuY2UgYmV0d2VlbiAyIHBvaW50c1xuICAgICAqIEBwYXJhbSBhIC0gdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIHNxdWFyZSBvZiB0aGUgZGlzdGFuY2UgYmV0d2VlbiBhIGFuZCBiXG4gICAgICovXG4gICAgZnVuY3Rpb24gZGlzdGFuY2VTcShhLCBiKSB7XG4gICAgICAgIGNvbnN0IGR4ID0gYVswXSAtIGJbMF07XG4gICAgICAgIGNvbnN0IGR5ID0gYVsxXSAtIGJbMV07XG4gICAgICAgIGNvbnN0IGR6ID0gYVsyXSAtIGJbMl07XG4gICAgICAgIHJldHVybiBkeCAqIGR4ICsgZHkgKiBkeSArIGR6ICogZHo7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBzcXVhcmUgb2YgdGhlIGRpc3RhbmNlIGJldHdlZW4gMiBwb2ludHMgKHNhbWUgYXMgZGlzdGFuY2VTcSlcbiAgICAgKiBAcGFyYW0gYSAtIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBzcXVhcmUgb2YgdGhlIGRpc3RhbmNlIGJldHdlZW4gYSBhbmQgYlxuICAgICAqL1xuICAgIGNvbnN0IGRpc3RTcSA9IGRpc3RhbmNlU3E7XG4gICAgLyoqXG4gICAgICogRGl2aWRlcyBhIHZlY3RvciBieSBpdHMgRXVjbGlkZWFuIGxlbmd0aCBhbmQgcmV0dXJucyB0aGUgcXVvdGllbnQuXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIG5vcm1hbGl6ZWQgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG5vcm1hbGl6ZSh2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIGNvbnN0IHYwID0gdlswXTtcbiAgICAgICAgY29uc3QgdjEgPSB2WzFdO1xuICAgICAgICBjb25zdCB2MiA9IHZbMl07XG4gICAgICAgIGNvbnN0IGxlbiA9IE1hdGguc3FydCh2MCAqIHYwICsgdjEgKiB2MSArIHYyICogdjIpO1xuICAgICAgICBpZiAobGVuID4gMC4wMDAwMSkge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0gdjAgLyBsZW47XG4gICAgICAgICAgICBuZXdEc3RbMV0gPSB2MSAvIGxlbjtcbiAgICAgICAgICAgIG5ld0RzdFsyXSA9IHYyIC8gbGVuO1xuICAgICAgICB9XG4gICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0gMDtcbiAgICAgICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIE5lZ2F0ZXMgYSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgLXYuXG4gICAgICovXG4gICAgZnVuY3Rpb24gbmVnYXRlKHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gLXZbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IC12WzFdO1xuICAgICAgICBuZXdEc3RbMl0gPSAtdlsyXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29waWVzIGEgdmVjdG9yLiAoc2FtZSBhcyB7QGxpbmsgdmVjMy5jbG9uZX0pXG4gICAgICogQWxzbyBzZWUge0BsaW5rIHZlYzMuY3JlYXRlfSBhbmQge0BsaW5rIHZlYzMuc2V0fVxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgY29weSBvZiB2LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGNvcHkodiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2WzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSB2WzFdO1xuICAgICAgICBuZXdEc3RbMl0gPSB2WzJdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDbG9uZXMgYSB2ZWN0b3IuIChzYW1lIGFzIHtAbGluayB2ZWMzLmNvcHl9KVxuICAgICAqIEFsc28gc2VlIHtAbGluayB2ZWMzLmNyZWF0ZX0gYW5kIHtAbGluayB2ZWMzLnNldH1cbiAgICAgKiBAcGFyYW0gdiAtIFRoZSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIGNvcHkgb2Ygdi5cbiAgICAgKi9cbiAgICBjb25zdCBjbG9uZSA9IGNvcHk7XG4gICAgLyoqXG4gICAgICogTXVsdGlwbGllcyBhIHZlY3RvciBieSBhbm90aGVyIHZlY3RvciAoY29tcG9uZW50LXdpc2UpOyBhc3N1bWVzIGEgYW5kXG4gICAgICogYiBoYXZlIHRoZSBzYW1lIGxlbmd0aC5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgdmVjdG9yIG9mIHByb2R1Y3RzIG9mIGVudHJpZXMgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBtdWx0aXBseShhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IGFbMF0gKiBiWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSBhWzFdICogYlsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gYVsyXSAqIGJbMl07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIE11bHRpcGxpZXMgYSB2ZWN0b3IgYnkgYW5vdGhlciB2ZWN0b3IgKGNvbXBvbmVudC13aXNlKTsgYXNzdW1lcyBhIGFuZFxuICAgICAqIGIgaGF2ZSB0aGUgc2FtZSBsZW5ndGguIChzYW1lIGFzIG11bClcbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgdmVjdG9yIG9mIHByb2R1Y3RzIG9mIGVudHJpZXMgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBjb25zdCBtdWwgPSBtdWx0aXBseTtcbiAgICAvKipcbiAgICAgKiBEaXZpZGVzIGEgdmVjdG9yIGJ5IGFub3RoZXIgdmVjdG9yIChjb21wb25lbnQtd2lzZSk7IGFzc3VtZXMgYSBhbmRcbiAgICAgKiBiIGhhdmUgdGhlIHNhbWUgbGVuZ3RoLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSB2ZWN0b3Igb2YgcXVvdGllbnRzIG9mIGVudHJpZXMgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBkaXZpZGUoYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBuZXdEc3RbMF0gPSBhWzBdIC8gYlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSAvIGJbMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IGFbMl0gLyBiWzJdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBEaXZpZGVzIGEgdmVjdG9yIGJ5IGFub3RoZXIgdmVjdG9yIChjb21wb25lbnQtd2lzZSk7IGFzc3VtZXMgYSBhbmRcbiAgICAgKiBiIGhhdmUgdGhlIHNhbWUgbGVuZ3RoLiAoc2FtZSBhcyBkaXZpZGUpXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHZlY3RvciBvZiBxdW90aWVudHMgb2YgZW50cmllcyBvZiBhIGFuZCBiLlxuICAgICAqL1xuICAgIGNvbnN0IGRpdiA9IGRpdmlkZTtcbiAgICAvKipcbiAgICAgKiBDcmVhdGVzIGEgcmFuZG9tIHZlY3RvclxuICAgICAqIEBwYXJhbSBzY2FsZSAtIERlZmF1bHQgMVxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHJhbmRvbSB2ZWN0b3IuXG4gICAgICovXG4gICAgZnVuY3Rpb24gcmFuZG9tKHNjYWxlID0gMSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBjb25zdCBhbmdsZSA9IE1hdGgucmFuZG9tKCkgKiAyICogTWF0aC5QSTtcbiAgICAgICAgY29uc3QgeiA9IE1hdGgucmFuZG9tKCkgKiAyIC0gMTtcbiAgICAgICAgY29uc3QgelNjYWxlID0gTWF0aC5zcXJ0KDEgLSB6ICogeikgKiBzY2FsZTtcbiAgICAgICAgbmV3RHN0WzBdID0gTWF0aC5jb3MoYW5nbGUpICogelNjYWxlO1xuICAgICAgICBuZXdEc3RbMV0gPSBNYXRoLnNpbihhbmdsZSkgKiB6U2NhbGU7XG4gICAgICAgIG5ld0RzdFsyXSA9IHogKiBzY2FsZTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogWmVybydzIGEgdmVjdG9yXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgemVyb2VkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiB6ZXJvKGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gMDtcbiAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgbmV3RHN0WzJdID0gMDtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogdHJhbnNmb3JtIHZlYzMgYnkgNHg0IG1hdHJpeFxuICAgICAqIEBwYXJhbSB2IC0gdGhlIHZlY3RvclxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gb3B0aW9uYWwgdmVjMyB0byBzdG9yZSByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgdGhlIHRyYW5zZm9ybWVkIHZlY3RvclxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHRyYW5zZm9ybU1hdDQodiwgbSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBjb25zdCB4ID0gdlswXTtcbiAgICAgICAgY29uc3QgeSA9IHZbMV07XG4gICAgICAgIGNvbnN0IHogPSB2WzJdO1xuICAgICAgICBjb25zdCB3ID0gKG1bM10gKiB4ICsgbVs3XSAqIHkgKyBtWzExXSAqIHogKyBtWzE1XSkgfHwgMTtcbiAgICAgICAgbmV3RHN0WzBdID0gKG1bMF0gKiB4ICsgbVs0XSAqIHkgKyBtWzhdICogeiArIG1bMTJdKSAvIHc7XG4gICAgICAgIG5ld0RzdFsxXSA9IChtWzFdICogeCArIG1bNV0gKiB5ICsgbVs5XSAqIHogKyBtWzEzXSkgLyB3O1xuICAgICAgICBuZXdEc3RbMl0gPSAobVsyXSAqIHggKyBtWzZdICogeSArIG1bMTBdICogeiArIG1bMTRdKSAvIHc7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFRyYW5zZm9ybSB2ZWMzIGJ5IHVwcGVyIDN4MyBtYXRyaXggaW5zaWRlIDR4NCBtYXRyaXguXG4gICAgICogQHBhcmFtIHYgLSBUaGUgZGlyZWN0aW9uLlxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gb3B0aW9uYWwgdmVjMyB0byBzdG9yZSByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHRyYW5zZm9ybWVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiB0cmFuc2Zvcm1NYXQ0VXBwZXIzeDModiwgbSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBjb25zdCB2MCA9IHZbMF07XG4gICAgICAgIGNvbnN0IHYxID0gdlsxXTtcbiAgICAgICAgY29uc3QgdjIgPSB2WzJdO1xuICAgICAgICBuZXdEc3RbMF0gPSB2MCAqIG1bMCAqIDQgKyAwXSArIHYxICogbVsxICogNCArIDBdICsgdjIgKiBtWzIgKiA0ICsgMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IHYwICogbVswICogNCArIDFdICsgdjEgKiBtWzEgKiA0ICsgMV0gKyB2MiAqIG1bMiAqIDQgKyAxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gdjAgKiBtWzAgKiA0ICsgMl0gKyB2MSAqIG1bMSAqIDQgKyAyXSArIHYyICogbVsyICogNCArIDJdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBUcmFuc2Zvcm1zIHZlYzMgYnkgM3gzIG1hdHJpeFxuICAgICAqXG4gICAgICogQHBhcmFtIHYgLSB0aGUgdmVjdG9yXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBkc3QgLSBvcHRpb25hbCB2ZWMzIHRvIHN0b3JlIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyB0aGUgdHJhbnNmb3JtZWQgdmVjdG9yXG4gICAgICovXG4gICAgZnVuY3Rpb24gdHJhbnNmb3JtTWF0Myh2LCBtLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIGNvbnN0IHggPSB2WzBdO1xuICAgICAgICBjb25zdCB5ID0gdlsxXTtcbiAgICAgICAgY29uc3QgeiA9IHZbMl07XG4gICAgICAgIG5ld0RzdFswXSA9IHggKiBtWzBdICsgeSAqIG1bNF0gKyB6ICogbVs4XTtcbiAgICAgICAgbmV3RHN0WzFdID0geCAqIG1bMV0gKyB5ICogbVs1XSArIHogKiBtWzldO1xuICAgICAgICBuZXdEc3RbMl0gPSB4ICogbVsyXSArIHkgKiBtWzZdICsgeiAqIG1bMTBdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBUcmFuc2Zvcm1zIHZlYzMgYnkgUXVhdGVybmlvblxuICAgICAqIEBwYXJhbSB2IC0gdGhlIHZlY3RvciB0byB0cmFuc2Zvcm1cbiAgICAgKiBAcGFyYW0gcSAtIHRoZSBxdWF0ZXJuaW9uIHRvIHRyYW5zZm9ybSBieVxuICAgICAqIEBwYXJhbSBkc3QgLSBvcHRpb25hbCB2ZWMzIHRvIHN0b3JlIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyB0aGUgdHJhbnNmb3JtZWRcbiAgICAgKi9cbiAgICBmdW5jdGlvbiB0cmFuc2Zvcm1RdWF0KHYsIHEsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgY29uc3QgcXggPSBxWzBdO1xuICAgICAgICBjb25zdCBxeSA9IHFbMV07XG4gICAgICAgIGNvbnN0IHF6ID0gcVsyXTtcbiAgICAgICAgY29uc3QgdzIgPSBxWzNdICogMjtcbiAgICAgICAgY29uc3QgeCA9IHZbMF07XG4gICAgICAgIGNvbnN0IHkgPSB2WzFdO1xuICAgICAgICBjb25zdCB6ID0gdlsyXTtcbiAgICAgICAgY29uc3QgdXZYID0gcXkgKiB6IC0gcXogKiB5O1xuICAgICAgICBjb25zdCB1dlkgPSBxeiAqIHggLSBxeCAqIHo7XG4gICAgICAgIGNvbnN0IHV2WiA9IHF4ICogeSAtIHF5ICogeDtcbiAgICAgICAgbmV3RHN0WzBdID0geCArIHV2WCAqIHcyICsgKHF5ICogdXZaIC0gcXogKiB1dlkpICogMjtcbiAgICAgICAgbmV3RHN0WzFdID0geSArIHV2WSAqIHcyICsgKHF6ICogdXZYIC0gcXggKiB1dlopICogMjtcbiAgICAgICAgbmV3RHN0WzJdID0geiArIHV2WiAqIHcyICsgKHF4ICogdXZZIC0gcXkgKiB1dlgpICogMjtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUmV0dXJucyB0aGUgdHJhbnNsYXRpb24gY29tcG9uZW50IG9mIGEgNC1ieS00IG1hdHJpeCBhcyBhIHZlY3RvciB3aXRoIDNcbiAgICAgKiBlbnRyaWVzLlxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSB0cmFuc2xhdGlvbiBjb21wb25lbnQgb2YgbS5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBnZXRUcmFuc2xhdGlvbihtLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IG1bMTJdO1xuICAgICAgICBuZXdEc3RbMV0gPSBtWzEzXTtcbiAgICAgICAgbmV3RHN0WzJdID0gbVsxNF07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJldHVybnMgYW4gYXhpcyBvZiBhIDR4NCBtYXRyaXggYXMgYSB2ZWN0b3Igd2l0aCAzIGVudHJpZXNcbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGF4aXMgLSBUaGUgYXhpcyAwID0geCwgMSA9IHksIDIgPSB6O1xuICAgICAqIEByZXR1cm5zIFRoZSBheGlzIGNvbXBvbmVudCBvZiBtLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGdldEF4aXMobSwgYXhpcywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBjb25zdCBvZmYgPSBheGlzICogNDtcbiAgICAgICAgbmV3RHN0WzBdID0gbVtvZmYgKyAwXTtcbiAgICAgICAgbmV3RHN0WzFdID0gbVtvZmYgKyAxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gbVtvZmYgKyAyXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUmV0dXJucyB0aGUgc2NhbGluZyBjb21wb25lbnQgb2YgdGhlIG1hdHJpeFxuICAgICAqIEBwYXJhbSBtIC0gVGhlIE1hdHJpeFxuICAgICAqIEBwYXJhbSBkc3QgLSBUaGUgdmVjdG9yIHRvIHNldC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBnZXRTY2FsaW5nKG0sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgY29uc3QgeHggPSBtWzBdO1xuICAgICAgICBjb25zdCB4eSA9IG1bMV07XG4gICAgICAgIGNvbnN0IHh6ID0gbVsyXTtcbiAgICAgICAgY29uc3QgeXggPSBtWzRdO1xuICAgICAgICBjb25zdCB5eSA9IG1bNV07XG4gICAgICAgIGNvbnN0IHl6ID0gbVs2XTtcbiAgICAgICAgY29uc3QgenggPSBtWzhdO1xuICAgICAgICBjb25zdCB6eSA9IG1bOV07XG4gICAgICAgIGNvbnN0IHp6ID0gbVsxMF07XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGguc3FydCh4eCAqIHh4ICsgeHkgKiB4eSArIHh6ICogeHopO1xuICAgICAgICBuZXdEc3RbMV0gPSBNYXRoLnNxcnQoeXggKiB5eCArIHl5ICogeXkgKyB5eiAqIHl6KTtcbiAgICAgICAgbmV3RHN0WzJdID0gTWF0aC5zcXJ0KHp4ICogenggKyB6eSAqIHp5ICsgenogKiB6eik7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJvdGF0ZSBhIDNEIHZlY3RvciBhcm91bmQgdGhlIHgtYXhpc1xuICAgICAqXG4gICAgICogQHBhcmFtIHtSZWFkb25seVZlYzN9IGEgVGhlIHZlYzMgcG9pbnQgdG8gcm90YXRlXG4gICAgICogQHBhcmFtIHtSZWFkb25seVZlYzN9IGIgVGhlIG9yaWdpbiBvZiB0aGUgcm90YXRpb25cbiAgICAgKiBAcGFyYW0ge051bWJlcn0gcmFkIFRoZSBhbmdsZSBvZiByb3RhdGlvbiBpbiByYWRpYW5zXG4gICAgICogQHBhcmFtIGRzdCAtIFRoZSB2ZWN0b3IgdG8gc2V0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIHRoZSByb3RhdGVkIHZlY3RvclxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0ZVgoYSwgYiwgcmFkLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIGNvbnN0IHAgPSBbXTtcbiAgICAgICAgY29uc3QgciA9IFtdO1xuICAgICAgICAvL1RyYW5zbGF0ZSBwb2ludCB0byB0aGUgb3JpZ2luXG4gICAgICAgIHBbMF0gPSBhWzBdIC0gYlswXTtcbiAgICAgICAgcFsxXSA9IGFbMV0gLSBiWzFdO1xuICAgICAgICBwWzJdID0gYVsyXSAtIGJbMl07XG4gICAgICAgIC8vcGVyZm9ybSByb3RhdGlvblxuICAgICAgICByWzBdID0gcFswXTtcbiAgICAgICAgclsxXSA9IHBbMV0gKiBNYXRoLmNvcyhyYWQpIC0gcFsyXSAqIE1hdGguc2luKHJhZCk7XG4gICAgICAgIHJbMl0gPSBwWzFdICogTWF0aC5zaW4ocmFkKSArIHBbMl0gKiBNYXRoLmNvcyhyYWQpO1xuICAgICAgICAvL3RyYW5zbGF0ZSB0byBjb3JyZWN0IHBvc2l0aW9uXG4gICAgICAgIG5ld0RzdFswXSA9IHJbMF0gKyBiWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSByWzFdICsgYlsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gclsyXSArIGJbMl07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJvdGF0ZSBhIDNEIHZlY3RvciBhcm91bmQgdGhlIHktYXhpc1xuICAgICAqXG4gICAgICogQHBhcmFtIHtSZWFkb25seVZlYzN9IGEgVGhlIHZlYzMgcG9pbnQgdG8gcm90YXRlXG4gICAgICogQHBhcmFtIHtSZWFkb25seVZlYzN9IGIgVGhlIG9yaWdpbiBvZiB0aGUgcm90YXRpb25cbiAgICAgKiBAcGFyYW0ge051bWJlcn0gcmFkIFRoZSBhbmdsZSBvZiByb3RhdGlvbiBpbiByYWRpYW5zXG4gICAgICogQHBhcmFtIGRzdCAtIFRoZSB2ZWN0b3IgdG8gc2V0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIHRoZSByb3RhdGVkIHZlY3RvclxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0ZVkoYSwgYiwgcmFkLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIGNvbnN0IHAgPSBbXTtcbiAgICAgICAgY29uc3QgciA9IFtdO1xuICAgICAgICAvLyB0cmFuc2xhdGUgcG9pbnQgdG8gdGhlIG9yaWdpblxuICAgICAgICBwWzBdID0gYVswXSAtIGJbMF07XG4gICAgICAgIHBbMV0gPSBhWzFdIC0gYlsxXTtcbiAgICAgICAgcFsyXSA9IGFbMl0gLSBiWzJdO1xuICAgICAgICAvLyBwZXJmb3JtIHJvdGF0aW9uXG4gICAgICAgIHJbMF0gPSBwWzJdICogTWF0aC5zaW4ocmFkKSArIHBbMF0gKiBNYXRoLmNvcyhyYWQpO1xuICAgICAgICByWzFdID0gcFsxXTtcbiAgICAgICAgclsyXSA9IHBbMl0gKiBNYXRoLmNvcyhyYWQpIC0gcFswXSAqIE1hdGguc2luKHJhZCk7XG4gICAgICAgIC8vIHRyYW5zbGF0ZSB0byBjb3JyZWN0IHBvc2l0aW9uXG4gICAgICAgIG5ld0RzdFswXSA9IHJbMF0gKyBiWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSByWzFdICsgYlsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gclsyXSArIGJbMl07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJvdGF0ZSBhIDNEIHZlY3RvciBhcm91bmQgdGhlIHotYXhpc1xuICAgICAqXG4gICAgICogQHBhcmFtIHtSZWFkb25seVZlYzN9IGEgVGhlIHZlYzMgcG9pbnQgdG8gcm90YXRlXG4gICAgICogQHBhcmFtIHtSZWFkb25seVZlYzN9IGIgVGhlIG9yaWdpbiBvZiB0aGUgcm90YXRpb25cbiAgICAgKiBAcGFyYW0ge051bWJlcn0gcmFkIFRoZSBhbmdsZSBvZiByb3RhdGlvbiBpbiByYWRpYW5zXG4gICAgICogQHBhcmFtIGRzdCAtIFRoZSB2ZWN0b3IgdG8gc2V0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIHt2ZWMzfSBvdXRcbiAgICAgKi9cbiAgICBmdW5jdGlvbiByb3RhdGVaKGEsIGIsIHJhZCwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMykpO1xuICAgICAgICBjb25zdCBwID0gW107XG4gICAgICAgIGNvbnN0IHIgPSBbXTtcbiAgICAgICAgLy8gdHJhbnNsYXRlIHBvaW50IHRvIHRoZSBvcmlnaW5cbiAgICAgICAgcFswXSA9IGFbMF0gLSBiWzBdO1xuICAgICAgICBwWzFdID0gYVsxXSAtIGJbMV07XG4gICAgICAgIHBbMl0gPSBhWzJdIC0gYlsyXTtcbiAgICAgICAgLy8gcGVyZm9ybSByb3RhdGlvblxuICAgICAgICByWzBdID0gcFswXSAqIE1hdGguY29zKHJhZCkgLSBwWzFdICogTWF0aC5zaW4ocmFkKTtcbiAgICAgICAgclsxXSA9IHBbMF0gKiBNYXRoLnNpbihyYWQpICsgcFsxXSAqIE1hdGguY29zKHJhZCk7XG4gICAgICAgIHJbMl0gPSBwWzJdO1xuICAgICAgICAvLyB0cmFuc2xhdGUgdG8gY29ycmVjdCBwb3NpdGlvblxuICAgICAgICBuZXdEc3RbMF0gPSByWzBdICsgYlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gclsxXSArIGJbMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IHJbMl0gKyBiWzJdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBUcmVhdCBhIDNEIHZlY3RvciBhcyBhIGRpcmVjdGlvbiBhbmQgc2V0IGl0J3MgbGVuZ3RoXG4gICAgICpcbiAgICAgKiBAcGFyYW0gYSBUaGUgdmVjMyB0byBsZW5ndGhlblxuICAgICAqIEBwYXJhbSBsZW4gVGhlIGxlbmd0aCBvZiB0aGUgcmVzdWx0aW5nIHZlY3RvclxuICAgICAqIEByZXR1cm5zIFRoZSBsZW5ndGhlbmVkIHZlY3RvclxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHNldExlbmd0aChhLCBsZW4sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDMpKTtcbiAgICAgICAgbm9ybWFsaXplKGEsIG5ld0RzdCk7XG4gICAgICAgIHJldHVybiBtdWxTY2FsYXIobmV3RHN0LCBsZW4sIG5ld0RzdCk7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEVuc3VyZSBhIHZlY3RvciBpcyBub3QgbG9uZ2VyIHRoYW4gYSBtYXggbGVuZ3RoXG4gICAgICpcbiAgICAgKiBAcGFyYW0gYSBUaGUgdmVjMyB0byBsaW1pdFxuICAgICAqIEBwYXJhbSBtYXhMZW4gVGhlIGxvbmdlc3QgbGVuZ3RoIG9mIHRoZSByZXN1bHRpbmcgdmVjdG9yXG4gICAgICogQHJldHVybnMgVGhlIHZlY3Rvciwgc2hvcnRlbmVkIHRvIG1heExlbiBpZiBpdCdzIHRvbyBsb25nXG4gICAgICovXG4gICAgZnVuY3Rpb24gdHJ1bmNhdGUoYSwgbWF4TGVuLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIGlmIChsZW5ndGgoYSkgPiBtYXhMZW4pIHtcbiAgICAgICAgICAgIHJldHVybiBzZXRMZW5ndGgoYSwgbWF4TGVuLCBuZXdEc3QpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBjb3B5KGEsIG5ld0RzdCk7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJldHVybiB0aGUgdmVjdG9yIGV4YWN0bHkgYmV0d2VlbiAyIGVuZHBvaW50IHZlY3RvcnNcbiAgICAgKlxuICAgICAqIEBwYXJhbSBhIEVuZHBvaW50IDFcbiAgICAgKiBAcGFyYW0gYiBFbmRwb2ludCAyXG4gICAgICogQHJldHVybnMgVGhlIHZlY3RvciBleGFjdGx5IHJlc2lkaW5nIGJldHdlZW4gZW5kcG9pbnRzIDEgYW5kIDJcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBtaWRwb2ludChhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigzKSk7XG4gICAgICAgIHJldHVybiBsZXJwKGEsIGIsIDAuNSwgbmV3RHN0KTtcbiAgICB9XG4gICAgcmV0dXJuIHtcbiAgICAgICAgY3JlYXRlLFxuICAgICAgICBmcm9tVmFsdWVzLFxuICAgICAgICBzZXQsXG4gICAgICAgIGNlaWwsXG4gICAgICAgIGZsb29yLFxuICAgICAgICByb3VuZCxcbiAgICAgICAgY2xhbXAsXG4gICAgICAgIGFkZCxcbiAgICAgICAgYWRkU2NhbGVkLFxuICAgICAgICBhbmdsZSxcbiAgICAgICAgc3VidHJhY3QsXG4gICAgICAgIHN1YixcbiAgICAgICAgZXF1YWxzQXBwcm94aW1hdGVseSxcbiAgICAgICAgZXF1YWxzLFxuICAgICAgICBsZXJwLFxuICAgICAgICBsZXJwVixcbiAgICAgICAgbWF4LFxuICAgICAgICBtaW4sXG4gICAgICAgIG11bFNjYWxhcixcbiAgICAgICAgc2NhbGUsXG4gICAgICAgIGRpdlNjYWxhcixcbiAgICAgICAgaW52ZXJzZSxcbiAgICAgICAgaW52ZXJ0LFxuICAgICAgICBjcm9zcyxcbiAgICAgICAgZG90LFxuICAgICAgICBsZW5ndGgsXG4gICAgICAgIGxlbixcbiAgICAgICAgbGVuZ3RoU3EsXG4gICAgICAgIGxlblNxLFxuICAgICAgICBkaXN0YW5jZSxcbiAgICAgICAgZGlzdCxcbiAgICAgICAgZGlzdGFuY2VTcSxcbiAgICAgICAgZGlzdFNxLFxuICAgICAgICBub3JtYWxpemUsXG4gICAgICAgIG5lZ2F0ZSxcbiAgICAgICAgY29weSxcbiAgICAgICAgY2xvbmUsXG4gICAgICAgIG11bHRpcGx5LFxuICAgICAgICBtdWwsXG4gICAgICAgIGRpdmlkZSxcbiAgICAgICAgZGl2LFxuICAgICAgICByYW5kb20sXG4gICAgICAgIHplcm8sXG4gICAgICAgIHRyYW5zZm9ybU1hdDQsXG4gICAgICAgIHRyYW5zZm9ybU1hdDRVcHBlcjN4MyxcbiAgICAgICAgdHJhbnNmb3JtTWF0MyxcbiAgICAgICAgdHJhbnNmb3JtUXVhdCxcbiAgICAgICAgZ2V0VHJhbnNsYXRpb24sXG4gICAgICAgIGdldEF4aXMsXG4gICAgICAgIGdldFNjYWxpbmcsXG4gICAgICAgIHJvdGF0ZVgsXG4gICAgICAgIHJvdGF0ZVksXG4gICAgICAgIHJvdGF0ZVosXG4gICAgICAgIHNldExlbmd0aCxcbiAgICAgICAgdHJ1bmNhdGUsXG4gICAgICAgIG1pZHBvaW50LFxuICAgIH07XG59XG5jb25zdCBjYWNoZSQ0ID0gbmV3IE1hcCgpO1xuZnVuY3Rpb24gZ2V0QVBJJDQoQ3Rvcikge1xuICAgIGxldCBhcGkgPSBjYWNoZSQ0LmdldChDdG9yKTtcbiAgICBpZiAoIWFwaSkge1xuICAgICAgICBhcGkgPSBnZXRBUElJbXBsJDQoQ3Rvcik7XG4gICAgICAgIGNhY2hlJDQuc2V0KEN0b3IsIGFwaSk7XG4gICAgfVxuICAgIHJldHVybiBhcGk7XG59XG5cbi8qXG4gKiBDb3B5cmlnaHQgMjAyMiBHcmVnZyBUYXZhcmVzXG4gKlxuICogUGVybWlzc2lvbiBpcyBoZXJlYnkgZ3JhbnRlZCwgZnJlZSBvZiBjaGFyZ2UsIHRvIGFueSBwZXJzb24gb2J0YWluaW5nIGFcbiAqIGNvcHkgb2YgdGhpcyBzb2Z0d2FyZSBhbmQgYXNzb2NpYXRlZCBkb2N1bWVudGF0aW9uIGZpbGVzICh0aGUgXCJTb2Z0d2FyZVwiKSxcbiAqIHRvIGRlYWwgaW4gdGhlIFNvZnR3YXJlIHdpdGhvdXQgcmVzdHJpY3Rpb24sIGluY2x1ZGluZyB3aXRob3V0IGxpbWl0YXRpb25cbiAqIHRoZSByaWdodHMgdG8gdXNlLCBjb3B5LCBtb2RpZnksIG1lcmdlLCBwdWJsaXNoLCBkaXN0cmlidXRlLCBzdWJsaWNlbnNlLFxuICogYW5kL29yIHNlbGwgY29waWVzIG9mIHRoZSBTb2Z0d2FyZSwgYW5kIHRvIHBlcm1pdCBwZXJzb25zIHRvIHdob20gdGhlXG4gKiBTb2Z0d2FyZSBpcyBmdXJuaXNoZWQgdG8gZG8gc28sIHN1YmplY3QgdG8gdGhlIGZvbGxvd2luZyBjb25kaXRpb25zOlxuICpcbiAqIFRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluXG4gKiBhbGwgY29waWVzIG9yIHN1YnN0YW50aWFsIHBvcnRpb25zIG9mIHRoZSBTb2Z0d2FyZS5cbiAqXG4gKiBUSEUgU09GVFdBUkUgSVMgUFJPVklERUQgXCJBUyBJU1wiLCBXSVRIT1VUIFdBUlJBTlRZIE9GIEFOWSBLSU5ELCBFWFBSRVNTIE9SXG4gKiBJTVBMSUVELCBJTkNMVURJTkcgQlVUIE5PVCBMSU1JVEVEIFRPIFRIRSBXQVJSQU5USUVTIE9GIE1FUkNIQU5UQUJJTElUWSxcbiAqIEZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFIEFORCBOT05JTkZSSU5HRU1FTlQuICBJTiBOTyBFVkVOVCBTSEFMTFxuICogVEhFIEFVVEhPUlMgT1IgQ09QWVJJR0hUIEhPTERFUlMgQkUgTElBQkxFIEZPUiBBTlkgQ0xBSU0sIERBTUFHRVMgT1IgT1RIRVJcbiAqIExJQUJJTElUWSwgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HXG4gKiBGUk9NLCBPVVQgT0YgT1IgSU4gQ09OTkVDVElPTiBXSVRIIFRIRSBTT0ZUV0FSRSBPUiBUSEUgVVNFIE9SIE9USEVSXG4gKiBERUFMSU5HUyBJTiBUSEUgU09GVFdBUkUuXG4gKi9cbi8qKlxuICogR2VuZXJhdGVzIGEgdHlwZWQgQVBJIGZvciBNYXQzXG4gKiAqL1xuZnVuY3Rpb24gZ2V0QVBJSW1wbCQzKEN0b3IpIHtcbiAgICBjb25zdCB2ZWMyID0gZ2V0QVBJJDUoQ3Rvcik7XG4gICAgY29uc3QgdmVjMyA9IGdldEFQSSQ0KEN0b3IpO1xuICAgIC8qKlxuICAgICAqIENyZWF0ZSBhIE1hdDMgZnJvbSB2YWx1ZXNcbiAgICAgKlxuICAgICAqIE5vdGU6IFNpbmNlIHBhc3NpbmcgaW4gYSByYXcgSmF2YVNjcmlwdCBhcnJheVxuICAgICAqIGlzIHZhbGlkIGluIGFsbCBjaXJjdW1zdGFuY2VzLCBpZiB5b3Ugd2FudCB0b1xuICAgICAqIGZvcmNlIGEgSmF2YVNjcmlwdCBhcnJheSBpbnRvIGEgTWF0MydzIHNwZWNpZmllZCB0eXBlXG4gICAgICogaXQgd291bGQgYmUgZmFzdGVyIHRvIHVzZVxuICAgICAqXG4gICAgICogYGBgXG4gICAgICogY29uc3QgbSA9IG1hdDMuY2xvbmUoc29tZUpTQXJyYXkpO1xuICAgICAqIGBgYFxuICAgICAqXG4gICAgICogQHBhcmFtIHYwIC0gdmFsdWUgZm9yIGVsZW1lbnQgMFxuICAgICAqIEBwYXJhbSB2MSAtIHZhbHVlIGZvciBlbGVtZW50IDFcbiAgICAgKiBAcGFyYW0gdjIgLSB2YWx1ZSBmb3IgZWxlbWVudCAyXG4gICAgICogQHBhcmFtIHYzIC0gdmFsdWUgZm9yIGVsZW1lbnQgM1xuICAgICAqIEBwYXJhbSB2NCAtIHZhbHVlIGZvciBlbGVtZW50IDRcbiAgICAgKiBAcGFyYW0gdjUgLSB2YWx1ZSBmb3IgZWxlbWVudCA1XG4gICAgICogQHBhcmFtIHY2IC0gdmFsdWUgZm9yIGVsZW1lbnQgNlxuICAgICAqIEBwYXJhbSB2NyAtIHZhbHVlIGZvciBlbGVtZW50IDdcbiAgICAgKiBAcGFyYW0gdjggLSB2YWx1ZSBmb3IgZWxlbWVudCA4XG4gICAgICogQHJldHVybnMgbWF0cml4IGNyZWF0ZWQgZnJvbSB2YWx1ZXMuXG4gICAgICovXG4gICAgZnVuY3Rpb24gY3JlYXRlKHYwLCB2MSwgdjIsIHYzLCB2NCwgdjUsIHY2LCB2NywgdjgpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gbmV3IEN0b3IoMTIpO1xuICAgICAgICAvLyB0byBtYWtlIHRoZSBhcnJheSBob21vZ2Vub3VzXG4gICAgICAgIG5ld0RzdFszXSA9IDA7XG4gICAgICAgIG5ld0RzdFs3XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMV0gPSAwO1xuICAgICAgICBpZiAodjAgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0gdjA7XG4gICAgICAgICAgICBpZiAodjEgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgIG5ld0RzdFsxXSA9IHYxO1xuICAgICAgICAgICAgICAgIGlmICh2MiAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgICAgIG5ld0RzdFsyXSA9IHYyO1xuICAgICAgICAgICAgICAgICAgICBpZiAodjMgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgbmV3RHN0WzRdID0gdjM7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAodjQgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFs1XSA9IHY0O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh2NSAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFs2XSA9IHY1O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodjYgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbmV3RHN0WzhdID0gdjY7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodjcgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFs5XSA9IHY3O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh2OCAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFsxMF0gPSB2ODtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogU2V0cyB0aGUgdmFsdWVzIG9mIGEgTWF0M1xuICAgICAqIEFsc28gc2VlIHtAbGluayBtYXQzLmNyZWF0ZX0gYW5kIHtAbGluayBtYXQzLmNvcHl9XG4gICAgICpcbiAgICAgKiBAcGFyYW0gdjAgLSB2YWx1ZSBmb3IgZWxlbWVudCAwXG4gICAgICogQHBhcmFtIHYxIC0gdmFsdWUgZm9yIGVsZW1lbnQgMVxuICAgICAqIEBwYXJhbSB2MiAtIHZhbHVlIGZvciBlbGVtZW50IDJcbiAgICAgKiBAcGFyYW0gdjMgLSB2YWx1ZSBmb3IgZWxlbWVudCAzXG4gICAgICogQHBhcmFtIHY0IC0gdmFsdWUgZm9yIGVsZW1lbnQgNFxuICAgICAqIEBwYXJhbSB2NSAtIHZhbHVlIGZvciBlbGVtZW50IDVcbiAgICAgKiBAcGFyYW0gdjYgLSB2YWx1ZSBmb3IgZWxlbWVudCA2XG4gICAgICogQHBhcmFtIHY3IC0gdmFsdWUgZm9yIGVsZW1lbnQgN1xuICAgICAqIEBwYXJhbSB2OCAtIHZhbHVlIGZvciBlbGVtZW50IDhcbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIE1hdDMgc2V0IGZyb20gdmFsdWVzLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHNldCh2MCwgdjEsIHYyLCB2MywgdjQsIHY1LCB2NiwgdjcsIHY4LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2MDtcbiAgICAgICAgbmV3RHN0WzFdID0gdjE7XG4gICAgICAgIG5ld0RzdFsyXSA9IHYyO1xuICAgICAgICBuZXdEc3RbM10gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSB2MztcbiAgICAgICAgbmV3RHN0WzVdID0gdjQ7XG4gICAgICAgIG5ld0RzdFs2XSA9IHY1O1xuICAgICAgICBuZXdEc3RbN10gPSAwO1xuICAgICAgICBuZXdEc3RbOF0gPSB2NjtcbiAgICAgICAgbmV3RHN0WzldID0gdjc7XG4gICAgICAgIG5ld0RzdFsxMF0gPSB2ODtcbiAgICAgICAgbmV3RHN0WzExXSA9IDA7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENyZWF0ZXMgYSBNYXQzIGZyb20gdGhlIHVwcGVyIGxlZnQgM3gzIHBhcnQgb2YgYSBNYXQ0XG4gICAgICogQHBhcmFtIG00IC0gc291cmNlIG1hdHJpeFxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgTWF0MyBtYWRlIGZyb20gbTRcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBmcm9tTWF0NChtNCwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gbTRbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IG00WzFdO1xuICAgICAgICBuZXdEc3RbMl0gPSBtNFsyXTtcbiAgICAgICAgbmV3RHN0WzNdID0gMDtcbiAgICAgICAgbmV3RHN0WzRdID0gbTRbNF07XG4gICAgICAgIG5ld0RzdFs1XSA9IG00WzVdO1xuICAgICAgICBuZXdEc3RbNl0gPSBtNFs2XTtcbiAgICAgICAgbmV3RHN0WzddID0gMDtcbiAgICAgICAgbmV3RHN0WzhdID0gbTRbOF07XG4gICAgICAgIG5ld0RzdFs5XSA9IG00WzldO1xuICAgICAgICBuZXdEc3RbMTBdID0gbTRbMTBdO1xuICAgICAgICBuZXdEc3RbMTFdID0gMDtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIE1hdDMgcm90YXRpb24gbWF0cml4IGZyb20gYSBxdWF0ZXJuaW9uXG4gICAgICogQHBhcmFtIHEgLSBxdWF0ZXJuaW9uIHRvIGNyZWF0ZSBtYXRyaXggZnJvbVxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgTWF0MyBtYWRlIGZyb20gcVxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGZyb21RdWF0KHEsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDEyKSk7XG4gICAgICAgIGNvbnN0IHggPSBxWzBdO1xuICAgICAgICBjb25zdCB5ID0gcVsxXTtcbiAgICAgICAgY29uc3QgeiA9IHFbMl07XG4gICAgICAgIGNvbnN0IHcgPSBxWzNdO1xuICAgICAgICBjb25zdCB4MiA9IHggKyB4O1xuICAgICAgICBjb25zdCB5MiA9IHkgKyB5O1xuICAgICAgICBjb25zdCB6MiA9IHogKyB6O1xuICAgICAgICBjb25zdCB4eCA9IHggKiB4MjtcbiAgICAgICAgY29uc3QgeXggPSB5ICogeDI7XG4gICAgICAgIGNvbnN0IHl5ID0geSAqIHkyO1xuICAgICAgICBjb25zdCB6eCA9IHogKiB4MjtcbiAgICAgICAgY29uc3QgenkgPSB6ICogeTI7XG4gICAgICAgIGNvbnN0IHp6ID0geiAqIHoyO1xuICAgICAgICBjb25zdCB3eCA9IHcgKiB4MjtcbiAgICAgICAgY29uc3Qgd3kgPSB3ICogeTI7XG4gICAgICAgIGNvbnN0IHd6ID0gdyAqIHoyO1xuICAgICAgICBuZXdEc3RbMF0gPSAxIC0geXkgLSB6ejtcbiAgICAgICAgbmV3RHN0WzFdID0geXggKyB3ejtcbiAgICAgICAgbmV3RHN0WzJdID0genggLSB3eTtcbiAgICAgICAgbmV3RHN0WzNdID0gMDtcbiAgICAgICAgbmV3RHN0WzRdID0geXggLSB3ejtcbiAgICAgICAgbmV3RHN0WzVdID0gMSAtIHh4IC0geno7XG4gICAgICAgIG5ld0RzdFs2XSA9IHp5ICsgd3g7XG4gICAgICAgIG5ld0RzdFs3XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IHp4ICsgd3k7XG4gICAgICAgIG5ld0RzdFs5XSA9IHp5IC0gd3g7XG4gICAgICAgIG5ld0RzdFsxMF0gPSAxIC0geHggLSB5eTtcbiAgICAgICAgbmV3RHN0WzExXSA9IDA7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIE5lZ2F0ZXMgYSBtYXRyaXguXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgLW0uXG4gICAgICovXG4gICAgZnVuY3Rpb24gbmVnYXRlKG0sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDEyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IC1tWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSAtbVsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gLW1bMl07XG4gICAgICAgIG5ld0RzdFs0XSA9IC1tWzRdO1xuICAgICAgICBuZXdEc3RbNV0gPSAtbVs1XTtcbiAgICAgICAgbmV3RHN0WzZdID0gLW1bNl07XG4gICAgICAgIG5ld0RzdFs4XSA9IC1tWzhdO1xuICAgICAgICBuZXdEc3RbOV0gPSAtbVs5XTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IC1tWzEwXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29waWVzIGEgbWF0cml4LiAoc2FtZSBhcyB7QGxpbmsgbWF0My5jbG9uZX0pXG4gICAgICogQWxzbyBzZWUge0BsaW5rIG1hdDMuY3JlYXRlfSBhbmQge0BsaW5rIG1hdDMuc2V0fVxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gVGhlIG1hdHJpeC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIGNvcHkgb2YgbS5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBjb3B5KG0sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDEyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IG1bMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IG1bMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IG1bMl07XG4gICAgICAgIG5ld0RzdFs0XSA9IG1bNF07XG4gICAgICAgIG5ld0RzdFs1XSA9IG1bNV07XG4gICAgICAgIG5ld0RzdFs2XSA9IG1bNl07XG4gICAgICAgIG5ld0RzdFs4XSA9IG1bOF07XG4gICAgICAgIG5ld0RzdFs5XSA9IG1bOV07XG4gICAgICAgIG5ld0RzdFsxMF0gPSBtWzEwXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29waWVzIGEgbWF0cml4IChzYW1lIGFzIHtAbGluayBtYXQzLmNvcHl9KVxuICAgICAqIEFsc28gc2VlIHtAbGluayBtYXQzLmNyZWF0ZX0gYW5kIHtAbGluayBtYXQzLnNldH1cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGRzdCAtIFRoZSBtYXRyaXguIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSBjb3B5IG9mIG0uXG4gICAgICovXG4gICAgY29uc3QgY2xvbmUgPSBjb3B5O1xuICAgIC8qKlxuICAgICAqIENoZWNrIGlmIDIgbWF0cmljZXMgYXJlIGFwcHJveGltYXRlbHkgZXF1YWxcbiAgICAgKiBAcGFyYW0gYSBPcGVyYW5kIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gYiBPcGVyYW5kIG1hdHJpeC5cbiAgICAgKiBAcmV0dXJucyB0cnVlIGlmIG1hdHJpY2VzIGFyZSBhcHByb3hpbWF0ZWx5IGVxdWFsXG4gICAgICovXG4gICAgZnVuY3Rpb24gZXF1YWxzQXBwcm94aW1hdGVseShhLCBiKSB7XG4gICAgICAgIHJldHVybiBNYXRoLmFicyhhWzBdIC0gYlswXSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzFdIC0gYlsxXSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzJdIC0gYlsyXSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzRdIC0gYls0XSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzVdIC0gYls1XSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzZdIC0gYls2XSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzhdIC0gYls4XSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzldIC0gYls5XSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzEwXSAtIGJbMTBdKSA8IEVQU0lMT047XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENoZWNrIGlmIDIgbWF0cmljZXMgYXJlIGV4YWN0bHkgZXF1YWxcbiAgICAgKiBAcGFyYW0gYSBPcGVyYW5kIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gYiBPcGVyYW5kIG1hdHJpeC5cbiAgICAgKiBAcmV0dXJucyB0cnVlIGlmIG1hdHJpY2VzIGFyZSBleGFjdGx5IGVxdWFsXG4gICAgICovXG4gICAgZnVuY3Rpb24gZXF1YWxzKGEsIGIpIHtcbiAgICAgICAgcmV0dXJuIGFbMF0gPT09IGJbMF0gJiZcbiAgICAgICAgICAgIGFbMV0gPT09IGJbMV0gJiZcbiAgICAgICAgICAgIGFbMl0gPT09IGJbMl0gJiZcbiAgICAgICAgICAgIGFbNF0gPT09IGJbNF0gJiZcbiAgICAgICAgICAgIGFbNV0gPT09IGJbNV0gJiZcbiAgICAgICAgICAgIGFbNl0gPT09IGJbNl0gJiZcbiAgICAgICAgICAgIGFbOF0gPT09IGJbOF0gJiZcbiAgICAgICAgICAgIGFbOV0gPT09IGJbOV0gJiZcbiAgICAgICAgICAgIGFbMTBdID09PSBiWzEwXTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIDMtYnktMyBpZGVudGl0eSBtYXRyaXguXG4gICAgICpcbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgMy1ieS0zIGlkZW50aXR5IG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBpZGVudGl0eShkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSAxO1xuICAgICAgICBuZXdEc3RbMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSAwO1xuICAgICAgICBuZXdEc3RbNV0gPSAxO1xuICAgICAgICBuZXdEc3RbNl0gPSAwO1xuICAgICAgICBuZXdEc3RbOF0gPSAwO1xuICAgICAgICBuZXdEc3RbOV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTBdID0gMTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogVGFrZXMgdGhlIHRyYW5zcG9zZSBvZiBhIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgdHJhbnNwb3NlIG9mIG0uXG4gICAgICovXG4gICAgZnVuY3Rpb24gdHJhbnNwb3NlKG0sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDEyKSk7XG4gICAgICAgIGlmIChuZXdEc3QgPT09IG0pIHtcbiAgICAgICAgICAgIGxldCB0O1xuICAgICAgICAgICAgLy8gMCAxIDJcbiAgICAgICAgICAgIC8vIDQgNSA2XG4gICAgICAgICAgICAvLyA4IDkgMTBcbiAgICAgICAgICAgIHQgPSBtWzFdO1xuICAgICAgICAgICAgbVsxXSA9IG1bNF07XG4gICAgICAgICAgICBtWzRdID0gdDtcbiAgICAgICAgICAgIHQgPSBtWzJdO1xuICAgICAgICAgICAgbVsyXSA9IG1bOF07XG4gICAgICAgICAgICBtWzhdID0gdDtcbiAgICAgICAgICAgIHQgPSBtWzZdO1xuICAgICAgICAgICAgbVs2XSA9IG1bOV07XG4gICAgICAgICAgICBtWzldID0gdDtcbiAgICAgICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgbTAwID0gbVswICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMDEgPSBtWzAgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0wMiA9IG1bMCAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTEwID0gbVsxICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMTEgPSBtWzEgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0xMiA9IG1bMSAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTIwID0gbVsyICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMjEgPSBtWzIgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0yMiA9IG1bMiAqIDQgKyAyXTtcbiAgICAgICAgbmV3RHN0WzBdID0gbTAwO1xuICAgICAgICBuZXdEc3RbMV0gPSBtMTA7XG4gICAgICAgIG5ld0RzdFsyXSA9IG0yMDtcbiAgICAgICAgbmV3RHN0WzRdID0gbTAxO1xuICAgICAgICBuZXdEc3RbNV0gPSBtMTE7XG4gICAgICAgIG5ld0RzdFs2XSA9IG0yMTtcbiAgICAgICAgbmV3RHN0WzhdID0gbTAyO1xuICAgICAgICBuZXdEc3RbOV0gPSBtMTI7XG4gICAgICAgIG5ld0RzdFsxMF0gPSBtMjI7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBpbnZlcnNlIG9mIGEgMy1ieS0zIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgaW52ZXJzZSBvZiBtLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGludmVyc2UobSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgY29uc3QgbTAwID0gbVswICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMDEgPSBtWzAgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0wMiA9IG1bMCAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTEwID0gbVsxICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMTEgPSBtWzEgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0xMiA9IG1bMSAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTIwID0gbVsyICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMjEgPSBtWzIgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0yMiA9IG1bMiAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgYjAxID0gbTIyICogbTExIC0gbTEyICogbTIxO1xuICAgICAgICBjb25zdCBiMTEgPSAtbTIyICogbTEwICsgbTEyICogbTIwO1xuICAgICAgICBjb25zdCBiMjEgPSBtMjEgKiBtMTAgLSBtMTEgKiBtMjA7XG4gICAgICAgIGNvbnN0IGludkRldCA9IDEgLyAobTAwICogYjAxICsgbTAxICogYjExICsgbTAyICogYjIxKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYjAxICogaW52RGV0O1xuICAgICAgICBuZXdEc3RbMV0gPSAoLW0yMiAqIG0wMSArIG0wMiAqIG0yMSkgKiBpbnZEZXQ7XG4gICAgICAgIG5ld0RzdFsyXSA9IChtMTIgKiBtMDEgLSBtMDIgKiBtMTEpICogaW52RGV0O1xuICAgICAgICBuZXdEc3RbNF0gPSBiMTEgKiBpbnZEZXQ7XG4gICAgICAgIG5ld0RzdFs1XSA9IChtMjIgKiBtMDAgLSBtMDIgKiBtMjApICogaW52RGV0O1xuICAgICAgICBuZXdEc3RbNl0gPSAoLW0xMiAqIG0wMCArIG0wMiAqIG0xMCkgKiBpbnZEZXQ7XG4gICAgICAgIG5ld0RzdFs4XSA9IGIyMSAqIGludkRldDtcbiAgICAgICAgbmV3RHN0WzldID0gKC1tMjEgKiBtMDAgKyBtMDEgKiBtMjApICogaW52RGV0O1xuICAgICAgICBuZXdEc3RbMTBdID0gKG0xMSAqIG0wMCAtIG0wMSAqIG0xMCkgKiBpbnZEZXQ7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGUgdGhlIGRldGVybWluYW50IG9mIGEgbWF0cml4XG4gICAgICogQHBhcmFtIG0gLSB0aGUgbWF0cml4XG4gICAgICogQHJldHVybnMgdGhlIGRldGVybWluYW50XG4gICAgICovXG4gICAgZnVuY3Rpb24gZGV0ZXJtaW5hbnQobSkge1xuICAgICAgICBjb25zdCBtMDAgPSBtWzAgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0wMSA9IG1bMCAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTAyID0gbVswICogNCArIDJdO1xuICAgICAgICBjb25zdCBtMTAgPSBtWzEgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0xMSA9IG1bMSAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTEyID0gbVsxICogNCArIDJdO1xuICAgICAgICBjb25zdCBtMjAgPSBtWzIgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0yMSA9IG1bMiAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTIyID0gbVsyICogNCArIDJdO1xuICAgICAgICByZXR1cm4gbTAwICogKG0xMSAqIG0yMiAtIG0yMSAqIG0xMikgLVxuICAgICAgICAgICAgbTEwICogKG0wMSAqIG0yMiAtIG0yMSAqIG0wMikgK1xuICAgICAgICAgICAgbTIwICogKG0wMSAqIG0xMiAtIG0xMSAqIG0wMik7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBpbnZlcnNlIG9mIGEgMy1ieS0zIG1hdHJpeC4gKHNhbWUgYXMgaW52ZXJzZSlcbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgaW52ZXJzZSBvZiBtLlxuICAgICAqL1xuICAgIGNvbnN0IGludmVydCA9IGludmVyc2U7XG4gICAgLyoqXG4gICAgICogTXVsdGlwbGllcyB0d28gMy1ieS0zIG1hdHJpY2VzIHdpdGggYSBvbiB0aGUgbGVmdCBhbmQgYiBvbiB0aGUgcmlnaHRcbiAgICAgKiBAcGFyYW0gYSAtIFRoZSBtYXRyaXggb24gdGhlIGxlZnQuXG4gICAgICogQHBhcmFtIGIgLSBUaGUgbWF0cml4IG9uIHRoZSByaWdodC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBtYXRyaXggcHJvZHVjdCBvZiBhIGFuZCBiLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG11bHRpcGx5KGEsIGIsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDEyKSk7XG4gICAgICAgIGNvbnN0IGEwMCA9IGFbMF07XG4gICAgICAgIGNvbnN0IGEwMSA9IGFbMV07XG4gICAgICAgIGNvbnN0IGEwMiA9IGFbMl07XG4gICAgICAgIGNvbnN0IGExMCA9IGFbNCArIDBdO1xuICAgICAgICBjb25zdCBhMTEgPSBhWzQgKyAxXTtcbiAgICAgICAgY29uc3QgYTEyID0gYVs0ICsgMl07XG4gICAgICAgIGNvbnN0IGEyMCA9IGFbOCArIDBdO1xuICAgICAgICBjb25zdCBhMjEgPSBhWzggKyAxXTtcbiAgICAgICAgY29uc3QgYTIyID0gYVs4ICsgMl07XG4gICAgICAgIGNvbnN0IGIwMCA9IGJbMF07XG4gICAgICAgIGNvbnN0IGIwMSA9IGJbMV07XG4gICAgICAgIGNvbnN0IGIwMiA9IGJbMl07XG4gICAgICAgIGNvbnN0IGIxMCA9IGJbNCArIDBdO1xuICAgICAgICBjb25zdCBiMTEgPSBiWzQgKyAxXTtcbiAgICAgICAgY29uc3QgYjEyID0gYls0ICsgMl07XG4gICAgICAgIGNvbnN0IGIyMCA9IGJbOCArIDBdO1xuICAgICAgICBjb25zdCBiMjEgPSBiWzggKyAxXTtcbiAgICAgICAgY29uc3QgYjIyID0gYls4ICsgMl07XG4gICAgICAgIG5ld0RzdFswXSA9IGEwMCAqIGIwMCArIGExMCAqIGIwMSArIGEyMCAqIGIwMjtcbiAgICAgICAgbmV3RHN0WzFdID0gYTAxICogYjAwICsgYTExICogYjAxICsgYTIxICogYjAyO1xuICAgICAgICBuZXdEc3RbMl0gPSBhMDIgKiBiMDAgKyBhMTIgKiBiMDEgKyBhMjIgKiBiMDI7XG4gICAgICAgIG5ld0RzdFs0XSA9IGEwMCAqIGIxMCArIGExMCAqIGIxMSArIGEyMCAqIGIxMjtcbiAgICAgICAgbmV3RHN0WzVdID0gYTAxICogYjEwICsgYTExICogYjExICsgYTIxICogYjEyO1xuICAgICAgICBuZXdEc3RbNl0gPSBhMDIgKiBiMTAgKyBhMTIgKiBiMTEgKyBhMjIgKiBiMTI7XG4gICAgICAgIG5ld0RzdFs4XSA9IGEwMCAqIGIyMCArIGExMCAqIGIyMSArIGEyMCAqIGIyMjtcbiAgICAgICAgbmV3RHN0WzldID0gYTAxICogYjIwICsgYTExICogYjIxICsgYTIxICogYjIyO1xuICAgICAgICBuZXdEc3RbMTBdID0gYTAyICogYjIwICsgYTEyICogYjIxICsgYTIyICogYjIyO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBNdWx0aXBsaWVzIHR3byAzLWJ5LTMgbWF0cmljZXMgd2l0aCBhIG9uIHRoZSBsZWZ0IGFuZCBiIG9uIHRoZSByaWdodCAoc2FtZSBhcyBtdWx0aXBseSlcbiAgICAgKiBAcGFyYW0gYSAtIFRoZSBtYXRyaXggb24gdGhlIGxlZnQuXG4gICAgICogQHBhcmFtIGIgLSBUaGUgbWF0cml4IG9uIHRoZSByaWdodC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBtYXRyaXggcHJvZHVjdCBvZiBhIGFuZCBiLlxuICAgICAqL1xuICAgIGNvbnN0IG11bCA9IG11bHRpcGx5O1xuICAgIC8qKlxuICAgICAqIFNldHMgdGhlIHRyYW5zbGF0aW9uIGNvbXBvbmVudCBvZiBhIDMtYnktMyBtYXRyaXggdG8gdGhlIGdpdmVuXG4gICAgICogdmVjdG9yLlxuICAgICAqIEBwYXJhbSBhIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gdiAtIFRoZSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgbWF0cml4IHdpdGggdHJhbnNsYXRpb24gc2V0LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHNldFRyYW5zbGF0aW9uKGEsIHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IGlkZW50aXR5KCkpO1xuICAgICAgICBpZiAoYSAhPT0gbmV3RHN0KSB7XG4gICAgICAgICAgICBuZXdEc3RbMF0gPSBhWzBdO1xuICAgICAgICAgICAgbmV3RHN0WzFdID0gYVsxXTtcbiAgICAgICAgICAgIG5ld0RzdFsyXSA9IGFbMl07XG4gICAgICAgICAgICBuZXdEc3RbNF0gPSBhWzRdO1xuICAgICAgICAgICAgbmV3RHN0WzVdID0gYVs1XTtcbiAgICAgICAgICAgIG5ld0RzdFs2XSA9IGFbNl07XG4gICAgICAgIH1cbiAgICAgICAgbmV3RHN0WzhdID0gdlswXTtcbiAgICAgICAgbmV3RHN0WzldID0gdlsxXTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IDE7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJldHVybnMgdGhlIHRyYW5zbGF0aW9uIGNvbXBvbmVudCBvZiBhIDMtYnktMyBtYXRyaXggYXMgYSB2ZWN0b3Igd2l0aCAzXG4gICAgICogZW50cmllcy5cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgdHJhbnNsYXRpb24gY29tcG9uZW50IG9mIG0uXG4gICAgICovXG4gICAgZnVuY3Rpb24gZ2V0VHJhbnNsYXRpb24obSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gdmVjMi5jcmVhdGUoKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IG1bOF07XG4gICAgICAgIG5ld0RzdFsxXSA9IG1bOV07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJldHVybnMgYW4gYXhpcyBvZiBhIDN4MyBtYXRyaXggYXMgYSB2ZWN0b3Igd2l0aCAyIGVudHJpZXNcbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGF4aXMgLSBUaGUgYXhpcyAwID0geCwgMSA9IHksXG4gICAgICogQHJldHVybnMgVGhlIGF4aXMgY29tcG9uZW50IG9mIG0uXG4gICAgICovXG4gICAgZnVuY3Rpb24gZ2V0QXhpcyhtLCBheGlzLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyB2ZWMyLmNyZWF0ZSgpKTtcbiAgICAgICAgY29uc3Qgb2ZmID0gYXhpcyAqIDQ7XG4gICAgICAgIG5ld0RzdFswXSA9IG1bb2ZmICsgMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IG1bb2ZmICsgMV07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFNldHMgYW4gYXhpcyBvZiBhIDN4MyBtYXRyaXggYXMgYSB2ZWN0b3Igd2l0aCAyIGVudHJpZXNcbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIHYgLSB0aGUgYXhpcyB2ZWN0b3JcbiAgICAgKiBAcGFyYW0gYXhpcyAtIFRoZSBheGlzICAwID0geCwgMSA9IHk7XG4gICAgICogQHBhcmFtIGRzdCAtIFRoZSBtYXRyaXggdG8gc2V0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBtYXRyaXggd2l0aCBheGlzIHNldC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzZXRBeGlzKG0sIHYsIGF4aXMsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID09PSBtID8gbSA6IGNvcHkobSwgZHN0KSk7XG4gICAgICAgIGNvbnN0IG9mZiA9IGF4aXMgKiA0O1xuICAgICAgICBuZXdEc3Rbb2ZmICsgMF0gPSB2WzBdO1xuICAgICAgICBuZXdEc3Rbb2ZmICsgMV0gPSB2WzFdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSZXR1cm5zIHRoZSBcIjJkXCIgc2NhbGluZyBjb21wb25lbnQgb2YgdGhlIG1hdHJpeFxuICAgICAqIEBwYXJhbSBtIC0gVGhlIE1hdHJpeFxuICAgICAqIEBwYXJhbSBkc3QgLSBUaGUgdmVjdG9yIHRvIHNldC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBnZXRTY2FsaW5nKG0sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IHZlYzIuY3JlYXRlKCkpO1xuICAgICAgICBjb25zdCB4eCA9IG1bMF07XG4gICAgICAgIGNvbnN0IHh5ID0gbVsxXTtcbiAgICAgICAgY29uc3QgeXggPSBtWzRdO1xuICAgICAgICBjb25zdCB5eSA9IG1bNV07XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGguc3FydCh4eCAqIHh4ICsgeHkgKiB4eSk7XG4gICAgICAgIG5ld0RzdFsxXSA9IE1hdGguc3FydCh5eCAqIHl4ICsgeXkgKiB5eSk7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJldHVybnMgdGhlIFwiM2RcIiBzY2FsaW5nIGNvbXBvbmVudCBvZiB0aGUgbWF0cml4XG4gICAgICogQHBhcmFtIG0gLSBUaGUgTWF0cml4XG4gICAgICogQHBhcmFtIGRzdCAtIFRoZSB2ZWN0b3IgdG8gc2V0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGdldDNEU2NhbGluZyhtLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyB2ZWMzLmNyZWF0ZSgpKTtcbiAgICAgICAgY29uc3QgeHggPSBtWzBdO1xuICAgICAgICBjb25zdCB4eSA9IG1bMV07XG4gICAgICAgIGNvbnN0IHh6ID0gbVsyXTtcbiAgICAgICAgY29uc3QgeXggPSBtWzRdO1xuICAgICAgICBjb25zdCB5eSA9IG1bNV07XG4gICAgICAgIGNvbnN0IHl6ID0gbVs2XTtcbiAgICAgICAgY29uc3QgenggPSBtWzhdO1xuICAgICAgICBjb25zdCB6eSA9IG1bOV07XG4gICAgICAgIGNvbnN0IHp6ID0gbVsxMF07XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGguc3FydCh4eCAqIHh4ICsgeHkgKiB4eSArIHh6ICogeHopO1xuICAgICAgICBuZXdEc3RbMV0gPSBNYXRoLnNxcnQoeXggKiB5eCArIHl5ICogeXkgKyB5eiAqIHl6KTtcbiAgICAgICAgbmV3RHN0WzJdID0gTWF0aC5zcXJ0KHp4ICogenggKyB6eSAqIHp5ICsgenogKiB6eik7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENyZWF0ZXMgYSAzLWJ5LTMgbWF0cml4IHdoaWNoIHRyYW5zbGF0ZXMgYnkgdGhlIGdpdmVuIHZlY3RvciB2LlxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3RvciBieSB3aGljaCB0byB0cmFuc2xhdGUuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgdHJhbnNsYXRpb24gbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHRyYW5zbGF0aW9uKHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDEyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IDE7XG4gICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgIG5ld0RzdFsyXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IDA7XG4gICAgICAgIG5ld0RzdFs1XSA9IDE7XG4gICAgICAgIG5ld0RzdFs2XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IHZbMF07XG4gICAgICAgIG5ld0RzdFs5XSA9IHZbMV07XG4gICAgICAgIG5ld0RzdFsxMF0gPSAxO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBUcmFuc2xhdGVzIHRoZSBnaXZlbiAzLWJ5LTMgbWF0cml4IGJ5IHRoZSBnaXZlbiB2ZWN0b3Igdi5cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yIGJ5IHdoaWNoIHRvIHRyYW5zbGF0ZS5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSB0cmFuc2xhdGVkIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiB0cmFuc2xhdGUobSwgdiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgY29uc3QgdjAgPSB2WzBdO1xuICAgICAgICBjb25zdCB2MSA9IHZbMV07XG4gICAgICAgIGNvbnN0IG0wMCA9IG1bMF07XG4gICAgICAgIGNvbnN0IG0wMSA9IG1bMV07XG4gICAgICAgIGNvbnN0IG0wMiA9IG1bMl07XG4gICAgICAgIGNvbnN0IG0xMCA9IG1bMSAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTExID0gbVsxICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMTIgPSBtWzEgKiA0ICsgMl07XG4gICAgICAgIGNvbnN0IG0yMCA9IG1bMiAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTIxID0gbVsyICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMjIgPSBtWzIgKiA0ICsgMl07XG4gICAgICAgIGlmIChtICE9PSBuZXdEc3QpIHtcbiAgICAgICAgICAgIG5ld0RzdFswXSA9IG0wMDtcbiAgICAgICAgICAgIG5ld0RzdFsxXSA9IG0wMTtcbiAgICAgICAgICAgIG5ld0RzdFsyXSA9IG0wMjtcbiAgICAgICAgICAgIG5ld0RzdFs0XSA9IG0xMDtcbiAgICAgICAgICAgIG5ld0RzdFs1XSA9IG0xMTtcbiAgICAgICAgICAgIG5ld0RzdFs2XSA9IG0xMjtcbiAgICAgICAgfVxuICAgICAgICBuZXdEc3RbOF0gPSBtMDAgKiB2MCArIG0xMCAqIHYxICsgbTIwO1xuICAgICAgICBuZXdEc3RbOV0gPSBtMDEgKiB2MCArIG0xMSAqIHYxICsgbTIxO1xuICAgICAgICBuZXdEc3RbMTBdID0gbTAyICogdjAgKyBtMTIgKiB2MSArIG0yMjtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIDMtYnktMyBtYXRyaXggd2hpY2ggcm90YXRlcyAgYnkgdGhlIGdpdmVuIGFuZ2xlLlxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIFRoZSBhbmdsZSBieSB3aGljaCB0byByb3RhdGUgKGluIHJhZGlhbnMpLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHJvdGF0aW9uIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiByb3RhdGlvbihhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgY29uc3QgYyA9IE1hdGguY29zKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgY29uc3QgcyA9IE1hdGguc2luKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYztcbiAgICAgICAgbmV3RHN0WzFdID0gcztcbiAgICAgICAgbmV3RHN0WzJdID0gMDtcbiAgICAgICAgbmV3RHN0WzRdID0gLXM7XG4gICAgICAgIG5ld0RzdFs1XSA9IGM7XG4gICAgICAgIG5ld0RzdFs2XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IDA7XG4gICAgICAgIG5ld0RzdFs5XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMF0gPSAxO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSb3RhdGVzIHRoZSBnaXZlbiAzLWJ5LTMgbWF0cml4ICBieSB0aGUgZ2l2ZW4gYW5nbGUuXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIFRoZSBhbmdsZSBieSB3aGljaCB0byByb3RhdGUgKGluIHJhZGlhbnMpLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHJvdGF0ZWQgbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0ZShtLCBhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgY29uc3QgbTAwID0gbVswICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMDEgPSBtWzAgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0wMiA9IG1bMCAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTEwID0gbVsxICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMTEgPSBtWzEgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0xMiA9IG1bMSAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgYyA9IE1hdGguY29zKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgY29uc3QgcyA9IE1hdGguc2luKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYyAqIG0wMCArIHMgKiBtMTA7XG4gICAgICAgIG5ld0RzdFsxXSA9IGMgKiBtMDEgKyBzICogbTExO1xuICAgICAgICBuZXdEc3RbMl0gPSBjICogbTAyICsgcyAqIG0xMjtcbiAgICAgICAgbmV3RHN0WzRdID0gYyAqIG0xMCAtIHMgKiBtMDA7XG4gICAgICAgIG5ld0RzdFs1XSA9IGMgKiBtMTEgLSBzICogbTAxO1xuICAgICAgICBuZXdEc3RbNl0gPSBjICogbTEyIC0gcyAqIG0wMjtcbiAgICAgICAgaWYgKG0gIT09IG5ld0RzdCkge1xuICAgICAgICAgICAgbmV3RHN0WzhdID0gbVs4XTtcbiAgICAgICAgICAgIG5ld0RzdFs5XSA9IG1bOV07XG4gICAgICAgICAgICBuZXdEc3RbMTBdID0gbVsxMF07XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIDMtYnktMyBtYXRyaXggd2hpY2ggcm90YXRlcyBhcm91bmQgdGhlIHgtYXhpcyBieSB0aGUgZ2l2ZW4gYW5nbGUuXG4gICAgICogQHBhcmFtIGFuZ2xlSW5SYWRpYW5zIC0gVGhlIGFuZ2xlIGJ5IHdoaWNoIHRvIHJvdGF0ZSAoaW4gcmFkaWFucykuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgcm90YXRpb24gbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0aW9uWChhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgY29uc3QgYyA9IE1hdGguY29zKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgY29uc3QgcyA9IE1hdGguc2luKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgbmV3RHN0WzBdID0gMTtcbiAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgbmV3RHN0WzJdID0gMDtcbiAgICAgICAgbmV3RHN0WzRdID0gMDtcbiAgICAgICAgbmV3RHN0WzVdID0gYztcbiAgICAgICAgbmV3RHN0WzZdID0gcztcbiAgICAgICAgbmV3RHN0WzhdID0gMDtcbiAgICAgICAgbmV3RHN0WzldID0gLXM7XG4gICAgICAgIG5ld0RzdFsxMF0gPSBjO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSb3RhdGVzIHRoZSBnaXZlbiAzLWJ5LTMgbWF0cml4IGFyb3VuZCB0aGUgeC1heGlzIGJ5IHRoZSBnaXZlblxuICAgICAqIGFuZ2xlLlxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gYW5nbGVJblJhZGlhbnMgLSBUaGUgYW5nbGUgYnkgd2hpY2ggdG8gcm90YXRlIChpbiByYWRpYW5zKS5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSByb3RhdGVkIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiByb3RhdGVYKG0sIGFuZ2xlSW5SYWRpYW5zLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxMikpO1xuICAgICAgICBjb25zdCBtMTAgPSBtWzRdO1xuICAgICAgICBjb25zdCBtMTEgPSBtWzVdO1xuICAgICAgICBjb25zdCBtMTIgPSBtWzZdO1xuICAgICAgICBjb25zdCBtMjAgPSBtWzhdO1xuICAgICAgICBjb25zdCBtMjEgPSBtWzldO1xuICAgICAgICBjb25zdCBtMjIgPSBtWzEwXTtcbiAgICAgICAgY29uc3QgYyA9IE1hdGguY29zKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgY29uc3QgcyA9IE1hdGguc2luKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgbmV3RHN0WzRdID0gYyAqIG0xMCArIHMgKiBtMjA7XG4gICAgICAgIG5ld0RzdFs1XSA9IGMgKiBtMTEgKyBzICogbTIxO1xuICAgICAgICBuZXdEc3RbNl0gPSBjICogbTEyICsgcyAqIG0yMjtcbiAgICAgICAgbmV3RHN0WzhdID0gYyAqIG0yMCAtIHMgKiBtMTA7XG4gICAgICAgIG5ld0RzdFs5XSA9IGMgKiBtMjEgLSBzICogbTExO1xuICAgICAgICBuZXdEc3RbMTBdID0gYyAqIG0yMiAtIHMgKiBtMTI7XG4gICAgICAgIGlmIChtICE9PSBuZXdEc3QpIHtcbiAgICAgICAgICAgIG5ld0RzdFswXSA9IG1bMF07XG4gICAgICAgICAgICBuZXdEc3RbMV0gPSBtWzFdO1xuICAgICAgICAgICAgbmV3RHN0WzJdID0gbVsyXTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDcmVhdGVzIGEgMy1ieS0zIG1hdHJpeCB3aGljaCByb3RhdGVzIGFyb3VuZCB0aGUgeS1heGlzIGJ5IHRoZSBnaXZlbiBhbmdsZS5cbiAgICAgKiBAcGFyYW0gYW5nbGVJblJhZGlhbnMgLSBUaGUgYW5nbGUgYnkgd2hpY2ggdG8gcm90YXRlIChpbiByYWRpYW5zKS5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSByb3RhdGlvbiBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gcm90YXRpb25ZKGFuZ2xlSW5SYWRpYW5zLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxMikpO1xuICAgICAgICBjb25zdCBjID0gTWF0aC5jb3MoYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBjb25zdCBzID0gTWF0aC5zaW4oYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBuZXdEc3RbMF0gPSBjO1xuICAgICAgICBuZXdEc3RbMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMl0gPSAtcztcbiAgICAgICAgbmV3RHN0WzRdID0gMDtcbiAgICAgICAgbmV3RHN0WzVdID0gMTtcbiAgICAgICAgbmV3RHN0WzZdID0gMDtcbiAgICAgICAgbmV3RHN0WzhdID0gcztcbiAgICAgICAgbmV3RHN0WzldID0gMDtcbiAgICAgICAgbmV3RHN0WzEwXSA9IGM7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJvdGF0ZXMgdGhlIGdpdmVuIDMtYnktMyBtYXRyaXggYXJvdW5kIHRoZSB5LWF4aXMgYnkgdGhlIGdpdmVuXG4gICAgICogYW5nbGUuXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIFRoZSBhbmdsZSBieSB3aGljaCB0byByb3RhdGUgKGluIHJhZGlhbnMpLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHJvdGF0ZWQgbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0ZVkobSwgYW5nbGVJblJhZGlhbnMsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDEyKSk7XG4gICAgICAgIGNvbnN0IG0wMCA9IG1bMCAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTAxID0gbVswICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMDIgPSBtWzAgKiA0ICsgMl07XG4gICAgICAgIGNvbnN0IG0yMCA9IG1bMiAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTIxID0gbVsyICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMjIgPSBtWzIgKiA0ICsgMl07XG4gICAgICAgIGNvbnN0IGMgPSBNYXRoLmNvcyhhbmdsZUluUmFkaWFucyk7XG4gICAgICAgIGNvbnN0IHMgPSBNYXRoLnNpbihhbmdsZUluUmFkaWFucyk7XG4gICAgICAgIG5ld0RzdFswXSA9IGMgKiBtMDAgLSBzICogbTIwO1xuICAgICAgICBuZXdEc3RbMV0gPSBjICogbTAxIC0gcyAqIG0yMTtcbiAgICAgICAgbmV3RHN0WzJdID0gYyAqIG0wMiAtIHMgKiBtMjI7XG4gICAgICAgIG5ld0RzdFs4XSA9IGMgKiBtMjAgKyBzICogbTAwO1xuICAgICAgICBuZXdEc3RbOV0gPSBjICogbTIxICsgcyAqIG0wMTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IGMgKiBtMjIgKyBzICogbTAyO1xuICAgICAgICBpZiAobSAhPT0gbmV3RHN0KSB7XG4gICAgICAgICAgICBuZXdEc3RbNF0gPSBtWzRdO1xuICAgICAgICAgICAgbmV3RHN0WzVdID0gbVs1XTtcbiAgICAgICAgICAgIG5ld0RzdFs2XSA9IG1bNl07XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIDMtYnktMyBtYXRyaXggd2hpY2ggcm90YXRlcyBhcm91bmQgdGhlIHotYXhpcyBieSB0aGUgZ2l2ZW4gYW5nbGUuXG4gICAgICogQHBhcmFtIGFuZ2xlSW5SYWRpYW5zIC0gVGhlIGFuZ2xlIGJ5IHdoaWNoIHRvIHJvdGF0ZSAoaW4gcmFkaWFucykuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgcm90YXRpb24gbWF0cml4LlxuICAgICAqL1xuICAgIGNvbnN0IHJvdGF0aW9uWiA9IHJvdGF0aW9uO1xuICAgIC8qKlxuICAgICAqIFJvdGF0ZXMgdGhlIGdpdmVuIDMtYnktMyBtYXRyaXggYXJvdW5kIHRoZSB6LWF4aXMgYnkgdGhlIGdpdmVuXG4gICAgICogYW5nbGUuXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIFRoZSBhbmdsZSBieSB3aGljaCB0byByb3RhdGUgKGluIHJhZGlhbnMpLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHJvdGF0ZWQgbWF0cml4LlxuICAgICAqL1xuICAgIGNvbnN0IHJvdGF0ZVogPSByb3RhdGU7XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIDMtYnktMyBtYXRyaXggd2hpY2ggc2NhbGVzIGluIGVhY2ggZGltZW5zaW9uIGJ5IGFuIGFtb3VudCBnaXZlbiBieVxuICAgICAqIHRoZSBjb3JyZXNwb25kaW5nIGVudHJ5IGluIHRoZSBnaXZlbiB2ZWN0b3I7IGFzc3VtZXMgdGhlIHZlY3RvciBoYXMgdHdvXG4gICAgICogZW50cmllcy5cbiAgICAgKiBAcGFyYW0gdiAtIEEgdmVjdG9yIG9mXG4gICAgICogICAgIDIgZW50cmllcyBzcGVjaWZ5aW5nIHRoZSBmYWN0b3IgYnkgd2hpY2ggdG8gc2NhbGUgaW4gZWFjaCBkaW1lbnNpb24uXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGluZyBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gc2NhbGluZyh2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2WzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSAwO1xuICAgICAgICBuZXdEc3RbNV0gPSB2WzFdO1xuICAgICAgICBuZXdEc3RbNl0gPSAwO1xuICAgICAgICBuZXdEc3RbOF0gPSAwO1xuICAgICAgICBuZXdEc3RbOV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTBdID0gMTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogU2NhbGVzIHRoZSBnaXZlbiAzLWJ5LTMgbWF0cml4IGluIGVhY2ggZGltZW5zaW9uIGJ5IGFuIGFtb3VudFxuICAgICAqIGdpdmVuIGJ5IHRoZSBjb3JyZXNwb25kaW5nIGVudHJ5IGluIHRoZSBnaXZlbiB2ZWN0b3I7IGFzc3VtZXMgdGhlIHZlY3RvciBoYXNcbiAgICAgKiB0d28gZW50cmllcy5cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXggdG8gYmUgbW9kaWZpZWQuXG4gICAgICogQHBhcmFtIHYgLSBBIHZlY3RvciBvZiAyIGVudHJpZXMgc3BlY2lmeWluZyB0aGVcbiAgICAgKiAgICAgZmFjdG9yIGJ5IHdoaWNoIHRvIHNjYWxlIGluIGVhY2ggZGltZW5zaW9uLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHNjYWxlZCBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gc2NhbGUobSwgdiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgY29uc3QgdjAgPSB2WzBdO1xuICAgICAgICBjb25zdCB2MSA9IHZbMV07XG4gICAgICAgIG5ld0RzdFswXSA9IHYwICogbVswICogNCArIDBdO1xuICAgICAgICBuZXdEc3RbMV0gPSB2MCAqIG1bMCAqIDQgKyAxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gdjAgKiBtWzAgKiA0ICsgMl07XG4gICAgICAgIG5ld0RzdFs0XSA9IHYxICogbVsxICogNCArIDBdO1xuICAgICAgICBuZXdEc3RbNV0gPSB2MSAqIG1bMSAqIDQgKyAxXTtcbiAgICAgICAgbmV3RHN0WzZdID0gdjEgKiBtWzEgKiA0ICsgMl07XG4gICAgICAgIGlmIChtICE9PSBuZXdEc3QpIHtcbiAgICAgICAgICAgIG5ld0RzdFs4XSA9IG1bOF07XG4gICAgICAgICAgICBuZXdEc3RbOV0gPSBtWzldO1xuICAgICAgICAgICAgbmV3RHN0WzEwXSA9IG1bMTBdO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENyZWF0ZXMgYSAzLWJ5LTMgbWF0cml4IHdoaWNoIHNjYWxlcyBpbiBlYWNoIGRpbWVuc2lvbiBieSBhbiBhbW91bnQgZ2l2ZW4gYnlcbiAgICAgKiB0aGUgY29ycmVzcG9uZGluZyBlbnRyeSBpbiB0aGUgZ2l2ZW4gdmVjdG9yOyBhc3N1bWVzIHRoZSB2ZWN0b3IgaGFzIHRocmVlXG4gICAgICogZW50cmllcy5cbiAgICAgKiBAcGFyYW0gdiAtIEEgdmVjdG9yIG9mXG4gICAgICogICAgIDMgZW50cmllcyBzcGVjaWZ5aW5nIHRoZSBmYWN0b3IgYnkgd2hpY2ggdG8gc2NhbGUgaW4gZWFjaCBkaW1lbnNpb24uXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGluZyBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gc2NhbGluZzNEKHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDEyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IHZbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgIG5ld0RzdFsyXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IDA7XG4gICAgICAgIG5ld0RzdFs1XSA9IHZbMV07XG4gICAgICAgIG5ld0RzdFs2XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IDA7XG4gICAgICAgIG5ld0RzdFs5XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMF0gPSB2WzJdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBTY2FsZXMgdGhlIGdpdmVuIDMtYnktMyBtYXRyaXggaW4gZWFjaCBkaW1lbnNpb24gYnkgYW4gYW1vdW50XG4gICAgICogZ2l2ZW4gYnkgdGhlIGNvcnJlc3BvbmRpbmcgZW50cnkgaW4gdGhlIGdpdmVuIHZlY3RvcjsgYXNzdW1lcyB0aGUgdmVjdG9yIGhhc1xuICAgICAqIHRocmVlIGVudHJpZXMuXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4IHRvIGJlIG1vZGlmaWVkLlxuICAgICAqIEBwYXJhbSB2IC0gQSB2ZWN0b3Igb2YgMyBlbnRyaWVzIHNwZWNpZnlpbmcgdGhlXG4gICAgICogICAgIGZhY3RvciBieSB3aGljaCB0byBzY2FsZSBpbiBlYWNoIGRpbWVuc2lvbi5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBzY2FsZWQgbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHNjYWxlM0QobSwgdiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgY29uc3QgdjAgPSB2WzBdO1xuICAgICAgICBjb25zdCB2MSA9IHZbMV07XG4gICAgICAgIGNvbnN0IHYyID0gdlsyXTtcbiAgICAgICAgbmV3RHN0WzBdID0gdjAgKiBtWzAgKiA0ICsgMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IHYwICogbVswICogNCArIDFdO1xuICAgICAgICBuZXdEc3RbMl0gPSB2MCAqIG1bMCAqIDQgKyAyXTtcbiAgICAgICAgbmV3RHN0WzRdID0gdjEgKiBtWzEgKiA0ICsgMF07XG4gICAgICAgIG5ld0RzdFs1XSA9IHYxICogbVsxICogNCArIDFdO1xuICAgICAgICBuZXdEc3RbNl0gPSB2MSAqIG1bMSAqIDQgKyAyXTtcbiAgICAgICAgbmV3RHN0WzhdID0gdjIgKiBtWzIgKiA0ICsgMF07XG4gICAgICAgIG5ld0RzdFs5XSA9IHYyICogbVsyICogNCArIDFdO1xuICAgICAgICBuZXdEc3RbMTBdID0gdjIgKiBtWzIgKiA0ICsgMl07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENyZWF0ZXMgYSAzLWJ5LTMgbWF0cml4IHdoaWNoIHNjYWxlcyB1bmlmb3JtbHkgaW4gdGhlIFggYW5kIFkgZGltZW5zaW9uc1xuICAgICAqIEBwYXJhbSBzIC0gQW1vdW50IHRvIHNjYWxlXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGluZyBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gdW5pZm9ybVNjYWxpbmcocywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gcztcbiAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgbmV3RHN0WzJdID0gMDtcbiAgICAgICAgbmV3RHN0WzRdID0gMDtcbiAgICAgICAgbmV3RHN0WzVdID0gcztcbiAgICAgICAgbmV3RHN0WzZdID0gMDtcbiAgICAgICAgbmV3RHN0WzhdID0gMDtcbiAgICAgICAgbmV3RHN0WzldID0gMDtcbiAgICAgICAgbmV3RHN0WzEwXSA9IDE7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFNjYWxlcyB0aGUgZ2l2ZW4gMy1ieS0zIG1hdHJpeCBpbiB0aGUgWCBhbmQgWSBkaW1lbnNpb24gYnkgYW4gYW1vdW50XG4gICAgICogZ2l2ZW4uXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4IHRvIGJlIG1vZGlmaWVkLlxuICAgICAqIEBwYXJhbSBzIC0gQW1vdW50IHRvIHNjYWxlLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHNjYWxlZCBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gdW5pZm9ybVNjYWxlKG0sIHMsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDEyKSk7XG4gICAgICAgIG5ld0RzdFswXSA9IHMgKiBtWzAgKiA0ICsgMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IHMgKiBtWzAgKiA0ICsgMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IHMgKiBtWzAgKiA0ICsgMl07XG4gICAgICAgIG5ld0RzdFs0XSA9IHMgKiBtWzEgKiA0ICsgMF07XG4gICAgICAgIG5ld0RzdFs1XSA9IHMgKiBtWzEgKiA0ICsgMV07XG4gICAgICAgIG5ld0RzdFs2XSA9IHMgKiBtWzEgKiA0ICsgMl07XG4gICAgICAgIGlmIChtICE9PSBuZXdEc3QpIHtcbiAgICAgICAgICAgIG5ld0RzdFs4XSA9IG1bOF07XG4gICAgICAgICAgICBuZXdEc3RbOV0gPSBtWzldO1xuICAgICAgICAgICAgbmV3RHN0WzEwXSA9IG1bMTBdO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENyZWF0ZXMgYSAzLWJ5LTMgbWF0cml4IHdoaWNoIHNjYWxlcyB1bmlmb3JtbHkgaW4gZWFjaCBkaW1lbnNpb25cbiAgICAgKiBAcGFyYW0gcyAtIEFtb3VudCB0byBzY2FsZVxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHNjYWxpbmcgbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHVuaWZvcm1TY2FsaW5nM0QocywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTIpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gcztcbiAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgbmV3RHN0WzJdID0gMDtcbiAgICAgICAgbmV3RHN0WzRdID0gMDtcbiAgICAgICAgbmV3RHN0WzVdID0gcztcbiAgICAgICAgbmV3RHN0WzZdID0gMDtcbiAgICAgICAgbmV3RHN0WzhdID0gMDtcbiAgICAgICAgbmV3RHN0WzldID0gMDtcbiAgICAgICAgbmV3RHN0WzEwXSA9IHM7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFNjYWxlcyB0aGUgZ2l2ZW4gMy1ieS0zIG1hdHJpeCBpbiBlYWNoIGRpbWVuc2lvbiBieSBhbiBhbW91bnRcbiAgICAgKiBnaXZlbi5cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXggdG8gYmUgbW9kaWZpZWQuXG4gICAgICogQHBhcmFtIHMgLSBBbW91bnQgdG8gc2NhbGUuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiB1bmlmb3JtU2NhbGUzRChtLCBzLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxMikpO1xuICAgICAgICBuZXdEc3RbMF0gPSBzICogbVswICogNCArIDBdO1xuICAgICAgICBuZXdEc3RbMV0gPSBzICogbVswICogNCArIDFdO1xuICAgICAgICBuZXdEc3RbMl0gPSBzICogbVswICogNCArIDJdO1xuICAgICAgICBuZXdEc3RbNF0gPSBzICogbVsxICogNCArIDBdO1xuICAgICAgICBuZXdEc3RbNV0gPSBzICogbVsxICogNCArIDFdO1xuICAgICAgICBuZXdEc3RbNl0gPSBzICogbVsxICogNCArIDJdO1xuICAgICAgICBuZXdEc3RbOF0gPSBzICogbVsyICogNCArIDBdO1xuICAgICAgICBuZXdEc3RbOV0gPSBzICogbVsyICogNCArIDFdO1xuICAgICAgICBuZXdEc3RbMTBdID0gcyAqIG1bMiAqIDQgKyAyXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgcmV0dXJuIHtcbiAgICAgICAgY2xvbmUsXG4gICAgICAgIGNyZWF0ZSxcbiAgICAgICAgc2V0LFxuICAgICAgICBmcm9tTWF0NCxcbiAgICAgICAgZnJvbVF1YXQsXG4gICAgICAgIG5lZ2F0ZSxcbiAgICAgICAgY29weSxcbiAgICAgICAgZXF1YWxzQXBwcm94aW1hdGVseSxcbiAgICAgICAgZXF1YWxzLFxuICAgICAgICBpZGVudGl0eSxcbiAgICAgICAgdHJhbnNwb3NlLFxuICAgICAgICBpbnZlcnNlLFxuICAgICAgICBpbnZlcnQsXG4gICAgICAgIGRldGVybWluYW50LFxuICAgICAgICBtdWwsXG4gICAgICAgIG11bHRpcGx5LFxuICAgICAgICBzZXRUcmFuc2xhdGlvbixcbiAgICAgICAgZ2V0VHJhbnNsYXRpb24sXG4gICAgICAgIGdldEF4aXMsXG4gICAgICAgIHNldEF4aXMsXG4gICAgICAgIGdldFNjYWxpbmcsXG4gICAgICAgIGdldDNEU2NhbGluZyxcbiAgICAgICAgdHJhbnNsYXRpb24sXG4gICAgICAgIHRyYW5zbGF0ZSxcbiAgICAgICAgcm90YXRpb24sXG4gICAgICAgIHJvdGF0ZSxcbiAgICAgICAgcm90YXRpb25YLFxuICAgICAgICByb3RhdGVYLFxuICAgICAgICByb3RhdGlvblksXG4gICAgICAgIHJvdGF0ZVksXG4gICAgICAgIHJvdGF0aW9uWixcbiAgICAgICAgcm90YXRlWixcbiAgICAgICAgc2NhbGluZyxcbiAgICAgICAgc2NhbGUsXG4gICAgICAgIHVuaWZvcm1TY2FsaW5nLFxuICAgICAgICB1bmlmb3JtU2NhbGUsXG4gICAgICAgIHNjYWxpbmczRCxcbiAgICAgICAgc2NhbGUzRCxcbiAgICAgICAgdW5pZm9ybVNjYWxpbmczRCxcbiAgICAgICAgdW5pZm9ybVNjYWxlM0QsXG4gICAgfTtcbn1cbmNvbnN0IGNhY2hlJDMgPSBuZXcgTWFwKCk7XG5mdW5jdGlvbiBnZXRBUEkkMyhDdG9yKSB7XG4gICAgbGV0IGFwaSA9IGNhY2hlJDMuZ2V0KEN0b3IpO1xuICAgIGlmICghYXBpKSB7XG4gICAgICAgIGFwaSA9IGdldEFQSUltcGwkMyhDdG9yKTtcbiAgICAgICAgY2FjaGUkMy5zZXQoQ3RvciwgYXBpKTtcbiAgICB9XG4gICAgcmV0dXJuIGFwaTtcbn1cblxuLyoqXG4gKiBHZW5lcmF0ZXMgYSB0eXBlZCBBUEkgZm9yIE1hdDRcbiAqICovXG5mdW5jdGlvbiBnZXRBUElJbXBsJDIoQ3Rvcikge1xuICAgIGNvbnN0IHZlYzMgPSBnZXRBUEkkNChDdG9yKTtcbiAgICAvKipcbiAgICAgKiA0eDQgTWF0cml4IG1hdGggbWF0aCBmdW5jdGlvbnMuXG4gICAgICpcbiAgICAgKiBBbG1vc3QgYWxsIGZ1bmN0aW9ucyB0YWtlIGFuIG9wdGlvbmFsIGBuZXdEc3RgIGFyZ3VtZW50LiBJZiBpdCBpcyBub3QgcGFzc2VkIGluIHRoZVxuICAgICAqIGZ1bmN0aW9ucyB3aWxsIGNyZWF0ZSBhIG5ldyBtYXRyaXguIEluIG90aGVyIHdvcmRzIHlvdSBjYW4gZG8gdGhpc1xuICAgICAqXG4gICAgICogICAgIGNvbnN0IG1hdCA9IG1hdDQudHJhbnNsYXRpb24oWzEsIDIsIDNdKTsgIC8vIENyZWF0ZXMgYSBuZXcgdHJhbnNsYXRpb24gbWF0cml4XG4gICAgICpcbiAgICAgKiBvclxuICAgICAqXG4gICAgICogICAgIGNvbnN0IG1hdCA9IG1hdDQuY3JlYXRlKCk7XG4gICAgICogICAgIG1hdDQudHJhbnNsYXRpb24oWzEsIDIsIDNdLCBtYXQpOyAgLy8gUHV0cyB0cmFuc2xhdGlvbiBtYXRyaXggaW4gbWF0LlxuICAgICAqXG4gICAgICogVGhlIGZpcnN0IHN0eWxlIGlzIG9mdGVuIGVhc2llciBidXQgZGVwZW5kaW5nIG9uIHdoZXJlIGl0J3MgdXNlZCBpdCBnZW5lcmF0ZXMgZ2FyYmFnZSB3aGVyZVxuICAgICAqIGFzIHRoZXJlIGlzIGFsbW9zdCBuZXZlciBhbGxvY2F0aW9uIHdpdGggdGhlIHNlY29uZCBzdHlsZS5cbiAgICAgKlxuICAgICAqIEl0IGlzIGFsd2F5cyBzYXZlIHRvIHBhc3MgYW55IG1hdHJpeCBhcyB0aGUgZGVzdGluYXRpb24uIFNvIGZvciBleGFtcGxlXG4gICAgICpcbiAgICAgKiAgICAgY29uc3QgbWF0ID0gbWF0NC5pZGVudGl0eSgpO1xuICAgICAqICAgICBjb25zdCB0cmFucyA9IG1hdDQudHJhbnNsYXRpb24oWzEsIDIsIDNdKTtcbiAgICAgKiAgICAgbWF0NC5tdWx0aXBseShtYXQsIHRyYW5zLCBtYXQpOyAgLy8gTXVsdGlwbGllcyBtYXQgKiB0cmFucyBhbmQgcHV0cyByZXN1bHQgaW4gbWF0LlxuICAgICAqXG4gICAgICovXG4gICAgLyoqXG4gICAgICogQ3JlYXRlIGEgTWF0NCBmcm9tIHZhbHVlc1xuICAgICAqXG4gICAgICogTm90ZTogU2luY2UgcGFzc2luZyBpbiBhIHJhdyBKYXZhU2NyaXB0IGFycmF5XG4gICAgICogaXMgdmFsaWQgaW4gYWxsIGNpcmN1bXN0YW5jZXMsIGlmIHlvdSB3YW50IHRvXG4gICAgICogZm9yY2UgYSBKYXZhU2NyaXB0IGFycmF5IGludG8gYSBNYXQ0J3Mgc3BlY2lmaWVkIHR5cGVcbiAgICAgKiBpdCB3b3VsZCBiZSBmYXN0ZXIgdG8gdXNlXG4gICAgICpcbiAgICAgKiBgYGBcbiAgICAgKiBjb25zdCBtID0gbWF0NC5jbG9uZShzb21lSlNBcnJheSk7XG4gICAgICogYGBgXG4gICAgICpcbiAgICAgKiBAcGFyYW0gdjAgLSB2YWx1ZSBmb3IgZWxlbWVudCAwXG4gICAgICogQHBhcmFtIHYxIC0gdmFsdWUgZm9yIGVsZW1lbnQgMVxuICAgICAqIEBwYXJhbSB2MiAtIHZhbHVlIGZvciBlbGVtZW50IDJcbiAgICAgKiBAcGFyYW0gdjMgLSB2YWx1ZSBmb3IgZWxlbWVudCAzXG4gICAgICogQHBhcmFtIHY0IC0gdmFsdWUgZm9yIGVsZW1lbnQgNFxuICAgICAqIEBwYXJhbSB2NSAtIHZhbHVlIGZvciBlbGVtZW50IDVcbiAgICAgKiBAcGFyYW0gdjYgLSB2YWx1ZSBmb3IgZWxlbWVudCA2XG4gICAgICogQHBhcmFtIHY3IC0gdmFsdWUgZm9yIGVsZW1lbnQgN1xuICAgICAqIEBwYXJhbSB2OCAtIHZhbHVlIGZvciBlbGVtZW50IDhcbiAgICAgKiBAcGFyYW0gdjkgLSB2YWx1ZSBmb3IgZWxlbWVudCA5XG4gICAgICogQHBhcmFtIHYxMCAtIHZhbHVlIGZvciBlbGVtZW50IDEwXG4gICAgICogQHBhcmFtIHYxMSAtIHZhbHVlIGZvciBlbGVtZW50IDExXG4gICAgICogQHBhcmFtIHYxMiAtIHZhbHVlIGZvciBlbGVtZW50IDEyXG4gICAgICogQHBhcmFtIHYxMyAtIHZhbHVlIGZvciBlbGVtZW50IDEzXG4gICAgICogQHBhcmFtIHYxNCAtIHZhbHVlIGZvciBlbGVtZW50IDE0XG4gICAgICogQHBhcmFtIHYxNSAtIHZhbHVlIGZvciBlbGVtZW50IDE1XG4gICAgICogQHJldHVybnMgY3JlYXRlZCBmcm9tIHZhbHVlcy5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBjcmVhdGUodjAsIHYxLCB2MiwgdjMsIHY0LCB2NSwgdjYsIHY3LCB2OCwgdjksIHYxMCwgdjExLCB2MTIsIHYxMywgdjE0LCB2MTUpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gbmV3IEN0b3IoMTYpO1xuICAgICAgICBpZiAodjAgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0gdjA7XG4gICAgICAgICAgICBpZiAodjEgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgIG5ld0RzdFsxXSA9IHYxO1xuICAgICAgICAgICAgICAgIGlmICh2MiAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgICAgIG5ld0RzdFsyXSA9IHYyO1xuICAgICAgICAgICAgICAgICAgICBpZiAodjMgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgbmV3RHN0WzNdID0gdjM7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAodjQgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFs0XSA9IHY0O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh2NSAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFs1XSA9IHY1O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodjYgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbmV3RHN0WzZdID0gdjY7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodjcgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFs3XSA9IHY3O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh2OCAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFs4XSA9IHY4O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodjkgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbmV3RHN0WzldID0gdjk7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodjEwICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXdEc3RbMTBdID0gdjEwO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh2MTEgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXdEc3RbMTFdID0gdjExO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodjEyICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFsxMl0gPSB2MTI7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodjEzICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXdEc3RbMTNdID0gdjEzO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh2MTQgIT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXdEc3RbMTRdID0gdjE0O1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodjE1ICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFsxNV0gPSB2MTU7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogU2V0cyB0aGUgdmFsdWVzIG9mIGEgTWF0NFxuICAgICAqIEFsc28gc2VlIHtAbGluayBtYXQ0LmNyZWF0ZX0gYW5kIHtAbGluayBtYXQ0LmNvcHl9XG4gICAgICpcbiAgICAgKiBAcGFyYW0gdjAgLSB2YWx1ZSBmb3IgZWxlbWVudCAwXG4gICAgICogQHBhcmFtIHYxIC0gdmFsdWUgZm9yIGVsZW1lbnQgMVxuICAgICAqIEBwYXJhbSB2MiAtIHZhbHVlIGZvciBlbGVtZW50IDJcbiAgICAgKiBAcGFyYW0gdjMgLSB2YWx1ZSBmb3IgZWxlbWVudCAzXG4gICAgICogQHBhcmFtIHY0IC0gdmFsdWUgZm9yIGVsZW1lbnQgNFxuICAgICAqIEBwYXJhbSB2NSAtIHZhbHVlIGZvciBlbGVtZW50IDVcbiAgICAgKiBAcGFyYW0gdjYgLSB2YWx1ZSBmb3IgZWxlbWVudCA2XG4gICAgICogQHBhcmFtIHY3IC0gdmFsdWUgZm9yIGVsZW1lbnQgN1xuICAgICAqIEBwYXJhbSB2OCAtIHZhbHVlIGZvciBlbGVtZW50IDhcbiAgICAgKiBAcGFyYW0gdjkgLSB2YWx1ZSBmb3IgZWxlbWVudCA5XG4gICAgICogQHBhcmFtIHYxMCAtIHZhbHVlIGZvciBlbGVtZW50IDEwXG4gICAgICogQHBhcmFtIHYxMSAtIHZhbHVlIGZvciBlbGVtZW50IDExXG4gICAgICogQHBhcmFtIHYxMiAtIHZhbHVlIGZvciBlbGVtZW50IDEyXG4gICAgICogQHBhcmFtIHYxMyAtIHZhbHVlIGZvciBlbGVtZW50IDEzXG4gICAgICogQHBhcmFtIHYxNCAtIHZhbHVlIGZvciBlbGVtZW50IDE0XG4gICAgICogQHBhcmFtIHYxNSAtIHZhbHVlIGZvciBlbGVtZW50IDE1XG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBNYXQ0IGNyZWF0ZWQgZnJvbSB2YWx1ZXMuXG4gICAgICovXG4gICAgZnVuY3Rpb24gc2V0KHYwLCB2MSwgdjIsIHYzLCB2NCwgdjUsIHY2LCB2NywgdjgsIHY5LCB2MTAsIHYxMSwgdjEyLCB2MTMsIHYxNCwgdjE1LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2MDtcbiAgICAgICAgbmV3RHN0WzFdID0gdjE7XG4gICAgICAgIG5ld0RzdFsyXSA9IHYyO1xuICAgICAgICBuZXdEc3RbM10gPSB2MztcbiAgICAgICAgbmV3RHN0WzRdID0gdjQ7XG4gICAgICAgIG5ld0RzdFs1XSA9IHY1O1xuICAgICAgICBuZXdEc3RbNl0gPSB2NjtcbiAgICAgICAgbmV3RHN0WzddID0gdjc7XG4gICAgICAgIG5ld0RzdFs4XSA9IHY4O1xuICAgICAgICBuZXdEc3RbOV0gPSB2OTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IHYxMDtcbiAgICAgICAgbmV3RHN0WzExXSA9IHYxMTtcbiAgICAgICAgbmV3RHN0WzEyXSA9IHYxMjtcbiAgICAgICAgbmV3RHN0WzEzXSA9IHYxMztcbiAgICAgICAgbmV3RHN0WzE0XSA9IHYxNDtcbiAgICAgICAgbmV3RHN0WzE1XSA9IHYxNTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIE1hdDQgZnJvbSBhIE1hdDNcbiAgICAgKiBAcGFyYW0gbTMgLSBzb3VyY2UgbWF0cml4XG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBNYXQ0IG1hZGUgZnJvbSBtM1xuICAgICAqL1xuICAgIGZ1bmN0aW9uIGZyb21NYXQzKG0zLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBuZXdEc3RbMF0gPSBtM1swXTtcbiAgICAgICAgbmV3RHN0WzFdID0gbTNbMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IG0zWzJdO1xuICAgICAgICBuZXdEc3RbM10gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSBtM1s0XTtcbiAgICAgICAgbmV3RHN0WzVdID0gbTNbNV07XG4gICAgICAgIG5ld0RzdFs2XSA9IG0zWzZdO1xuICAgICAgICBuZXdEc3RbN10gPSAwO1xuICAgICAgICBuZXdEc3RbOF0gPSBtM1s4XTtcbiAgICAgICAgbmV3RHN0WzldID0gbTNbOV07XG4gICAgICAgIG5ld0RzdFsxMF0gPSBtM1sxMF07XG4gICAgICAgIG5ld0RzdFsxMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTJdID0gMDtcbiAgICAgICAgbmV3RHN0WzEzXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxNF0gPSAwO1xuICAgICAgICBuZXdEc3RbMTVdID0gMTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIE1hdDQgcm90YXRpb24gbWF0cml4IGZyb20gYSBxdWF0ZXJuaW9uXG4gICAgICogQHBhcmFtIHEgLSBxdWF0ZXJuaW9uIHRvIGNyZWF0ZSBtYXRyaXggZnJvbVxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgTWF0NCBtYWRlIGZyb20gcVxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGZyb21RdWF0KHEsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDE2KSk7XG4gICAgICAgIGNvbnN0IHggPSBxWzBdO1xuICAgICAgICBjb25zdCB5ID0gcVsxXTtcbiAgICAgICAgY29uc3QgeiA9IHFbMl07XG4gICAgICAgIGNvbnN0IHcgPSBxWzNdO1xuICAgICAgICBjb25zdCB4MiA9IHggKyB4O1xuICAgICAgICBjb25zdCB5MiA9IHkgKyB5O1xuICAgICAgICBjb25zdCB6MiA9IHogKyB6O1xuICAgICAgICBjb25zdCB4eCA9IHggKiB4MjtcbiAgICAgICAgY29uc3QgeXggPSB5ICogeDI7XG4gICAgICAgIGNvbnN0IHl5ID0geSAqIHkyO1xuICAgICAgICBjb25zdCB6eCA9IHogKiB4MjtcbiAgICAgICAgY29uc3QgenkgPSB6ICogeTI7XG4gICAgICAgIGNvbnN0IHp6ID0geiAqIHoyO1xuICAgICAgICBjb25zdCB3eCA9IHcgKiB4MjtcbiAgICAgICAgY29uc3Qgd3kgPSB3ICogeTI7XG4gICAgICAgIGNvbnN0IHd6ID0gdyAqIHoyO1xuICAgICAgICBuZXdEc3RbMF0gPSAxIC0geXkgLSB6ejtcbiAgICAgICAgbmV3RHN0WzFdID0geXggKyB3ejtcbiAgICAgICAgbmV3RHN0WzJdID0genggLSB3eTtcbiAgICAgICAgbmV3RHN0WzNdID0gMDtcbiAgICAgICAgbmV3RHN0WzRdID0geXggLSB3ejtcbiAgICAgICAgbmV3RHN0WzVdID0gMSAtIHh4IC0geno7XG4gICAgICAgIG5ld0RzdFs2XSA9IHp5ICsgd3g7XG4gICAgICAgIG5ld0RzdFs3XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IHp4ICsgd3k7XG4gICAgICAgIG5ld0RzdFs5XSA9IHp5IC0gd3g7XG4gICAgICAgIG5ld0RzdFsxMF0gPSAxIC0geHggLSB5eTtcbiAgICAgICAgbmV3RHN0WzExXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMl0gPSAwO1xuICAgICAgICBuZXdEc3RbMTNdID0gMDtcbiAgICAgICAgbmV3RHN0WzE0XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxNV0gPSAxO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBOZWdhdGVzIGEgbWF0cml4LlxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIC1tLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG5lZ2F0ZShtLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBuZXdEc3RbMF0gPSAtbVswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gLW1bMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IC1tWzJdO1xuICAgICAgICBuZXdEc3RbM10gPSAtbVszXTtcbiAgICAgICAgbmV3RHN0WzRdID0gLW1bNF07XG4gICAgICAgIG5ld0RzdFs1XSA9IC1tWzVdO1xuICAgICAgICBuZXdEc3RbNl0gPSAtbVs2XTtcbiAgICAgICAgbmV3RHN0WzddID0gLW1bN107XG4gICAgICAgIG5ld0RzdFs4XSA9IC1tWzhdO1xuICAgICAgICBuZXdEc3RbOV0gPSAtbVs5XTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IC1tWzEwXTtcbiAgICAgICAgbmV3RHN0WzExXSA9IC1tWzExXTtcbiAgICAgICAgbmV3RHN0WzEyXSA9IC1tWzEyXTtcbiAgICAgICAgbmV3RHN0WzEzXSA9IC1tWzEzXTtcbiAgICAgICAgbmV3RHN0WzE0XSA9IC1tWzE0XTtcbiAgICAgICAgbmV3RHN0WzE1XSA9IC1tWzE1XTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29waWVzIGEgbWF0cml4LiAoc2FtZSBhcyB7QGxpbmsgbWF0NC5jbG9uZX0pXG4gICAgICogQWxzbyBzZWUge0BsaW5rIG1hdDQuY3JlYXRlfSBhbmQge0BsaW5rIG1hdDQuc2V0fVxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gVGhlIG1hdHJpeC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIGNvcHkgb2YgbS5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBjb3B5KG0sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDE2KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IG1bMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IG1bMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IG1bMl07XG4gICAgICAgIG5ld0RzdFszXSA9IG1bM107XG4gICAgICAgIG5ld0RzdFs0XSA9IG1bNF07XG4gICAgICAgIG5ld0RzdFs1XSA9IG1bNV07XG4gICAgICAgIG5ld0RzdFs2XSA9IG1bNl07XG4gICAgICAgIG5ld0RzdFs3XSA9IG1bN107XG4gICAgICAgIG5ld0RzdFs4XSA9IG1bOF07XG4gICAgICAgIG5ld0RzdFs5XSA9IG1bOV07XG4gICAgICAgIG5ld0RzdFsxMF0gPSBtWzEwXTtcbiAgICAgICAgbmV3RHN0WzExXSA9IG1bMTFdO1xuICAgICAgICBuZXdEc3RbMTJdID0gbVsxMl07XG4gICAgICAgIG5ld0RzdFsxM10gPSBtWzEzXTtcbiAgICAgICAgbmV3RHN0WzE0XSA9IG1bMTRdO1xuICAgICAgICBuZXdEc3RbMTVdID0gbVsxNV07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvcGllcyBhIG1hdHJpeCAoc2FtZSBhcyB7QGxpbmsgbWF0NC5jb3B5fSlcbiAgICAgKiBBbHNvIHNlZSB7QGxpbmsgbWF0NC5jcmVhdGV9IGFuZCB7QGxpbmsgbWF0NC5zZXR9XG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBkc3QgLSBUaGUgbWF0cml4LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgY29weSBvZiBtLlxuICAgICAqL1xuICAgIGNvbnN0IGNsb25lID0gY29weTtcbiAgICAvKipcbiAgICAgKiBDaGVjayBpZiAyIG1hdHJpY2VzIGFyZSBhcHByb3hpbWF0ZWx5IGVxdWFsXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgbWF0cml4LlxuICAgICAqIEByZXR1cm5zIHRydWUgaWYgbWF0cmljZXMgYXJlIGFwcHJveGltYXRlbHkgZXF1YWxcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBlcXVhbHNBcHByb3hpbWF0ZWx5KGEsIGIpIHtcbiAgICAgICAgcmV0dXJuIE1hdGguYWJzKGFbMF0gLSBiWzBdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbMV0gLSBiWzFdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbMl0gLSBiWzJdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbM10gLSBiWzNdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbNF0gLSBiWzRdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbNV0gLSBiWzVdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbNl0gLSBiWzZdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbN10gLSBiWzddKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbOF0gLSBiWzhdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbOV0gLSBiWzldKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbMTBdIC0gYlsxMF0pIDwgRVBTSUxPTiAmJlxuICAgICAgICAgICAgTWF0aC5hYnMoYVsxMV0gLSBiWzExXSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzEyXSAtIGJbMTJdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbMTNdIC0gYlsxM10pIDwgRVBTSUxPTiAmJlxuICAgICAgICAgICAgTWF0aC5hYnMoYVsxNF0gLSBiWzE0XSkgPCBFUFNJTE9OICYmXG4gICAgICAgICAgICBNYXRoLmFicyhhWzE1XSAtIGJbMTVdKSA8IEVQU0lMT047XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENoZWNrIGlmIDIgbWF0cmljZXMgYXJlIGV4YWN0bHkgZXF1YWxcbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgbWF0cml4LlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCBtYXRyaXguXG4gICAgICogQHJldHVybnMgdHJ1ZSBpZiBtYXRyaWNlcyBhcmUgZXhhY3RseSBlcXVhbFxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGVxdWFscyhhLCBiKSB7XG4gICAgICAgIHJldHVybiBhWzBdID09PSBiWzBdICYmXG4gICAgICAgICAgICBhWzFdID09PSBiWzFdICYmXG4gICAgICAgICAgICBhWzJdID09PSBiWzJdICYmXG4gICAgICAgICAgICBhWzNdID09PSBiWzNdICYmXG4gICAgICAgICAgICBhWzRdID09PSBiWzRdICYmXG4gICAgICAgICAgICBhWzVdID09PSBiWzVdICYmXG4gICAgICAgICAgICBhWzZdID09PSBiWzZdICYmXG4gICAgICAgICAgICBhWzddID09PSBiWzddICYmXG4gICAgICAgICAgICBhWzhdID09PSBiWzhdICYmXG4gICAgICAgICAgICBhWzldID09PSBiWzldICYmXG4gICAgICAgICAgICBhWzEwXSA9PT0gYlsxMF0gJiZcbiAgICAgICAgICAgIGFbMTFdID09PSBiWzExXSAmJlxuICAgICAgICAgICAgYVsxMl0gPT09IGJbMTJdICYmXG4gICAgICAgICAgICBhWzEzXSA9PT0gYlsxM10gJiZcbiAgICAgICAgICAgIGFbMTRdID09PSBiWzE0XSAmJlxuICAgICAgICAgICAgYVsxNV0gPT09IGJbMTVdO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDcmVhdGVzIGEgNC1ieS00IGlkZW50aXR5IG1hdHJpeC5cbiAgICAgKlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSA0LWJ5LTQgaWRlbnRpdHkgbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGlkZW50aXR5KGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDE2KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IDE7XG4gICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgIG5ld0RzdFsyXSA9IDA7XG4gICAgICAgIG5ld0RzdFszXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IDA7XG4gICAgICAgIG5ld0RzdFs1XSA9IDE7XG4gICAgICAgIG5ld0RzdFs2XSA9IDA7XG4gICAgICAgIG5ld0RzdFs3XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IDA7XG4gICAgICAgIG5ld0RzdFs5XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMF0gPSAxO1xuICAgICAgICBuZXdEc3RbMTFdID0gMDtcbiAgICAgICAgbmV3RHN0WzEyXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxM10gPSAwO1xuICAgICAgICBuZXdEc3RbMTRdID0gMDtcbiAgICAgICAgbmV3RHN0WzE1XSA9IDE7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFRha2VzIHRoZSB0cmFuc3Bvc2Ugb2YgYSBtYXRyaXguXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHRyYW5zcG9zZSBvZiBtLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHRyYW5zcG9zZShtLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBpZiAobmV3RHN0ID09PSBtKSB7XG4gICAgICAgICAgICBsZXQgdDtcbiAgICAgICAgICAgIHQgPSBtWzFdO1xuICAgICAgICAgICAgbVsxXSA9IG1bNF07XG4gICAgICAgICAgICBtWzRdID0gdDtcbiAgICAgICAgICAgIHQgPSBtWzJdO1xuICAgICAgICAgICAgbVsyXSA9IG1bOF07XG4gICAgICAgICAgICBtWzhdID0gdDtcbiAgICAgICAgICAgIHQgPSBtWzNdO1xuICAgICAgICAgICAgbVszXSA9IG1bMTJdO1xuICAgICAgICAgICAgbVsxMl0gPSB0O1xuICAgICAgICAgICAgdCA9IG1bNl07XG4gICAgICAgICAgICBtWzZdID0gbVs5XTtcbiAgICAgICAgICAgIG1bOV0gPSB0O1xuICAgICAgICAgICAgdCA9IG1bN107XG4gICAgICAgICAgICBtWzddID0gbVsxM107XG4gICAgICAgICAgICBtWzEzXSA9IHQ7XG4gICAgICAgICAgICB0ID0gbVsxMV07XG4gICAgICAgICAgICBtWzExXSA9IG1bMTRdO1xuICAgICAgICAgICAgbVsxNF0gPSB0O1xuICAgICAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICAgICAgfVxuICAgICAgICBjb25zdCBtMDAgPSBtWzAgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0wMSA9IG1bMCAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTAyID0gbVswICogNCArIDJdO1xuICAgICAgICBjb25zdCBtMDMgPSBtWzAgKiA0ICsgM107XG4gICAgICAgIGNvbnN0IG0xMCA9IG1bMSAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTExID0gbVsxICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMTIgPSBtWzEgKiA0ICsgMl07XG4gICAgICAgIGNvbnN0IG0xMyA9IG1bMSAqIDQgKyAzXTtcbiAgICAgICAgY29uc3QgbTIwID0gbVsyICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMjEgPSBtWzIgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0yMiA9IG1bMiAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTIzID0gbVsyICogNCArIDNdO1xuICAgICAgICBjb25zdCBtMzAgPSBtWzMgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0zMSA9IG1bMyAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTMyID0gbVszICogNCArIDJdO1xuICAgICAgICBjb25zdCBtMzMgPSBtWzMgKiA0ICsgM107XG4gICAgICAgIG5ld0RzdFswXSA9IG0wMDtcbiAgICAgICAgbmV3RHN0WzFdID0gbTEwO1xuICAgICAgICBuZXdEc3RbMl0gPSBtMjA7XG4gICAgICAgIG5ld0RzdFszXSA9IG0zMDtcbiAgICAgICAgbmV3RHN0WzRdID0gbTAxO1xuICAgICAgICBuZXdEc3RbNV0gPSBtMTE7XG4gICAgICAgIG5ld0RzdFs2XSA9IG0yMTtcbiAgICAgICAgbmV3RHN0WzddID0gbTMxO1xuICAgICAgICBuZXdEc3RbOF0gPSBtMDI7XG4gICAgICAgIG5ld0RzdFs5XSA9IG0xMjtcbiAgICAgICAgbmV3RHN0WzEwXSA9IG0yMjtcbiAgICAgICAgbmV3RHN0WzExXSA9IG0zMjtcbiAgICAgICAgbmV3RHN0WzEyXSA9IG0wMztcbiAgICAgICAgbmV3RHN0WzEzXSA9IG0xMztcbiAgICAgICAgbmV3RHN0WzE0XSA9IG0yMztcbiAgICAgICAgbmV3RHN0WzE1XSA9IG0zMztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIGludmVyc2Ugb2YgYSA0LWJ5LTQgbWF0cml4LlxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBpbnZlcnNlIG9mIG0uXG4gICAgICovXG4gICAgZnVuY3Rpb24gaW52ZXJzZShtLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBjb25zdCBtMDAgPSBtWzAgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0wMSA9IG1bMCAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTAyID0gbVswICogNCArIDJdO1xuICAgICAgICBjb25zdCBtMDMgPSBtWzAgKiA0ICsgM107XG4gICAgICAgIGNvbnN0IG0xMCA9IG1bMSAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTExID0gbVsxICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMTIgPSBtWzEgKiA0ICsgMl07XG4gICAgICAgIGNvbnN0IG0xMyA9IG1bMSAqIDQgKyAzXTtcbiAgICAgICAgY29uc3QgbTIwID0gbVsyICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMjEgPSBtWzIgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0yMiA9IG1bMiAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTIzID0gbVsyICogNCArIDNdO1xuICAgICAgICBjb25zdCBtMzAgPSBtWzMgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0zMSA9IG1bMyAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTMyID0gbVszICogNCArIDJdO1xuICAgICAgICBjb25zdCBtMzMgPSBtWzMgKiA0ICsgM107XG4gICAgICAgIGNvbnN0IHRtcDAgPSBtMjIgKiBtMzM7XG4gICAgICAgIGNvbnN0IHRtcDEgPSBtMzIgKiBtMjM7XG4gICAgICAgIGNvbnN0IHRtcDIgPSBtMTIgKiBtMzM7XG4gICAgICAgIGNvbnN0IHRtcDMgPSBtMzIgKiBtMTM7XG4gICAgICAgIGNvbnN0IHRtcDQgPSBtMTIgKiBtMjM7XG4gICAgICAgIGNvbnN0IHRtcDUgPSBtMjIgKiBtMTM7XG4gICAgICAgIGNvbnN0IHRtcDYgPSBtMDIgKiBtMzM7XG4gICAgICAgIGNvbnN0IHRtcDcgPSBtMzIgKiBtMDM7XG4gICAgICAgIGNvbnN0IHRtcDggPSBtMDIgKiBtMjM7XG4gICAgICAgIGNvbnN0IHRtcDkgPSBtMjIgKiBtMDM7XG4gICAgICAgIGNvbnN0IHRtcDEwID0gbTAyICogbTEzO1xuICAgICAgICBjb25zdCB0bXAxMSA9IG0xMiAqIG0wMztcbiAgICAgICAgY29uc3QgdG1wMTIgPSBtMjAgKiBtMzE7XG4gICAgICAgIGNvbnN0IHRtcDEzID0gbTMwICogbTIxO1xuICAgICAgICBjb25zdCB0bXAxNCA9IG0xMCAqIG0zMTtcbiAgICAgICAgY29uc3QgdG1wMTUgPSBtMzAgKiBtMTE7XG4gICAgICAgIGNvbnN0IHRtcDE2ID0gbTEwICogbTIxO1xuICAgICAgICBjb25zdCB0bXAxNyA9IG0yMCAqIG0xMTtcbiAgICAgICAgY29uc3QgdG1wMTggPSBtMDAgKiBtMzE7XG4gICAgICAgIGNvbnN0IHRtcDE5ID0gbTMwICogbTAxO1xuICAgICAgICBjb25zdCB0bXAyMCA9IG0wMCAqIG0yMTtcbiAgICAgICAgY29uc3QgdG1wMjEgPSBtMjAgKiBtMDE7XG4gICAgICAgIGNvbnN0IHRtcDIyID0gbTAwICogbTExO1xuICAgICAgICBjb25zdCB0bXAyMyA9IG0xMCAqIG0wMTtcbiAgICAgICAgY29uc3QgdDAgPSAodG1wMCAqIG0xMSArIHRtcDMgKiBtMjEgKyB0bXA0ICogbTMxKSAtXG4gICAgICAgICAgICAodG1wMSAqIG0xMSArIHRtcDIgKiBtMjEgKyB0bXA1ICogbTMxKTtcbiAgICAgICAgY29uc3QgdDEgPSAodG1wMSAqIG0wMSArIHRtcDYgKiBtMjEgKyB0bXA5ICogbTMxKSAtXG4gICAgICAgICAgICAodG1wMCAqIG0wMSArIHRtcDcgKiBtMjEgKyB0bXA4ICogbTMxKTtcbiAgICAgICAgY29uc3QgdDIgPSAodG1wMiAqIG0wMSArIHRtcDcgKiBtMTEgKyB0bXAxMCAqIG0zMSkgLVxuICAgICAgICAgICAgKHRtcDMgKiBtMDEgKyB0bXA2ICogbTExICsgdG1wMTEgKiBtMzEpO1xuICAgICAgICBjb25zdCB0MyA9ICh0bXA1ICogbTAxICsgdG1wOCAqIG0xMSArIHRtcDExICogbTIxKSAtXG4gICAgICAgICAgICAodG1wNCAqIG0wMSArIHRtcDkgKiBtMTEgKyB0bXAxMCAqIG0yMSk7XG4gICAgICAgIGNvbnN0IGQgPSAxIC8gKG0wMCAqIHQwICsgbTEwICogdDEgKyBtMjAgKiB0MiArIG0zMCAqIHQzKTtcbiAgICAgICAgbmV3RHN0WzBdID0gZCAqIHQwO1xuICAgICAgICBuZXdEc3RbMV0gPSBkICogdDE7XG4gICAgICAgIG5ld0RzdFsyXSA9IGQgKiB0MjtcbiAgICAgICAgbmV3RHN0WzNdID0gZCAqIHQzO1xuICAgICAgICBuZXdEc3RbNF0gPSBkICogKCh0bXAxICogbTEwICsgdG1wMiAqIG0yMCArIHRtcDUgKiBtMzApIC1cbiAgICAgICAgICAgICh0bXAwICogbTEwICsgdG1wMyAqIG0yMCArIHRtcDQgKiBtMzApKTtcbiAgICAgICAgbmV3RHN0WzVdID0gZCAqICgodG1wMCAqIG0wMCArIHRtcDcgKiBtMjAgKyB0bXA4ICogbTMwKSAtXG4gICAgICAgICAgICAodG1wMSAqIG0wMCArIHRtcDYgKiBtMjAgKyB0bXA5ICogbTMwKSk7XG4gICAgICAgIG5ld0RzdFs2XSA9IGQgKiAoKHRtcDMgKiBtMDAgKyB0bXA2ICogbTEwICsgdG1wMTEgKiBtMzApIC1cbiAgICAgICAgICAgICh0bXAyICogbTAwICsgdG1wNyAqIG0xMCArIHRtcDEwICogbTMwKSk7XG4gICAgICAgIG5ld0RzdFs3XSA9IGQgKiAoKHRtcDQgKiBtMDAgKyB0bXA5ICogbTEwICsgdG1wMTAgKiBtMjApIC1cbiAgICAgICAgICAgICh0bXA1ICogbTAwICsgdG1wOCAqIG0xMCArIHRtcDExICogbTIwKSk7XG4gICAgICAgIG5ld0RzdFs4XSA9IGQgKiAoKHRtcDEyICogbTEzICsgdG1wMTUgKiBtMjMgKyB0bXAxNiAqIG0zMykgLVxuICAgICAgICAgICAgKHRtcDEzICogbTEzICsgdG1wMTQgKiBtMjMgKyB0bXAxNyAqIG0zMykpO1xuICAgICAgICBuZXdEc3RbOV0gPSBkICogKCh0bXAxMyAqIG0wMyArIHRtcDE4ICogbTIzICsgdG1wMjEgKiBtMzMpIC1cbiAgICAgICAgICAgICh0bXAxMiAqIG0wMyArIHRtcDE5ICogbTIzICsgdG1wMjAgKiBtMzMpKTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IGQgKiAoKHRtcDE0ICogbTAzICsgdG1wMTkgKiBtMTMgKyB0bXAyMiAqIG0zMykgLVxuICAgICAgICAgICAgKHRtcDE1ICogbTAzICsgdG1wMTggKiBtMTMgKyB0bXAyMyAqIG0zMykpO1xuICAgICAgICBuZXdEc3RbMTFdID0gZCAqICgodG1wMTcgKiBtMDMgKyB0bXAyMCAqIG0xMyArIHRtcDIzICogbTIzKSAtXG4gICAgICAgICAgICAodG1wMTYgKiBtMDMgKyB0bXAyMSAqIG0xMyArIHRtcDIyICogbTIzKSk7XG4gICAgICAgIG5ld0RzdFsxMl0gPSBkICogKCh0bXAxNCAqIG0yMiArIHRtcDE3ICogbTMyICsgdG1wMTMgKiBtMTIpIC1cbiAgICAgICAgICAgICh0bXAxNiAqIG0zMiArIHRtcDEyICogbTEyICsgdG1wMTUgKiBtMjIpKTtcbiAgICAgICAgbmV3RHN0WzEzXSA9IGQgKiAoKHRtcDIwICogbTMyICsgdG1wMTIgKiBtMDIgKyB0bXAxOSAqIG0yMikgLVxuICAgICAgICAgICAgKHRtcDE4ICogbTIyICsgdG1wMjEgKiBtMzIgKyB0bXAxMyAqIG0wMikpO1xuICAgICAgICBuZXdEc3RbMTRdID0gZCAqICgodG1wMTggKiBtMTIgKyB0bXAyMyAqIG0zMiArIHRtcDE1ICogbTAyKSAtXG4gICAgICAgICAgICAodG1wMjIgKiBtMzIgKyB0bXAxNCAqIG0wMiArIHRtcDE5ICogbTEyKSk7XG4gICAgICAgIG5ld0RzdFsxNV0gPSBkICogKCh0bXAyMiAqIG0yMiArIHRtcDE2ICogbTAyICsgdG1wMjEgKiBtMTIpIC1cbiAgICAgICAgICAgICh0bXAyMCAqIG0xMiArIHRtcDIzICogbTIyICsgdG1wMTcgKiBtMDIpKTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZSB0aGUgZGV0ZXJtaW5hbnQgb2YgYSBtYXRyaXhcbiAgICAgKiBAcGFyYW0gbSAtIHRoZSBtYXRyaXhcbiAgICAgKiBAcmV0dXJucyB0aGUgZGV0ZXJtaW5hbnRcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBkZXRlcm1pbmFudChtKSB7XG4gICAgICAgIGNvbnN0IG0wMCA9IG1bMCAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTAxID0gbVswICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMDIgPSBtWzAgKiA0ICsgMl07XG4gICAgICAgIGNvbnN0IG0wMyA9IG1bMCAqIDQgKyAzXTtcbiAgICAgICAgY29uc3QgbTEwID0gbVsxICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMTEgPSBtWzEgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0xMiA9IG1bMSAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTEzID0gbVsxICogNCArIDNdO1xuICAgICAgICBjb25zdCBtMjAgPSBtWzIgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0yMSA9IG1bMiAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTIyID0gbVsyICogNCArIDJdO1xuICAgICAgICBjb25zdCBtMjMgPSBtWzIgKiA0ICsgM107XG4gICAgICAgIGNvbnN0IG0zMCA9IG1bMyAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTMxID0gbVszICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMzIgPSBtWzMgKiA0ICsgMl07XG4gICAgICAgIGNvbnN0IG0zMyA9IG1bMyAqIDQgKyAzXTtcbiAgICAgICAgY29uc3QgdG1wMCA9IG0yMiAqIG0zMztcbiAgICAgICAgY29uc3QgdG1wMSA9IG0zMiAqIG0yMztcbiAgICAgICAgY29uc3QgdG1wMiA9IG0xMiAqIG0zMztcbiAgICAgICAgY29uc3QgdG1wMyA9IG0zMiAqIG0xMztcbiAgICAgICAgY29uc3QgdG1wNCA9IG0xMiAqIG0yMztcbiAgICAgICAgY29uc3QgdG1wNSA9IG0yMiAqIG0xMztcbiAgICAgICAgY29uc3QgdG1wNiA9IG0wMiAqIG0zMztcbiAgICAgICAgY29uc3QgdG1wNyA9IG0zMiAqIG0wMztcbiAgICAgICAgY29uc3QgdG1wOCA9IG0wMiAqIG0yMztcbiAgICAgICAgY29uc3QgdG1wOSA9IG0yMiAqIG0wMztcbiAgICAgICAgY29uc3QgdG1wMTAgPSBtMDIgKiBtMTM7XG4gICAgICAgIGNvbnN0IHRtcDExID0gbTEyICogbTAzO1xuICAgICAgICBjb25zdCB0MCA9ICh0bXAwICogbTExICsgdG1wMyAqIG0yMSArIHRtcDQgKiBtMzEpIC1cbiAgICAgICAgICAgICh0bXAxICogbTExICsgdG1wMiAqIG0yMSArIHRtcDUgKiBtMzEpO1xuICAgICAgICBjb25zdCB0MSA9ICh0bXAxICogbTAxICsgdG1wNiAqIG0yMSArIHRtcDkgKiBtMzEpIC1cbiAgICAgICAgICAgICh0bXAwICogbTAxICsgdG1wNyAqIG0yMSArIHRtcDggKiBtMzEpO1xuICAgICAgICBjb25zdCB0MiA9ICh0bXAyICogbTAxICsgdG1wNyAqIG0xMSArIHRtcDEwICogbTMxKSAtXG4gICAgICAgICAgICAodG1wMyAqIG0wMSArIHRtcDYgKiBtMTEgKyB0bXAxMSAqIG0zMSk7XG4gICAgICAgIGNvbnN0IHQzID0gKHRtcDUgKiBtMDEgKyB0bXA4ICogbTExICsgdG1wMTEgKiBtMjEpIC1cbiAgICAgICAgICAgICh0bXA0ICogbTAxICsgdG1wOSAqIG0xMSArIHRtcDEwICogbTIxKTtcbiAgICAgICAgcmV0dXJuIG0wMCAqIHQwICsgbTEwICogdDEgKyBtMjAgKiB0MiArIG0zMCAqIHQzO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgaW52ZXJzZSBvZiBhIDQtYnktNCBtYXRyaXguIChzYW1lIGFzIGludmVyc2UpXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIGludmVyc2Ugb2YgbS5cbiAgICAgKi9cbiAgICBjb25zdCBpbnZlcnQgPSBpbnZlcnNlO1xuICAgIC8qKlxuICAgICAqIE11bHRpcGxpZXMgdHdvIDQtYnktNCBtYXRyaWNlcyB3aXRoIGEgb24gdGhlIGxlZnQgYW5kIGIgb24gdGhlIHJpZ2h0XG4gICAgICogQHBhcmFtIGEgLSBUaGUgbWF0cml4IG9uIHRoZSBsZWZ0LlxuICAgICAqIEBwYXJhbSBiIC0gVGhlIG1hdHJpeCBvbiB0aGUgcmlnaHQuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgbWF0cml4IHByb2R1Y3Qgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBtdWx0aXBseShhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBjb25zdCBhMDAgPSBhWzBdO1xuICAgICAgICBjb25zdCBhMDEgPSBhWzFdO1xuICAgICAgICBjb25zdCBhMDIgPSBhWzJdO1xuICAgICAgICBjb25zdCBhMDMgPSBhWzNdO1xuICAgICAgICBjb25zdCBhMTAgPSBhWzQgKyAwXTtcbiAgICAgICAgY29uc3QgYTExID0gYVs0ICsgMV07XG4gICAgICAgIGNvbnN0IGExMiA9IGFbNCArIDJdO1xuICAgICAgICBjb25zdCBhMTMgPSBhWzQgKyAzXTtcbiAgICAgICAgY29uc3QgYTIwID0gYVs4ICsgMF07XG4gICAgICAgIGNvbnN0IGEyMSA9IGFbOCArIDFdO1xuICAgICAgICBjb25zdCBhMjIgPSBhWzggKyAyXTtcbiAgICAgICAgY29uc3QgYTIzID0gYVs4ICsgM107XG4gICAgICAgIGNvbnN0IGEzMCA9IGFbMTIgKyAwXTtcbiAgICAgICAgY29uc3QgYTMxID0gYVsxMiArIDFdO1xuICAgICAgICBjb25zdCBhMzIgPSBhWzEyICsgMl07XG4gICAgICAgIGNvbnN0IGEzMyA9IGFbMTIgKyAzXTtcbiAgICAgICAgY29uc3QgYjAwID0gYlswXTtcbiAgICAgICAgY29uc3QgYjAxID0gYlsxXTtcbiAgICAgICAgY29uc3QgYjAyID0gYlsyXTtcbiAgICAgICAgY29uc3QgYjAzID0gYlszXTtcbiAgICAgICAgY29uc3QgYjEwID0gYls0ICsgMF07XG4gICAgICAgIGNvbnN0IGIxMSA9IGJbNCArIDFdO1xuICAgICAgICBjb25zdCBiMTIgPSBiWzQgKyAyXTtcbiAgICAgICAgY29uc3QgYjEzID0gYls0ICsgM107XG4gICAgICAgIGNvbnN0IGIyMCA9IGJbOCArIDBdO1xuICAgICAgICBjb25zdCBiMjEgPSBiWzggKyAxXTtcbiAgICAgICAgY29uc3QgYjIyID0gYls4ICsgMl07XG4gICAgICAgIGNvbnN0IGIyMyA9IGJbOCArIDNdO1xuICAgICAgICBjb25zdCBiMzAgPSBiWzEyICsgMF07XG4gICAgICAgIGNvbnN0IGIzMSA9IGJbMTIgKyAxXTtcbiAgICAgICAgY29uc3QgYjMyID0gYlsxMiArIDJdO1xuICAgICAgICBjb25zdCBiMzMgPSBiWzEyICsgM107XG4gICAgICAgIG5ld0RzdFswXSA9IGEwMCAqIGIwMCArIGExMCAqIGIwMSArIGEyMCAqIGIwMiArIGEzMCAqIGIwMztcbiAgICAgICAgbmV3RHN0WzFdID0gYTAxICogYjAwICsgYTExICogYjAxICsgYTIxICogYjAyICsgYTMxICogYjAzO1xuICAgICAgICBuZXdEc3RbMl0gPSBhMDIgKiBiMDAgKyBhMTIgKiBiMDEgKyBhMjIgKiBiMDIgKyBhMzIgKiBiMDM7XG4gICAgICAgIG5ld0RzdFszXSA9IGEwMyAqIGIwMCArIGExMyAqIGIwMSArIGEyMyAqIGIwMiArIGEzMyAqIGIwMztcbiAgICAgICAgbmV3RHN0WzRdID0gYTAwICogYjEwICsgYTEwICogYjExICsgYTIwICogYjEyICsgYTMwICogYjEzO1xuICAgICAgICBuZXdEc3RbNV0gPSBhMDEgKiBiMTAgKyBhMTEgKiBiMTEgKyBhMjEgKiBiMTIgKyBhMzEgKiBiMTM7XG4gICAgICAgIG5ld0RzdFs2XSA9IGEwMiAqIGIxMCArIGExMiAqIGIxMSArIGEyMiAqIGIxMiArIGEzMiAqIGIxMztcbiAgICAgICAgbmV3RHN0WzddID0gYTAzICogYjEwICsgYTEzICogYjExICsgYTIzICogYjEyICsgYTMzICogYjEzO1xuICAgICAgICBuZXdEc3RbOF0gPSBhMDAgKiBiMjAgKyBhMTAgKiBiMjEgKyBhMjAgKiBiMjIgKyBhMzAgKiBiMjM7XG4gICAgICAgIG5ld0RzdFs5XSA9IGEwMSAqIGIyMCArIGExMSAqIGIyMSArIGEyMSAqIGIyMiArIGEzMSAqIGIyMztcbiAgICAgICAgbmV3RHN0WzEwXSA9IGEwMiAqIGIyMCArIGExMiAqIGIyMSArIGEyMiAqIGIyMiArIGEzMiAqIGIyMztcbiAgICAgICAgbmV3RHN0WzExXSA9IGEwMyAqIGIyMCArIGExMyAqIGIyMSArIGEyMyAqIGIyMiArIGEzMyAqIGIyMztcbiAgICAgICAgbmV3RHN0WzEyXSA9IGEwMCAqIGIzMCArIGExMCAqIGIzMSArIGEyMCAqIGIzMiArIGEzMCAqIGIzMztcbiAgICAgICAgbmV3RHN0WzEzXSA9IGEwMSAqIGIzMCArIGExMSAqIGIzMSArIGEyMSAqIGIzMiArIGEzMSAqIGIzMztcbiAgICAgICAgbmV3RHN0WzE0XSA9IGEwMiAqIGIzMCArIGExMiAqIGIzMSArIGEyMiAqIGIzMiArIGEzMiAqIGIzMztcbiAgICAgICAgbmV3RHN0WzE1XSA9IGEwMyAqIGIzMCArIGExMyAqIGIzMSArIGEyMyAqIGIzMiArIGEzMyAqIGIzMztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogTXVsdGlwbGllcyB0d28gNC1ieS00IG1hdHJpY2VzIHdpdGggYSBvbiB0aGUgbGVmdCBhbmQgYiBvbiB0aGUgcmlnaHQgKHNhbWUgYXMgbXVsdGlwbHkpXG4gICAgICogQHBhcmFtIGEgLSBUaGUgbWF0cml4IG9uIHRoZSBsZWZ0LlxuICAgICAqIEBwYXJhbSBiIC0gVGhlIG1hdHJpeCBvbiB0aGUgcmlnaHQuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgbWF0cml4IHByb2R1Y3Qgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBjb25zdCBtdWwgPSBtdWx0aXBseTtcbiAgICAvKipcbiAgICAgKiBTZXRzIHRoZSB0cmFuc2xhdGlvbiBjb21wb25lbnQgb2YgYSA0LWJ5LTQgbWF0cml4IHRvIHRoZSBnaXZlblxuICAgICAqIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIG1hdHJpeCB3aXRoIHRyYW5zbGF0aW9uIHNldC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzZXRUcmFuc2xhdGlvbihhLCB2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBpZGVudGl0eSgpKTtcbiAgICAgICAgaWYgKGEgIT09IG5ld0RzdCkge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0gYVswXTtcbiAgICAgICAgICAgIG5ld0RzdFsxXSA9IGFbMV07XG4gICAgICAgICAgICBuZXdEc3RbMl0gPSBhWzJdO1xuICAgICAgICAgICAgbmV3RHN0WzNdID0gYVszXTtcbiAgICAgICAgICAgIG5ld0RzdFs0XSA9IGFbNF07XG4gICAgICAgICAgICBuZXdEc3RbNV0gPSBhWzVdO1xuICAgICAgICAgICAgbmV3RHN0WzZdID0gYVs2XTtcbiAgICAgICAgICAgIG5ld0RzdFs3XSA9IGFbN107XG4gICAgICAgICAgICBuZXdEc3RbOF0gPSBhWzhdO1xuICAgICAgICAgICAgbmV3RHN0WzldID0gYVs5XTtcbiAgICAgICAgICAgIG5ld0RzdFsxMF0gPSBhWzEwXTtcbiAgICAgICAgICAgIG5ld0RzdFsxMV0gPSBhWzExXTtcbiAgICAgICAgfVxuICAgICAgICBuZXdEc3RbMTJdID0gdlswXTtcbiAgICAgICAgbmV3RHN0WzEzXSA9IHZbMV07XG4gICAgICAgIG5ld0RzdFsxNF0gPSB2WzJdO1xuICAgICAgICBuZXdEc3RbMTVdID0gMTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLy8vKipcbiAgICAvLyAqIFJldHVybnMgdGhlIHRyYW5zbGF0aW9uIGNvbXBvbmVudCBvZiBhIDQtYnktNCBtYXRyaXggYXMgYSB2ZWN0b3Igd2l0aCAzXG4gICAgLy8gKiBlbnRyaWVzLlxuICAgIC8vICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgIC8vICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAvLyAqIEByZXR1cm5zIFRoZSB0cmFuc2xhdGlvbiBjb21wb25lbnQgb2YgbS5cbiAgICAvLyAqL1xuICAgIGZ1bmN0aW9uIGdldFRyYW5zbGF0aW9uKG0sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IHZlYzMuY3JlYXRlKCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSBtWzEyXTtcbiAgICAgICAgbmV3RHN0WzFdID0gbVsxM107XG4gICAgICAgIG5ld0RzdFsyXSA9IG1bMTRdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSZXR1cm5zIGFuIGF4aXMgb2YgYSA0eDQgbWF0cml4IGFzIGEgdmVjdG9yIHdpdGggMyBlbnRyaWVzXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBheGlzIC0gVGhlIGF4aXMgMCA9IHgsIDEgPSB5LCAyID0gejtcbiAgICAgKiBAcmV0dXJucyBUaGUgYXhpcyBjb21wb25lbnQgb2YgbS5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBnZXRBeGlzKG0sIGF4aXMsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IHZlYzMuY3JlYXRlKCkpO1xuICAgICAgICBjb25zdCBvZmYgPSBheGlzICogNDtcbiAgICAgICAgbmV3RHN0WzBdID0gbVtvZmYgKyAwXTtcbiAgICAgICAgbmV3RHN0WzFdID0gbVtvZmYgKyAxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gbVtvZmYgKyAyXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogU2V0cyBhbiBheGlzIG9mIGEgNHg0IG1hdHJpeCBhcyBhIHZlY3RvciB3aXRoIDMgZW50cmllc1xuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gdiAtIHRoZSBheGlzIHZlY3RvclxuICAgICAqIEBwYXJhbSBheGlzIC0gVGhlIGF4aXMgIDAgPSB4LCAxID0geSwgMiA9IHo7XG4gICAgICogQHBhcmFtIGRzdCAtIFRoZSBtYXRyaXggdG8gc2V0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBtYXRyaXggd2l0aCBheGlzIHNldC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzZXRBeGlzKG0sIHYsIGF4aXMsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID09PSBtKSA/IGRzdCA6IGNvcHkobSwgZHN0KTtcbiAgICAgICAgY29uc3Qgb2ZmID0gYXhpcyAqIDQ7XG4gICAgICAgIG5ld0RzdFtvZmYgKyAwXSA9IHZbMF07XG4gICAgICAgIG5ld0RzdFtvZmYgKyAxXSA9IHZbMV07XG4gICAgICAgIG5ld0RzdFtvZmYgKyAyXSA9IHZbMl07XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJldHVybnMgdGhlIFwiM2RcIiBzY2FsaW5nIGNvbXBvbmVudCBvZiB0aGUgbWF0cml4XG4gICAgICogQHBhcmFtIG0gLSBUaGUgTWF0cml4XG4gICAgICogQHBhcmFtIGRzdCAtIFRoZSB2ZWN0b3IgdG8gc2V0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGdldFNjYWxpbmcobSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gdmVjMy5jcmVhdGUoKSk7XG4gICAgICAgIGNvbnN0IHh4ID0gbVswXTtcbiAgICAgICAgY29uc3QgeHkgPSBtWzFdO1xuICAgICAgICBjb25zdCB4eiA9IG1bMl07XG4gICAgICAgIGNvbnN0IHl4ID0gbVs0XTtcbiAgICAgICAgY29uc3QgeXkgPSBtWzVdO1xuICAgICAgICBjb25zdCB5eiA9IG1bNl07XG4gICAgICAgIGNvbnN0IHp4ID0gbVs4XTtcbiAgICAgICAgY29uc3QgenkgPSBtWzldO1xuICAgICAgICBjb25zdCB6eiA9IG1bMTBdO1xuICAgICAgICBuZXdEc3RbMF0gPSBNYXRoLnNxcnQoeHggKiB4eCArIHh5ICogeHkgKyB4eiAqIHh6KTtcbiAgICAgICAgbmV3RHN0WzFdID0gTWF0aC5zcXJ0KHl4ICogeXggKyB5eSAqIHl5ICsgeXogKiB5eik7XG4gICAgICAgIG5ld0RzdFsyXSA9IE1hdGguc3FydCh6eCAqIHp4ICsgenkgKiB6eSArIHp6ICogenopO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyBhIDQtYnktNCBwZXJzcGVjdGl2ZSB0cmFuc2Zvcm1hdGlvbiBtYXRyaXggZ2l2ZW4gdGhlIGFuZ3VsYXIgaGVpZ2h0XG4gICAgICogb2YgdGhlIGZydXN0dW0sIHRoZSBhc3BlY3QgcmF0aW8sIGFuZCB0aGUgbmVhciBhbmQgZmFyIGNsaXBwaW5nIHBsYW5lcy4gIFRoZVxuICAgICAqIGFyZ3VtZW50cyBkZWZpbmUgYSBmcnVzdHVtIGV4dGVuZGluZyBpbiB0aGUgbmVnYXRpdmUgeiBkaXJlY3Rpb24uICBUaGUgZ2l2ZW5cbiAgICAgKiBhbmdsZSBpcyB0aGUgdmVydGljYWwgYW5nbGUgb2YgdGhlIGZydXN0dW0sIGFuZCB0aGUgaG9yaXpvbnRhbCBhbmdsZSBpc1xuICAgICAqIGRldGVybWluZWQgdG8gcHJvZHVjZSB0aGUgZ2l2ZW4gYXNwZWN0IHJhdGlvLiAgVGhlIGFyZ3VtZW50cyBuZWFyIGFuZCBmYXIgYXJlXG4gICAgICogdGhlIGRpc3RhbmNlcyB0byB0aGUgbmVhciBhbmQgZmFyIGNsaXBwaW5nIHBsYW5lcy4gIE5vdGUgdGhhdCBuZWFyIGFuZCBmYXJcbiAgICAgKiBhcmUgbm90IHogY29vcmRpbmF0ZXMsIGJ1dCByYXRoZXIgdGhleSBhcmUgZGlzdGFuY2VzIGFsb25nIHRoZSBuZWdhdGl2ZVxuICAgICAqIHotYXhpcy4gIFRoZSBtYXRyaXggZ2VuZXJhdGVkIHNlbmRzIHRoZSB2aWV3aW5nIGZydXN0dW0gdG8gdGhlIHVuaXQgYm94LlxuICAgICAqIFdlIGFzc3VtZSBhIHVuaXQgYm94IGV4dGVuZGluZyBmcm9tIC0xIHRvIDEgaW4gdGhlIHggYW5kIHkgZGltZW5zaW9ucyBhbmRcbiAgICAgKiBmcm9tIDAgdG8gMSBpbiB0aGUgeiBkaW1lbnNpb24uXG4gICAgICpcbiAgICAgKiBOb3RlOiBJZiB5b3UgcGFzcyBgSW5maW5pdHlgIGZvciB6RmFyIHRoZW4gaXQgd2lsbCBwcm9kdWNlIGEgcHJvamVjdGlvbiBtYXRyaXhcbiAgICAgKiByZXR1cm5zIC1JbmZpbml0eSBmb3IgWiB3aGVuIHRyYW5zZm9ybWluZyBjb29yZGluYXRlcyB3aXRoIFogPD0gMCBhbmQgK0luZmluaXR5IGZvciBaXG4gICAgICogb3RoZXJ3aXNlLlxuICAgICAqXG4gICAgICogQHBhcmFtIGZpZWxkT2ZWaWV3WUluUmFkaWFucyAtIFRoZSBjYW1lcmEgYW5nbGUgZnJvbSB0b3AgdG8gYm90dG9tIChpbiByYWRpYW5zKS5cbiAgICAgKiBAcGFyYW0gYXNwZWN0IC0gVGhlIGFzcGVjdCByYXRpbyB3aWR0aCAvIGhlaWdodC5cbiAgICAgKiBAcGFyYW0gek5lYXIgLSBUaGUgZGVwdGggKG5lZ2F0aXZlIHogY29vcmRpbmF0ZSlcbiAgICAgKiAgICAgb2YgdGhlIG5lYXIgY2xpcHBpbmcgcGxhbmUuXG4gICAgICogQHBhcmFtIHpGYXIgLSBUaGUgZGVwdGggKG5lZ2F0aXZlIHogY29vcmRpbmF0ZSlcbiAgICAgKiAgICAgb2YgdGhlIGZhciBjbGlwcGluZyBwbGFuZS5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBwZXJzcGVjdGl2ZSBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gcGVyc3BlY3RpdmUoZmllbGRPZlZpZXdZSW5SYWRpYW5zLCBhc3BlY3QsIHpOZWFyLCB6RmFyLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBjb25zdCBmID0gTWF0aC50YW4oTWF0aC5QSSAqIDAuNSAtIDAuNSAqIGZpZWxkT2ZWaWV3WUluUmFkaWFucyk7XG4gICAgICAgIG5ld0RzdFswXSA9IGYgLyBhc3BlY3Q7XG4gICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgIG5ld0RzdFsyXSA9IDA7XG4gICAgICAgIG5ld0RzdFszXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IDA7XG4gICAgICAgIG5ld0RzdFs1XSA9IGY7XG4gICAgICAgIG5ld0RzdFs2XSA9IDA7XG4gICAgICAgIG5ld0RzdFs3XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IDA7XG4gICAgICAgIG5ld0RzdFs5XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMV0gPSAtMTtcbiAgICAgICAgbmV3RHN0WzEyXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxM10gPSAwO1xuICAgICAgICBuZXdEc3RbMTVdID0gMDtcbiAgICAgICAgaWYgKE51bWJlci5pc0Zpbml0ZSh6RmFyKSkge1xuICAgICAgICAgICAgY29uc3QgcmFuZ2VJbnYgPSAxIC8gKHpOZWFyIC0gekZhcik7XG4gICAgICAgICAgICBuZXdEc3RbMTBdID0gekZhciAqIHJhbmdlSW52O1xuICAgICAgICAgICAgbmV3RHN0WzE0XSA9IHpGYXIgKiB6TmVhciAqIHJhbmdlSW52O1xuICAgICAgICB9XG4gICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgbmV3RHN0WzEwXSA9IC0xO1xuICAgICAgICAgICAgbmV3RHN0WzE0XSA9IC16TmVhcjtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyBhIDQtYnktNCByZXZlcnNlLXogcGVyc3BlY3RpdmUgdHJhbnNmb3JtYXRpb24gbWF0cml4IGdpdmVuIHRoZSBhbmd1bGFyIGhlaWdodFxuICAgICAqIG9mIHRoZSBmcnVzdHVtLCB0aGUgYXNwZWN0IHJhdGlvLCBhbmQgdGhlIG5lYXIgYW5kIGZhciBjbGlwcGluZyBwbGFuZXMuICBUaGVcbiAgICAgKiBhcmd1bWVudHMgZGVmaW5lIGEgZnJ1c3R1bSBleHRlbmRpbmcgaW4gdGhlIG5lZ2F0aXZlIHogZGlyZWN0aW9uLiAgVGhlIGdpdmVuXG4gICAgICogYW5nbGUgaXMgdGhlIHZlcnRpY2FsIGFuZ2xlIG9mIHRoZSBmcnVzdHVtLCBhbmQgdGhlIGhvcml6b250YWwgYW5nbGUgaXNcbiAgICAgKiBkZXRlcm1pbmVkIHRvIHByb2R1Y2UgdGhlIGdpdmVuIGFzcGVjdCByYXRpby4gIFRoZSBhcmd1bWVudHMgbmVhciBhbmQgZmFyIGFyZVxuICAgICAqIHRoZSBkaXN0YW5jZXMgdG8gdGhlIG5lYXIgYW5kIGZhciBjbGlwcGluZyBwbGFuZXMuICBOb3RlIHRoYXQgbmVhciBhbmQgZmFyXG4gICAgICogYXJlIG5vdCB6IGNvb3JkaW5hdGVzLCBidXQgcmF0aGVyIHRoZXkgYXJlIGRpc3RhbmNlcyBhbG9uZyB0aGUgbmVnYXRpdmVcbiAgICAgKiB6LWF4aXMuICBUaGUgbWF0cml4IGdlbmVyYXRlZCBzZW5kcyB0aGUgdmlld2luZyBmcnVzdHVtIHRvIHRoZSB1bml0IGJveC5cbiAgICAgKiBXZSBhc3N1bWUgYSB1bml0IGJveCBleHRlbmRpbmcgZnJvbSAtMSB0byAxIGluIHRoZSB4IGFuZCB5IGRpbWVuc2lvbnMgYW5kXG4gICAgICogZnJvbSAxIChhdCAtek5lYXIpIHRvIDAgKGF0IC16RmFyKSBpbiB0aGUgeiBkaW1lbnNpb24uXG4gICAgICpcbiAgICAgKiBAcGFyYW0gZmllbGRPZlZpZXdZSW5SYWRpYW5zIC0gVGhlIGNhbWVyYSBhbmdsZSBmcm9tIHRvcCB0byBib3R0b20gKGluIHJhZGlhbnMpLlxuICAgICAqIEBwYXJhbSBhc3BlY3QgLSBUaGUgYXNwZWN0IHJhdGlvIHdpZHRoIC8gaGVpZ2h0LlxuICAgICAqIEBwYXJhbSB6TmVhciAtIFRoZSBkZXB0aCAobmVnYXRpdmUgeiBjb29yZGluYXRlKVxuICAgICAqICAgICBvZiB0aGUgbmVhciBjbGlwcGluZyBwbGFuZS5cbiAgICAgKiBAcGFyYW0gekZhciAtIFRoZSBkZXB0aCAobmVnYXRpdmUgeiBjb29yZGluYXRlKVxuICAgICAqICAgICBvZiB0aGUgZmFyIGNsaXBwaW5nIHBsYW5lLiAoZGVmYXVsdCA9IEluZmluaXR5KVxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHBlcnNwZWN0aXZlIG1hdHJpeC5cbiAgICAgKi8gZnVuY3Rpb24gcGVyc3BlY3RpdmVSZXZlcnNlWihmaWVsZE9mVmlld1lJblJhZGlhbnMsIGFzcGVjdCwgek5lYXIsIHpGYXIgPSBJbmZpbml0eSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgY29uc3QgZiA9IDEgLyBNYXRoLnRhbihmaWVsZE9mVmlld1lJblJhZGlhbnMgKiAwLjUpO1xuICAgICAgICBuZXdEc3RbMF0gPSBmIC8gYXNwZWN0O1xuICAgICAgICBuZXdEc3RbMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICBuZXdEc3RbM10gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSAwO1xuICAgICAgICBuZXdEc3RbNV0gPSBmO1xuICAgICAgICBuZXdEc3RbNl0gPSAwO1xuICAgICAgICBuZXdEc3RbN10gPSAwO1xuICAgICAgICBuZXdEc3RbOF0gPSAwO1xuICAgICAgICBuZXdEc3RbOV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTFdID0gLTE7XG4gICAgICAgIG5ld0RzdFsxMl0gPSAwO1xuICAgICAgICBuZXdEc3RbMTNdID0gMDtcbiAgICAgICAgbmV3RHN0WzE1XSA9IDA7XG4gICAgICAgIGlmICh6RmFyID09PSBJbmZpbml0eSkge1xuICAgICAgICAgICAgbmV3RHN0WzEwXSA9IDA7XG4gICAgICAgICAgICBuZXdEc3RbMTRdID0gek5lYXI7XG4gICAgICAgIH1cbiAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICBjb25zdCByYW5nZUludiA9IDEgLyAoekZhciAtIHpOZWFyKTtcbiAgICAgICAgICAgIG5ld0RzdFsxMF0gPSB6TmVhciAqIHJhbmdlSW52O1xuICAgICAgICAgICAgbmV3RHN0WzE0XSA9IHpGYXIgKiB6TmVhciAqIHJhbmdlSW52O1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIGEgNC1ieS00IG9ydGhvZ29uYWwgdHJhbnNmb3JtYXRpb24gbWF0cml4IHRoYXQgdHJhbnNmb3JtcyBmcm9tXG4gICAgICogdGhlIGdpdmVuIHRoZSBsZWZ0LCByaWdodCwgYm90dG9tLCBhbmQgdG9wIGRpbWVuc2lvbnMgdG8gLTEgKzEgaW4geCwgYW5kIHlcbiAgICAgKiBhbmQgMCB0byArMSBpbiB6LlxuICAgICAqIEBwYXJhbSBsZWZ0IC0gTGVmdCBzaWRlIG9mIHRoZSBuZWFyIGNsaXBwaW5nIHBsYW5lIHZpZXdwb3J0LlxuICAgICAqIEBwYXJhbSByaWdodCAtIFJpZ2h0IHNpZGUgb2YgdGhlIG5lYXIgY2xpcHBpbmcgcGxhbmUgdmlld3BvcnQuXG4gICAgICogQHBhcmFtIGJvdHRvbSAtIEJvdHRvbSBvZiB0aGUgbmVhciBjbGlwcGluZyBwbGFuZSB2aWV3cG9ydC5cbiAgICAgKiBAcGFyYW0gdG9wIC0gVG9wIG9mIHRoZSBuZWFyIGNsaXBwaW5nIHBsYW5lIHZpZXdwb3J0LlxuICAgICAqIEBwYXJhbSBuZWFyIC0gVGhlIGRlcHRoIChuZWdhdGl2ZSB6IGNvb3JkaW5hdGUpXG4gICAgICogICAgIG9mIHRoZSBuZWFyIGNsaXBwaW5nIHBsYW5lLlxuICAgICAqIEBwYXJhbSBmYXIgLSBUaGUgZGVwdGggKG5lZ2F0aXZlIHogY29vcmRpbmF0ZSlcbiAgICAgKiAgICAgb2YgdGhlIGZhciBjbGlwcGluZyBwbGFuZS5cbiAgICAgKiBAcGFyYW0gZHN0IC0gT3V0cHV0IG1hdHJpeC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgb3J0aG9ncmFwaGljIHByb2plY3Rpb24gbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG9ydGhvKGxlZnQsIHJpZ2h0LCBib3R0b20sIHRvcCwgbmVhciwgZmFyLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBuZXdEc3RbMF0gPSAyIC8gKHJpZ2h0IC0gbGVmdCk7XG4gICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgIG5ld0RzdFsyXSA9IDA7XG4gICAgICAgIG5ld0RzdFszXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IDA7XG4gICAgICAgIG5ld0RzdFs1XSA9IDIgLyAodG9wIC0gYm90dG9tKTtcbiAgICAgICAgbmV3RHN0WzZdID0gMDtcbiAgICAgICAgbmV3RHN0WzddID0gMDtcbiAgICAgICAgbmV3RHN0WzhdID0gMDtcbiAgICAgICAgbmV3RHN0WzldID0gMDtcbiAgICAgICAgbmV3RHN0WzEwXSA9IDEgLyAobmVhciAtIGZhcik7XG4gICAgICAgIG5ld0RzdFsxMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTJdID0gKHJpZ2h0ICsgbGVmdCkgLyAobGVmdCAtIHJpZ2h0KTtcbiAgICAgICAgbmV3RHN0WzEzXSA9ICh0b3AgKyBib3R0b20pIC8gKGJvdHRvbSAtIHRvcCk7XG4gICAgICAgIG5ld0RzdFsxNF0gPSBuZWFyIC8gKG5lYXIgLSBmYXIpO1xuICAgICAgICBuZXdEc3RbMTVdID0gMTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgYSA0LWJ5LTQgcGVyc3BlY3RpdmUgdHJhbnNmb3JtYXRpb24gbWF0cml4IGdpdmVuIHRoZSBsZWZ0LCByaWdodCxcbiAgICAgKiB0b3AsIGJvdHRvbSwgbmVhciBhbmQgZmFyIGNsaXBwaW5nIHBsYW5lcy4gVGhlIGFyZ3VtZW50cyBkZWZpbmUgYSBmcnVzdHVtXG4gICAgICogZXh0ZW5kaW5nIGluIHRoZSBuZWdhdGl2ZSB6IGRpcmVjdGlvbi4gVGhlIGFyZ3VtZW50cyBuZWFyIGFuZCBmYXIgYXJlIHRoZVxuICAgICAqIGRpc3RhbmNlcyB0byB0aGUgbmVhciBhbmQgZmFyIGNsaXBwaW5nIHBsYW5lcy4gTm90ZSB0aGF0IG5lYXIgYW5kIGZhciBhcmUgbm90XG4gICAgICogeiBjb29yZGluYXRlcywgYnV0IHJhdGhlciB0aGV5IGFyZSBkaXN0YW5jZXMgYWxvbmcgdGhlIG5lZ2F0aXZlIHotYXhpcy4gVGhlXG4gICAgICogbWF0cml4IGdlbmVyYXRlZCBzZW5kcyB0aGUgdmlld2luZyBmcnVzdHVtIHRvIHRoZSB1bml0IGJveC4gV2UgYXNzdW1lIGEgdW5pdFxuICAgICAqIGJveCBleHRlbmRpbmcgZnJvbSAtMSB0byAxIGluIHRoZSB4IGFuZCB5IGRpbWVuc2lvbnMgYW5kIGZyb20gMCB0byAxIGluIHRoZSB6XG4gICAgICogZGltZW5zaW9uLlxuICAgICAqIEBwYXJhbSBsZWZ0IC0gVGhlIHggY29vcmRpbmF0ZSBvZiB0aGUgbGVmdCBwbGFuZSBvZiB0aGUgYm94LlxuICAgICAqIEBwYXJhbSByaWdodCAtIFRoZSB4IGNvb3JkaW5hdGUgb2YgdGhlIHJpZ2h0IHBsYW5lIG9mIHRoZSBib3guXG4gICAgICogQHBhcmFtIGJvdHRvbSAtIFRoZSB5IGNvb3JkaW5hdGUgb2YgdGhlIGJvdHRvbSBwbGFuZSBvZiB0aGUgYm94LlxuICAgICAqIEBwYXJhbSB0b3AgLSBUaGUgeSBjb29yZGluYXRlIG9mIHRoZSByaWdodCBwbGFuZSBvZiB0aGUgYm94LlxuICAgICAqIEBwYXJhbSBuZWFyIC0gVGhlIG5lZ2F0aXZlIHogY29vcmRpbmF0ZSBvZiB0aGUgbmVhciBwbGFuZSBvZiB0aGUgYm94LlxuICAgICAqIEBwYXJhbSBmYXIgLSBUaGUgbmVnYXRpdmUgeiBjb29yZGluYXRlIG9mIHRoZSBmYXIgcGxhbmUgb2YgdGhlIGJveC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gT3V0cHV0IG1hdHJpeC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgcGVyc3BlY3RpdmUgcHJvamVjdGlvbiBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gZnJ1c3R1bShsZWZ0LCByaWdodCwgYm90dG9tLCB0b3AsIG5lYXIsIGZhciwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgY29uc3QgZHggPSAocmlnaHQgLSBsZWZ0KTtcbiAgICAgICAgY29uc3QgZHkgPSAodG9wIC0gYm90dG9tKTtcbiAgICAgICAgY29uc3QgZHogPSAobmVhciAtIGZhcik7XG4gICAgICAgIG5ld0RzdFswXSA9IDIgKiBuZWFyIC8gZHg7XG4gICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgIG5ld0RzdFsyXSA9IDA7XG4gICAgICAgIG5ld0RzdFszXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IDA7XG4gICAgICAgIG5ld0RzdFs1XSA9IDIgKiBuZWFyIC8gZHk7XG4gICAgICAgIG5ld0RzdFs2XSA9IDA7XG4gICAgICAgIG5ld0RzdFs3XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IChsZWZ0ICsgcmlnaHQpIC8gZHg7XG4gICAgICAgIG5ld0RzdFs5XSA9ICh0b3AgKyBib3R0b20pIC8gZHk7XG4gICAgICAgIG5ld0RzdFsxMF0gPSBmYXIgLyBkejtcbiAgICAgICAgbmV3RHN0WzExXSA9IC0xO1xuICAgICAgICBuZXdEc3RbMTJdID0gMDtcbiAgICAgICAgbmV3RHN0WzEzXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxNF0gPSBuZWFyICogZmFyIC8gZHo7XG4gICAgICAgIG5ld0RzdFsxNV0gPSAwO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyBhIDQtYnktNCByZXZlcnNlLXogcGVyc3BlY3RpdmUgdHJhbnNmb3JtYXRpb24gbWF0cml4IGdpdmVuIHRoZSBsZWZ0LCByaWdodCxcbiAgICAgKiB0b3AsIGJvdHRvbSwgbmVhciBhbmQgZmFyIGNsaXBwaW5nIHBsYW5lcy4gVGhlIGFyZ3VtZW50cyBkZWZpbmUgYSBmcnVzdHVtXG4gICAgICogZXh0ZW5kaW5nIGluIHRoZSBuZWdhdGl2ZSB6IGRpcmVjdGlvbi4gVGhlIGFyZ3VtZW50cyBuZWFyIGFuZCBmYXIgYXJlIHRoZVxuICAgICAqIGRpc3RhbmNlcyB0byB0aGUgbmVhciBhbmQgZmFyIGNsaXBwaW5nIHBsYW5lcy4gTm90ZSB0aGF0IG5lYXIgYW5kIGZhciBhcmUgbm90XG4gICAgICogeiBjb29yZGluYXRlcywgYnV0IHJhdGhlciB0aGV5IGFyZSBkaXN0YW5jZXMgYWxvbmcgdGhlIG5lZ2F0aXZlIHotYXhpcy4gVGhlXG4gICAgICogbWF0cml4IGdlbmVyYXRlZCBzZW5kcyB0aGUgdmlld2luZyBmcnVzdHVtIHRvIHRoZSB1bml0IGJveC4gV2UgYXNzdW1lIGEgdW5pdFxuICAgICAqIGJveCBleHRlbmRpbmcgZnJvbSAtMSB0byAxIGluIHRoZSB4IGFuZCB5IGRpbWVuc2lvbnMgYW5kIGZyb20gMSAoLW5lYXIpIHRvIDAgKC1mYXIpIGluIHRoZSB6XG4gICAgICogZGltZW5zaW9uLlxuICAgICAqIEBwYXJhbSBsZWZ0IC0gVGhlIHggY29vcmRpbmF0ZSBvZiB0aGUgbGVmdCBwbGFuZSBvZiB0aGUgYm94LlxuICAgICAqIEBwYXJhbSByaWdodCAtIFRoZSB4IGNvb3JkaW5hdGUgb2YgdGhlIHJpZ2h0IHBsYW5lIG9mIHRoZSBib3guXG4gICAgICogQHBhcmFtIGJvdHRvbSAtIFRoZSB5IGNvb3JkaW5hdGUgb2YgdGhlIGJvdHRvbSBwbGFuZSBvZiB0aGUgYm94LlxuICAgICAqIEBwYXJhbSB0b3AgLSBUaGUgeSBjb29yZGluYXRlIG9mIHRoZSByaWdodCBwbGFuZSBvZiB0aGUgYm94LlxuICAgICAqIEBwYXJhbSBuZWFyIC0gVGhlIG5lZ2F0aXZlIHogY29vcmRpbmF0ZSBvZiB0aGUgbmVhciBwbGFuZSBvZiB0aGUgYm94LlxuICAgICAqIEBwYXJhbSBmYXIgLSBUaGUgbmVnYXRpdmUgeiBjb29yZGluYXRlIG9mIHRoZSBmYXIgcGxhbmUgb2YgdGhlIGJveC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gT3V0cHV0IG1hdHJpeC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgcGVyc3BlY3RpdmUgcHJvamVjdGlvbiBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gZnJ1c3R1bVJldmVyc2VaKGxlZnQsIHJpZ2h0LCBib3R0b20sIHRvcCwgbmVhciwgZmFyID0gSW5maW5pdHksIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDE2KSk7XG4gICAgICAgIGNvbnN0IGR4ID0gKHJpZ2h0IC0gbGVmdCk7XG4gICAgICAgIGNvbnN0IGR5ID0gKHRvcCAtIGJvdHRvbSk7XG4gICAgICAgIG5ld0RzdFswXSA9IDIgKiBuZWFyIC8gZHg7XG4gICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgIG5ld0RzdFsyXSA9IDA7XG4gICAgICAgIG5ld0RzdFszXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IDA7XG4gICAgICAgIG5ld0RzdFs1XSA9IDIgKiBuZWFyIC8gZHk7XG4gICAgICAgIG5ld0RzdFs2XSA9IDA7XG4gICAgICAgIG5ld0RzdFs3XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IChsZWZ0ICsgcmlnaHQpIC8gZHg7XG4gICAgICAgIG5ld0RzdFs5XSA9ICh0b3AgKyBib3R0b20pIC8gZHk7XG4gICAgICAgIG5ld0RzdFsxMV0gPSAtMTtcbiAgICAgICAgbmV3RHN0WzEyXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxM10gPSAwO1xuICAgICAgICBuZXdEc3RbMTVdID0gMDtcbiAgICAgICAgaWYgKGZhciA9PT0gSW5maW5pdHkpIHtcbiAgICAgICAgICAgIG5ld0RzdFsxMF0gPSAwO1xuICAgICAgICAgICAgbmV3RHN0WzE0XSA9IG5lYXI7XG4gICAgICAgIH1cbiAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICBjb25zdCByYW5nZUludiA9IDEgLyAoZmFyIC0gbmVhcik7XG4gICAgICAgICAgICBuZXdEc3RbMTBdID0gbmVhciAqIHJhbmdlSW52O1xuICAgICAgICAgICAgbmV3RHN0WzE0XSA9IGZhciAqIG5lYXIgKiByYW5nZUludjtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICBjb25zdCB4QXhpcyA9IHZlYzMuY3JlYXRlKCk7XG4gICAgY29uc3QgeUF4aXMgPSB2ZWMzLmNyZWF0ZSgpO1xuICAgIGNvbnN0IHpBeGlzID0gdmVjMy5jcmVhdGUoKTtcbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyBhIDQtYnktNCBhaW0gdHJhbnNmb3JtYXRpb24uXG4gICAgICpcbiAgICAgKiBUaGlzIGlzIGEgbWF0cml4IHdoaWNoIHBvc2l0aW9ucyBhbiBvYmplY3QgYWltaW5nIGRvd24gcG9zaXRpdmUgWi5cbiAgICAgKiB0b3dhcmQgdGhlIHRhcmdldC5cbiAgICAgKlxuICAgICAqIE5vdGU6IHRoaXMgaXMgKipOT1QqKiB0aGUgaW52ZXJzZSBvZiBsb29rQXQgYXMgbG9va0F0IGxvb2tzIGF0IG5lZ2F0aXZlIFouXG4gICAgICpcbiAgICAgKiBAcGFyYW0gcG9zaXRpb24gLSBUaGUgcG9zaXRpb24gb2YgdGhlIG9iamVjdC5cbiAgICAgKiBAcGFyYW0gdGFyZ2V0IC0gVGhlIHBvc2l0aW9uIG1lYW50IHRvIGJlIGFpbWVkIGF0LlxuICAgICAqIEBwYXJhbSB1cCAtIEEgdmVjdG9yIHBvaW50aW5nIHVwLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIGFpbSBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gYWltKHBvc2l0aW9uLCB0YXJnZXQsIHVwLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICB2ZWMzLm5vcm1hbGl6ZSh2ZWMzLnN1YnRyYWN0KHRhcmdldCwgcG9zaXRpb24sIHpBeGlzKSwgekF4aXMpO1xuICAgICAgICB2ZWMzLm5vcm1hbGl6ZSh2ZWMzLmNyb3NzKHVwLCB6QXhpcywgeEF4aXMpLCB4QXhpcyk7XG4gICAgICAgIHZlYzMubm9ybWFsaXplKHZlYzMuY3Jvc3MoekF4aXMsIHhBeGlzLCB5QXhpcyksIHlBeGlzKTtcbiAgICAgICAgbmV3RHN0WzBdID0geEF4aXNbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IHhBeGlzWzFdO1xuICAgICAgICBuZXdEc3RbMl0gPSB4QXhpc1syXTtcbiAgICAgICAgbmV3RHN0WzNdID0gMDtcbiAgICAgICAgbmV3RHN0WzRdID0geUF4aXNbMF07XG4gICAgICAgIG5ld0RzdFs1XSA9IHlBeGlzWzFdO1xuICAgICAgICBuZXdEc3RbNl0gPSB5QXhpc1syXTtcbiAgICAgICAgbmV3RHN0WzddID0gMDtcbiAgICAgICAgbmV3RHN0WzhdID0gekF4aXNbMF07XG4gICAgICAgIG5ld0RzdFs5XSA9IHpBeGlzWzFdO1xuICAgICAgICBuZXdEc3RbMTBdID0gekF4aXNbMl07XG4gICAgICAgIG5ld0RzdFsxMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTJdID0gcG9zaXRpb25bMF07XG4gICAgICAgIG5ld0RzdFsxM10gPSBwb3NpdGlvblsxXTtcbiAgICAgICAgbmV3RHN0WzE0XSA9IHBvc2l0aW9uWzJdO1xuICAgICAgICBuZXdEc3RbMTVdID0gMTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgYSA0LWJ5LTQgY2FtZXJhIGFpbSB0cmFuc2Zvcm1hdGlvbi5cbiAgICAgKlxuICAgICAqIFRoaXMgaXMgYSBtYXRyaXggd2hpY2ggcG9zaXRpb25zIGFuIG9iamVjdCBhaW1pbmcgZG93biBuZWdhdGl2ZSBaLlxuICAgICAqIHRvd2FyZCB0aGUgdGFyZ2V0LlxuICAgICAqXG4gICAgICogTm90ZTogdGhpcyBpcyB0aGUgaW52ZXJzZSBvZiBgbG9va0F0YFxuICAgICAqXG4gICAgICogQHBhcmFtIGV5ZSAtIFRoZSBwb3NpdGlvbiBvZiB0aGUgb2JqZWN0LlxuICAgICAqIEBwYXJhbSB0YXJnZXQgLSBUaGUgcG9zaXRpb24gbWVhbnQgdG8gYmUgYWltZWQgYXQuXG4gICAgICogQHBhcmFtIHVwIC0gQSB2ZWN0b3IgcG9pbnRpbmcgdXAuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgYWltIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBjYW1lcmFBaW0oZXllLCB0YXJnZXQsIHVwLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICB2ZWMzLm5vcm1hbGl6ZSh2ZWMzLnN1YnRyYWN0KGV5ZSwgdGFyZ2V0LCB6QXhpcyksIHpBeGlzKTtcbiAgICAgICAgdmVjMy5ub3JtYWxpemUodmVjMy5jcm9zcyh1cCwgekF4aXMsIHhBeGlzKSwgeEF4aXMpO1xuICAgICAgICB2ZWMzLm5vcm1hbGl6ZSh2ZWMzLmNyb3NzKHpBeGlzLCB4QXhpcywgeUF4aXMpLCB5QXhpcyk7XG4gICAgICAgIG5ld0RzdFswXSA9IHhBeGlzWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSB4QXhpc1sxXTtcbiAgICAgICAgbmV3RHN0WzJdID0geEF4aXNbMl07XG4gICAgICAgIG5ld0RzdFszXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IHlBeGlzWzBdO1xuICAgICAgICBuZXdEc3RbNV0gPSB5QXhpc1sxXTtcbiAgICAgICAgbmV3RHN0WzZdID0geUF4aXNbMl07XG4gICAgICAgIG5ld0RzdFs3XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IHpBeGlzWzBdO1xuICAgICAgICBuZXdEc3RbOV0gPSB6QXhpc1sxXTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IHpBeGlzWzJdO1xuICAgICAgICBuZXdEc3RbMTFdID0gMDtcbiAgICAgICAgbmV3RHN0WzEyXSA9IGV5ZVswXTtcbiAgICAgICAgbmV3RHN0WzEzXSA9IGV5ZVsxXTtcbiAgICAgICAgbmV3RHN0WzE0XSA9IGV5ZVsyXTtcbiAgICAgICAgbmV3RHN0WzE1XSA9IDE7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIGEgNC1ieS00IHZpZXcgdHJhbnNmb3JtYXRpb24uXG4gICAgICpcbiAgICAgKiBUaGlzIGlzIGEgdmlldyBtYXRyaXggd2hpY2ggdHJhbnNmb3JtcyBhbGwgb3RoZXIgb2JqZWN0c1xuICAgICAqIHRvIGJlIGluIHRoZSBzcGFjZSBvZiB0aGUgdmlldyBkZWZpbmVkIGJ5IHRoZSBwYXJhbWV0ZXJzLlxuICAgICAqXG4gICAgICogQHBhcmFtIGV5ZSAtIFRoZSBwb3NpdGlvbiBvZiB0aGUgb2JqZWN0LlxuICAgICAqIEBwYXJhbSB0YXJnZXQgLSBUaGUgcG9zaXRpb24gbWVhbnQgdG8gYmUgYWltZWQgYXQuXG4gICAgICogQHBhcmFtIHVwIC0gQSB2ZWN0b3IgcG9pbnRpbmcgdXAuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgbG9vay1hdCBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gbG9va0F0KGV5ZSwgdGFyZ2V0LCB1cCwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgdmVjMy5ub3JtYWxpemUodmVjMy5zdWJ0cmFjdChleWUsIHRhcmdldCwgekF4aXMpLCB6QXhpcyk7XG4gICAgICAgIHZlYzMubm9ybWFsaXplKHZlYzMuY3Jvc3ModXAsIHpBeGlzLCB4QXhpcyksIHhBeGlzKTtcbiAgICAgICAgdmVjMy5ub3JtYWxpemUodmVjMy5jcm9zcyh6QXhpcywgeEF4aXMsIHlBeGlzKSwgeUF4aXMpO1xuICAgICAgICBuZXdEc3RbMF0gPSB4QXhpc1swXTtcbiAgICAgICAgbmV3RHN0WzFdID0geUF4aXNbMF07XG4gICAgICAgIG5ld0RzdFsyXSA9IHpBeGlzWzBdO1xuICAgICAgICBuZXdEc3RbM10gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSB4QXhpc1sxXTtcbiAgICAgICAgbmV3RHN0WzVdID0geUF4aXNbMV07XG4gICAgICAgIG5ld0RzdFs2XSA9IHpBeGlzWzFdO1xuICAgICAgICBuZXdEc3RbN10gPSAwO1xuICAgICAgICBuZXdEc3RbOF0gPSB4QXhpc1syXTtcbiAgICAgICAgbmV3RHN0WzldID0geUF4aXNbMl07XG4gICAgICAgIG5ld0RzdFsxMF0gPSB6QXhpc1syXTtcbiAgICAgICAgbmV3RHN0WzExXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMl0gPSAtKHhBeGlzWzBdICogZXllWzBdICsgeEF4aXNbMV0gKiBleWVbMV0gKyB4QXhpc1syXSAqIGV5ZVsyXSk7XG4gICAgICAgIG5ld0RzdFsxM10gPSAtKHlBeGlzWzBdICogZXllWzBdICsgeUF4aXNbMV0gKiBleWVbMV0gKyB5QXhpc1syXSAqIGV5ZVsyXSk7XG4gICAgICAgIG5ld0RzdFsxNF0gPSAtKHpBeGlzWzBdICogZXllWzBdICsgekF4aXNbMV0gKiBleWVbMV0gKyB6QXhpc1syXSAqIGV5ZVsyXSk7XG4gICAgICAgIG5ld0RzdFsxNV0gPSAxO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDcmVhdGVzIGEgNC1ieS00IG1hdHJpeCB3aGljaCB0cmFuc2xhdGVzIGJ5IHRoZSBnaXZlbiB2ZWN0b3Igdi5cbiAgICAgKiBAcGFyYW0gdiAtIFRoZSB2ZWN0b3IgYnlcbiAgICAgKiAgICAgd2hpY2ggdG8gdHJhbnNsYXRlLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHRyYW5zbGF0aW9uIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiB0cmFuc2xhdGlvbih2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBuZXdEc3RbMF0gPSAxO1xuICAgICAgICBuZXdEc3RbMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICBuZXdEc3RbM10gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSAwO1xuICAgICAgICBuZXdEc3RbNV0gPSAxO1xuICAgICAgICBuZXdEc3RbNl0gPSAwO1xuICAgICAgICBuZXdEc3RbN10gPSAwO1xuICAgICAgICBuZXdEc3RbOF0gPSAwO1xuICAgICAgICBuZXdEc3RbOV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTBdID0gMTtcbiAgICAgICAgbmV3RHN0WzExXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMl0gPSB2WzBdO1xuICAgICAgICBuZXdEc3RbMTNdID0gdlsxXTtcbiAgICAgICAgbmV3RHN0WzE0XSA9IHZbMl07XG4gICAgICAgIG5ld0RzdFsxNV0gPSAxO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBUcmFuc2xhdGVzIHRoZSBnaXZlbiA0LWJ5LTQgbWF0cml4IGJ5IHRoZSBnaXZlbiB2ZWN0b3Igdi5cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yIGJ5XG4gICAgICogICAgIHdoaWNoIHRvIHRyYW5zbGF0ZS5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSB0cmFuc2xhdGVkIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiB0cmFuc2xhdGUobSwgdiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgY29uc3QgdjAgPSB2WzBdO1xuICAgICAgICBjb25zdCB2MSA9IHZbMV07XG4gICAgICAgIGNvbnN0IHYyID0gdlsyXTtcbiAgICAgICAgY29uc3QgbTAwID0gbVswXTtcbiAgICAgICAgY29uc3QgbTAxID0gbVsxXTtcbiAgICAgICAgY29uc3QgbTAyID0gbVsyXTtcbiAgICAgICAgY29uc3QgbTAzID0gbVszXTtcbiAgICAgICAgY29uc3QgbTEwID0gbVsxICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMTEgPSBtWzEgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0xMiA9IG1bMSAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTEzID0gbVsxICogNCArIDNdO1xuICAgICAgICBjb25zdCBtMjAgPSBtWzIgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0yMSA9IG1bMiAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTIyID0gbVsyICogNCArIDJdO1xuICAgICAgICBjb25zdCBtMjMgPSBtWzIgKiA0ICsgM107XG4gICAgICAgIGNvbnN0IG0zMCA9IG1bMyAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTMxID0gbVszICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMzIgPSBtWzMgKiA0ICsgMl07XG4gICAgICAgIGNvbnN0IG0zMyA9IG1bMyAqIDQgKyAzXTtcbiAgICAgICAgaWYgKG0gIT09IG5ld0RzdCkge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0gbTAwO1xuICAgICAgICAgICAgbmV3RHN0WzFdID0gbTAxO1xuICAgICAgICAgICAgbmV3RHN0WzJdID0gbTAyO1xuICAgICAgICAgICAgbmV3RHN0WzNdID0gbTAzO1xuICAgICAgICAgICAgbmV3RHN0WzRdID0gbTEwO1xuICAgICAgICAgICAgbmV3RHN0WzVdID0gbTExO1xuICAgICAgICAgICAgbmV3RHN0WzZdID0gbTEyO1xuICAgICAgICAgICAgbmV3RHN0WzddID0gbTEzO1xuICAgICAgICAgICAgbmV3RHN0WzhdID0gbTIwO1xuICAgICAgICAgICAgbmV3RHN0WzldID0gbTIxO1xuICAgICAgICAgICAgbmV3RHN0WzEwXSA9IG0yMjtcbiAgICAgICAgICAgIG5ld0RzdFsxMV0gPSBtMjM7XG4gICAgICAgIH1cbiAgICAgICAgbmV3RHN0WzEyXSA9IG0wMCAqIHYwICsgbTEwICogdjEgKyBtMjAgKiB2MiArIG0zMDtcbiAgICAgICAgbmV3RHN0WzEzXSA9IG0wMSAqIHYwICsgbTExICogdjEgKyBtMjEgKiB2MiArIG0zMTtcbiAgICAgICAgbmV3RHN0WzE0XSA9IG0wMiAqIHYwICsgbTEyICogdjEgKyBtMjIgKiB2MiArIG0zMjtcbiAgICAgICAgbmV3RHN0WzE1XSA9IG0wMyAqIHYwICsgbTEzICogdjEgKyBtMjMgKiB2MiArIG0zMztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIDQtYnktNCBtYXRyaXggd2hpY2ggcm90YXRlcyBhcm91bmQgdGhlIHgtYXhpcyBieSB0aGUgZ2l2ZW4gYW5nbGUuXG4gICAgICogQHBhcmFtIGFuZ2xlSW5SYWRpYW5zIC0gVGhlIGFuZ2xlIGJ5IHdoaWNoIHRvIHJvdGF0ZSAoaW4gcmFkaWFucykuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgcm90YXRpb24gbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0aW9uWChhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgY29uc3QgYyA9IE1hdGguY29zKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgY29uc3QgcyA9IE1hdGguc2luKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgbmV3RHN0WzBdID0gMTtcbiAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgbmV3RHN0WzJdID0gMDtcbiAgICAgICAgbmV3RHN0WzNdID0gMDtcbiAgICAgICAgbmV3RHN0WzRdID0gMDtcbiAgICAgICAgbmV3RHN0WzVdID0gYztcbiAgICAgICAgbmV3RHN0WzZdID0gcztcbiAgICAgICAgbmV3RHN0WzddID0gMDtcbiAgICAgICAgbmV3RHN0WzhdID0gMDtcbiAgICAgICAgbmV3RHN0WzldID0gLXM7XG4gICAgICAgIG5ld0RzdFsxMF0gPSBjO1xuICAgICAgICBuZXdEc3RbMTFdID0gMDtcbiAgICAgICAgbmV3RHN0WzEyXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxM10gPSAwO1xuICAgICAgICBuZXdEc3RbMTRdID0gMDtcbiAgICAgICAgbmV3RHN0WzE1XSA9IDE7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJvdGF0ZXMgdGhlIGdpdmVuIDQtYnktNCBtYXRyaXggYXJvdW5kIHRoZSB4LWF4aXMgYnkgdGhlIGdpdmVuXG4gICAgICogYW5nbGUuXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIFRoZSBhbmdsZSBieSB3aGljaCB0byByb3RhdGUgKGluIHJhZGlhbnMpLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHJvdGF0ZWQgbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0ZVgobSwgYW5nbGVJblJhZGlhbnMsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDE2KSk7XG4gICAgICAgIGNvbnN0IG0xMCA9IG1bNF07XG4gICAgICAgIGNvbnN0IG0xMSA9IG1bNV07XG4gICAgICAgIGNvbnN0IG0xMiA9IG1bNl07XG4gICAgICAgIGNvbnN0IG0xMyA9IG1bN107XG4gICAgICAgIGNvbnN0IG0yMCA9IG1bOF07XG4gICAgICAgIGNvbnN0IG0yMSA9IG1bOV07XG4gICAgICAgIGNvbnN0IG0yMiA9IG1bMTBdO1xuICAgICAgICBjb25zdCBtMjMgPSBtWzExXTtcbiAgICAgICAgY29uc3QgYyA9IE1hdGguY29zKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgY29uc3QgcyA9IE1hdGguc2luKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgbmV3RHN0WzRdID0gYyAqIG0xMCArIHMgKiBtMjA7XG4gICAgICAgIG5ld0RzdFs1XSA9IGMgKiBtMTEgKyBzICogbTIxO1xuICAgICAgICBuZXdEc3RbNl0gPSBjICogbTEyICsgcyAqIG0yMjtcbiAgICAgICAgbmV3RHN0WzddID0gYyAqIG0xMyArIHMgKiBtMjM7XG4gICAgICAgIG5ld0RzdFs4XSA9IGMgKiBtMjAgLSBzICogbTEwO1xuICAgICAgICBuZXdEc3RbOV0gPSBjICogbTIxIC0gcyAqIG0xMTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IGMgKiBtMjIgLSBzICogbTEyO1xuICAgICAgICBuZXdEc3RbMTFdID0gYyAqIG0yMyAtIHMgKiBtMTM7XG4gICAgICAgIGlmIChtICE9PSBuZXdEc3QpIHtcbiAgICAgICAgICAgIG5ld0RzdFswXSA9IG1bMF07XG4gICAgICAgICAgICBuZXdEc3RbMV0gPSBtWzFdO1xuICAgICAgICAgICAgbmV3RHN0WzJdID0gbVsyXTtcbiAgICAgICAgICAgIG5ld0RzdFszXSA9IG1bM107XG4gICAgICAgICAgICBuZXdEc3RbMTJdID0gbVsxMl07XG4gICAgICAgICAgICBuZXdEc3RbMTNdID0gbVsxM107XG4gICAgICAgICAgICBuZXdEc3RbMTRdID0gbVsxNF07XG4gICAgICAgICAgICBuZXdEc3RbMTVdID0gbVsxNV07XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIDQtYnktNCBtYXRyaXggd2hpY2ggcm90YXRlcyBhcm91bmQgdGhlIHktYXhpcyBieSB0aGUgZ2l2ZW4gYW5nbGUuXG4gICAgICogQHBhcmFtIGFuZ2xlSW5SYWRpYW5zIC0gVGhlIGFuZ2xlIGJ5IHdoaWNoIHRvIHJvdGF0ZSAoaW4gcmFkaWFucykuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgcm90YXRpb24gbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0aW9uWShhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgY29uc3QgYyA9IE1hdGguY29zKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgY29uc3QgcyA9IE1hdGguc2luKGFuZ2xlSW5SYWRpYW5zKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYztcbiAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgbmV3RHN0WzJdID0gLXM7XG4gICAgICAgIG5ld0RzdFszXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IDA7XG4gICAgICAgIG5ld0RzdFs1XSA9IDE7XG4gICAgICAgIG5ld0RzdFs2XSA9IDA7XG4gICAgICAgIG5ld0RzdFs3XSA9IDA7XG4gICAgICAgIG5ld0RzdFs4XSA9IHM7XG4gICAgICAgIG5ld0RzdFs5XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMF0gPSBjO1xuICAgICAgICBuZXdEc3RbMTFdID0gMDtcbiAgICAgICAgbmV3RHN0WzEyXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxM10gPSAwO1xuICAgICAgICBuZXdEc3RbMTRdID0gMDtcbiAgICAgICAgbmV3RHN0WzE1XSA9IDE7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJvdGF0ZXMgdGhlIGdpdmVuIDQtYnktNCBtYXRyaXggYXJvdW5kIHRoZSB5LWF4aXMgYnkgdGhlIGdpdmVuXG4gICAgICogYW5nbGUuXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIFRoZSBhbmdsZSBieSB3aGljaCB0byByb3RhdGUgKGluIHJhZGlhbnMpLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHJvdGF0ZWQgbWF0cml4LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0ZVkobSwgYW5nbGVJblJhZGlhbnMsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDE2KSk7XG4gICAgICAgIGNvbnN0IG0wMCA9IG1bMCAqIDQgKyAwXTtcbiAgICAgICAgY29uc3QgbTAxID0gbVswICogNCArIDFdO1xuICAgICAgICBjb25zdCBtMDIgPSBtWzAgKiA0ICsgMl07XG4gICAgICAgIGNvbnN0IG0wMyA9IG1bMCAqIDQgKyAzXTtcbiAgICAgICAgY29uc3QgbTIwID0gbVsyICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMjEgPSBtWzIgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0yMiA9IG1bMiAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTIzID0gbVsyICogNCArIDNdO1xuICAgICAgICBjb25zdCBjID0gTWF0aC5jb3MoYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBjb25zdCBzID0gTWF0aC5zaW4oYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBuZXdEc3RbMF0gPSBjICogbTAwIC0gcyAqIG0yMDtcbiAgICAgICAgbmV3RHN0WzFdID0gYyAqIG0wMSAtIHMgKiBtMjE7XG4gICAgICAgIG5ld0RzdFsyXSA9IGMgKiBtMDIgLSBzICogbTIyO1xuICAgICAgICBuZXdEc3RbM10gPSBjICogbTAzIC0gcyAqIG0yMztcbiAgICAgICAgbmV3RHN0WzhdID0gYyAqIG0yMCArIHMgKiBtMDA7XG4gICAgICAgIG5ld0RzdFs5XSA9IGMgKiBtMjEgKyBzICogbTAxO1xuICAgICAgICBuZXdEc3RbMTBdID0gYyAqIG0yMiArIHMgKiBtMDI7XG4gICAgICAgIG5ld0RzdFsxMV0gPSBjICogbTIzICsgcyAqIG0wMztcbiAgICAgICAgaWYgKG0gIT09IG5ld0RzdCkge1xuICAgICAgICAgICAgbmV3RHN0WzRdID0gbVs0XTtcbiAgICAgICAgICAgIG5ld0RzdFs1XSA9IG1bNV07XG4gICAgICAgICAgICBuZXdEc3RbNl0gPSBtWzZdO1xuICAgICAgICAgICAgbmV3RHN0WzddID0gbVs3XTtcbiAgICAgICAgICAgIG5ld0RzdFsxMl0gPSBtWzEyXTtcbiAgICAgICAgICAgIG5ld0RzdFsxM10gPSBtWzEzXTtcbiAgICAgICAgICAgIG5ld0RzdFsxNF0gPSBtWzE0XTtcbiAgICAgICAgICAgIG5ld0RzdFsxNV0gPSBtWzE1XTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDcmVhdGVzIGEgNC1ieS00IG1hdHJpeCB3aGljaCByb3RhdGVzIGFyb3VuZCB0aGUgei1heGlzIGJ5IHRoZSBnaXZlbiBhbmdsZS5cbiAgICAgKiBAcGFyYW0gYW5nbGVJblJhZGlhbnMgLSBUaGUgYW5nbGUgYnkgd2hpY2ggdG8gcm90YXRlIChpbiByYWRpYW5zKS5cbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSByb3RhdGlvbiBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gcm90YXRpb25aKGFuZ2xlSW5SYWRpYW5zLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBjb25zdCBjID0gTWF0aC5jb3MoYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBjb25zdCBzID0gTWF0aC5zaW4oYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBuZXdEc3RbMF0gPSBjO1xuICAgICAgICBuZXdEc3RbMV0gPSBzO1xuICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICBuZXdEc3RbM10gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSAtcztcbiAgICAgICAgbmV3RHN0WzVdID0gYztcbiAgICAgICAgbmV3RHN0WzZdID0gMDtcbiAgICAgICAgbmV3RHN0WzddID0gMDtcbiAgICAgICAgbmV3RHN0WzhdID0gMDtcbiAgICAgICAgbmV3RHN0WzldID0gMDtcbiAgICAgICAgbmV3RHN0WzEwXSA9IDE7XG4gICAgICAgIG5ld0RzdFsxMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTJdID0gMDtcbiAgICAgICAgbmV3RHN0WzEzXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxNF0gPSAwO1xuICAgICAgICBuZXdEc3RbMTVdID0gMTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUm90YXRlcyB0aGUgZ2l2ZW4gNC1ieS00IG1hdHJpeCBhcm91bmQgdGhlIHotYXhpcyBieSB0aGUgZ2l2ZW5cbiAgICAgKiBhbmdsZS5cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGFuZ2xlSW5SYWRpYW5zIC0gVGhlIGFuZ2xlIGJ5IHdoaWNoIHRvIHJvdGF0ZSAoaW4gcmFkaWFucykuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgcm90YXRlZCBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gcm90YXRlWihtLCBhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgY29uc3QgbTAwID0gbVswICogNCArIDBdO1xuICAgICAgICBjb25zdCBtMDEgPSBtWzAgKiA0ICsgMV07XG4gICAgICAgIGNvbnN0IG0wMiA9IG1bMCAqIDQgKyAyXTtcbiAgICAgICAgY29uc3QgbTAzID0gbVswICogNCArIDNdO1xuICAgICAgICBjb25zdCBtMTAgPSBtWzEgKiA0ICsgMF07XG4gICAgICAgIGNvbnN0IG0xMSA9IG1bMSAqIDQgKyAxXTtcbiAgICAgICAgY29uc3QgbTEyID0gbVsxICogNCArIDJdO1xuICAgICAgICBjb25zdCBtMTMgPSBtWzEgKiA0ICsgM107XG4gICAgICAgIGNvbnN0IGMgPSBNYXRoLmNvcyhhbmdsZUluUmFkaWFucyk7XG4gICAgICAgIGNvbnN0IHMgPSBNYXRoLnNpbihhbmdsZUluUmFkaWFucyk7XG4gICAgICAgIG5ld0RzdFswXSA9IGMgKiBtMDAgKyBzICogbTEwO1xuICAgICAgICBuZXdEc3RbMV0gPSBjICogbTAxICsgcyAqIG0xMTtcbiAgICAgICAgbmV3RHN0WzJdID0gYyAqIG0wMiArIHMgKiBtMTI7XG4gICAgICAgIG5ld0RzdFszXSA9IGMgKiBtMDMgKyBzICogbTEzO1xuICAgICAgICBuZXdEc3RbNF0gPSBjICogbTEwIC0gcyAqIG0wMDtcbiAgICAgICAgbmV3RHN0WzVdID0gYyAqIG0xMSAtIHMgKiBtMDE7XG4gICAgICAgIG5ld0RzdFs2XSA9IGMgKiBtMTIgLSBzICogbTAyO1xuICAgICAgICBuZXdEc3RbN10gPSBjICogbTEzIC0gcyAqIG0wMztcbiAgICAgICAgaWYgKG0gIT09IG5ld0RzdCkge1xuICAgICAgICAgICAgbmV3RHN0WzhdID0gbVs4XTtcbiAgICAgICAgICAgIG5ld0RzdFs5XSA9IG1bOV07XG4gICAgICAgICAgICBuZXdEc3RbMTBdID0gbVsxMF07XG4gICAgICAgICAgICBuZXdEc3RbMTFdID0gbVsxMV07XG4gICAgICAgICAgICBuZXdEc3RbMTJdID0gbVsxMl07XG4gICAgICAgICAgICBuZXdEc3RbMTNdID0gbVsxM107XG4gICAgICAgICAgICBuZXdEc3RbMTRdID0gbVsxNF07XG4gICAgICAgICAgICBuZXdEc3RbMTVdID0gbVsxNV07XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIDQtYnktNCBtYXRyaXggd2hpY2ggcm90YXRlcyBhcm91bmQgdGhlIGdpdmVuIGF4aXMgYnkgdGhlIGdpdmVuXG4gICAgICogYW5nbGUuXG4gICAgICogQHBhcmFtIGF4aXMgLSBUaGUgYXhpc1xuICAgICAqICAgICBhYm91dCB3aGljaCB0byByb3RhdGUuXG4gICAgICogQHBhcmFtIGFuZ2xlSW5SYWRpYW5zIC0gVGhlIGFuZ2xlIGJ5IHdoaWNoIHRvIHJvdGF0ZSAoaW4gcmFkaWFucykuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIG1hdHJpeCB3aGljaCByb3RhdGVzIGFuZ2xlIHJhZGlhbnNcbiAgICAgKiAgICAgYXJvdW5kIHRoZSBheGlzLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGF4aXNSb3RhdGlvbihheGlzLCBhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgbGV0IHggPSBheGlzWzBdO1xuICAgICAgICBsZXQgeSA9IGF4aXNbMV07XG4gICAgICAgIGxldCB6ID0gYXhpc1syXTtcbiAgICAgICAgY29uc3QgbiA9IE1hdGguc3FydCh4ICogeCArIHkgKiB5ICsgeiAqIHopO1xuICAgICAgICB4IC89IG47XG4gICAgICAgIHkgLz0gbjtcbiAgICAgICAgeiAvPSBuO1xuICAgICAgICBjb25zdCB4eCA9IHggKiB4O1xuICAgICAgICBjb25zdCB5eSA9IHkgKiB5O1xuICAgICAgICBjb25zdCB6eiA9IHogKiB6O1xuICAgICAgICBjb25zdCBjID0gTWF0aC5jb3MoYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBjb25zdCBzID0gTWF0aC5zaW4oYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBjb25zdCBvbmVNaW51c0Nvc2luZSA9IDEgLSBjO1xuICAgICAgICBuZXdEc3RbMF0gPSB4eCArICgxIC0geHgpICogYztcbiAgICAgICAgbmV3RHN0WzFdID0geCAqIHkgKiBvbmVNaW51c0Nvc2luZSArIHogKiBzO1xuICAgICAgICBuZXdEc3RbMl0gPSB4ICogeiAqIG9uZU1pbnVzQ29zaW5lIC0geSAqIHM7XG4gICAgICAgIG5ld0RzdFszXSA9IDA7XG4gICAgICAgIG5ld0RzdFs0XSA9IHggKiB5ICogb25lTWludXNDb3NpbmUgLSB6ICogcztcbiAgICAgICAgbmV3RHN0WzVdID0geXkgKyAoMSAtIHl5KSAqIGM7XG4gICAgICAgIG5ld0RzdFs2XSA9IHkgKiB6ICogb25lTWludXNDb3NpbmUgKyB4ICogcztcbiAgICAgICAgbmV3RHN0WzddID0gMDtcbiAgICAgICAgbmV3RHN0WzhdID0geCAqIHogKiBvbmVNaW51c0Nvc2luZSArIHkgKiBzO1xuICAgICAgICBuZXdEc3RbOV0gPSB5ICogeiAqIG9uZU1pbnVzQ29zaW5lIC0geCAqIHM7XG4gICAgICAgIG5ld0RzdFsxMF0gPSB6eiArICgxIC0genopICogYztcbiAgICAgICAgbmV3RHN0WzExXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMl0gPSAwO1xuICAgICAgICBuZXdEc3RbMTNdID0gMDtcbiAgICAgICAgbmV3RHN0WzE0XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxNV0gPSAxO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDcmVhdGVzIGEgNC1ieS00IG1hdHJpeCB3aGljaCByb3RhdGVzIGFyb3VuZCB0aGUgZ2l2ZW4gYXhpcyBieSB0aGUgZ2l2ZW5cbiAgICAgKiBhbmdsZS4gKHNhbWUgYXMgYXhpc1JvdGF0aW9uKVxuICAgICAqIEBwYXJhbSBheGlzIC0gVGhlIGF4aXNcbiAgICAgKiAgICAgYWJvdXQgd2hpY2ggdG8gcm90YXRlLlxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIFRoZSBhbmdsZSBieSB3aGljaCB0byByb3RhdGUgKGluIHJhZGlhbnMpLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSBtYXRyaXggd2hpY2ggcm90YXRlcyBhbmdsZSByYWRpYW5zXG4gICAgICogICAgIGFyb3VuZCB0aGUgYXhpcy5cbiAgICAgKi9cbiAgICBjb25zdCByb3RhdGlvbiA9IGF4aXNSb3RhdGlvbjtcbiAgICAvKipcbiAgICAgKiBSb3RhdGVzIHRoZSBnaXZlbiA0LWJ5LTQgbWF0cml4IGFyb3VuZCB0aGUgZ2l2ZW4gYXhpcyBieSB0aGVcbiAgICAgKiBnaXZlbiBhbmdsZS5cbiAgICAgKiBAcGFyYW0gbSAtIFRoZSBtYXRyaXguXG4gICAgICogQHBhcmFtIGF4aXMgLSBUaGUgYXhpc1xuICAgICAqICAgICBhYm91dCB3aGljaCB0byByb3RhdGUuXG4gICAgICogQHBhcmFtIGFuZ2xlSW5SYWRpYW5zIC0gVGhlIGFuZ2xlIGJ5IHdoaWNoIHRvIHJvdGF0ZSAoaW4gcmFkaWFucykuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgcm90YXRlZCBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gYXhpc1JvdGF0ZShtLCBheGlzLCBhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgbGV0IHggPSBheGlzWzBdO1xuICAgICAgICBsZXQgeSA9IGF4aXNbMV07XG4gICAgICAgIGxldCB6ID0gYXhpc1syXTtcbiAgICAgICAgY29uc3QgbiA9IE1hdGguc3FydCh4ICogeCArIHkgKiB5ICsgeiAqIHopO1xuICAgICAgICB4IC89IG47XG4gICAgICAgIHkgLz0gbjtcbiAgICAgICAgeiAvPSBuO1xuICAgICAgICBjb25zdCB4eCA9IHggKiB4O1xuICAgICAgICBjb25zdCB5eSA9IHkgKiB5O1xuICAgICAgICBjb25zdCB6eiA9IHogKiB6O1xuICAgICAgICBjb25zdCBjID0gTWF0aC5jb3MoYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBjb25zdCBzID0gTWF0aC5zaW4oYW5nbGVJblJhZGlhbnMpO1xuICAgICAgICBjb25zdCBvbmVNaW51c0Nvc2luZSA9IDEgLSBjO1xuICAgICAgICBjb25zdCByMDAgPSB4eCArICgxIC0geHgpICogYztcbiAgICAgICAgY29uc3QgcjAxID0geCAqIHkgKiBvbmVNaW51c0Nvc2luZSArIHogKiBzO1xuICAgICAgICBjb25zdCByMDIgPSB4ICogeiAqIG9uZU1pbnVzQ29zaW5lIC0geSAqIHM7XG4gICAgICAgIGNvbnN0IHIxMCA9IHggKiB5ICogb25lTWludXNDb3NpbmUgLSB6ICogcztcbiAgICAgICAgY29uc3QgcjExID0geXkgKyAoMSAtIHl5KSAqIGM7XG4gICAgICAgIGNvbnN0IHIxMiA9IHkgKiB6ICogb25lTWludXNDb3NpbmUgKyB4ICogcztcbiAgICAgICAgY29uc3QgcjIwID0geCAqIHogKiBvbmVNaW51c0Nvc2luZSArIHkgKiBzO1xuICAgICAgICBjb25zdCByMjEgPSB5ICogeiAqIG9uZU1pbnVzQ29zaW5lIC0geCAqIHM7XG4gICAgICAgIGNvbnN0IHIyMiA9IHp6ICsgKDEgLSB6eikgKiBjO1xuICAgICAgICBjb25zdCBtMDAgPSBtWzBdO1xuICAgICAgICBjb25zdCBtMDEgPSBtWzFdO1xuICAgICAgICBjb25zdCBtMDIgPSBtWzJdO1xuICAgICAgICBjb25zdCBtMDMgPSBtWzNdO1xuICAgICAgICBjb25zdCBtMTAgPSBtWzRdO1xuICAgICAgICBjb25zdCBtMTEgPSBtWzVdO1xuICAgICAgICBjb25zdCBtMTIgPSBtWzZdO1xuICAgICAgICBjb25zdCBtMTMgPSBtWzddO1xuICAgICAgICBjb25zdCBtMjAgPSBtWzhdO1xuICAgICAgICBjb25zdCBtMjEgPSBtWzldO1xuICAgICAgICBjb25zdCBtMjIgPSBtWzEwXTtcbiAgICAgICAgY29uc3QgbTIzID0gbVsxMV07XG4gICAgICAgIG5ld0RzdFswXSA9IHIwMCAqIG0wMCArIHIwMSAqIG0xMCArIHIwMiAqIG0yMDtcbiAgICAgICAgbmV3RHN0WzFdID0gcjAwICogbTAxICsgcjAxICogbTExICsgcjAyICogbTIxO1xuICAgICAgICBuZXdEc3RbMl0gPSByMDAgKiBtMDIgKyByMDEgKiBtMTIgKyByMDIgKiBtMjI7XG4gICAgICAgIG5ld0RzdFszXSA9IHIwMCAqIG0wMyArIHIwMSAqIG0xMyArIHIwMiAqIG0yMztcbiAgICAgICAgbmV3RHN0WzRdID0gcjEwICogbTAwICsgcjExICogbTEwICsgcjEyICogbTIwO1xuICAgICAgICBuZXdEc3RbNV0gPSByMTAgKiBtMDEgKyByMTEgKiBtMTEgKyByMTIgKiBtMjE7XG4gICAgICAgIG5ld0RzdFs2XSA9IHIxMCAqIG0wMiArIHIxMSAqIG0xMiArIHIxMiAqIG0yMjtcbiAgICAgICAgbmV3RHN0WzddID0gcjEwICogbTAzICsgcjExICogbTEzICsgcjEyICogbTIzO1xuICAgICAgICBuZXdEc3RbOF0gPSByMjAgKiBtMDAgKyByMjEgKiBtMTAgKyByMjIgKiBtMjA7XG4gICAgICAgIG5ld0RzdFs5XSA9IHIyMCAqIG0wMSArIHIyMSAqIG0xMSArIHIyMiAqIG0yMTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IHIyMCAqIG0wMiArIHIyMSAqIG0xMiArIHIyMiAqIG0yMjtcbiAgICAgICAgbmV3RHN0WzExXSA9IHIyMCAqIG0wMyArIHIyMSAqIG0xMyArIHIyMiAqIG0yMztcbiAgICAgICAgaWYgKG0gIT09IG5ld0RzdCkge1xuICAgICAgICAgICAgbmV3RHN0WzEyXSA9IG1bMTJdO1xuICAgICAgICAgICAgbmV3RHN0WzEzXSA9IG1bMTNdO1xuICAgICAgICAgICAgbmV3RHN0WzE0XSA9IG1bMTRdO1xuICAgICAgICAgICAgbmV3RHN0WzE1XSA9IG1bMTVdO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJvdGF0ZXMgdGhlIGdpdmVuIDQtYnktNCBtYXRyaXggYXJvdW5kIHRoZSBnaXZlbiBheGlzIGJ5IHRoZVxuICAgICAqIGdpdmVuIGFuZ2xlLiAoc2FtZSBhcyByb3RhdGUpXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4LlxuICAgICAqIEBwYXJhbSBheGlzIC0gVGhlIGF4aXNcbiAgICAgKiAgICAgYWJvdXQgd2hpY2ggdG8gcm90YXRlLlxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIFRoZSBhbmdsZSBieSB3aGljaCB0byByb3RhdGUgKGluIHJhZGlhbnMpLlxuICAgICAqIEBwYXJhbSBkc3QgLSBtYXRyaXggdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIHJvdGF0ZWQgbWF0cml4LlxuICAgICAqL1xuICAgIGNvbnN0IHJvdGF0ZSA9IGF4aXNSb3RhdGU7XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIDQtYnktNCBtYXRyaXggd2hpY2ggc2NhbGVzIGluIGVhY2ggZGltZW5zaW9uIGJ5IGFuIGFtb3VudCBnaXZlbiBieVxuICAgICAqIHRoZSBjb3JyZXNwb25kaW5nIGVudHJ5IGluIHRoZSBnaXZlbiB2ZWN0b3I7IGFzc3VtZXMgdGhlIHZlY3RvciBoYXMgdGhyZWVcbiAgICAgKiBlbnRyaWVzLlxuICAgICAqIEBwYXJhbSB2IC0gQSB2ZWN0b3Igb2ZcbiAgICAgKiAgICAgdGhyZWUgZW50cmllcyBzcGVjaWZ5aW5nIHRoZSBmYWN0b3IgYnkgd2hpY2ggdG8gc2NhbGUgaW4gZWFjaCBkaW1lbnNpb24uXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGluZyBtYXRyaXguXG4gICAgICovXG4gICAgZnVuY3Rpb24gc2NhbGluZyh2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2WzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICBuZXdEc3RbM10gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSAwO1xuICAgICAgICBuZXdEc3RbNV0gPSB2WzFdO1xuICAgICAgICBuZXdEc3RbNl0gPSAwO1xuICAgICAgICBuZXdEc3RbN10gPSAwO1xuICAgICAgICBuZXdEc3RbOF0gPSAwO1xuICAgICAgICBuZXdEc3RbOV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTBdID0gdlsyXTtcbiAgICAgICAgbmV3RHN0WzExXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMl0gPSAwO1xuICAgICAgICBuZXdEc3RbMTNdID0gMDtcbiAgICAgICAgbmV3RHN0WzE0XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxNV0gPSAxO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBTY2FsZXMgdGhlIGdpdmVuIDQtYnktNCBtYXRyaXggaW4gZWFjaCBkaW1lbnNpb24gYnkgYW4gYW1vdW50XG4gICAgICogZ2l2ZW4gYnkgdGhlIGNvcnJlc3BvbmRpbmcgZW50cnkgaW4gdGhlIGdpdmVuIHZlY3RvcjsgYXNzdW1lcyB0aGUgdmVjdG9yIGhhc1xuICAgICAqIHRocmVlIGVudHJpZXMuXG4gICAgICogQHBhcmFtIG0gLSBUaGUgbWF0cml4IHRvIGJlIG1vZGlmaWVkLlxuICAgICAqIEBwYXJhbSB2IC0gQSB2ZWN0b3Igb2YgdGhyZWUgZW50cmllcyBzcGVjaWZ5aW5nIHRoZVxuICAgICAqICAgICBmYWN0b3IgYnkgd2hpY2ggdG8gc2NhbGUgaW4gZWFjaCBkaW1lbnNpb24uXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzY2FsZShtLCB2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBjb25zdCB2MCA9IHZbMF07XG4gICAgICAgIGNvbnN0IHYxID0gdlsxXTtcbiAgICAgICAgY29uc3QgdjIgPSB2WzJdO1xuICAgICAgICBuZXdEc3RbMF0gPSB2MCAqIG1bMCAqIDQgKyAwXTtcbiAgICAgICAgbmV3RHN0WzFdID0gdjAgKiBtWzAgKiA0ICsgMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IHYwICogbVswICogNCArIDJdO1xuICAgICAgICBuZXdEc3RbM10gPSB2MCAqIG1bMCAqIDQgKyAzXTtcbiAgICAgICAgbmV3RHN0WzRdID0gdjEgKiBtWzEgKiA0ICsgMF07XG4gICAgICAgIG5ld0RzdFs1XSA9IHYxICogbVsxICogNCArIDFdO1xuICAgICAgICBuZXdEc3RbNl0gPSB2MSAqIG1bMSAqIDQgKyAyXTtcbiAgICAgICAgbmV3RHN0WzddID0gdjEgKiBtWzEgKiA0ICsgM107XG4gICAgICAgIG5ld0RzdFs4XSA9IHYyICogbVsyICogNCArIDBdO1xuICAgICAgICBuZXdEc3RbOV0gPSB2MiAqIG1bMiAqIDQgKyAxXTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IHYyICogbVsyICogNCArIDJdO1xuICAgICAgICBuZXdEc3RbMTFdID0gdjIgKiBtWzIgKiA0ICsgM107XG4gICAgICAgIGlmIChtICE9PSBuZXdEc3QpIHtcbiAgICAgICAgICAgIG5ld0RzdFsxMl0gPSBtWzEyXTtcbiAgICAgICAgICAgIG5ld0RzdFsxM10gPSBtWzEzXTtcbiAgICAgICAgICAgIG5ld0RzdFsxNF0gPSBtWzE0XTtcbiAgICAgICAgICAgIG5ld0RzdFsxNV0gPSBtWzE1XTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDcmVhdGVzIGEgNC1ieS00IG1hdHJpeCB3aGljaCBzY2FsZXMgYSB1bmlmb3JtIGFtb3VudCBpbiBlYWNoIGRpbWVuc2lvbi5cbiAgICAgKiBAcGFyYW0gcyAtIHRoZSBhbW91bnQgdG8gc2NhbGVcbiAgICAgKiBAcGFyYW0gZHN0IC0gbWF0cml4IHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBzY2FsaW5nIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiB1bmlmb3JtU2NhbGluZyhzLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3RvcigxNikpO1xuICAgICAgICBuZXdEc3RbMF0gPSBzO1xuICAgICAgICBuZXdEc3RbMV0gPSAwO1xuICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICBuZXdEc3RbM10gPSAwO1xuICAgICAgICBuZXdEc3RbNF0gPSAwO1xuICAgICAgICBuZXdEc3RbNV0gPSBzO1xuICAgICAgICBuZXdEc3RbNl0gPSAwO1xuICAgICAgICBuZXdEc3RbN10gPSAwO1xuICAgICAgICBuZXdEc3RbOF0gPSAwO1xuICAgICAgICBuZXdEc3RbOV0gPSAwO1xuICAgICAgICBuZXdEc3RbMTBdID0gcztcbiAgICAgICAgbmV3RHN0WzExXSA9IDA7XG4gICAgICAgIG5ld0RzdFsxMl0gPSAwO1xuICAgICAgICBuZXdEc3RbMTNdID0gMDtcbiAgICAgICAgbmV3RHN0WzE0XSA9IDA7XG4gICAgICAgIG5ld0RzdFsxNV0gPSAxO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBTY2FsZXMgdGhlIGdpdmVuIDQtYnktNCBtYXRyaXggaW4gZWFjaCBkaW1lbnNpb24gYnkgYSB1bmlmb3JtIHNjYWxlLlxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeCB0byBiZSBtb2RpZmllZC5cbiAgICAgKiBAcGFyYW0gcyAtIFRoZSBhbW91bnQgdG8gc2NhbGUuXG4gICAgICogQHBhcmFtIGRzdCAtIG1hdHJpeCB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIG1hdHJpeC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiB1bmlmb3JtU2NhbGUobSwgcywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoMTYpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gcyAqIG1bMCAqIDQgKyAwXTtcbiAgICAgICAgbmV3RHN0WzFdID0gcyAqIG1bMCAqIDQgKyAxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gcyAqIG1bMCAqIDQgKyAyXTtcbiAgICAgICAgbmV3RHN0WzNdID0gcyAqIG1bMCAqIDQgKyAzXTtcbiAgICAgICAgbmV3RHN0WzRdID0gcyAqIG1bMSAqIDQgKyAwXTtcbiAgICAgICAgbmV3RHN0WzVdID0gcyAqIG1bMSAqIDQgKyAxXTtcbiAgICAgICAgbmV3RHN0WzZdID0gcyAqIG1bMSAqIDQgKyAyXTtcbiAgICAgICAgbmV3RHN0WzddID0gcyAqIG1bMSAqIDQgKyAzXTtcbiAgICAgICAgbmV3RHN0WzhdID0gcyAqIG1bMiAqIDQgKyAwXTtcbiAgICAgICAgbmV3RHN0WzldID0gcyAqIG1bMiAqIDQgKyAxXTtcbiAgICAgICAgbmV3RHN0WzEwXSA9IHMgKiBtWzIgKiA0ICsgMl07XG4gICAgICAgIG5ld0RzdFsxMV0gPSBzICogbVsyICogNCArIDNdO1xuICAgICAgICBpZiAobSAhPT0gbmV3RHN0KSB7XG4gICAgICAgICAgICBuZXdEc3RbMTJdID0gbVsxMl07XG4gICAgICAgICAgICBuZXdEc3RbMTNdID0gbVsxM107XG4gICAgICAgICAgICBuZXdEc3RbMTRdID0gbVsxNF07XG4gICAgICAgICAgICBuZXdEc3RbMTVdID0gbVsxNV07XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgcmV0dXJuIHtcbiAgICAgICAgY3JlYXRlLFxuICAgICAgICBzZXQsXG4gICAgICAgIGZyb21NYXQzLFxuICAgICAgICBmcm9tUXVhdCxcbiAgICAgICAgbmVnYXRlLFxuICAgICAgICBjb3B5LFxuICAgICAgICBjbG9uZSxcbiAgICAgICAgZXF1YWxzQXBwcm94aW1hdGVseSxcbiAgICAgICAgZXF1YWxzLFxuICAgICAgICBpZGVudGl0eSxcbiAgICAgICAgdHJhbnNwb3NlLFxuICAgICAgICBpbnZlcnNlLFxuICAgICAgICBkZXRlcm1pbmFudCxcbiAgICAgICAgaW52ZXJ0LFxuICAgICAgICBtdWx0aXBseSxcbiAgICAgICAgbXVsLFxuICAgICAgICBzZXRUcmFuc2xhdGlvbixcbiAgICAgICAgZ2V0VHJhbnNsYXRpb24sXG4gICAgICAgIGdldEF4aXMsXG4gICAgICAgIHNldEF4aXMsXG4gICAgICAgIGdldFNjYWxpbmcsXG4gICAgICAgIHBlcnNwZWN0aXZlLFxuICAgICAgICBwZXJzcGVjdGl2ZVJldmVyc2VaLFxuICAgICAgICBvcnRobyxcbiAgICAgICAgZnJ1c3R1bSxcbiAgICAgICAgZnJ1c3R1bVJldmVyc2VaLFxuICAgICAgICBhaW0sXG4gICAgICAgIGNhbWVyYUFpbSxcbiAgICAgICAgbG9va0F0LFxuICAgICAgICB0cmFuc2xhdGlvbixcbiAgICAgICAgdHJhbnNsYXRlLFxuICAgICAgICByb3RhdGlvblgsXG4gICAgICAgIHJvdGF0ZVgsXG4gICAgICAgIHJvdGF0aW9uWSxcbiAgICAgICAgcm90YXRlWSxcbiAgICAgICAgcm90YXRpb25aLFxuICAgICAgICByb3RhdGVaLFxuICAgICAgICBheGlzUm90YXRpb24sXG4gICAgICAgIHJvdGF0aW9uLFxuICAgICAgICBheGlzUm90YXRlLFxuICAgICAgICByb3RhdGUsXG4gICAgICAgIHNjYWxpbmcsXG4gICAgICAgIHNjYWxlLFxuICAgICAgICB1bmlmb3JtU2NhbGluZyxcbiAgICAgICAgdW5pZm9ybVNjYWxlLFxuICAgIH07XG59XG5jb25zdCBjYWNoZSQyID0gbmV3IE1hcCgpO1xuZnVuY3Rpb24gZ2V0QVBJJDIoQ3Rvcikge1xuICAgIGxldCBhcGkgPSBjYWNoZSQyLmdldChDdG9yKTtcbiAgICBpZiAoIWFwaSkge1xuICAgICAgICBhcGkgPSBnZXRBUElJbXBsJDIoQ3Rvcik7XG4gICAgICAgIGNhY2hlJDIuc2V0KEN0b3IsIGFwaSk7XG4gICAgfVxuICAgIHJldHVybiBhcGk7XG59XG5cbi8qXG4gKiBDb3B5cmlnaHQgMjAyMiBHcmVnZyBUYXZhcmVzXG4gKlxuICogUGVybWlzc2lvbiBpcyBoZXJlYnkgZ3JhbnRlZCwgZnJlZSBvZiBjaGFyZ2UsIHRvIGFueSBwZXJzb24gb2J0YWluaW5nIGFcbiAqIGNvcHkgb2YgdGhpcyBzb2Z0d2FyZSBhbmQgYXNzb2NpYXRlZCBkb2N1bWVudGF0aW9uIGZpbGVzICh0aGUgXCJTb2Z0d2FyZVwiKSxcbiAqIHRvIGRlYWwgaW4gdGhlIFNvZnR3YXJlIHdpdGhvdXQgcmVzdHJpY3Rpb24sIGluY2x1ZGluZyB3aXRob3V0IGxpbWl0YXRpb25cbiAqIHRoZSByaWdodHMgdG8gdXNlLCBjb3B5LCBtb2RpZnksIG1lcmdlLCBwdWJsaXNoLCBkaXN0cmlidXRlLCBzdWJsaWNlbnNlLFxuICogYW5kL29yIHNlbGwgY29waWVzIG9mIHRoZSBTb2Z0d2FyZSwgYW5kIHRvIHBlcm1pdCBwZXJzb25zIHRvIHdob20gdGhlXG4gKiBTb2Z0d2FyZSBpcyBmdXJuaXNoZWQgdG8gZG8gc28sIHN1YmplY3QgdG8gdGhlIGZvbGxvd2luZyBjb25kaXRpb25zOlxuICpcbiAqIFRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluXG4gKiBhbGwgY29waWVzIG9yIHN1YnN0YW50aWFsIHBvcnRpb25zIG9mIHRoZSBTb2Z0d2FyZS5cbiAqXG4gKiBUSEUgU09GVFdBUkUgSVMgUFJPVklERUQgXCJBUyBJU1wiLCBXSVRIT1VUIFdBUlJBTlRZIE9GIEFOWSBLSU5ELCBFWFBSRVNTIE9SXG4gKiBJTVBMSUVELCBJTkNMVURJTkcgQlVUIE5PVCBMSU1JVEVEIFRPIFRIRSBXQVJSQU5USUVTIE9GIE1FUkNIQU5UQUJJTElUWSxcbiAqIEZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFIEFORCBOT05JTkZSSU5HRU1FTlQuICBJTiBOTyBFVkVOVCBTSEFMTFxuICogVEhFIEFVVEhPUlMgT1IgQ09QWVJJR0hUIEhPTERFUlMgQkUgTElBQkxFIEZPUiBBTlkgQ0xBSU0sIERBTUFHRVMgT1IgT1RIRVJcbiAqIExJQUJJTElUWSwgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HXG4gKiBGUk9NLCBPVVQgT0YgT1IgSU4gQ09OTkVDVElPTiBXSVRIIFRIRSBTT0ZUV0FSRSBPUiBUSEUgVVNFIE9SIE9USEVSXG4gKiBERUFMSU5HUyBJTiBUSEUgU09GVFdBUkUuXG4gKi9cbi8qKlxuICogR2VuZXJhdGVzIGFtIHR5cGVkIEFQSSBmb3IgUXVkXG4gKiAqL1xuZnVuY3Rpb24gZ2V0QVBJSW1wbCQxKEN0b3IpIHtcbiAgICBjb25zdCB2ZWMzID0gZ2V0QVBJJDQoQ3Rvcik7XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIHF1YXQ0OyBtYXkgYmUgY2FsbGVkIHdpdGggeCwgeSwgeiB0byBzZXQgaW5pdGlhbCB2YWx1ZXMuXG4gICAgICogQHBhcmFtIHggLSBJbml0aWFsIHggdmFsdWUuXG4gICAgICogQHBhcmFtIHkgLSBJbml0aWFsIHkgdmFsdWUuXG4gICAgICogQHBhcmFtIHogLSBJbml0aWFsIHogdmFsdWUuXG4gICAgICogQHBhcmFtIHcgLSBJbml0aWFsIHcgdmFsdWUuXG4gICAgICogQHJldHVybnMgdGhlIGNyZWF0ZWQgdmVjdG9yXG4gICAgICovXG4gICAgZnVuY3Rpb24gY3JlYXRlKHgsIHksIHosIHcpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gbmV3IEN0b3IoNCk7XG4gICAgICAgIGlmICh4ICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgIG5ld0RzdFswXSA9IHg7XG4gICAgICAgICAgICBpZiAoeSAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgbmV3RHN0WzFdID0geTtcbiAgICAgICAgICAgICAgICBpZiAoeiAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgICAgIG5ld0RzdFsyXSA9IHo7XG4gICAgICAgICAgICAgICAgICAgIGlmICh3ICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFszXSA9IHc7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIFF1YXQ7IG1heSBiZSBjYWxsZWQgd2l0aCB4LCB5LCB6IHRvIHNldCBpbml0aWFsIHZhbHVlcy4gKHNhbWUgYXMgY3JlYXRlKVxuICAgICAqIEBwYXJhbSB4IC0gSW5pdGlhbCB4IHZhbHVlLlxuICAgICAqIEBwYXJhbSB5IC0gSW5pdGlhbCB5IHZhbHVlLlxuICAgICAqIEBwYXJhbSB6IC0gSW5pdGlhbCB6IHZhbHVlLlxuICAgICAqIEBwYXJhbSB6IC0gSW5pdGlhbCB3IHZhbHVlLlxuICAgICAqIEByZXR1cm5zIHRoZSBjcmVhdGVkIHZlY3RvclxuICAgICAqL1xuICAgIGNvbnN0IGZyb21WYWx1ZXMgPSBjcmVhdGU7XG4gICAgLyoqXG4gICAgICogU2V0cyB0aGUgdmFsdWVzIG9mIGEgUXVhdFxuICAgICAqIEFsc28gc2VlIHtAbGluayBxdWF0LmNyZWF0ZX0gYW5kIHtAbGluayBxdWF0LmNvcHl9XG4gICAgICpcbiAgICAgKiBAcGFyYW0geCBmaXJzdCB2YWx1ZVxuICAgICAqIEBwYXJhbSB5IHNlY29uZCB2YWx1ZVxuICAgICAqIEBwYXJhbSB6IHRoaXJkIHZhbHVlXG4gICAgICogQHBhcmFtIHcgZm91cnRoIHZhbHVlXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB3aXRoIGl0cyBlbGVtZW50cyBzZXQuXG4gICAgICovXG4gICAgZnVuY3Rpb24gc2V0KHgsIHksIHosIHcsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0geDtcbiAgICAgICAgbmV3RHN0WzFdID0geTtcbiAgICAgICAgbmV3RHN0WzJdID0gejtcbiAgICAgICAgbmV3RHN0WzNdID0gdztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogU2V0cyBhIHF1YXRlcm5pb24gZnJvbSB0aGUgZ2l2ZW4gYW5nbGUgYW5kICBheGlzLFxuICAgICAqIHRoZW4gcmV0dXJucyBpdC5cbiAgICAgKlxuICAgICAqIEBwYXJhbSBheGlzIC0gdGhlIGF4aXMgdG8gcm90YXRlIGFyb3VuZFxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIHRoZSBhbmdsZVxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBxdWF0ZXJuaW9uIHRoYXQgcmVwcmVzZW50cyB0aGUgZ2l2ZW4gYXhpcyBhbmQgYW5nbGVcbiAgICAgKiovXG4gICAgZnVuY3Rpb24gZnJvbUF4aXNBbmdsZShheGlzLCBhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBjb25zdCBoYWxmQW5nbGUgPSBhbmdsZUluUmFkaWFucyAqIDAuNTtcbiAgICAgICAgY29uc3QgcyA9IE1hdGguc2luKGhhbGZBbmdsZSk7XG4gICAgICAgIG5ld0RzdFswXSA9IHMgKiBheGlzWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSBzICogYXhpc1sxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gcyAqIGF4aXNbMl07XG4gICAgICAgIG5ld0RzdFszXSA9IE1hdGguY29zKGhhbGZBbmdsZSk7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEdldHMgdGhlIHJvdGF0aW9uIGF4aXMgYW5kIGFuZ2xlXG4gICAgICogQHBhcmFtIHEgLSBxdWF0ZXJuaW9uIHRvIGNvbXB1dGUgZnJvbVxuICAgICAqIEBwYXJhbSBkc3QgLSBWZWMzIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm4gYW5nbGUgYW5kIGF4aXNcbiAgICAgKi9cbiAgICBmdW5jdGlvbiB0b0F4aXNBbmdsZShxLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyB2ZWMzLmNyZWF0ZSgzKSk7XG4gICAgICAgIGNvbnN0IGFuZ2xlID0gTWF0aC5hY29zKHFbM10pICogMjtcbiAgICAgICAgY29uc3QgcyA9IE1hdGguc2luKGFuZ2xlICogMC41KTtcbiAgICAgICAgaWYgKHMgPiBFUFNJTE9OKSB7XG4gICAgICAgICAgICBuZXdEc3RbMF0gPSBxWzBdIC8gcztcbiAgICAgICAgICAgIG5ld0RzdFsxXSA9IHFbMV0gLyBzO1xuICAgICAgICAgICAgbmV3RHN0WzJdID0gcVsyXSAvIHM7XG4gICAgICAgIH1cbiAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICBuZXdEc3RbMF0gPSAxO1xuICAgICAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgICAgIG5ld0RzdFsyXSA9IDA7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHsgYW5nbGUsIGF4aXM6IG5ld0RzdCB9O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSZXR1cm5zIHRoZSBhbmdsZSBpbiBkZWdyZWVzIGJldHdlZW4gdHdvIHJvdGF0aW9ucyBhIGFuZCBiLlxuICAgICAqIEBwYXJhbSBhIC0gcXVhdGVybmlvbiBhXG4gICAgICogQHBhcmFtIGIgLSBxdWF0ZXJuaW9uIGJcbiAgICAgKiBAcmV0dXJuIGFuZ2xlIGluIHJhZGlhbnMgYmV0d2VlbiB0aGUgdHdvIHF1YXRlcm5pb25zXG4gICAgICovXG4gICAgZnVuY3Rpb24gYW5nbGUoYSwgYikge1xuICAgICAgICBjb25zdCBkID0gZG90KGEsIGIpO1xuICAgICAgICByZXR1cm4gTWF0aC5hY29zKDIgKiBkICogZCAtIDEpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBNdWx0aXBsaWVzIHR3byBxdWF0ZXJuaW9uc1xuICAgICAqXG4gICAgICogQHBhcmFtIGEgLSB0aGUgZmlyc3QgcXVhdGVybmlvblxuICAgICAqIEBwYXJhbSBiIC0gdGhlIHNlY29uZCBxdWF0ZXJuaW9uXG4gICAgICogQHBhcmFtIGRzdCAtIHF1YXRlcm5pb24gdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSBxdWF0ZXJuaW9uIHRoYXQgaXMgdGhlIHJlc3VsdCBvZiBhICogYlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG11bHRpcGx5KGEsIGIsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgY29uc3QgYXggPSBhWzBdO1xuICAgICAgICBjb25zdCBheSA9IGFbMV07XG4gICAgICAgIGNvbnN0IGF6ID0gYVsyXTtcbiAgICAgICAgY29uc3QgYXcgPSBhWzNdO1xuICAgICAgICBjb25zdCBieCA9IGJbMF07XG4gICAgICAgIGNvbnN0IGJ5ID0gYlsxXTtcbiAgICAgICAgY29uc3QgYnogPSBiWzJdO1xuICAgICAgICBjb25zdCBidyA9IGJbM107XG4gICAgICAgIG5ld0RzdFswXSA9IGF4ICogYncgKyBhdyAqIGJ4ICsgYXkgKiBieiAtIGF6ICogYnk7XG4gICAgICAgIG5ld0RzdFsxXSA9IGF5ICogYncgKyBhdyAqIGJ5ICsgYXogKiBieCAtIGF4ICogYno7XG4gICAgICAgIG5ld0RzdFsyXSA9IGF6ICogYncgKyBhdyAqIGJ6ICsgYXggKiBieSAtIGF5ICogYng7XG4gICAgICAgIG5ld0RzdFszXSA9IGF3ICogYncgLSBheCAqIGJ4IC0gYXkgKiBieSAtIGF6ICogYno7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIE11bHRpcGxpZXMgdHdvIHF1YXRlcm5pb25zXG4gICAgICpcbiAgICAgKiBAcGFyYW0gYSAtIHRoZSBmaXJzdCBxdWF0ZXJuaW9uXG4gICAgICogQHBhcmFtIGIgLSB0aGUgc2Vjb25kIHF1YXRlcm5pb25cbiAgICAgKiBAcGFyYW0gZHN0IC0gcXVhdGVybmlvbiB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHF1YXRlcm5pb24gdGhhdCBpcyB0aGUgcmVzdWx0IG9mIGEgKiBiXG4gICAgICovXG4gICAgY29uc3QgbXVsID0gbXVsdGlwbHk7XG4gICAgLyoqXG4gICAgICogUm90YXRlcyB0aGUgZ2l2ZW4gcXVhdGVybmlvbiBhcm91bmQgdGhlIFggYXhpcyBieSB0aGUgZ2l2ZW4gYW5nbGUuXG4gICAgICogQHBhcmFtIHEgLSBxdWF0ZXJuaW9uIHRvIHJvdGF0ZVxuICAgICAqIEBwYXJhbSBhbmdsZUluUmFkaWFucyAtIFRoZSBhbmdsZSBieSB3aGljaCB0byByb3RhdGVcbiAgICAgKiBAcGFyYW0gZHN0IC0gcXVhdGVybmlvbiB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHF1YXRlcm5pb24gdGhhdCBpcyB0aGUgcmVzdWx0IG9mIGEgKiBiXG4gICAgICovXG4gICAgZnVuY3Rpb24gcm90YXRlWChxLCBhbmdsZUluUmFkaWFucywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBjb25zdCBoYWxmQW5nbGUgPSBhbmdsZUluUmFkaWFucyAqIDAuNTtcbiAgICAgICAgY29uc3QgcXggPSBxWzBdO1xuICAgICAgICBjb25zdCBxeSA9IHFbMV07XG4gICAgICAgIGNvbnN0IHF6ID0gcVsyXTtcbiAgICAgICAgY29uc3QgcXcgPSBxWzNdO1xuICAgICAgICBjb25zdCBieCA9IE1hdGguc2luKGhhbGZBbmdsZSk7XG4gICAgICAgIGNvbnN0IGJ3ID0gTWF0aC5jb3MoaGFsZkFuZ2xlKTtcbiAgICAgICAgbmV3RHN0WzBdID0gcXggKiBidyArIHF3ICogYng7XG4gICAgICAgIG5ld0RzdFsxXSA9IHF5ICogYncgKyBxeiAqIGJ4O1xuICAgICAgICBuZXdEc3RbMl0gPSBxeiAqIGJ3IC0gcXkgKiBieDtcbiAgICAgICAgbmV3RHN0WzNdID0gcXcgKiBidyAtIHF4ICogYng7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFJvdGF0ZXMgdGhlIGdpdmVuIHF1YXRlcm5pb24gYXJvdW5kIHRoZSBZIGF4aXMgYnkgdGhlIGdpdmVuIGFuZ2xlLlxuICAgICAqIEBwYXJhbSBxIC0gcXVhdGVybmlvbiB0byByb3RhdGVcbiAgICAgKiBAcGFyYW0gYW5nbGVJblJhZGlhbnMgLSBUaGUgYW5nbGUgYnkgd2hpY2ggdG8gcm90YXRlXG4gICAgICogQHBhcmFtIGRzdCAtIHF1YXRlcm5pb24gdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSBxdWF0ZXJuaW9uIHRoYXQgaXMgdGhlIHJlc3VsdCBvZiBhICogYlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHJvdGF0ZVkocSwgYW5nbGVJblJhZGlhbnMsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgY29uc3QgaGFsZkFuZ2xlID0gYW5nbGVJblJhZGlhbnMgKiAwLjU7XG4gICAgICAgIGNvbnN0IHF4ID0gcVswXTtcbiAgICAgICAgY29uc3QgcXkgPSBxWzFdO1xuICAgICAgICBjb25zdCBxeiA9IHFbMl07XG4gICAgICAgIGNvbnN0IHF3ID0gcVszXTtcbiAgICAgICAgY29uc3QgYnkgPSBNYXRoLnNpbihoYWxmQW5nbGUpO1xuICAgICAgICBjb25zdCBidyA9IE1hdGguY29zKGhhbGZBbmdsZSk7XG4gICAgICAgIG5ld0RzdFswXSA9IHF4ICogYncgLSBxeiAqIGJ5O1xuICAgICAgICBuZXdEc3RbMV0gPSBxeSAqIGJ3ICsgcXcgKiBieTtcbiAgICAgICAgbmV3RHN0WzJdID0gcXogKiBidyArIHF4ICogYnk7XG4gICAgICAgIG5ld0RzdFszXSA9IHF3ICogYncgLSBxeSAqIGJ5O1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSb3RhdGVzIHRoZSBnaXZlbiBxdWF0ZXJuaW9uIGFyb3VuZCB0aGUgWiBheGlzIGJ5IHRoZSBnaXZlbiBhbmdsZS5cbiAgICAgKiBAcGFyYW0gcSAtIHF1YXRlcm5pb24gdG8gcm90YXRlXG4gICAgICogQHBhcmFtIGFuZ2xlSW5SYWRpYW5zIC0gVGhlIGFuZ2xlIGJ5IHdoaWNoIHRvIHJvdGF0ZVxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgcXVhdGVybmlvbiB0aGF0IGlzIHRoZSByZXN1bHQgb2YgYSAqIGJcbiAgICAgKi9cbiAgICBmdW5jdGlvbiByb3RhdGVaKHEsIGFuZ2xlSW5SYWRpYW5zLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIGNvbnN0IGhhbGZBbmdsZSA9IGFuZ2xlSW5SYWRpYW5zICogMC41O1xuICAgICAgICBjb25zdCBxeCA9IHFbMF07XG4gICAgICAgIGNvbnN0IHF5ID0gcVsxXTtcbiAgICAgICAgY29uc3QgcXogPSBxWzJdO1xuICAgICAgICBjb25zdCBxdyA9IHFbM107XG4gICAgICAgIGNvbnN0IGJ6ID0gTWF0aC5zaW4oaGFsZkFuZ2xlKTtcbiAgICAgICAgY29uc3QgYncgPSBNYXRoLmNvcyhoYWxmQW5nbGUpO1xuICAgICAgICBuZXdEc3RbMF0gPSBxeCAqIGJ3ICsgcXkgKiBiejtcbiAgICAgICAgbmV3RHN0WzFdID0gcXkgKiBidyAtIHF4ICogYno7XG4gICAgICAgIG5ld0RzdFsyXSA9IHF6ICogYncgKyBxdyAqIGJ6O1xuICAgICAgICBuZXdEc3RbM10gPSBxdyAqIGJ3IC0gcXogKiBiejtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogU3BoZXJpY2FsbHkgbGluZWFyIGludGVycG9sYXRlIGJldHdlZW4gdHdvIHF1YXRlcm5pb25zXG4gICAgICpcbiAgICAgKiBAcGFyYW0gYSAtIHN0YXJ0aW5nIHZhbHVlXG4gICAgICogQHBhcmFtIGIgLSBlbmRpbmcgdmFsdWVcbiAgICAgKiBAcGFyYW0gdCAtIHZhbHVlIHdoZXJlIDAgPSBhIGFuZCAxID0gYlxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgcXVhdGVybmlvbiB0aGF0IGlzIHRoZSByZXN1bHQgb2YgYSAqIGJcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzbGVycChhLCBiLCB0LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIGNvbnN0IGF4ID0gYVswXTtcbiAgICAgICAgY29uc3QgYXkgPSBhWzFdO1xuICAgICAgICBjb25zdCBheiA9IGFbMl07XG4gICAgICAgIGNvbnN0IGF3ID0gYVszXTtcbiAgICAgICAgbGV0IGJ4ID0gYlswXTtcbiAgICAgICAgbGV0IGJ5ID0gYlsxXTtcbiAgICAgICAgbGV0IGJ6ID0gYlsyXTtcbiAgICAgICAgbGV0IGJ3ID0gYlszXTtcbiAgICAgICAgbGV0IGNvc09tZWdhID0gYXggKiBieCArIGF5ICogYnkgKyBheiAqIGJ6ICsgYXcgKiBidztcbiAgICAgICAgaWYgKGNvc09tZWdhIDwgMCkge1xuICAgICAgICAgICAgY29zT21lZ2EgPSAtY29zT21lZ2E7XG4gICAgICAgICAgICBieCA9IC1ieDtcbiAgICAgICAgICAgIGJ5ID0gLWJ5O1xuICAgICAgICAgICAgYnogPSAtYno7XG4gICAgICAgICAgICBidyA9IC1idztcbiAgICAgICAgfVxuICAgICAgICBsZXQgc2NhbGUwO1xuICAgICAgICBsZXQgc2NhbGUxO1xuICAgICAgICBpZiAoMS4wIC0gY29zT21lZ2EgPiBFUFNJTE9OKSB7XG4gICAgICAgICAgICBjb25zdCBvbWVnYSA9IE1hdGguYWNvcyhjb3NPbWVnYSk7XG4gICAgICAgICAgICBjb25zdCBzaW5PbWVnYSA9IE1hdGguc2luKG9tZWdhKTtcbiAgICAgICAgICAgIHNjYWxlMCA9IE1hdGguc2luKCgxIC0gdCkgKiBvbWVnYSkgLyBzaW5PbWVnYTtcbiAgICAgICAgICAgIHNjYWxlMSA9IE1hdGguc2luKHQgKiBvbWVnYSkgLyBzaW5PbWVnYTtcbiAgICAgICAgfVxuICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgIHNjYWxlMCA9IDEuMCAtIHQ7XG4gICAgICAgICAgICBzY2FsZTEgPSB0O1xuICAgICAgICB9XG4gICAgICAgIG5ld0RzdFswXSA9IHNjYWxlMCAqIGF4ICsgc2NhbGUxICogYng7XG4gICAgICAgIG5ld0RzdFsxXSA9IHNjYWxlMCAqIGF5ICsgc2NhbGUxICogYnk7XG4gICAgICAgIG5ld0RzdFsyXSA9IHNjYWxlMCAqIGF6ICsgc2NhbGUxICogYno7XG4gICAgICAgIG5ld0RzdFszXSA9IHNjYWxlMCAqIGF3ICsgc2NhbGUxICogYnc7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGUgdGhlIGludmVyc2Ugb2YgYSBxdWF0ZXJuaW9uXG4gICAgICpcbiAgICAgKiBAcGFyYW0gcSAtIHF1YXRlcm5pb24gdG8gY29tcHV0ZSB0aGUgaW52ZXJzZSBvZlxuICAgICAqIEByZXR1cm5zIEEgcXVhdGVybmlvbiB0aGF0IGlzIHRoZSByZXN1bHQgb2YgYSAqIGJcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBpbnZlcnNlKHEsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgY29uc3QgYTAgPSBxWzBdO1xuICAgICAgICBjb25zdCBhMSA9IHFbMV07XG4gICAgICAgIGNvbnN0IGEyID0gcVsyXTtcbiAgICAgICAgY29uc3QgYTMgPSBxWzNdO1xuICAgICAgICBjb25zdCBkb3QgPSBhMCAqIGEwICsgYTEgKiBhMSArIGEyICogYTIgKyBhMyAqIGEzO1xuICAgICAgICBjb25zdCBpbnZEb3QgPSBkb3QgPyAxIC8gZG90IDogMDtcbiAgICAgICAgbmV3RHN0WzBdID0gLWEwICogaW52RG90O1xuICAgICAgICBuZXdEc3RbMV0gPSAtYTEgKiBpbnZEb3Q7XG4gICAgICAgIG5ld0RzdFsyXSA9IC1hMiAqIGludkRvdDtcbiAgICAgICAgbmV3RHN0WzNdID0gYTMgKiBpbnZEb3Q7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGUgdGhlIGNvbmp1Z2F0ZSBvZiBhIHF1YXRlcm5pb25cbiAgICAgKiBGb3IgcXVhdGVybmlvbnMgd2l0aCBhIG1hZ25pdHVkZSBvZiAxIChhIHVuaXQgcXVhdGVybmlvbilcbiAgICAgKiB0aGlzIHJldHVybnMgdGhlIHNhbWUgYXMgdGhlIGludmVyc2UgYnV0IGlzIGZhc3RlciB0byBjYWxjdWxhdGUuXG4gICAgICpcbiAgICAgKiBAcGFyYW0gcSAtIHF1YXRlcm5pb24gdG8gY29tcHV0ZSB0aGUgY29uanVnYXRlIG9mLlxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBjb25qdWdhdGUgb2YgcVxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGNvbmp1Z2F0ZShxLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IC1xWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSAtcVsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gLXFbMl07XG4gICAgICAgIG5ld0RzdFszXSA9IHFbM107XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENyZWF0ZXMgYSBxdWF0ZXJuaW9uIGZyb20gdGhlIGdpdmVuIHJvdGF0aW9uIG1hdHJpeC5cbiAgICAgKlxuICAgICAqIFRoZSBjcmVhdGVkIHF1YXRlcm5pb24gaXMgbm90IG5vcm1hbGl6ZWQuXG4gICAgICpcbiAgICAgKiBAcGFyYW0gbSAtIHJvdGF0aW9uIG1hdHJpeFxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIHRoZSByZXN1bHRcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBmcm9tTWF0KG0sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgLypcbiAgICAgICAgMCAxIDJcbiAgICAgICAgMyA0IDVcbiAgICAgICAgNiA3IDhcbiAgICAgIFxuICAgICAgICAwIDEgMlxuICAgICAgICA0IDUgNlxuICAgICAgICA4IDkgMTBcbiAgICAgICAgICovXG4gICAgICAgIC8vIEFsZ29yaXRobSBpbiBLZW4gU2hvZW1ha2UncyBhcnRpY2xlIGluIDE5ODcgU0lHR1JBUEggY291cnNlIG5vdGVzXG4gICAgICAgIC8vIGFydGljbGUgXCJRdWF0ZXJuaW9uIENhbGN1bHVzIGFuZCBGYXN0IEFuaW1hdGlvblwiLlxuICAgICAgICBjb25zdCB0cmFjZSA9IG1bMF0gKyBtWzVdICsgbVsxMF07XG4gICAgICAgIGlmICh0cmFjZSA+IDAuMCkge1xuICAgICAgICAgICAgLy8gfHd8ID4gMS8yLCBtYXkgYXMgd2VsbCBjaG9vc2UgdyA+IDEvMlxuICAgICAgICAgICAgY29uc3Qgcm9vdCA9IE1hdGguc3FydCh0cmFjZSArIDEpOyAvLyAyd1xuICAgICAgICAgICAgbmV3RHN0WzNdID0gMC41ICogcm9vdDtcbiAgICAgICAgICAgIGNvbnN0IGludlJvb3QgPSAwLjUgLyByb290OyAvLyAxLyg0dylcbiAgICAgICAgICAgIG5ld0RzdFswXSA9IChtWzZdIC0gbVs5XSkgKiBpbnZSb290O1xuICAgICAgICAgICAgbmV3RHN0WzFdID0gKG1bOF0gLSBtWzJdKSAqIGludlJvb3Q7XG4gICAgICAgICAgICBuZXdEc3RbMl0gPSAobVsxXSAtIG1bNF0pICogaW52Um9vdDtcbiAgICAgICAgfVxuICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgIC8vIHx3fCA8PSAxLzJcbiAgICAgICAgICAgIGxldCBpID0gMDtcbiAgICAgICAgICAgIGlmIChtWzVdID4gbVswXSkge1xuICAgICAgICAgICAgICAgIGkgPSAxO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgaWYgKG1bMTBdID4gbVtpICogNCArIGldKSB7XG4gICAgICAgICAgICAgICAgaSA9IDI7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBjb25zdCBqID0gKGkgKyAxKSAlIDM7XG4gICAgICAgICAgICBjb25zdCBrID0gKGkgKyAyKSAlIDM7XG4gICAgICAgICAgICBjb25zdCByb290ID0gTWF0aC5zcXJ0KG1baSAqIDQgKyBpXSAtIG1baiAqIDQgKyBqXSAtIG1bayAqIDQgKyBrXSArIDEuMCk7XG4gICAgICAgICAgICBuZXdEc3RbaV0gPSAwLjUgKiByb290O1xuICAgICAgICAgICAgY29uc3QgaW52Um9vdCA9IDAuNSAvIHJvb3Q7XG4gICAgICAgICAgICBuZXdEc3RbM10gPSAobVtqICogNCArIGtdIC0gbVtrICogNCArIGpdKSAqIGludlJvb3Q7XG4gICAgICAgICAgICBuZXdEc3Rbal0gPSAobVtqICogNCArIGldICsgbVtpICogNCArIGpdKSAqIGludlJvb3Q7XG4gICAgICAgICAgICBuZXdEc3Rba10gPSAobVtrICogNCArIGldICsgbVtpICogNCArIGtdKSAqIGludlJvb3Q7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIHF1YXRlcm5pb24gZnJvbSB0aGUgZ2l2ZW4gZXVsZXIgYW5nbGUgeCwgeSwgeiB1c2luZyB0aGUgcHJvdmlkZWQgaW50cmluc2ljIG9yZGVyIGZvciB0aGUgY29udmVyc2lvbi5cbiAgICAgKlxuICAgICAqIEBwYXJhbSB4QW5nbGVJblJhZGlhbnMgLSBhbmdsZSB0byByb3RhdGUgYXJvdW5kIFggYXhpcyBpbiByYWRpYW5zLlxuICAgICAqIEBwYXJhbSB5QW5nbGVJblJhZGlhbnMgLSBhbmdsZSB0byByb3RhdGUgYXJvdW5kIFkgYXhpcyBpbiByYWRpYW5zLlxuICAgICAqIEBwYXJhbSB6QW5nbGVJblJhZGlhbnMgLSBhbmdsZSB0byByb3RhdGUgYXJvdW5kIFogYXhpcyBpbiByYWRpYW5zLlxuICAgICAqIEBwYXJhbSBvcmRlciAtIG9yZGVyIHRvIGFwcGx5IGV1bGVyIGFuZ2xlc1xuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgcXVhdGVybmlvbiByZXByZXNlbnRpbmcgdGhlIHNhbWUgcm90YXRpb24gYXMgdGhlIGV1bGVyIGFuZ2xlcyBhcHBsaWVkIGluIHRoZSBnaXZlbiBvcmRlclxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGZyb21FdWxlcih4QW5nbGVJblJhZGlhbnMsIHlBbmdsZUluUmFkaWFucywgekFuZ2xlSW5SYWRpYW5zLCBvcmRlciwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBjb25zdCB4SGFsZkFuZ2xlID0geEFuZ2xlSW5SYWRpYW5zICogMC41O1xuICAgICAgICBjb25zdCB5SGFsZkFuZ2xlID0geUFuZ2xlSW5SYWRpYW5zICogMC41O1xuICAgICAgICBjb25zdCB6SGFsZkFuZ2xlID0gekFuZ2xlSW5SYWRpYW5zICogMC41O1xuICAgICAgICBjb25zdCBzeCA9IE1hdGguc2luKHhIYWxmQW5nbGUpO1xuICAgICAgICBjb25zdCBjeCA9IE1hdGguY29zKHhIYWxmQW5nbGUpO1xuICAgICAgICBjb25zdCBzeSA9IE1hdGguc2luKHlIYWxmQW5nbGUpO1xuICAgICAgICBjb25zdCBjeSA9IE1hdGguY29zKHlIYWxmQW5nbGUpO1xuICAgICAgICBjb25zdCBzeiA9IE1hdGguc2luKHpIYWxmQW5nbGUpO1xuICAgICAgICBjb25zdCBjeiA9IE1hdGguY29zKHpIYWxmQW5nbGUpO1xuICAgICAgICBzd2l0Y2ggKG9yZGVyKSB7XG4gICAgICAgICAgICBjYXNlICd4eXonOlxuICAgICAgICAgICAgICAgIG5ld0RzdFswXSA9IHN4ICogY3kgKiBjeiArIGN4ICogc3kgKiBzejtcbiAgICAgICAgICAgICAgICBuZXdEc3RbMV0gPSBjeCAqIHN5ICogY3ogLSBzeCAqIGN5ICogc3o7XG4gICAgICAgICAgICAgICAgbmV3RHN0WzJdID0gY3ggKiBjeSAqIHN6ICsgc3ggKiBzeSAqIGN6O1xuICAgICAgICAgICAgICAgIG5ld0RzdFszXSA9IGN4ICogY3kgKiBjeiAtIHN4ICogc3kgKiBzejtcbiAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgIGNhc2UgJ3h6eSc6XG4gICAgICAgICAgICAgICAgbmV3RHN0WzBdID0gc3ggKiBjeSAqIGN6IC0gY3ggKiBzeSAqIHN6O1xuICAgICAgICAgICAgICAgIG5ld0RzdFsxXSA9IGN4ICogc3kgKiBjeiAtIHN4ICogY3kgKiBzejtcbiAgICAgICAgICAgICAgICBuZXdEc3RbMl0gPSBjeCAqIGN5ICogc3ogKyBzeCAqIHN5ICogY3o7XG4gICAgICAgICAgICAgICAgbmV3RHN0WzNdID0gY3ggKiBjeSAqIGN6ICsgc3ggKiBzeSAqIHN6O1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgY2FzZSAneXh6JzpcbiAgICAgICAgICAgICAgICBuZXdEc3RbMF0gPSBzeCAqIGN5ICogY3ogKyBjeCAqIHN5ICogc3o7XG4gICAgICAgICAgICAgICAgbmV3RHN0WzFdID0gY3ggKiBzeSAqIGN6IC0gc3ggKiBjeSAqIHN6O1xuICAgICAgICAgICAgICAgIG5ld0RzdFsyXSA9IGN4ICogY3kgKiBzeiAtIHN4ICogc3kgKiBjejtcbiAgICAgICAgICAgICAgICBuZXdEc3RbM10gPSBjeCAqIGN5ICogY3ogKyBzeCAqIHN5ICogc3o7XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICBjYXNlICd5engnOlxuICAgICAgICAgICAgICAgIG5ld0RzdFswXSA9IHN4ICogY3kgKiBjeiArIGN4ICogc3kgKiBzejtcbiAgICAgICAgICAgICAgICBuZXdEc3RbMV0gPSBjeCAqIHN5ICogY3ogKyBzeCAqIGN5ICogc3o7XG4gICAgICAgICAgICAgICAgbmV3RHN0WzJdID0gY3ggKiBjeSAqIHN6IC0gc3ggKiBzeSAqIGN6O1xuICAgICAgICAgICAgICAgIG5ld0RzdFszXSA9IGN4ICogY3kgKiBjeiAtIHN4ICogc3kgKiBzejtcbiAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgIGNhc2UgJ3p4eSc6XG4gICAgICAgICAgICAgICAgbmV3RHN0WzBdID0gc3ggKiBjeSAqIGN6IC0gY3ggKiBzeSAqIHN6O1xuICAgICAgICAgICAgICAgIG5ld0RzdFsxXSA9IGN4ICogc3kgKiBjeiArIHN4ICogY3kgKiBzejtcbiAgICAgICAgICAgICAgICBuZXdEc3RbMl0gPSBjeCAqIGN5ICogc3ogKyBzeCAqIHN5ICogY3o7XG4gICAgICAgICAgICAgICAgbmV3RHN0WzNdID0gY3ggKiBjeSAqIGN6IC0gc3ggKiBzeSAqIHN6O1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgY2FzZSAnenl4JzpcbiAgICAgICAgICAgICAgICBuZXdEc3RbMF0gPSBzeCAqIGN5ICogY3ogLSBjeCAqIHN5ICogc3o7XG4gICAgICAgICAgICAgICAgbmV3RHN0WzFdID0gY3ggKiBzeSAqIGN6ICsgc3ggKiBjeSAqIHN6O1xuICAgICAgICAgICAgICAgIG5ld0RzdFsyXSA9IGN4ICogY3kgKiBzeiAtIHN4ICogc3kgKiBjejtcbiAgICAgICAgICAgICAgICBuZXdEc3RbM10gPSBjeCAqIGN5ICogY3ogKyBzeCAqIHN5ICogc3o7XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICBkZWZhdWx0OlxuICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biByb3RhdGlvbiBvcmRlcjogJHtvcmRlcn1gKTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb3BpZXMgYSBxdWF0ZXJuaW9uLiAoc2FtZSBhcyB7QGxpbmsgcXVhdC5jbG9uZX0pXG4gICAgICogQWxzbyBzZWUge0BsaW5rIHF1YXQuY3JlYXRlfSBhbmQge0BsaW5rIHF1YXQuc2V0fVxuICAgICAqIEBwYXJhbSBxIC0gVGhlIHF1YXRlcm5pb24uXG4gICAgICogQHBhcmFtIGRzdCAtIHF1YXRlcm5pb24gdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSBxdWF0ZXJuaW9uIHRoYXQgaXMgYSBjb3B5IG9mIHFcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBjb3B5KHEsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gcVswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gcVsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gcVsyXTtcbiAgICAgICAgbmV3RHN0WzNdID0gcVszXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ2xvbmVzIGEgcXVhdGVybmlvbi4gKHNhbWUgYXMge0BsaW5rIHF1YXQuY29weX0pXG4gICAgICogQWxzbyBzZWUge0BsaW5rIHF1YXQuY3JlYXRlfSBhbmQge0BsaW5rIHF1YXQuc2V0fVxuICAgICAqIEBwYXJhbSBxIC0gVGhlIHF1YXRlcm5pb24uXG4gICAgICogQHBhcmFtIGRzdCAtIHF1YXRlcm5pb24gdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSBjb3B5IG9mIHEuXG4gICAgICovXG4gICAgY29uc3QgY2xvbmUgPSBjb3B5O1xuICAgIC8qKlxuICAgICAqIEFkZHMgdHdvIHF1YXRlcm5pb25zOyBhc3N1bWVzIGEgYW5kIGIgaGF2ZSB0aGUgc2FtZSBkaW1lbnNpb24uXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHF1YXRlcm5pb24uXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHF1YXRlcm5pb24uXG4gICAgICogQHBhcmFtIGRzdCAtIHF1YXRlcm5pb24gdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSBxdWF0ZXJuaW9uIHRoYXQgaXMgdGhlIHN1bSBvZiBhIGFuZCBiLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGFkZChhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IGFbMF0gKyBiWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSBhWzFdICsgYlsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gYVsyXSArIGJbMl07XG4gICAgICAgIG5ld0RzdFszXSA9IGFbM10gKyBiWzNdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBTdWJ0cmFjdHMgdHdvIHF1YXRlcm5pb25zLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCBxdWF0ZXJuaW9uLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCBxdWF0ZXJuaW9uLlxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgcXVhdGVybmlvbiB0aGF0IGlzIHRoZSBkaWZmZXJlbmNlIG9mIGEgYW5kIGIuXG4gICAgICovXG4gICAgZnVuY3Rpb24gc3VidHJhY3QoYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSBhWzBdIC0gYlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSAtIGJbMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IGFbMl0gLSBiWzJdO1xuICAgICAgICBuZXdEc3RbM10gPSBhWzNdIC0gYlszXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogU3VidHJhY3RzIHR3byBxdWF0ZXJuaW9ucy5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgcXVhdGVybmlvbi5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgcXVhdGVybmlvbi5cbiAgICAgKiBAcGFyYW0gZHN0IC0gcXVhdGVybmlvbiB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHF1YXRlcm5pb24gdGhhdCBpcyB0aGUgZGlmZmVyZW5jZSBvZiBhIGFuZCBiLlxuICAgICAqL1xuICAgIGNvbnN0IHN1YiA9IHN1YnRyYWN0O1xuICAgIC8qKlxuICAgICAqIE11bHRpcGxpZXMgYSBxdWF0ZXJuaW9uIGJ5IGEgc2NhbGFyLlxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHF1YXRlcm5pb24uXG4gICAgICogQHBhcmFtIGsgLSBUaGUgc2NhbGFyLlxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBzY2FsZWQgcXVhdGVybmlvbi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBtdWxTY2FsYXIodiwgaywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2WzBdICogaztcbiAgICAgICAgbmV3RHN0WzFdID0gdlsxXSAqIGs7XG4gICAgICAgIG5ld0RzdFsyXSA9IHZbMl0gKiBrO1xuICAgICAgICBuZXdEc3RbM10gPSB2WzNdICogaztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogTXVsdGlwbGllcyBhIHF1YXRlcm5pb24gYnkgYSBzY2FsYXIuIChzYW1lIGFzIG11bFNjYWxhcilcbiAgICAgKiBAcGFyYW0gdiAtIFRoZSBxdWF0ZXJuaW9uLlxuICAgICAqIEBwYXJhbSBrIC0gVGhlIHNjYWxhci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gcXVhdGVybmlvbiB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIHF1YXRlcm5pb24uXG4gICAgICovXG4gICAgY29uc3Qgc2NhbGUgPSBtdWxTY2FsYXI7XG4gICAgLyoqXG4gICAgICogRGl2aWRlcyBhIHZlY3RvciBieSBhIHNjYWxhci5cbiAgICAgKiBAcGFyYW0gdiAtIFRoZSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGsgLSBUaGUgc2NhbGFyLlxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBzY2FsZWQgcXVhdGVybmlvbi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBkaXZTY2FsYXIodiwgaywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2WzBdIC8gaztcbiAgICAgICAgbmV3RHN0WzFdID0gdlsxXSAvIGs7XG4gICAgICAgIG5ld0RzdFsyXSA9IHZbMl0gLyBrO1xuICAgICAgICBuZXdEc3RbM10gPSB2WzNdIC8gaztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIGRvdCBwcm9kdWN0IG9mIHR3byBxdWF0ZXJuaW9uc1xuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCBxdWF0ZXJuaW9uLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCBxdWF0ZXJuaW9uLlxuICAgICAqIEByZXR1cm5zIGRvdCBwcm9kdWN0XG4gICAgICovXG4gICAgZnVuY3Rpb24gZG90KGEsIGIpIHtcbiAgICAgICAgcmV0dXJuIChhWzBdICogYlswXSkgKyAoYVsxXSAqIGJbMV0pICsgKGFbMl0gKiBiWzJdKSArIChhWzNdICogYlszXSk7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFBlcmZvcm1zIGxpbmVhciBpbnRlcnBvbGF0aW9uIG9uIHR3byBxdWF0ZXJuaW9ucy5cbiAgICAgKiBHaXZlbiBxdWF0ZXJuaW9ucyBhIGFuZCBiIGFuZCBpbnRlcnBvbGF0aW9uIGNvZWZmaWNpZW50IHQsIHJldHVybnNcbiAgICAgKiBhICsgdCAqIChiIC0gYSkuXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHF1YXRlcm5pb24uXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHF1YXRlcm5pb24uXG4gICAgICogQHBhcmFtIHQgLSBJbnRlcnBvbGF0aW9uIGNvZWZmaWNpZW50LlxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBsaW5lYXIgaW50ZXJwb2xhdGVkIHJlc3VsdC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBsZXJwKGEsIGIsIHQsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSArIHQgKiAoYlswXSAtIGFbMF0pO1xuICAgICAgICBuZXdEc3RbMV0gPSBhWzFdICsgdCAqIChiWzFdIC0gYVsxXSk7XG4gICAgICAgIG5ld0RzdFsyXSA9IGFbMl0gKyB0ICogKGJbMl0gLSBhWzJdKTtcbiAgICAgICAgbmV3RHN0WzNdID0gYVszXSArIHQgKiAoYlszXSAtIGFbM10pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgbGVuZ3RoIG9mIHF1YXRlcm5pb25cbiAgICAgKiBAcGFyYW0gdiAtIHF1YXRlcm5pb24uXG4gICAgICogQHJldHVybnMgbGVuZ3RoIG9mIHF1YXRlcm5pb24uXG4gICAgICovXG4gICAgZnVuY3Rpb24gbGVuZ3RoKHYpIHtcbiAgICAgICAgY29uc3QgdjAgPSB2WzBdO1xuICAgICAgICBjb25zdCB2MSA9IHZbMV07XG4gICAgICAgIGNvbnN0IHYyID0gdlsyXTtcbiAgICAgICAgY29uc3QgdjMgPSB2WzNdO1xuICAgICAgICByZXR1cm4gTWF0aC5zcXJ0KHYwICogdjAgKyB2MSAqIHYxICsgdjIgKiB2MiArIHYzICogdjMpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgbGVuZ3RoIG9mIHF1YXRlcm5pb24gKHNhbWUgYXMgbGVuZ3RoKVxuICAgICAqIEBwYXJhbSB2IC0gcXVhdGVybmlvbi5cbiAgICAgKiBAcmV0dXJucyBsZW5ndGggb2YgcXVhdGVybmlvbi5cbiAgICAgKi9cbiAgICBjb25zdCBsZW4gPSBsZW5ndGg7XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIHNxdWFyZSBvZiB0aGUgbGVuZ3RoIG9mIHF1YXRlcm5pb25cbiAgICAgKiBAcGFyYW0gdiAtIHF1YXRlcm5pb24uXG4gICAgICogQHJldHVybnMgc3F1YXJlIG9mIHRoZSBsZW5ndGggb2YgcXVhdGVybmlvbi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBsZW5ndGhTcSh2KSB7XG4gICAgICAgIGNvbnN0IHYwID0gdlswXTtcbiAgICAgICAgY29uc3QgdjEgPSB2WzFdO1xuICAgICAgICBjb25zdCB2MiA9IHZbMl07XG4gICAgICAgIGNvbnN0IHYzID0gdlszXTtcbiAgICAgICAgcmV0dXJuIHYwICogdjAgKyB2MSAqIHYxICsgdjIgKiB2MiArIHYzICogdjM7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBzcXVhcmUgb2YgdGhlIGxlbmd0aCBvZiBxdWF0ZXJuaW9uIChzYW1lIGFzIGxlbmd0aFNxKVxuICAgICAqIEBwYXJhbSB2IC0gcXVhdGVybmlvbi5cbiAgICAgKiBAcmV0dXJucyBzcXVhcmUgb2YgdGhlIGxlbmd0aCBvZiBxdWF0ZXJuaW9uLlxuICAgICAqL1xuICAgIGNvbnN0IGxlblNxID0gbGVuZ3RoU3E7XG4gICAgLyoqXG4gICAgICogRGl2aWRlcyBhIHF1YXRlcm5pb24gYnkgaXRzIEV1Y2xpZGVhbiBsZW5ndGggYW5kIHJldHVybnMgdGhlIHF1b3RpZW50LlxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHF1YXRlcm5pb24uXG4gICAgICogQHBhcmFtIGRzdCAtIHF1YXRlcm5pb24gdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgVGhlIG5vcm1hbGl6ZWQgcXVhdGVybmlvbi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBub3JtYWxpemUodiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBjb25zdCB2MCA9IHZbMF07XG4gICAgICAgIGNvbnN0IHYxID0gdlsxXTtcbiAgICAgICAgY29uc3QgdjIgPSB2WzJdO1xuICAgICAgICBjb25zdCB2MyA9IHZbM107XG4gICAgICAgIGNvbnN0IGxlbiA9IE1hdGguc3FydCh2MCAqIHYwICsgdjEgKiB2MSArIHYyICogdjIgKyB2MyAqIHYzKTtcbiAgICAgICAgaWYgKGxlbiA+IDAuMDAwMDEpIHtcbiAgICAgICAgICAgIG5ld0RzdFswXSA9IHYwIC8gbGVuO1xuICAgICAgICAgICAgbmV3RHN0WzFdID0gdjEgLyBsZW47XG4gICAgICAgICAgICBuZXdEc3RbMl0gPSB2MiAvIGxlbjtcbiAgICAgICAgICAgIG5ld0RzdFszXSA9IHYzIC8gbGVuO1xuICAgICAgICB9XG4gICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0gMDtcbiAgICAgICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICAgICAgbmV3RHN0WzNdID0gMTtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDaGVjayBpZiAyIHF1YXRlcm5pb25zIGFyZSBhcHByb3hpbWF0ZWx5IGVxdWFsXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHF1YXRlcm5pb24uXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHF1YXRlcm5pb24uXG4gICAgICogQHJldHVybnMgdHJ1ZSBpZiBxdWF0ZXJuaW9ucyBhcmUgYXBwcm94aW1hdGVseSBlcXVhbFxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGVxdWFsc0FwcHJveGltYXRlbHkoYSwgYikge1xuICAgICAgICByZXR1cm4gTWF0aC5hYnMoYVswXSAtIGJbMF0pIDwgRVBTSUxPTiAmJlxuICAgICAgICAgICAgTWF0aC5hYnMoYVsxXSAtIGJbMV0pIDwgRVBTSUxPTiAmJlxuICAgICAgICAgICAgTWF0aC5hYnMoYVsyXSAtIGJbMl0pIDwgRVBTSUxPTiAmJlxuICAgICAgICAgICAgTWF0aC5hYnMoYVszXSAtIGJbM10pIDwgRVBTSUxPTjtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ2hlY2sgaWYgMiBxdWF0ZXJuaW9ucyBhcmUgZXhhY3RseSBlcXVhbFxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCBxdWF0ZXJuaW9uLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCBxdWF0ZXJuaW9uLlxuICAgICAqIEByZXR1cm5zIHRydWUgaWYgcXVhdGVybmlvbnMgYXJlIGV4YWN0bHkgZXF1YWxcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBlcXVhbHMoYSwgYikge1xuICAgICAgICByZXR1cm4gYVswXSA9PT0gYlswXSAmJiBhWzFdID09PSBiWzFdICYmIGFbMl0gPT09IGJbMl0gJiYgYVszXSA9PT0gYlszXTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhbiBpZGVudGl0eSBxdWF0ZXJuaW9uXG4gICAgICogQHBhcmFtIGRzdCAtIHF1YXRlcm5pb24gdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgYW4gaWRlbnRpdHkgcXVhdGVybmlvblxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGlkZW50aXR5KGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gMDtcbiAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgbmV3RHN0WzJdID0gMDtcbiAgICAgICAgbmV3RHN0WzNdID0gMTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgY29uc3QgdGVtcFZlYzMgPSB2ZWMzLmNyZWF0ZSgpO1xuICAgIGNvbnN0IHhVbml0VmVjMyA9IHZlYzMuY3JlYXRlKCk7XG4gICAgY29uc3QgeVVuaXRWZWMzID0gdmVjMy5jcmVhdGUoKTtcbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyBhIHF1YXRlcm5pb24gdG8gcmVwcmVzZW50IHRoZSBzaG9ydGVzdCByb3RhdGlvbiBmcm9tIG9uZSB2ZWN0b3IgdG8gYW5vdGhlci5cbiAgICAgKlxuICAgICAqIEBwYXJhbSBhVW5pdCAtIHRoZSBzdGFydCB2ZWN0b3JcbiAgICAgKiBAcGFyYW0gYlVuaXQgLSB0aGUgZW5kIHZlY3RvclxuICAgICAqIEBwYXJhbSBkc3QgLSBxdWF0ZXJuaW9uIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIHRoZSByZXN1bHRcbiAgICAgKi9cbiAgICBmdW5jdGlvbiByb3RhdGlvblRvKGFVbml0LCBiVW5pdCwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBjb25zdCBkb3QgPSB2ZWMzLmRvdChhVW5pdCwgYlVuaXQpO1xuICAgICAgICBpZiAoZG90IDwgLTAuOTk5OTk5KSB7XG4gICAgICAgICAgICB2ZWMzLmNyb3NzKHhVbml0VmVjMywgYVVuaXQsIHRlbXBWZWMzKTtcbiAgICAgICAgICAgIGlmICh2ZWMzLmxlbih0ZW1wVmVjMykgPCAwLjAwMDAwMSkge1xuICAgICAgICAgICAgICAgIHZlYzMuY3Jvc3MoeVVuaXRWZWMzLCBhVW5pdCwgdGVtcFZlYzMpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdmVjMy5ub3JtYWxpemUodGVtcFZlYzMsIHRlbXBWZWMzKTtcbiAgICAgICAgICAgIGZyb21BeGlzQW5nbGUodGVtcFZlYzMsIE1hdGguUEksIG5ld0RzdCk7XG4gICAgICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgICAgICB9XG4gICAgICAgIGVsc2UgaWYgKGRvdCA+IDAuOTk5OTk5KSB7XG4gICAgICAgICAgICBuZXdEc3RbMF0gPSAwO1xuICAgICAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgICAgIG5ld0RzdFsyXSA9IDA7XG4gICAgICAgICAgICBuZXdEc3RbM10gPSAxO1xuICAgICAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICAgICAgfVxuICAgICAgICBlbHNlIHtcbiAgICAgICAgICAgIHZlYzMuY3Jvc3MoYVVuaXQsIGJVbml0LCB0ZW1wVmVjMyk7XG4gICAgICAgICAgICBuZXdEc3RbMF0gPSB0ZW1wVmVjM1swXTtcbiAgICAgICAgICAgIG5ld0RzdFsxXSA9IHRlbXBWZWMzWzFdO1xuICAgICAgICAgICAgbmV3RHN0WzJdID0gdGVtcFZlYzNbMl07XG4gICAgICAgICAgICBuZXdEc3RbM10gPSAxICsgZG90O1xuICAgICAgICAgICAgcmV0dXJuIG5vcm1hbGl6ZShuZXdEc3QsIG5ld0RzdCk7XG4gICAgICAgIH1cbiAgICB9XG4gICAgY29uc3QgdGVtcFF1YXQxID0gbmV3IEN0b3IoNCk7XG4gICAgY29uc3QgdGVtcFF1YXQyID0gbmV3IEN0b3IoNCk7XG4gICAgLyoqXG4gICAgICogUGVyZm9ybXMgYSBzcGhlcmljYWwgbGluZWFyIGludGVycG9sYXRpb24gd2l0aCB0d28gY29udHJvbCBwb2ludHNcbiAgICAgKlxuICAgICAqIEBwYXJhbSBhIC0gdGhlIGZpcnN0IHF1YXRlcm5pb25cbiAgICAgKiBAcGFyYW0gYiAtIHRoZSBzZWNvbmQgcXVhdGVybmlvblxuICAgICAqIEBwYXJhbSBjIC0gdGhlIHRoaXJkIHF1YXRlcm5pb25cbiAgICAgKiBAcGFyYW0gZCAtIHRoZSBmb3VydGggcXVhdGVybmlvblxuICAgICAqIEBwYXJhbSB0IC0gSW50ZXJwb2xhdGlvbiBjb2VmZmljaWVudCAwIHRvIDFcbiAgICAgKiBAcmV0dXJucyByZXN1bHRcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzcWxlcnAoYSwgYiwgYywgZCwgdCwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBzbGVycChhLCBkLCB0LCB0ZW1wUXVhdDEpO1xuICAgICAgICBzbGVycChiLCBjLCB0LCB0ZW1wUXVhdDIpO1xuICAgICAgICBzbGVycCh0ZW1wUXVhdDEsIHRlbXBRdWF0MiwgMiAqIHQgKiAoMSAtIHQpLCBuZXdEc3QpO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICByZXR1cm4ge1xuICAgICAgICBjcmVhdGUsXG4gICAgICAgIGZyb21WYWx1ZXMsXG4gICAgICAgIHNldCxcbiAgICAgICAgZnJvbUF4aXNBbmdsZSxcbiAgICAgICAgdG9BeGlzQW5nbGUsXG4gICAgICAgIGFuZ2xlLFxuICAgICAgICBtdWx0aXBseSxcbiAgICAgICAgbXVsLFxuICAgICAgICByb3RhdGVYLFxuICAgICAgICByb3RhdGVZLFxuICAgICAgICByb3RhdGVaLFxuICAgICAgICBzbGVycCxcbiAgICAgICAgaW52ZXJzZSxcbiAgICAgICAgY29uanVnYXRlLFxuICAgICAgICBmcm9tTWF0LFxuICAgICAgICBmcm9tRXVsZXIsXG4gICAgICAgIGNvcHksXG4gICAgICAgIGNsb25lLFxuICAgICAgICBhZGQsXG4gICAgICAgIHN1YnRyYWN0LFxuICAgICAgICBzdWIsXG4gICAgICAgIG11bFNjYWxhcixcbiAgICAgICAgc2NhbGUsXG4gICAgICAgIGRpdlNjYWxhcixcbiAgICAgICAgZG90LFxuICAgICAgICBsZXJwLFxuICAgICAgICBsZW5ndGgsXG4gICAgICAgIGxlbixcbiAgICAgICAgbGVuZ3RoU3EsXG4gICAgICAgIGxlblNxLFxuICAgICAgICBub3JtYWxpemUsXG4gICAgICAgIGVxdWFsc0FwcHJveGltYXRlbHksXG4gICAgICAgIGVxdWFscyxcbiAgICAgICAgaWRlbnRpdHksXG4gICAgICAgIHJvdGF0aW9uVG8sXG4gICAgICAgIHNxbGVycCxcbiAgICB9O1xufVxuY29uc3QgY2FjaGUkMSA9IG5ldyBNYXAoKTtcbi8qKlxuICpcbiAqIFF1YXQ0IG1hdGggZnVuY3Rpb25zLlxuICpcbiAqIEFsbW9zdCBhbGwgZnVuY3Rpb25zIHRha2UgYW4gb3B0aW9uYWwgYG5ld0RzdGAgYXJndW1lbnQuIElmIGl0IGlzIG5vdCBwYXNzZWQgaW4gdGhlXG4gKiBmdW5jdGlvbnMgd2lsbCBjcmVhdGUgYSBuZXcgYFF1YXQ0YC4gSW4gb3RoZXIgd29yZHMgeW91IGNhbiBkbyB0aGlzXG4gKlxuICogICAgIGNvbnN0IHYgPSBxdWF0NC5jcm9zcyh2MSwgdjIpOyAgLy8gQ3JlYXRlcyBhIG5ldyBRdWF0NCB3aXRoIHRoZSBjcm9zcyBwcm9kdWN0IG9mIHYxIHggdjIuXG4gKlxuICogb3JcbiAqXG4gKiAgICAgY29uc3QgdiA9IHF1YXQ0LmNyZWF0ZSgpO1xuICogICAgIHF1YXQ0LmNyb3NzKHYxLCB2Miwgdik7ICAvLyBQdXRzIHRoZSBjcm9zcyBwcm9kdWN0IG9mIHYxIHggdjIgaW4gdlxuICpcbiAqIFRoZSBmaXJzdCBzdHlsZSBpcyBvZnRlbiBlYXNpZXIgYnV0IGRlcGVuZGluZyBvbiB3aGVyZSBpdCdzIHVzZWQgaXQgZ2VuZXJhdGVzIGdhcmJhZ2Ugd2hlcmVcbiAqIGFzIHRoZXJlIGlzIGFsbW9zdCBuZXZlciBhbGxvY2F0aW9uIHdpdGggdGhlIHNlY29uZCBzdHlsZS5cbiAqXG4gKiBJdCBpcyBhbHdheXMgc2FmZSB0byBwYXNzIGFueSB2ZWN0b3IgYXMgdGhlIGRlc3RpbmF0aW9uLiBTbyBmb3IgZXhhbXBsZVxuICpcbiAqICAgICBxdWF0NC5jcm9zcyh2MSwgdjIsIHYxKTsgIC8vIFB1dHMgdGhlIGNyb3NzIHByb2R1Y3Qgb2YgdjEgeCB2MiBpbiB2MVxuICpcbiAqL1xuZnVuY3Rpb24gZ2V0QVBJJDEoQ3Rvcikge1xuICAgIGxldCBhcGkgPSBjYWNoZSQxLmdldChDdG9yKTtcbiAgICBpZiAoIWFwaSkge1xuICAgICAgICBhcGkgPSBnZXRBUElJbXBsJDEoQ3Rvcik7XG4gICAgICAgIGNhY2hlJDEuc2V0KEN0b3IsIGFwaSk7XG4gICAgfVxuICAgIHJldHVybiBhcGk7XG59XG5cbi8qXG4gKiBDb3B5cmlnaHQgMjAyMiBHcmVnZyBUYXZhcmVzXG4gKlxuICogUGVybWlzc2lvbiBpcyBoZXJlYnkgZ3JhbnRlZCwgZnJlZSBvZiBjaGFyZ2UsIHRvIGFueSBwZXJzb24gb2J0YWluaW5nIGFcbiAqIGNvcHkgb2YgdGhpcyBzb2Z0d2FyZSBhbmQgYXNzb2NpYXRlZCBkb2N1bWVudGF0aW9uIGZpbGVzICh0aGUgXCJTb2Z0d2FyZVwiKSxcbiAqIHRvIGRlYWwgaW4gdGhlIFNvZnR3YXJlIHdpdGhvdXQgcmVzdHJpY3Rpb24sIGluY2x1ZGluZyB3aXRob3V0IGxpbWl0YXRpb25cbiAqIHRoZSByaWdodHMgdG8gdXNlLCBjb3B5LCBtb2RpZnksIG1lcmdlLCBwdWJsaXNoLCBkaXN0cmlidXRlLCBzdWJsaWNlbnNlLFxuICogYW5kL29yIHNlbGwgY29waWVzIG9mIHRoZSBTb2Z0d2FyZSwgYW5kIHRvIHBlcm1pdCBwZXJzb25zIHRvIHdob20gdGhlXG4gKiBTb2Z0d2FyZSBpcyBmdXJuaXNoZWQgdG8gZG8gc28sIHN1YmplY3QgdG8gdGhlIGZvbGxvd2luZyBjb25kaXRpb25zOlxuICpcbiAqIFRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluXG4gKiBhbGwgY29waWVzIG9yIHN1YnN0YW50aWFsIHBvcnRpb25zIG9mIHRoZSBTb2Z0d2FyZS5cbiAqXG4gKiBUSEUgU09GVFdBUkUgSVMgUFJPVklERUQgXCJBUyBJU1wiLCBXSVRIT1VUIFdBUlJBTlRZIE9GIEFOWSBLSU5ELCBFWFBSRVNTIE9SXG4gKiBJTVBMSUVELCBJTkNMVURJTkcgQlVUIE5PVCBMSU1JVEVEIFRPIFRIRSBXQVJSQU5USUVTIE9GIE1FUkNIQU5UQUJJTElUWSxcbiAqIEZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFIEFORCBOT05JTkZSSU5HRU1FTlQuICBJTiBOTyBFVkVOVCBTSEFMTFxuICogVEhFIEFVVEhPUlMgT1IgQ09QWVJJR0hUIEhPTERFUlMgQkUgTElBQkxFIEZPUiBBTlkgQ0xBSU0sIERBTUFHRVMgT1IgT1RIRVJcbiAqIExJQUJJTElUWSwgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HXG4gKiBGUk9NLCBPVVQgT0YgT1IgSU4gQ09OTkVDVElPTiBXSVRIIFRIRSBTT0ZUV0FSRSBPUiBUSEUgVVNFIE9SIE9USEVSXG4gKiBERUFMSU5HUyBJTiBUSEUgU09GVFdBUkUuXG4gKi9cbi8qKlxuICogR2VuZXJhdGVzIGFtIHR5cGVkIEFQSSBmb3IgVmVjNFxuICogKi9cbmZ1bmN0aW9uIGdldEFQSUltcGwoQ3Rvcikge1xuICAgIC8qKlxuICAgICAqIENyZWF0ZXMgYSB2ZWM0OyBtYXkgYmUgY2FsbGVkIHdpdGggeCwgeSwgeiB0byBzZXQgaW5pdGlhbCB2YWx1ZXMuXG4gICAgICogQHBhcmFtIHggLSBJbml0aWFsIHggdmFsdWUuXG4gICAgICogQHBhcmFtIHkgLSBJbml0aWFsIHkgdmFsdWUuXG4gICAgICogQHBhcmFtIHogLSBJbml0aWFsIHogdmFsdWUuXG4gICAgICogQHBhcmFtIHcgLSBJbml0aWFsIHcgdmFsdWUuXG4gICAgICogQHJldHVybnMgdGhlIGNyZWF0ZWQgdmVjdG9yXG4gICAgICovXG4gICAgZnVuY3Rpb24gY3JlYXRlKHgsIHksIHosIHcpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gbmV3IEN0b3IoNCk7XG4gICAgICAgIGlmICh4ICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgIG5ld0RzdFswXSA9IHg7XG4gICAgICAgICAgICBpZiAoeSAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgbmV3RHN0WzFdID0geTtcbiAgICAgICAgICAgICAgICBpZiAoeiAhPT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICAgICAgICAgIG5ld0RzdFsyXSA9IHo7XG4gICAgICAgICAgICAgICAgICAgIGlmICh3ICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIG5ld0RzdFszXSA9IHc7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ3JlYXRlcyBhIHZlYzQ7IG1heSBiZSBjYWxsZWQgd2l0aCB4LCB5LCB6IHRvIHNldCBpbml0aWFsIHZhbHVlcy4gKHNhbWUgYXMgY3JlYXRlKVxuICAgICAqIEBwYXJhbSB4IC0gSW5pdGlhbCB4IHZhbHVlLlxuICAgICAqIEBwYXJhbSB5IC0gSW5pdGlhbCB5IHZhbHVlLlxuICAgICAqIEBwYXJhbSB6IC0gSW5pdGlhbCB6IHZhbHVlLlxuICAgICAqIEBwYXJhbSB6IC0gSW5pdGlhbCB3IHZhbHVlLlxuICAgICAqIEByZXR1cm5zIHRoZSBjcmVhdGVkIHZlY3RvclxuICAgICAqL1xuICAgIGNvbnN0IGZyb21WYWx1ZXMgPSBjcmVhdGU7XG4gICAgLyoqXG4gICAgICogU2V0cyB0aGUgdmFsdWVzIG9mIGEgVmVjNFxuICAgICAqIEFsc28gc2VlIHtAbGluayB2ZWM0LmNyZWF0ZX0gYW5kIHtAbGluayB2ZWM0LmNvcHl9XG4gICAgICpcbiAgICAgKiBAcGFyYW0geCBmaXJzdCB2YWx1ZVxuICAgICAqIEBwYXJhbSB5IHNlY29uZCB2YWx1ZVxuICAgICAqIEBwYXJhbSB6IHRoaXJkIHZhbHVlXG4gICAgICogQHBhcmFtIHcgZm91cnRoIHZhbHVlXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB3aXRoIGl0cyBlbGVtZW50cyBzZXQuXG4gICAgICovXG4gICAgZnVuY3Rpb24gc2V0KHgsIHksIHosIHcsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0geDtcbiAgICAgICAgbmV3RHN0WzFdID0geTtcbiAgICAgICAgbmV3RHN0WzJdID0gejtcbiAgICAgICAgbmV3RHN0WzNdID0gdztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQXBwbGllcyBNYXRoLmNlaWwgdG8gZWFjaCBlbGVtZW50IG9mIHZlY3RvclxuICAgICAqIEBwYXJhbSB2IC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB0aGF0IGlzIHRoZSBjZWlsIG9mIGVhY2ggZWxlbWVudCBvZiB2LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGNlaWwodiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSBNYXRoLmNlaWwodlswXSk7XG4gICAgICAgIG5ld0RzdFsxXSA9IE1hdGguY2VpbCh2WzFdKTtcbiAgICAgICAgbmV3RHN0WzJdID0gTWF0aC5jZWlsKHZbMl0pO1xuICAgICAgICBuZXdEc3RbM10gPSBNYXRoLmNlaWwodlszXSk7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEFwcGxpZXMgTWF0aC5mbG9vciB0byBlYWNoIGVsZW1lbnQgb2YgdmVjdG9yXG4gICAgICogQHBhcmFtIHYgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHRoYXQgaXMgdGhlIGZsb29yIG9mIGVhY2ggZWxlbWVudCBvZiB2LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGZsb29yKHYsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gTWF0aC5mbG9vcih2WzBdKTtcbiAgICAgICAgbmV3RHN0WzFdID0gTWF0aC5mbG9vcih2WzFdKTtcbiAgICAgICAgbmV3RHN0WzJdID0gTWF0aC5mbG9vcih2WzJdKTtcbiAgICAgICAgbmV3RHN0WzNdID0gTWF0aC5mbG9vcih2WzNdKTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQXBwbGllcyBNYXRoLnJvdW5kIHRvIGVhY2ggZWxlbWVudCBvZiB2ZWN0b3JcbiAgICAgKiBAcGFyYW0gdiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSB2ZWN0b3IgdGhhdCBpcyB0aGUgcm91bmQgb2YgZWFjaCBlbGVtZW50IG9mIHYuXG4gICAgICovXG4gICAgZnVuY3Rpb24gcm91bmQodiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSBNYXRoLnJvdW5kKHZbMF0pO1xuICAgICAgICBuZXdEc3RbMV0gPSBNYXRoLnJvdW5kKHZbMV0pO1xuICAgICAgICBuZXdEc3RbMl0gPSBNYXRoLnJvdW5kKHZbMl0pO1xuICAgICAgICBuZXdEc3RbM10gPSBNYXRoLnJvdW5kKHZbM10pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDbGFtcCBlYWNoIGVsZW1lbnQgb2YgdmVjdG9yIGJldHdlZW4gbWluIGFuZCBtYXhcbiAgICAgKiBAcGFyYW0gdiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBtYXggLSBNaW4gdmFsdWUsIGRlZmF1bHQgMFxuICAgICAqIEBwYXJhbSBtaW4gLSBNYXggdmFsdWUsIGRlZmF1bHQgMVxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSB2ZWN0b3IgdGhhdCB0aGUgY2xhbXBlZCB2YWx1ZSBvZiBlYWNoIGVsZW1lbnQgb2Ygdi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBjbGFtcCh2LCBtaW4gPSAwLCBtYXggPSAxLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGgubWluKG1heCwgTWF0aC5tYXgobWluLCB2WzBdKSk7XG4gICAgICAgIG5ld0RzdFsxXSA9IE1hdGgubWluKG1heCwgTWF0aC5tYXgobWluLCB2WzFdKSk7XG4gICAgICAgIG5ld0RzdFsyXSA9IE1hdGgubWluKG1heCwgTWF0aC5tYXgobWluLCB2WzJdKSk7XG4gICAgICAgIG5ld0RzdFszXSA9IE1hdGgubWluKG1heCwgTWF0aC5tYXgobWluLCB2WzNdKSk7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIEFkZHMgdHdvIHZlY3RvcnM7IGFzc3VtZXMgYSBhbmQgYiBoYXZlIHRoZSBzYW1lIGRpbWVuc2lvbi5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB0aGF0IGlzIHRoZSBzdW0gb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBhZGQoYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSBhWzBdICsgYlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSArIGJbMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IGFbMl0gKyBiWzJdO1xuICAgICAgICBuZXdEc3RbM10gPSBhWzNdICsgYlszXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQWRkcyB0d28gdmVjdG9ycywgc2NhbGluZyB0aGUgMm5kOyBhc3N1bWVzIGEgYW5kIGIgaGF2ZSB0aGUgc2FtZSBkaW1lbnNpb24uXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBzY2FsZSAtIEFtb3VudCB0byBzY2FsZSBiXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBBIHZlY3RvciB0aGF0IGlzIHRoZSBzdW0gb2YgYSArIGIgKiBzY2FsZS5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBhZGRTY2FsZWQoYSwgYiwgc2NhbGUsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSArIGJbMF0gKiBzY2FsZTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSArIGJbMV0gKiBzY2FsZTtcbiAgICAgICAgbmV3RHN0WzJdID0gYVsyXSArIGJbMl0gKiBzY2FsZTtcbiAgICAgICAgbmV3RHN0WzNdID0gYVszXSArIGJbM10gKiBzY2FsZTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogU3VidHJhY3RzIHR3byB2ZWN0b3JzLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgdmVjdG9yIHRoYXQgaXMgdGhlIGRpZmZlcmVuY2Ugb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBzdWJ0cmFjdChhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IGFbMF0gLSBiWzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSBhWzFdIC0gYlsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gYVsyXSAtIGJbMl07XG4gICAgICAgIG5ld0RzdFszXSA9IGFbM10gLSBiWzNdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBTdWJ0cmFjdHMgdHdvIHZlY3RvcnMuXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSB2ZWN0b3IgdGhhdCBpcyB0aGUgZGlmZmVyZW5jZSBvZiBhIGFuZCBiLlxuICAgICAqL1xuICAgIGNvbnN0IHN1YiA9IHN1YnRyYWN0O1xuICAgIC8qKlxuICAgICAqIENoZWNrIGlmIDIgdmVjdG9ycyBhcmUgYXBwcm94aW1hdGVseSBlcXVhbFxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyB0cnVlIGlmIHZlY3RvcnMgYXJlIGFwcHJveGltYXRlbHkgZXF1YWxcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBlcXVhbHNBcHByb3hpbWF0ZWx5KGEsIGIpIHtcbiAgICAgICAgcmV0dXJuIE1hdGguYWJzKGFbMF0gLSBiWzBdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbMV0gLSBiWzFdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbMl0gLSBiWzJdKSA8IEVQU0lMT04gJiZcbiAgICAgICAgICAgIE1hdGguYWJzKGFbM10gLSBiWzNdKSA8IEVQU0lMT047XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENoZWNrIGlmIDIgdmVjdG9ycyBhcmUgZXhhY3RseSBlcXVhbFxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyB0cnVlIGlmIHZlY3RvcnMgYXJlIGV4YWN0bHkgZXF1YWxcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBlcXVhbHMoYSwgYikge1xuICAgICAgICByZXR1cm4gYVswXSA9PT0gYlswXSAmJiBhWzFdID09PSBiWzFdICYmIGFbMl0gPT09IGJbMl0gJiYgYVszXSA9PT0gYlszXTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUGVyZm9ybXMgbGluZWFyIGludGVycG9sYXRpb24gb24gdHdvIHZlY3RvcnMuXG4gICAgICogR2l2ZW4gdmVjdG9ycyBhIGFuZCBiIGFuZCBpbnRlcnBvbGF0aW9uIGNvZWZmaWNpZW50IHQsIHJldHVybnNcbiAgICAgKiBhICsgdCAqIChiIC0gYSkuXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSB0IC0gSW50ZXJwb2xhdGlvbiBjb2VmZmljaWVudC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBsaW5lYXIgaW50ZXJwb2xhdGVkIHJlc3VsdC5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBsZXJwKGEsIGIsIHQsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSArIHQgKiAoYlswXSAtIGFbMF0pO1xuICAgICAgICBuZXdEc3RbMV0gPSBhWzFdICsgdCAqIChiWzFdIC0gYVsxXSk7XG4gICAgICAgIG5ld0RzdFsyXSA9IGFbMl0gKyB0ICogKGJbMl0gLSBhWzJdKTtcbiAgICAgICAgbmV3RHN0WzNdID0gYVszXSArIHQgKiAoYlszXSAtIGFbM10pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBQZXJmb3JtcyBsaW5lYXIgaW50ZXJwb2xhdGlvbiBvbiB0d28gdmVjdG9ycy5cbiAgICAgKiBHaXZlbiB2ZWN0b3JzIGEgYW5kIGIgYW5kIGludGVycG9sYXRpb24gY29lZmZpY2llbnQgdmVjdG9yIHQsIHJldHVybnNcbiAgICAgKiBhICsgdCAqIChiIC0gYSkuXG4gICAgICogQHBhcmFtIGEgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSB0IC0gSW50ZXJwb2xhdGlvbiBjb2VmZmljaWVudHMgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgdGhlIGxpbmVhciBpbnRlcnBvbGF0ZWQgcmVzdWx0LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGxlcnBWKGEsIGIsIHQsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSArIHRbMF0gKiAoYlswXSAtIGFbMF0pO1xuICAgICAgICBuZXdEc3RbMV0gPSBhWzFdICsgdFsxXSAqIChiWzFdIC0gYVsxXSk7XG4gICAgICAgIG5ld0RzdFsyXSA9IGFbMl0gKyB0WzJdICogKGJbMl0gLSBhWzJdKTtcbiAgICAgICAgbmV3RHN0WzNdID0gYVszXSArIHRbM10gKiAoYlszXSAtIGFbM10pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSZXR1cm4gbWF4IHZhbHVlcyBvZiB0d28gdmVjdG9ycy5cbiAgICAgKiBHaXZlbiB2ZWN0b3JzIGEgYW5kIGIgcmV0dXJuc1xuICAgICAqIFttYXgoYVswXSwgYlswXSksIG1heChhWzFdLCBiWzFdKSwgbWF4KGFbMl0sIGJbMl0pXS5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgbWF4IGNvbXBvbmVudHMgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG1heChhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGgubWF4KGFbMF0sIGJbMF0pO1xuICAgICAgICBuZXdEc3RbMV0gPSBNYXRoLm1heChhWzFdLCBiWzFdKTtcbiAgICAgICAgbmV3RHN0WzJdID0gTWF0aC5tYXgoYVsyXSwgYlsyXSk7XG4gICAgICAgIG5ld0RzdFszXSA9IE1hdGgubWF4KGFbM10sIGJbM10pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBSZXR1cm4gbWluIHZhbHVlcyBvZiB0d28gdmVjdG9ycy5cbiAgICAgKiBHaXZlbiB2ZWN0b3JzIGEgYW5kIGIgcmV0dXJuc1xuICAgICAqIFttaW4oYVswXSwgYlswXSksIG1pbihhWzFdLCBiWzFdKSwgbWluKGFbMl0sIGJbMl0pXS5cbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgbWluIGNvbXBvbmVudHMgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG1pbihhLCBiLCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IE1hdGgubWluKGFbMF0sIGJbMF0pO1xuICAgICAgICBuZXdEc3RbMV0gPSBNYXRoLm1pbihhWzFdLCBiWzFdKTtcbiAgICAgICAgbmV3RHN0WzJdID0gTWF0aC5taW4oYVsyXSwgYlsyXSk7XG4gICAgICAgIG5ld0RzdFszXSA9IE1hdGgubWluKGFbM10sIGJbM10pO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBNdWx0aXBsaWVzIGEgdmVjdG9yIGJ5IGEgc2NhbGFyLlxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gayAtIFRoZSBzY2FsYXIuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBtdWxTY2FsYXIodiwgaywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2WzBdICogaztcbiAgICAgICAgbmV3RHN0WzFdID0gdlsxXSAqIGs7XG4gICAgICAgIG5ld0RzdFsyXSA9IHZbMl0gKiBrO1xuICAgICAgICBuZXdEc3RbM10gPSB2WzNdICogaztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogTXVsdGlwbGllcyBhIHZlY3RvciBieSBhIHNjYWxhci4gKHNhbWUgYXMgbXVsU2NhbGFyKVxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gayAtIFRoZSBzY2FsYXIuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBjb25zdCBzY2FsZSA9IG11bFNjYWxhcjtcbiAgICAvKipcbiAgICAgKiBEaXZpZGVzIGEgdmVjdG9yIGJ5IGEgc2NhbGFyLlxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gayAtIFRoZSBzY2FsYXIuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgc2NhbGVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBkaXZTY2FsYXIodiwgaywgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSB2WzBdIC8gaztcbiAgICAgICAgbmV3RHN0WzFdID0gdlsxXSAvIGs7XG4gICAgICAgIG5ld0RzdFsyXSA9IHZbMl0gLyBrO1xuICAgICAgICBuZXdEc3RbM10gPSB2WzNdIC8gaztcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogSW52ZXJzZSBhIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gdiAtIFRoZSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgaW52ZXJ0ZWQgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGludmVyc2UodiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSAxIC8gdlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gMSAvIHZbMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IDEgLyB2WzJdO1xuICAgICAgICBuZXdEc3RbM10gPSAxIC8gdlszXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogSW52ZXJ0IGEgdmVjdG9yLiAoc2FtZSBhcyBpbnZlcnNlKVxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBpbnZlcnRlZCB2ZWN0b3IuXG4gICAgICovXG4gICAgY29uc3QgaW52ZXJ0ID0gaW52ZXJzZTtcbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgdHdvIHZlY3RvcnNcbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHJldHVybnMgZG90IHByb2R1Y3RcbiAgICAgKi9cbiAgICBmdW5jdGlvbiBkb3QoYSwgYikge1xuICAgICAgICByZXR1cm4gKGFbMF0gKiBiWzBdKSArIChhWzFdICogYlsxXSkgKyAoYVsyXSAqIGJbMl0pICsgKGFbM10gKiBiWzNdKTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIGxlbmd0aCBvZiB2ZWN0b3JcbiAgICAgKiBAcGFyYW0gdiAtIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBsZW5ndGggb2YgdmVjdG9yLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIGxlbmd0aCh2KSB7XG4gICAgICAgIGNvbnN0IHYwID0gdlswXTtcbiAgICAgICAgY29uc3QgdjEgPSB2WzFdO1xuICAgICAgICBjb25zdCB2MiA9IHZbMl07XG4gICAgICAgIGNvbnN0IHYzID0gdlszXTtcbiAgICAgICAgcmV0dXJuIE1hdGguc3FydCh2MCAqIHYwICsgdjEgKiB2MSArIHYyICogdjIgKyB2MyAqIHYzKTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIGxlbmd0aCBvZiB2ZWN0b3IgKHNhbWUgYXMgbGVuZ3RoKVxuICAgICAqIEBwYXJhbSB2IC0gdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIGxlbmd0aCBvZiB2ZWN0b3IuXG4gICAgICovXG4gICAgY29uc3QgbGVuID0gbGVuZ3RoO1xuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBzcXVhcmUgb2YgdGhlIGxlbmd0aCBvZiB2ZWN0b3JcbiAgICAgKiBAcGFyYW0gdiAtIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBzcXVhcmUgb2YgdGhlIGxlbmd0aCBvZiB2ZWN0b3IuXG4gICAgICovXG4gICAgZnVuY3Rpb24gbGVuZ3RoU3Eodikge1xuICAgICAgICBjb25zdCB2MCA9IHZbMF07XG4gICAgICAgIGNvbnN0IHYxID0gdlsxXTtcbiAgICAgICAgY29uc3QgdjIgPSB2WzJdO1xuICAgICAgICBjb25zdCB2MyA9IHZbM107XG4gICAgICAgIHJldHVybiB2MCAqIHYwICsgdjEgKiB2MSArIHYyICogdjIgKyB2MyAqIHYzO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgc3F1YXJlIG9mIHRoZSBsZW5ndGggb2YgdmVjdG9yIChzYW1lIGFzIGxlbmd0aFNxKVxuICAgICAqIEBwYXJhbSB2IC0gdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIHNxdWFyZSBvZiB0aGUgbGVuZ3RoIG9mIHZlY3Rvci5cbiAgICAgKi9cbiAgICBjb25zdCBsZW5TcSA9IGxlbmd0aFNxO1xuICAgIC8qKlxuICAgICAqIENvbXB1dGVzIHRoZSBkaXN0YW5jZSBiZXR3ZWVuIDIgcG9pbnRzXG4gICAgICogQHBhcmFtIGEgLSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSB2ZWN0b3IuXG4gICAgICogQHJldHVybnMgZGlzdGFuY2UgYmV0d2VlbiBhIGFuZCBiXG4gICAgICovXG4gICAgZnVuY3Rpb24gZGlzdGFuY2UoYSwgYikge1xuICAgICAgICBjb25zdCBkeCA9IGFbMF0gLSBiWzBdO1xuICAgICAgICBjb25zdCBkeSA9IGFbMV0gLSBiWzFdO1xuICAgICAgICBjb25zdCBkeiA9IGFbMl0gLSBiWzJdO1xuICAgICAgICBjb25zdCBkdyA9IGFbM10gLSBiWzNdO1xuICAgICAgICByZXR1cm4gTWF0aC5zcXJ0KGR4ICogZHggKyBkeSAqIGR5ICsgZHogKiBkeiArIGR3ICogZHcpO1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgZGlzdGFuY2UgYmV0d2VlbiAyIHBvaW50cyAoc2FtZSBhcyBkaXN0YW5jZSlcbiAgICAgKiBAcGFyYW0gYSAtIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gYiAtIHZlY3Rvci5cbiAgICAgKiBAcmV0dXJucyBkaXN0YW5jZSBiZXR3ZWVuIGEgYW5kIGJcbiAgICAgKi9cbiAgICBjb25zdCBkaXN0ID0gZGlzdGFuY2U7XG4gICAgLyoqXG4gICAgICogQ29tcHV0ZXMgdGhlIHNxdWFyZSBvZiB0aGUgZGlzdGFuY2UgYmV0d2VlbiAyIHBvaW50c1xuICAgICAqIEBwYXJhbSBhIC0gdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gdmVjdG9yLlxuICAgICAqIEByZXR1cm5zIHNxdWFyZSBvZiB0aGUgZGlzdGFuY2UgYmV0d2VlbiBhIGFuZCBiXG4gICAgICovXG4gICAgZnVuY3Rpb24gZGlzdGFuY2VTcShhLCBiKSB7XG4gICAgICAgIGNvbnN0IGR4ID0gYVswXSAtIGJbMF07XG4gICAgICAgIGNvbnN0IGR5ID0gYVsxXSAtIGJbMV07XG4gICAgICAgIGNvbnN0IGR6ID0gYVsyXSAtIGJbMl07XG4gICAgICAgIGNvbnN0IGR3ID0gYVszXSAtIGJbM107XG4gICAgICAgIHJldHVybiBkeCAqIGR4ICsgZHkgKiBkeSArIGR6ICogZHogKyBkdyAqIGR3O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb21wdXRlcyB0aGUgc3F1YXJlIG9mIHRoZSBkaXN0YW5jZSBiZXR3ZWVuIDIgcG9pbnRzIChzYW1lIGFzIGRpc3RhbmNlU3EpXG4gICAgICogQHBhcmFtIGEgLSB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSB2ZWN0b3IuXG4gICAgICogQHJldHVybnMgc3F1YXJlIG9mIHRoZSBkaXN0YW5jZSBiZXR3ZWVuIGEgYW5kIGJcbiAgICAgKi9cbiAgICBjb25zdCBkaXN0U3EgPSBkaXN0YW5jZVNxO1xuICAgIC8qKlxuICAgICAqIERpdmlkZXMgYSB2ZWN0b3IgYnkgaXRzIEV1Y2xpZGVhbiBsZW5ndGggYW5kIHJldHVybnMgdGhlIHF1b3RpZW50LlxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSBub3JtYWxpemVkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBub3JtYWxpemUodiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBjb25zdCB2MCA9IHZbMF07XG4gICAgICAgIGNvbnN0IHYxID0gdlsxXTtcbiAgICAgICAgY29uc3QgdjIgPSB2WzJdO1xuICAgICAgICBjb25zdCB2MyA9IHZbM107XG4gICAgICAgIGNvbnN0IGxlbiA9IE1hdGguc3FydCh2MCAqIHYwICsgdjEgKiB2MSArIHYyICogdjIgKyB2MyAqIHYzKTtcbiAgICAgICAgaWYgKGxlbiA+IDAuMDAwMDEpIHtcbiAgICAgICAgICAgIG5ld0RzdFswXSA9IHYwIC8gbGVuO1xuICAgICAgICAgICAgbmV3RHN0WzFdID0gdjEgLyBsZW47XG4gICAgICAgICAgICBuZXdEc3RbMl0gPSB2MiAvIGxlbjtcbiAgICAgICAgICAgIG5ld0RzdFszXSA9IHYzIC8gbGVuO1xuICAgICAgICB9XG4gICAgICAgIGVsc2Uge1xuICAgICAgICAgICAgbmV3RHN0WzBdID0gMDtcbiAgICAgICAgICAgIG5ld0RzdFsxXSA9IDA7XG4gICAgICAgICAgICBuZXdEc3RbMl0gPSAwO1xuICAgICAgICAgICAgbmV3RHN0WzNdID0gMDtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBOZWdhdGVzIGEgdmVjdG9yLlxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIC12LlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG5lZ2F0ZSh2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IC12WzBdO1xuICAgICAgICBuZXdEc3RbMV0gPSAtdlsxXTtcbiAgICAgICAgbmV3RHN0WzJdID0gLXZbMl07XG4gICAgICAgIG5ld0RzdFszXSA9IC12WzNdO1xuICAgICAgICByZXR1cm4gbmV3RHN0O1xuICAgIH1cbiAgICAvKipcbiAgICAgKiBDb3BpZXMgYSB2ZWN0b3IuIChzYW1lIGFzIHtAbGluayB2ZWM0LmNsb25lfSlcbiAgICAgKiBBbHNvIHNlZSB7QGxpbmsgdmVjNC5jcmVhdGV9IGFuZCB7QGxpbmsgdmVjNC5zZXR9XG4gICAgICogQHBhcmFtIHYgLSBUaGUgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBkc3QgLSB2ZWN0b3IgdG8gaG9sZCByZXN1bHQuIElmIG5vdCBwYXNzZWQgaW4gYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgQSBjb3B5IG9mIHYuXG4gICAgICovXG4gICAgZnVuY3Rpb24gY29weSh2LCBkc3QpIHtcbiAgICAgICAgY29uc3QgbmV3RHN0ID0gKGRzdCA/PyBuZXcgQ3Rvcig0KSk7XG4gICAgICAgIG5ld0RzdFswXSA9IHZbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IHZbMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IHZbMl07XG4gICAgICAgIG5ld0RzdFszXSA9IHZbM107XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIENsb25lcyBhIHZlY3Rvci4gKHNhbWUgYXMge0BsaW5rIHZlYzQuY29weX0pXG4gICAgICogQWxzbyBzZWUge0BsaW5rIHZlYzQuY3JlYXRlfSBhbmQge0BsaW5rIHZlYzQuc2V0fVxuICAgICAqIEBwYXJhbSB2IC0gVGhlIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIEEgY29weSBvZiB2LlxuICAgICAqL1xuICAgIGNvbnN0IGNsb25lID0gY29weTtcbiAgICAvKipcbiAgICAgKiBNdWx0aXBsaWVzIGEgdmVjdG9yIGJ5IGFub3RoZXIgdmVjdG9yIChjb21wb25lbnQtd2lzZSk7IGFzc3VtZXMgYSBhbmRcbiAgICAgKiBiIGhhdmUgdGhlIHNhbWUgbGVuZ3RoLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSB2ZWN0b3Igb2YgcHJvZHVjdHMgb2YgZW50cmllcyBvZiBhIGFuZCBiLlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG11bHRpcGx5KGEsIGIsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gYVswXSAqIGJbMF07XG4gICAgICAgIG5ld0RzdFsxXSA9IGFbMV0gKiBiWzFdO1xuICAgICAgICBuZXdEc3RbMl0gPSBhWzJdICogYlsyXTtcbiAgICAgICAgbmV3RHN0WzNdID0gYVszXSAqIGJbM107XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIE11bHRpcGxpZXMgYSB2ZWN0b3IgYnkgYW5vdGhlciB2ZWN0b3IgKGNvbXBvbmVudC13aXNlKTsgYXNzdW1lcyBhIGFuZFxuICAgICAqIGIgaGF2ZSB0aGUgc2FtZSBsZW5ndGguIChzYW1lIGFzIG11bClcbiAgICAgKiBAcGFyYW0gYSAtIE9wZXJhbmQgdmVjdG9yLlxuICAgICAqIEBwYXJhbSBiIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgdmVjdG9yIG9mIHByb2R1Y3RzIG9mIGVudHJpZXMgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBjb25zdCBtdWwgPSBtdWx0aXBseTtcbiAgICAvKipcbiAgICAgKiBEaXZpZGVzIGEgdmVjdG9yIGJ5IGFub3RoZXIgdmVjdG9yIChjb21wb25lbnQtd2lzZSk7IGFzc3VtZXMgYSBhbmRcbiAgICAgKiBiIGhhdmUgdGhlIHNhbWUgbGVuZ3RoLlxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSB2ZWN0b3Igb2YgcXVvdGllbnRzIG9mIGVudHJpZXMgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiBkaXZpZGUoYSwgYiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBuZXdEc3RbMF0gPSBhWzBdIC8gYlswXTtcbiAgICAgICAgbmV3RHN0WzFdID0gYVsxXSAvIGJbMV07XG4gICAgICAgIG5ld0RzdFsyXSA9IGFbMl0gLyBiWzJdO1xuICAgICAgICBuZXdEc3RbM10gPSBhWzNdIC8gYlszXTtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogRGl2aWRlcyBhIHZlY3RvciBieSBhbm90aGVyIHZlY3RvciAoY29tcG9uZW50LXdpc2UpOyBhc3N1bWVzIGEgYW5kXG4gICAgICogYiBoYXZlIHRoZSBzYW1lIGxlbmd0aC4gKHNhbWUgYXMgZGl2aWRlKVxuICAgICAqIEBwYXJhbSBhIC0gT3BlcmFuZCB2ZWN0b3IuXG4gICAgICogQHBhcmFtIGIgLSBPcGVyYW5kIHZlY3Rvci5cbiAgICAgKiBAcGFyYW0gZHN0IC0gdmVjdG9yIHRvIGhvbGQgcmVzdWx0LiBJZiBub3QgcGFzc2VkIGluIGEgbmV3IG9uZSBpcyBjcmVhdGVkLlxuICAgICAqIEByZXR1cm5zIFRoZSB2ZWN0b3Igb2YgcXVvdGllbnRzIG9mIGVudHJpZXMgb2YgYSBhbmQgYi5cbiAgICAgKi9cbiAgICBjb25zdCBkaXYgPSBkaXZpZGU7XG4gICAgLyoqXG4gICAgICogWmVybydzIGEgdmVjdG9yXG4gICAgICogQHBhcmFtIGRzdCAtIHZlY3RvciB0byBob2xkIHJlc3VsdC4gSWYgbm90IHBhc3NlZCBpbiBhIG5ldyBvbmUgaXMgY3JlYXRlZC5cbiAgICAgKiBAcmV0dXJucyBUaGUgemVyb2VkIHZlY3Rvci5cbiAgICAgKi9cbiAgICBmdW5jdGlvbiB6ZXJvKGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgbmV3RHN0WzBdID0gMDtcbiAgICAgICAgbmV3RHN0WzFdID0gMDtcbiAgICAgICAgbmV3RHN0WzJdID0gMDtcbiAgICAgICAgbmV3RHN0WzNdID0gMDtcbiAgICAgICAgcmV0dXJuIG5ld0RzdDtcbiAgICB9XG4gICAgLyoqXG4gICAgICogdHJhbnNmb3JtIHZlYzQgYnkgNHg0IG1hdHJpeFxuICAgICAqIEBwYXJhbSB2IC0gdGhlIHZlY3RvclxuICAgICAqIEBwYXJhbSBtIC0gVGhlIG1hdHJpeC5cbiAgICAgKiBAcGFyYW0gZHN0IC0gb3B0aW9uYWwgdmVjNCB0byBzdG9yZSByZXN1bHQuIElmIG5vdCBwYXNzZWQgYSBuZXcgb25lIGlzIGNyZWF0ZWQuXG4gICAgICogQHJldHVybnMgdGhlIHRyYW5zZm9ybWVkIHZlY3RvclxuICAgICAqL1xuICAgIGZ1bmN0aW9uIHRyYW5zZm9ybU1hdDQodiwgbSwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBjb25zdCB4ID0gdlswXTtcbiAgICAgICAgY29uc3QgeSA9IHZbMV07XG4gICAgICAgIGNvbnN0IHogPSB2WzJdO1xuICAgICAgICBjb25zdCB3ID0gdlszXTtcbiAgICAgICAgbmV3RHN0WzBdID0gbVswXSAqIHggKyBtWzRdICogeSArIG1bOF0gKiB6ICsgbVsxMl0gKiB3O1xuICAgICAgICBuZXdEc3RbMV0gPSBtWzFdICogeCArIG1bNV0gKiB5ICsgbVs5XSAqIHogKyBtWzEzXSAqIHc7XG4gICAgICAgIG5ld0RzdFsyXSA9IG1bMl0gKiB4ICsgbVs2XSAqIHkgKyBtWzEwXSAqIHogKyBtWzE0XSAqIHc7XG4gICAgICAgIG5ld0RzdFszXSA9IG1bM10gKiB4ICsgbVs3XSAqIHkgKyBtWzExXSAqIHogKyBtWzE1XSAqIHc7XG4gICAgICAgIHJldHVybiBuZXdEc3Q7XG4gICAgfVxuICAgIC8qKlxuICAgICAqIFRyZWF0IGEgNEQgdmVjdG9yIGFzIGEgZGlyZWN0aW9uIGFuZCBzZXQgaXQncyBsZW5ndGhcbiAgICAgKlxuICAgICAqIEBwYXJhbSBhIFRoZSB2ZWM0IHRvIGxlbmd0aGVuXG4gICAgICogQHBhcmFtIGxlbiBUaGUgbGVuZ3RoIG9mIHRoZSByZXN1bHRpbmcgdmVjdG9yXG4gICAgICogQHJldHVybnMgVGhlIGxlbmd0aGVuZWQgdmVjdG9yXG4gICAgICovXG4gICAgZnVuY3Rpb24gc2V0TGVuZ3RoKGEsIGxlbiwgZHN0KSB7XG4gICAgICAgIGNvbnN0IG5ld0RzdCA9IChkc3QgPz8gbmV3IEN0b3IoNCkpO1xuICAgICAgICBub3JtYWxpemUoYSwgbmV3RHN0KTtcbiAgICAgICAgcmV0dXJuIG11bFNjYWxhcihuZXdEc3QsIGxlbiwgbmV3RHN0KTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogRW5zdXJlIGEgdmVjdG9yIGlzIG5vdCBsb25nZXIgdGhhbiBhIG1heCBsZW5ndGhcbiAgICAgKlxuICAgICAqIEBwYXJhbSBhIFRoZSB2ZWM0IHRvIGxpbWl0XG4gICAgICogQHBhcmFtIG1heExlbiBUaGUgbG9uZ2VzdCBsZW5ndGggb2YgdGhlIHJlc3VsdGluZyB2ZWN0b3JcbiAgICAgKiBAcmV0dXJucyBUaGUgdmVjdG9yLCBzaG9ydGVuZWQgdG8gbWF4TGVuIGlmIGl0J3MgdG9vIGxvbmdcbiAgICAgKi9cbiAgICBmdW5jdGlvbiB0cnVuY2F0ZShhLCBtYXhMZW4sIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgaWYgKGxlbmd0aChhKSA+IG1heExlbikge1xuICAgICAgICAgICAgcmV0dXJuIHNldExlbmd0aChhLCBtYXhMZW4sIG5ld0RzdCk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIGNvcHkoYSwgbmV3RHN0KTtcbiAgICB9XG4gICAgLyoqXG4gICAgICogUmV0dXJuIHRoZSB2ZWN0b3IgZXhhY3RseSBiZXR3ZWVuIDIgZW5kcG9pbnQgdmVjdG9yc1xuICAgICAqXG4gICAgICogQHBhcmFtIGEgRW5kcG9pbnQgMVxuICAgICAqIEBwYXJhbSBiIEVuZHBvaW50IDJcbiAgICAgKiBAcmV0dXJucyBUaGUgdmVjdG9yIGV4YWN0bHkgcmVzaWRpbmcgYmV0d2VlbiBlbmRwb2ludHMgMSBhbmQgMlxuICAgICAqL1xuICAgIGZ1bmN0aW9uIG1pZHBvaW50KGEsIGIsIGRzdCkge1xuICAgICAgICBjb25zdCBuZXdEc3QgPSAoZHN0ID8/IG5ldyBDdG9yKDQpKTtcbiAgICAgICAgcmV0dXJuIGxlcnAoYSwgYiwgMC41LCBuZXdEc3QpO1xuICAgIH1cbiAgICByZXR1cm4ge1xuICAgICAgICBjcmVhdGUsXG4gICAgICAgIGZyb21WYWx1ZXMsXG4gICAgICAgIHNldCxcbiAgICAgICAgY2VpbCxcbiAgICAgICAgZmxvb3IsXG4gICAgICAgIHJvdW5kLFxuICAgICAgICBjbGFtcCxcbiAgICAgICAgYWRkLFxuICAgICAgICBhZGRTY2FsZWQsXG4gICAgICAgIHN1YnRyYWN0LFxuICAgICAgICBzdWIsXG4gICAgICAgIGVxdWFsc0FwcHJveGltYXRlbHksXG4gICAgICAgIGVxdWFscyxcbiAgICAgICAgbGVycCxcbiAgICAgICAgbGVycFYsXG4gICAgICAgIG1heCxcbiAgICAgICAgbWluLFxuICAgICAgICBtdWxTY2FsYXIsXG4gICAgICAgIHNjYWxlLFxuICAgICAgICBkaXZTY2FsYXIsXG4gICAgICAgIGludmVyc2UsXG4gICAgICAgIGludmVydCxcbiAgICAgICAgZG90LFxuICAgICAgICBsZW5ndGgsXG4gICAgICAgIGxlbixcbiAgICAgICAgbGVuZ3RoU3EsXG4gICAgICAgIGxlblNxLFxuICAgICAgICBkaXN0YW5jZSxcbiAgICAgICAgZGlzdCxcbiAgICAgICAgZGlzdGFuY2VTcSxcbiAgICAgICAgZGlzdFNxLFxuICAgICAgICBub3JtYWxpemUsXG4gICAgICAgIG5lZ2F0ZSxcbiAgICAgICAgY29weSxcbiAgICAgICAgY2xvbmUsXG4gICAgICAgIG11bHRpcGx5LFxuICAgICAgICBtdWwsXG4gICAgICAgIGRpdmlkZSxcbiAgICAgICAgZGl2LFxuICAgICAgICB6ZXJvLFxuICAgICAgICB0cmFuc2Zvcm1NYXQ0LFxuICAgICAgICBzZXRMZW5ndGgsXG4gICAgICAgIHRydW5jYXRlLFxuICAgICAgICBtaWRwb2ludCxcbiAgICB9O1xufVxuY29uc3QgY2FjaGUgPSBuZXcgTWFwKCk7XG4vKipcbiAqXG4gKiBWZWM0IG1hdGggZnVuY3Rpb25zLlxuICpcbiAqIEFsbW9zdCBhbGwgZnVuY3Rpb25zIHRha2UgYW4gb3B0aW9uYWwgYG5ld0RzdGAgYXJndW1lbnQuIElmIGl0IGlzIG5vdCBwYXNzZWQgaW4gdGhlXG4gKiBmdW5jdGlvbnMgd2lsbCBjcmVhdGUgYSBuZXcgYFZlYzRgLiBJbiBvdGhlciB3b3JkcyB5b3UgY2FuIGRvIHRoaXNcbiAqXG4gKiAgICAgY29uc3QgdiA9IHZlYzQuY3Jvc3ModjEsIHYyKTsgIC8vIENyZWF0ZXMgYSBuZXcgVmVjNCB3aXRoIHRoZSBjcm9zcyBwcm9kdWN0IG9mIHYxIHggdjIuXG4gKlxuICogb3JcbiAqXG4gKiAgICAgY29uc3QgdiA9IHZlYzQuY3JlYXRlKCk7XG4gKiAgICAgdmVjNC5jcm9zcyh2MSwgdjIsIHYpOyAgLy8gUHV0cyB0aGUgY3Jvc3MgcHJvZHVjdCBvZiB2MSB4IHYyIGluIHZcbiAqXG4gKiBUaGUgZmlyc3Qgc3R5bGUgaXMgb2Z0ZW4gZWFzaWVyIGJ1dCBkZXBlbmRpbmcgb24gd2hlcmUgaXQncyB1c2VkIGl0IGdlbmVyYXRlcyBnYXJiYWdlIHdoZXJlXG4gKiBhcyB0aGVyZSBpcyBhbG1vc3QgbmV2ZXIgYWxsb2NhdGlvbiB3aXRoIHRoZSBzZWNvbmQgc3R5bGUuXG4gKlxuICogSXQgaXMgYWx3YXlzIHNhZmUgdG8gcGFzcyBhbnkgdmVjdG9yIGFzIHRoZSBkZXN0aW5hdGlvbi4gU28gZm9yIGV4YW1wbGVcbiAqXG4gKiAgICAgdmVjNC5jcm9zcyh2MSwgdjIsIHYxKTsgIC8vIFB1dHMgdGhlIGNyb3NzIHByb2R1Y3Qgb2YgdjEgeCB2MiBpbiB2MVxuICpcbiAqL1xuZnVuY3Rpb24gZ2V0QVBJKEN0b3IpIHtcbiAgICBsZXQgYXBpID0gY2FjaGUuZ2V0KEN0b3IpO1xuICAgIGlmICghYXBpKSB7XG4gICAgICAgIGFwaSA9IGdldEFQSUltcGwoQ3Rvcik7XG4gICAgICAgIGNhY2hlLnNldChDdG9yLCBhcGkpO1xuICAgIH1cbiAgICByZXR1cm4gYXBpO1xufVxuXG4vKipcbiAqIFNvbWUgZG9jc1xuICogQG5hbWVzcGFjZSB3Z3B1LW1hdHJpeFxuICovXG4vKipcbiAqIEdlbmVyYXRlIHdncHUtbWF0cml4IEFQSSBmb3IgdHlwZVxuICovXG5mdW5jdGlvbiB3Z3B1TWF0cml4QVBJKE1hdDNDdG9yLCBNYXQ0Q3RvciwgUXVhdEN0b3IsIFZlYzJDdG9yLCBWZWMzQ3RvciwgVmVjNEN0b3IpIHtcbiAgICByZXR1cm4ge1xuICAgICAgICAvKiogQG5hbWVzcGFjZSBtYXQzICovXG4gICAgICAgIG1hdDM6IGdldEFQSSQzKE1hdDNDdG9yKSxcbiAgICAgICAgLyoqIEBuYW1lc3BhY2UgbWF0NCAqL1xuICAgICAgICBtYXQ0OiBnZXRBUEkkMihNYXQ0Q3RvciksXG4gICAgICAgIC8qKiBAbmFtZXNwYWNlIHF1YXQgKi9cbiAgICAgICAgcXVhdDogZ2V0QVBJJDEoUXVhdEN0b3IpLFxuICAgICAgICAvKiogQG5hbWVzcGFjZSB2ZWMyICovXG4gICAgICAgIHZlYzI6IGdldEFQSSQ1KFZlYzJDdG9yKSxcbiAgICAgICAgLyoqIEBuYW1lc3BhY2UgdmVjMyAqL1xuICAgICAgICB2ZWMzOiBnZXRBUEkkNChWZWMzQ3RvciksXG4gICAgICAgIC8qKiBAbmFtZXNwYWNlIHZlYzQgKi9cbiAgICAgICAgdmVjNDogZ2V0QVBJKFZlYzRDdG9yKSxcbiAgICB9O1xufVxuY29uc3QgeyBcbi8qKlxuICogM3gzIE1hdHJpeCBmdW5jdGlvbnMgdGhhdCBkZWZhdWx0IHRvIHJldHVybmluZyBgRmxvYXQzMkFycmF5YFxuICogQG5hbWVzcGFjZVxuICovXG5tYXQzLCBcbi8qKlxuICogNHg0IE1hdHJpeCBmdW5jdGlvbnMgdGhhdCBkZWZhdWx0IHRvIHJldHVybmluZyBgRmxvYXQzMkFycmF5YFxuICogQG5hbWVzcGFjZVxuICovXG5tYXQ0LCBcbi8qKlxuICogUXVhdGVybmlvbiBmdW5jdGlvbnMgdGhhdCBkZWZhdWx0IHRvIHJldHVybmluZyBgRmxvYXQzMkFycmF5YFxuICogQG5hbWVzcGFjZVxuICovXG5xdWF0LCBcbi8qKlxuICogVmVjMiBmdW5jdGlvbnMgdGhhdCBkZWZhdWx0IHRvIHJldHVybmluZyBgRmxvYXQzMkFycmF5YFxuICogQG5hbWVzcGFjZVxuICovXG52ZWMyLCBcbi8qKlxuICogVmVjMyBmdW5jdGlvbnMgdGhhdCBkZWZhdWx0IHRvIHJldHVybmluZyBgRmxvYXQzMkFycmF5YFxuICogQG5hbWVzcGFjZVxuICovXG52ZWMzLCBcbi8qKlxuICogVmVjMyBmdW5jdGlvbnMgdGhhdCBkZWZhdWx0IHRvIHJldHVybmluZyBgRmxvYXQzMkFycmF5YFxuICogQG5hbWVzcGFjZVxuICovXG52ZWM0LCB9ID0gd2dwdU1hdHJpeEFQSShGbG9hdDMyQXJyYXksIEZsb2F0MzJBcnJheSwgRmxvYXQzMkFycmF5LCBGbG9hdDMyQXJyYXksIEZsb2F0MzJBcnJheSwgRmxvYXQzMkFycmF5KTtcbmNvbnN0IHsgXG4vKipcbiAqIDN4MyBNYXRyaXggZnVuY3Rpb25zIHRoYXQgZGVmYXVsdCB0byByZXR1cm5pbmcgYEZsb2F0NjRBcnJheWBcbiAqIEBuYW1lc3BhY2VcbiAqL1xubWF0MzogbWF0M2QsIFxuLyoqXG4gKiA0eDQgTWF0cml4IGZ1bmN0aW9ucyB0aGF0IGRlZmF1bHQgdG8gcmV0dXJuaW5nIGBGbG9hdDY0QXJyYXlgXG4gKiBAbmFtZXNwYWNlXG4gKi9cbm1hdDQ6IG1hdDRkLCBcbi8qKlxuICogUXVhdGVybmlvbiBmdW5jdGlvbnMgdGhhdCBkZWZhdWx0IHRvIHJldHVybmluZyBgRmxvYXQ2NEFycmF5YFxuICogQG5hbWVzcGFjZVxuICovXG5xdWF0OiBxdWF0ZCwgXG4vKipcbiAqIFZlYzIgZnVuY3Rpb25zIHRoYXQgZGVmYXVsdCB0byByZXR1cm5pbmcgYEZsb2F0NjRBcnJheWBcbiAqIEBuYW1lc3BhY2VcbiAqL1xudmVjMjogdmVjMmQsIFxuLyoqXG4gKiBWZWMzIGZ1bmN0aW9ucyB0aGF0IGRlZmF1bHQgdG8gcmV0dXJuaW5nIGBGbG9hdDY0QXJyYXlgXG4gKiBAbmFtZXNwYWNlXG4gKi9cbnZlYzM6IHZlYzNkLCBcbi8qKlxuICogVmVjMyBmdW5jdGlvbnMgdGhhdCBkZWZhdWx0IHRvIHJldHVybmluZyBgRmxvYXQ2NEFycmF5YFxuICogQG5hbWVzcGFjZVxuICovXG52ZWM0OiB2ZWM0ZCwgfSA9IHdncHVNYXRyaXhBUEkoRmxvYXQ2NEFycmF5LCBGbG9hdDY0QXJyYXksIEZsb2F0NjRBcnJheSwgRmxvYXQ2NEFycmF5LCBGbG9hdDY0QXJyYXksIEZsb2F0NjRBcnJheSk7XG5jb25zdCB7IFxuLyoqXG4gKiAzeDMgTWF0cml4IGZ1bmN0aW9ucyB0aGF0IGRlZmF1bHQgdG8gcmV0dXJuaW5nIGBudW1iZXJbXWBcbiAqIEBuYW1lc3BhY2VcbiAqL1xubWF0MzogbWF0M24sIFxuLyoqXG4gKiA0eDQgTWF0cml4IGZ1bmN0aW9ucyB0aGF0IGRlZmF1bHQgdG8gcmV0dXJuaW5nIGBudW1iZXJbXWBcbiAqIEBuYW1lc3BhY2VcbiAqL1xubWF0NDogbWF0NG4sIFxuLyoqXG4gKiBRdWF0ZXJuaW9uIGZ1bmN0aW9ucyB0aGF0IGRlZmF1bHQgdG8gcmV0dXJuaW5nIGBudW1iZXJbXWBcbiAqIEBuYW1lc3BhY2VcbiAqL1xucXVhdDogcXVhdG4sIFxuLyoqXG4gKiBWZWMyIGZ1bmN0aW9ucyB0aGF0IGRlZmF1bHQgdG8gcmV0dXJuaW5nIGBudW1iZXJbXWBcbiAqIEBuYW1lc3BhY2VcbiAqL1xudmVjMjogdmVjMm4sIFxuLyoqXG4gKiBWZWMzIGZ1bmN0aW9ucyB0aGF0IGRlZmF1bHQgdG8gcmV0dXJuaW5nIGBudW1iZXJbXWBcbiAqIEBuYW1lc3BhY2VcbiAqL1xudmVjMzogdmVjM24sIFxuLyoqXG4gKiBWZWMzIGZ1bmN0aW9ucyB0aGF0IGRlZmF1bHQgdG8gcmV0dXJuaW5nIGBudW1iZXJbXWBcbiAqIEBuYW1lc3BhY2VcbiAqL1xudmVjNDogdmVjNG4sIH0gPSB3Z3B1TWF0cml4QVBJKFplcm9BcnJheSwgQXJyYXksIEFycmF5LCBBcnJheSwgQXJyYXksIEFycmF5KTtcblxuZXhwb3J0IHsgbWF0MywgbWF0M2QsIG1hdDNuLCBtYXQ0LCBtYXQ0ZCwgbWF0NG4sIHF1YXQsIHF1YXRkLCBxdWF0biwgdXRpbHMsIHZlYzIsIHZlYzJkLCB2ZWMybiwgdmVjMywgdmVjM2QsIHZlYzNuLCB2ZWM0LCB2ZWM0ZCwgdmVjNG4gfTtcbi8vIyBzb3VyY2VNYXBwaW5nVVJMPXdncHUtbWF0cml4Lm1vZHVsZS5qcy5tYXBcbiIsIi8vIFRoZSBtb2R1bGUgY2FjaGVcbnZhciBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX18gPSB7fTtcblxuLy8gVGhlIHJlcXVpcmUgZnVuY3Rpb25cbmZ1bmN0aW9uIF9fd2VicGFja19yZXF1aXJlX18obW9kdWxlSWQpIHtcblx0Ly8gQ2hlY2sgaWYgbW9kdWxlIGlzIGluIGNhY2hlXG5cdHZhciBjYWNoZWRNb2R1bGUgPSBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX19bbW9kdWxlSWRdO1xuXHRpZiAoY2FjaGVkTW9kdWxlICE9PSB1bmRlZmluZWQpIHtcblx0XHRyZXR1cm4gY2FjaGVkTW9kdWxlLmV4cG9ydHM7XG5cdH1cblx0Ly8gQ3JlYXRlIGEgbmV3IG1vZHVsZSAoYW5kIHB1dCBpdCBpbnRvIHRoZSBjYWNoZSlcblx0dmFyIG1vZHVsZSA9IF9fd2VicGFja19tb2R1bGVfY2FjaGVfX1ttb2R1bGVJZF0gPSB7XG5cdFx0Ly8gbm8gbW9kdWxlLmlkIG5lZWRlZFxuXHRcdC8vIG5vIG1vZHVsZS5sb2FkZWQgbmVlZGVkXG5cdFx0ZXhwb3J0czoge31cblx0fTtcblxuXHQvLyBFeGVjdXRlIHRoZSBtb2R1bGUgZnVuY3Rpb25cblx0X193ZWJwYWNrX21vZHVsZXNfX1ttb2R1bGVJZF0obW9kdWxlLCBtb2R1bGUuZXhwb3J0cywgX193ZWJwYWNrX3JlcXVpcmVfXyk7XG5cblx0Ly8gUmV0dXJuIHRoZSBleHBvcnRzIG9mIHRoZSBtb2R1bGVcblx0cmV0dXJuIG1vZHVsZS5leHBvcnRzO1xufVxuXG4iLCIvLyBkZWZpbmUgZ2V0dGVyIGZ1bmN0aW9ucyBmb3IgaGFybW9ueSBleHBvcnRzXG5fX3dlYnBhY2tfcmVxdWlyZV9fLmQgPSAoZXhwb3J0cywgZGVmaW5pdGlvbikgPT4ge1xuXHRmb3IodmFyIGtleSBpbiBkZWZpbml0aW9uKSB7XG5cdFx0aWYoX193ZWJwYWNrX3JlcXVpcmVfXy5vKGRlZmluaXRpb24sIGtleSkgJiYgIV9fd2VicGFja19yZXF1aXJlX18ubyhleHBvcnRzLCBrZXkpKSB7XG5cdFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywga2V5LCB7IGVudW1lcmFibGU6IHRydWUsIGdldDogZGVmaW5pdGlvbltrZXldIH0pO1xuXHRcdH1cblx0fVxufTsiLCJfX3dlYnBhY2tfcmVxdWlyZV9fLm8gPSAob2JqLCBwcm9wKSA9PiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKG9iaiwgcHJvcCkpIiwiLy8gZGVmaW5lIF9fZXNNb2R1bGUgb24gZXhwb3J0c1xuX193ZWJwYWNrX3JlcXVpcmVfXy5yID0gKGV4cG9ydHMpID0+IHtcblx0aWYodHlwZW9mIFN5bWJvbCAhPT0gJ3VuZGVmaW5lZCcgJiYgU3ltYm9sLnRvU3RyaW5nVGFnKSB7XG5cdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFN5bWJvbC50b1N0cmluZ1RhZywgeyB2YWx1ZTogJ01vZHVsZScgfSk7XG5cdH1cblx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsICdfX2VzTW9kdWxlJywgeyB2YWx1ZTogdHJ1ZSB9KTtcbn07IiwiaW1wb3J0IHsgYmxhY2ssIGJsdWUsIENhcnRDb250cm9sLCBEcmFnLCBHbG9iYWxHcmF2aXR5LCBQaHlzT2JqZWN0LCBQaXZvdCwgcmVkLCBTaGFwZXMsIHNpbXVsYXRlLCB3aGl0ZSB9IGZyb20gXCIuL3BoeXNpY3NcIjtcclxuaW1wb3J0IHsgcmVxdWVzdFJlbmRlcmVyIH0gZnJvbSBcIi4vcmVuZGVyZXJcIjtcclxuaW1wb3J0IHsgbWF0NCwgdmVjMiwgVmVjMiwgdmVjNCwgVmVjNCwgdmVjNGQgfSBmcm9tIFwid2dwdS1tYXRyaXhcIjtcclxuXHJcbmNvbnN0IG9iamVjdHM6IFBoeXNPYmplY3RbXSA9IFtdO1xyXG5cclxuZnVuY3Rpb24gY3JlYXRlQ2lyY2xlKHg6IG51bWJlciwgeTogbnVtYmVyLCByYWRpdXM6IG51bWJlciwgY29sb3I6IFZlYzQpOiBQaHlzT2JqZWN0IHtcclxuICAgIGNvbnN0IG9iamVjdDogUGh5c09iamVjdCA9IHtcclxuICAgICAgICBtYXNzOiAxLFxyXG4gICAgICAgIHBvc2l0aW9uOiB2ZWMyLmZyb21WYWx1ZXMoeCwgeSksXHJcbiAgICAgICAgdmVsb2NpdHk6IHZlYzIuemVybygpLFxyXG4gICAgICAgIGZvcmNlczogW10sXHJcbiAgICAgICAgY29uc3RyYWludHM6IFtdLFxyXG4gICAgICAgIHNoYXBlOiB7XHJcbiAgICAgICAgICAgIHZlcnRpY2VzOiBbXHJcbiAgICAgICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHZlYzQuZnJvbVZhbHVlcygtcmFkaXVzLzIsIC1yYWRpdXMvMiwgMCwgMSksXHJcbiAgICAgICAgICAgICAgICAgICAgY29sb3I6IGNvbG9yLFxyXG4gICAgICAgICAgICAgICAgICAgIHNoYXBlOiBTaGFwZXMuY2lyY2xlLFxyXG4gICAgICAgICAgICAgICAgICAgIHV2OiB2ZWMyLmZyb21WYWx1ZXMoMCwgMSlcclxuICAgICAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHZlYzQuZnJvbVZhbHVlcyhyYWRpdXMvMiwgLXJhZGl1cy8yLCAwLCAxKSxcclxuICAgICAgICAgICAgICAgICAgICBjb2xvcjogY29sb3IsXHJcbiAgICAgICAgICAgICAgICAgICAgc2hhcGU6IFNoYXBlcy5jaXJjbGUsXHJcbiAgICAgICAgICAgICAgICAgICAgdXY6IHZlYzIuZnJvbVZhbHVlcygxLCAxKVxyXG4gICAgICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICAgICAgICBwb3NpdGlvbjogdmVjNC5mcm9tVmFsdWVzKC1yYWRpdXMvMiwgcmFkaXVzLzIsIDAsIDEpLFxyXG4gICAgICAgICAgICAgICAgICAgIGNvbG9yOiBjb2xvcixcclxuICAgICAgICAgICAgICAgICAgICBzaGFwZTogU2hhcGVzLmNpcmNsZSxcclxuICAgICAgICAgICAgICAgICAgICB1djogdmVjMi5mcm9tVmFsdWVzKDAsIDApXHJcbiAgICAgICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgICAgICAgIHBvc2l0aW9uOiB2ZWM0LmZyb21WYWx1ZXMocmFkaXVzLzIsIHJhZGl1cy8yLCAwLCAxKSxcclxuICAgICAgICAgICAgICAgICAgICBjb2xvcjogY29sb3IsXHJcbiAgICAgICAgICAgICAgICAgICAgc2hhcGU6IFNoYXBlcy5jaXJjbGUsXHJcbiAgICAgICAgICAgICAgICAgICAgdXY6IHZlYzIuZnJvbVZhbHVlcygxLCAwKVxyXG4gICAgICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgXSxcclxuICAgICAgICAgICAgaW5kaWNlczogW1xyXG4gICAgICAgICAgICAgICAgMCwgMiwgMSwgMSwgMiwgM1xyXG4gICAgICAgICAgICBdXHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIG9iamVjdHMucHVzaChvYmplY3QpO1xyXG4gICAgcmV0dXJuIG9iamVjdDtcclxufVxyXG5cclxuZnVuY3Rpb24gY3JlYXRlUmVjdCh4OiBudW1iZXIsIHk6IG51bWJlciwgd2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIsIGNvbG9yOiBWZWM0KTogUGh5c09iamVjdCB7XHJcbiAgICBjb25zdCBvYmplY3Q6IFBoeXNPYmplY3QgPSB7XHJcbiAgICAgICAgbWFzczogMixcclxuICAgICAgICBwb3NpdGlvbjogdmVjMi5mcm9tVmFsdWVzKHgsIHkpLFxyXG4gICAgICAgIHZlbG9jaXR5OiB2ZWMyLnplcm8oKSxcclxuICAgICAgICBmb3JjZXM6IFtdLFxyXG4gICAgICAgIGNvbnN0cmFpbnRzOiBbXSxcclxuICAgICAgICBzaGFwZToge1xyXG4gICAgICAgICAgICB2ZXJ0aWNlczogW1xyXG4gICAgICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgICAgICAgIHBvc2l0aW9uOiB2ZWM0LmZyb21WYWx1ZXMoLXdpZHRoLzIsIC1oZWlnaHQvMiwgMCwgMSksXHJcbiAgICAgICAgICAgICAgICAgICAgY29sb3I6IGNvbG9yLFxyXG4gICAgICAgICAgICAgICAgICAgIHNoYXBlOiBTaGFwZXMucmVjdCxcclxuICAgICAgICAgICAgICAgICAgICB1djogdmVjMi5mcm9tVmFsdWVzKDAsIDEpXHJcbiAgICAgICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgICAgICAgIHBvc2l0aW9uOiB2ZWM0LmZyb21WYWx1ZXMod2lkdGgvMiwgLWhlaWdodC8yLCAwLCAxKSxcclxuICAgICAgICAgICAgICAgICAgICBjb2xvcjogY29sb3IsXHJcbiAgICAgICAgICAgICAgICAgICAgc2hhcGU6IFNoYXBlcy5yZWN0LFxyXG4gICAgICAgICAgICAgICAgICAgIHV2OiB2ZWMyLmZyb21WYWx1ZXMoMSwgMSlcclxuICAgICAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHZlYzQuZnJvbVZhbHVlcygtd2lkdGgvMiwgaGVpZ2h0LzIsIDAsIDEpLFxyXG4gICAgICAgICAgICAgICAgICAgIGNvbG9yOiBjb2xvcixcclxuICAgICAgICAgICAgICAgICAgICBzaGFwZTogU2hhcGVzLnJlY3QsXHJcbiAgICAgICAgICAgICAgICAgICAgdXY6IHZlYzIuZnJvbVZhbHVlcygwLCAwKVxyXG4gICAgICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICAgICAgICBwb3NpdGlvbjogdmVjNC5mcm9tVmFsdWVzKHdpZHRoLzIsIGhlaWdodC8yLCAwLCAxKSxcclxuICAgICAgICAgICAgICAgICAgICBjb2xvcjogY29sb3IsXHJcbiAgICAgICAgICAgICAgICAgICAgc2hhcGU6IFNoYXBlcy5yZWN0LFxyXG4gICAgICAgICAgICAgICAgICAgIHV2OiB2ZWMyLmZyb21WYWx1ZXMoMSwgMClcclxuICAgICAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIF0sXHJcbiAgICAgICAgICAgIGluZGljZXM6IFtcclxuICAgICAgICAgICAgICAgIDAsIDEsIDIsIDEsIDIsIDNcclxuICAgICAgICAgICAgXVxyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICBvYmplY3RzLnB1c2gob2JqZWN0KTtcclxuICAgIHJldHVybiBvYmplY3Q7XHJcbn1cclxuXHJcbmNvbnN0IHZpZXdNYXRyaXggPSBtYXQ0LmlkZW50aXR5KCk7XHJcbmxldCBoZWlnaHQgPSAwO1xyXG5sZXQgd2lkdGggPSAwO1xyXG5cclxuZnVuY3Rpb24gYWJzb2x1dGVUb05vcm1hbGl6ZWQoeDogbnVtYmVyLCB5OiBudW1iZXIpIHtcclxuICAgIGNvbnN0IHhOb3JtID0gKHggLyAod2lkdGgpIC0gLjUpICogMjtcclxuICAgIGNvbnN0IHlOb3JtID0gKC15IC8gKGhlaWdodCkgKyAuNSkgKiAyO1xyXG4gICAgY29uc3QgdkNhbnZhcyA9IHZlYzQuZnJvbVZhbHVlcyh4Tm9ybSwgeU5vcm0sIDAsIDEpO1xyXG4gICAgY29uc3QgdldvcmxkID0gdmVjNC50cmFuc2Zvcm1NYXQ0KHZDYW52YXMsbWF0NC5pbnZlcnNlKHZpZXdNYXRyaXgpKTtcclxuICAgIHJldHVybiBbdldvcmxkWzBdLHZXb3JsZFsxXV07XHJcbn1cclxuXHJcbmZ1bmN0aW9uIGRlbHRhVG9Ob3JtYWxpemVkKHg6IG51bWJlciwgeTogbnVtYmVyKSB7XHJcbiAgICBjb25zdCByZXQgPSBbeCAvIHdpZHRoICogKDIgLyB2aWV3TWF0cml4WzBdKSwgLXkgLyBoZWlnaHQgKiAoMiAvIHZpZXdNYXRyaXhbNV0pIF07XHJcbiAgICByZXR1cm4gcmV0O1xyXG59XHJcblxyXG5hc3luYyBmdW5jdGlvbiBzdGFydCgpIHtcclxuICAgIGNvbnN0IGNhbnZhcyA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoXCJjYW52YXNcIik7XHJcbiAgICB3aWR0aCA9IGNhbnZhcy53aWR0aDtcclxuICAgIGhlaWdodCA9IGNhbnZhcy5oZWlnaHQ7XHJcbiAgICBjYW52YXMuYWRkRXZlbnRMaXN0ZW5lcihcIm1vdXNld2hlZWxcIiwgKGV2OiBXaGVlbEV2ZW50KSA9PiB7XHJcbiAgICAgICAgZXYucHJldmVudERlZmF1bHQoKTtcclxuICAgICAgICBpZiAoZXYuY3RybEtleSkge1xyXG4gICAgICAgICAgICBpZiAoIShldi5kZWx0YVkgPiAwICYmIHZpZXdNYXRyaXhbMF0gPD0gLjgpICYmICEoZXYuZGVsdGFZIDwgMCAmJiB2aWV3TWF0cml4WzBdID49IDQpKSB7XHJcbiAgICAgICAgICAgICAgICBsZXQgW3gsIHldID0gYWJzb2x1dGVUb05vcm1hbGl6ZWQoZXYub2Zmc2V0WCwgZXYub2Zmc2V0WSk7XHJcbiAgICAgICAgICAgICAgICBjb25zdCBtb3VzZVRvT3JpZ2luID0gbWF0NC50cmFuc2xhdGlvbih2ZWM0LmZyb21WYWx1ZXMoeCwgeSwgMCwgMCkpO1xyXG4gICAgICAgICAgICAgICAgY29uc3QgZGVsdGFab29tID0gZXYuZGVsdGFZIC8gNjA7XHJcbiAgICAgICAgICAgICAgICBjb25zdCB6b29tID0gbWF0NC51bmlmb3JtU2NhbGUobWF0NC5pZGVudGl0eSgpLCAxIC0gZGVsdGFab29tKTtcclxuICAgICAgICAgICAgICAgIFt4LCB5XSA9IGFic29sdXRlVG9Ob3JtYWxpemVkKGV2Lm9mZnNldFgsIGV2Lm9mZnNldFkpO1xyXG4gICAgICAgICAgICAgICAgY29uc3QgdW5kb01vdXNlVG9PcmlnaW4gPSBtYXQ0LnRyYW5zbGF0aW9uKHZlYzQuZnJvbVZhbHVlcygteCwgLXksIDAsIDApKTtcclxuICAgICAgICAgICAgICAgIG1hdDQubXVsdGlwbHkodmlld01hdHJpeCwgbW91c2VUb09yaWdpbiwgdmlld01hdHJpeCk7XHJcbiAgICAgICAgICAgICAgICBtYXQ0Lm11bHRpcGx5KHZpZXdNYXRyaXgsIHpvb20sIHZpZXdNYXRyaXgpO1xyXG4gICAgICAgICAgICAgICAgbWF0NC5tdWx0aXBseSh2aWV3TWF0cml4LCB1bmRvTW91c2VUb09yaWdpbiwgdmlld01hdHJpeCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9KTtcclxuICAgIGNvbnN0IG1vdXNlUG9zID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJtb3VzZVBvc1wiKTtcclxuICAgIGNhbnZhcy5hZGRFdmVudExpc3RlbmVyKFwibW91c2Vtb3ZlXCIsIChldjogTW91c2VFdmVudCkgPT4ge1xyXG4gICAgICAgIGNvbnN0IFt4LCB5XSA9IGFic29sdXRlVG9Ob3JtYWxpemVkKGV2Lm9mZnNldFgsIGV2Lm9mZnNldFkpO1xyXG4gICAgICAgIG1vdXNlUG9zLmlubmVyVGV4dCA9IGBTY3JlZW46ICR7ZXYub2Zmc2V0WH0sICR7ZXYub2Zmc2V0WX0sIENhbWVyYTogJHt4fSwgJHt5fWA7XHJcbiAgICAgICAgaWYgKGV2LmJ1dHRvbnMgPT09IDEpIHtcclxuICAgICAgICAgICAgbGV0IFt4LHldID0gZGVsdGFUb05vcm1hbGl6ZWQoZXYubW92ZW1lbnRYLCBldi5tb3ZlbWVudFkpO1xyXG4gICAgICAgICAgICBjb25zdCB0cmFuc2xhdGlvbiA9IG1hdDQudHJhbnNsYXRpb24odmVjNC5mcm9tVmFsdWVzKHgsIHksIDAsIDApKTtcclxuICAgICAgICAgICAgbWF0NC5tdWx0aXBseSh2aWV3TWF0cml4LCB0cmFuc2xhdGlvbiwgdmlld01hdHJpeCk7XHJcbiAgICAgICAgfVxyXG4gICAgfSk7XHJcbiAgICBcclxuICAgIGNvbnN0IGVudmlyb25tZW50Qm1wID0gYXdhaXQgZmV0Y2goXCIvZW52aXJvbm1lbnQuYm1wXCIpLnRoZW4ocmVzcCA9PiByZXNwLmJsb2IoKSkudGhlbihibG9iID0+IGNyZWF0ZUltYWdlQml0bWFwKGJsb2IpKTtcclxuICAgIGNvbnN0IGRpcnRCbXAgPSBhd2FpdCBmZXRjaChcIi9kaXJ0X3RleHR1cmUuanBnXCIpLnRoZW4ocmVzcCA9PiByZXNwLmJsb2IoKSkudGhlbihibG9iID0+IGNyZWF0ZUltYWdlQml0bWFwKGJsb2IpKTtcclxuICAgIGNvbnN0IHJlbmRlcmVyID0gYXdhaXQgcmVxdWVzdFJlbmRlcmVyKGNhbnZhcywgW2Vudmlyb25tZW50Qm1wLCBkaXJ0Qm1wXSk7XHJcblxyXG4gICAgY3JlYXRlUmVjdCgwLCAwLCAyLCAyLCByZWQpO1xyXG4gICAgXHJcbiAgICByZW5kZXJlci5hZGRWZXJ0ZXhCdWZmZXIob2JqZWN0cyk7XHJcbiAgICByZW5kZXJlci52aWV3TWF0cml4ID0gdmlld01hdHJpeDtcclxuICAgIHJlbmRlcmVyLnJlbmRlcigpO1xyXG5cclxuICAgIGxldCBwYXVzZWQgPSB0cnVlO1xyXG4gICAgbGV0IGZyYW1lSGFuZGxlID0gMDtcclxuICAgIGxldCBwcmV2VGltZSA9IHBlcmZvcm1hbmNlLm5vdygpO1xyXG4gICAgY29uc3QgZnJhbWUgPSAodGltZTogRE9NSGlnaFJlc1RpbWVTdGFtcCkgPT4ge1xyXG4gICAgICAgIGNvbnN0IGRlbHRhVGltZSA9ICh0aW1lIC0gcHJldlRpbWUpIC8gMTAwMDtcclxuICAgICAgICBwcmV2VGltZSA9IHRpbWU7XHJcblxyXG4gICAgICAgIHNpbXVsYXRlKG9iamVjdHMsIGRlbHRhVGltZSk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgcmVuZGVyZXIuYWRkVmVydGV4QnVmZmVyKG9iamVjdHMpO1xyXG4gICAgICAgIHJlbmRlcmVyLnZpZXdNYXRyaXggPSB2aWV3TWF0cml4O1xyXG4gICAgICAgIHJlbmRlcmVyLnJlbmRlcigpO1xyXG4gICAgICAgIGlmICghcGF1c2VkKSB7XHJcbiAgICAgICAgICAgIGZyYW1lSGFuZGxlID0gcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGZyYW1lKTtcclxuICAgICAgICB9XHJcbiAgICB9XHJcblxyXG4gICAgY29uc3QgcGF1c2VCdXR0b24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcInBhdXNlXCIpO1xyXG4gICAgcGF1c2VCdXR0b24uYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IHtcclxuICAgICAgICBpZiAocGF1c2VkKSB7XHJcbiAgICAgICAgICAgIHByZXZUaW1lID0gcGVyZm9ybWFuY2Uubm93KCk7XHJcbiAgICAgICAgICAgIGZyYW1lSGFuZGxlID0gcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGZyYW1lKTtcclxuICAgICAgICAgICAgcGF1c2VCdXR0b24uaW5uZXJUZXh0ID0gXCJQYXVzZVwiO1xyXG4gICAgICAgICAgICBwYXVzZWQgPSBmYWxzZTtcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICBjYW5jZWxBbmltYXRpb25GcmFtZShmcmFtZUhhbmRsZSk7XHJcbiAgICAgICAgICAgIHBhdXNlQnV0dG9uLmlubmVyVGV4dCA9IFwiUGxheVwiXHJcbiAgICAgICAgICAgIHBhdXNlZCA9IHRydWU7XHJcbiAgICAgICAgfVxyXG4gICAgfSk7XHJcbn1cclxuXHJcbnN0YXJ0KCk7Il0sIm5hbWVzIjpbXSwic291cmNlUm9vdCI6IiJ9