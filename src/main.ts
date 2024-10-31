import { black, blue, CartControl, Drag, GlobalGravity, PhysObject, Pivot, red, Shapes, simulate, white } from "./physics";
import { requestRenderer } from "./renderer";
import { mat4, vec2, Vec2, vec4, Vec4, vec4d } from "wgpu-matrix";

const objects: PhysObject[] = [];

function createCircle(x: number, y: number, radius: number, color: Vec4): PhysObject {
    const object: PhysObject = {
        mass: 1,
        position: vec2.fromValues(x, y),
        velocity: vec2.zero(),
        forces: [],
        constraints: [],
        shape: {
            vertices: [
                {
                    position: vec4.fromValues(-radius/2, -radius/2, 0, 1),
                    color: color,
                    shape: Shapes.circle,
                    uv: vec2.fromValues(0, 1)
                },
                {
                    position: vec4.fromValues(radius/2, -radius/2, 0, 1),
                    color: color,
                    shape: Shapes.circle,
                    uv: vec2.fromValues(1, 1)
                },
                {
                    position: vec4.fromValues(-radius/2, radius/2, 0, 1),
                    color: color,
                    shape: Shapes.circle,
                    uv: vec2.fromValues(0, 0)
                },
                {
                    position: vec4.fromValues(radius/2, radius/2, 0, 1),
                    color: color,
                    shape: Shapes.circle,
                    uv: vec2.fromValues(1, 0)
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

function createRect(x: number, y: number, width: number, height: number, color: Vec4): PhysObject {
    const object: PhysObject = {
        mass: 2,
        position: vec2.fromValues(x, y),
        velocity: vec2.zero(),
        forces: [],
        constraints: [],
        shape: {
            vertices: [
                {
                    position: vec4.fromValues(-width/2, -height/2, 0, 1),
                    color: color,
                    shape: Shapes.rect,
                    uv: vec2.fromValues(0, 1)
                },
                {
                    position: vec4.fromValues(width/2, -height/2, 0, 1),
                    color: color,
                    shape: Shapes.rect,
                    uv: vec2.fromValues(1, 1)
                },
                {
                    position: vec4.fromValues(-width/2, height/2, 0, 1),
                    color: color,
                    shape: Shapes.rect,
                    uv: vec2.fromValues(0, 0)
                },
                {
                    position: vec4.fromValues(width/2, height/2, 0, 1),
                    color: color,
                    shape: Shapes.rect,
                    uv: vec2.fromValues(1, 0)
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

const viewMatrix = mat4.identity();
let height = 0;
let width = 0;

function absoluteToNormalized(x: number, y: number) {
    const xNorm = (x / (width) - .5) * 2;
    const yNorm = (-y / (height) + .5) * 2;
    const vCanvas = vec4.fromValues(xNorm, yNorm, 0, 1);
    const vWorld = vec4.transformMat4(vCanvas,mat4.inverse(viewMatrix));
    return [vWorld[0],vWorld[1]];
}

function deltaToNormalized(x: number, y: number) {
    const ret = [x / width * (2 / viewMatrix[0]), -y / height * (2 / viewMatrix[5]) ];
    return ret;
}

async function start() {
    const canvas = document.querySelector("canvas");
    width = canvas.width;
    height = canvas.height;
    canvas.addEventListener("mousewheel", (ev: WheelEvent) => {
        ev.preventDefault();
        if (ev.ctrlKey) {
            if (!(ev.deltaY > 0 && viewMatrix[0] <= .8) && !(ev.deltaY < 0 && viewMatrix[0] >= 4)) {
                let [x, y] = absoluteToNormalized(ev.offsetX, ev.offsetY);
                const mouseToOrigin = mat4.translation(vec4.fromValues(x, y, 0, 0));
                const deltaZoom = ev.deltaY / 60;
                const zoom = mat4.uniformScale(mat4.identity(), 1 - deltaZoom);
                [x, y] = absoluteToNormalized(ev.offsetX, ev.offsetY);
                const undoMouseToOrigin = mat4.translation(vec4.fromValues(-x, -y, 0, 0));
                mat4.multiply(viewMatrix, mouseToOrigin, viewMatrix);
                mat4.multiply(viewMatrix, zoom, viewMatrix);
                mat4.multiply(viewMatrix, undoMouseToOrigin, viewMatrix);
            }
        }
    });
    const mousePos = document.getElementById("mousePos");
    canvas.addEventListener("mousemove", (ev: MouseEvent) => {
        const [x, y] = absoluteToNormalized(ev.offsetX, ev.offsetY);
        mousePos.innerText = `Screen: ${ev.offsetX}, ${ev.offsetY}, Camera: ${x}, ${y}`;
        if (ev.buttons === 1) {
            let [x,y] = deltaToNormalized(ev.movementX, ev.movementY);
            const translation = mat4.translation(vec4.fromValues(x, y, 0, 0));
            mat4.multiply(viewMatrix, translation, viewMatrix);
        }
    });
    
    const environmentBmp = await fetch("/environment.bmp").then(resp => resp.blob()).then(blob => createImageBitmap(blob));
    const dirtBmp = await fetch("/dirt_texture.jpg").then(resp => resp.blob()).then(blob => createImageBitmap(blob));
    const renderer = await requestRenderer(canvas, [environmentBmp, dirtBmp]);

    createRect(0, 0, 2, 2, red);
    
    renderer.addVertexBuffer(objects);
    renderer.viewMatrix = viewMatrix;
    renderer.render();

    let paused = true;
    let frameHandle = 0;
    let prevTime = performance.now();
    const frame = (time: DOMHighResTimeStamp) => {
        const deltaTime = (time - prevTime) / 1000;
        prevTime = time;

        simulate(objects, deltaTime);
        
        renderer.addVertexBuffer(objects);
        renderer.viewMatrix = viewMatrix;
        renderer.render();
        if (!paused) {
            frameHandle = requestAnimationFrame(frame);
        }
    }

    const pauseButton = document.getElementById("pause");
    pauseButton.addEventListener("click", () => {
        if (paused) {
            prevTime = performance.now();
            frameHandle = requestAnimationFrame(frame);
            pauseButton.innerText = "Pause";
            paused = false;
        } else {
            cancelAnimationFrame(frameHandle);
            pauseButton.innerText = "Play"
            paused = true;
        }
    });
}

start();