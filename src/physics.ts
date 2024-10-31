import { vec4, Vec4, Vec2, vec2 } from "wgpu-matrix";

export const white = vec4.fromValues(1, 1, 1, 1);
export const red = vec4.fromValues(1, 0, 0, 1);
export const blue = vec4.fromValues(0, 0, 1, 1);
export const green = vec4.fromValues(0, 1, 0, 1);
export const black = vec4.fromValues(0, 0, 0, 1);

export interface Vertex {
    position: Vec2,
    color: Vec4,
    uv: Vec2,
    shape: Shapes
}

export enum Shapes {
    rect = 0,
    circle = 1,
}

export interface Shape {
    vertices: Vertex[],
    indices: number[]
}

export interface PhysObject {
    position: Vec2,
    velocity: Vec2,
    shape: Shape,
    mass: number,
    forces: Force[],
    constraints: Constraint[],
    sumAcceleration?: Vec2,
}

export interface Constraint {
    applyConstraint(acceleration: Vec2): Vec2,
}

export interface Force {
    getAcceleration: (object: PhysObject) => Vec2,
}

export function simulate(objects: PhysObject[], deltaTime: number) {
    objects.forEach(obj => {
        let acceleration = vec2.zero();
        obj.forces.forEach(force => {
            const a = force.getAcceleration(obj);
            vec2.add(acceleration, a, acceleration);
        });
        
        obj.constraints.forEach(constraint => {
            acceleration = constraint.applyConstraint(acceleration);
        });
        if (obj.constraints.length > 0) {
            // console.log("net accel", acceleration[0], acceleration[1], "vel", obj.velocity[0], obj.velocity[1]);
        }

        obj.velocity = vec2.addScaled(obj.velocity, acceleration, deltaTime);
        obj.position = vec2.addScaled(obj.position, obj.velocity, deltaTime);
    });
}

export class GlobalGravity implements Force {
    acceleration: number
    constructor(acceleration: number) {
        this.acceleration = acceleration;
    }
    getAcceleration(): Vec2 {
        return vec2.fromValues(0, this.acceleration);
    }
}

enum InputDirection {
    left = -1,
    zero = 0,
    right = 1,
}

export class CartControl implements Force {
    force: number;
    direction: number
    constructor(force: number) {
        this.force = force;
        this.direction = InputDirection.zero;
        document.addEventListener("keypress",(event) => {
            if (event.key == "a" && this.direction > -1) {
                this.direction += InputDirection.left;
            } else if (event.key == "d" && this.direction < 1) {
                this.direction += InputDirection.right;
            }
        });
        document.addEventListener("keyup",(event) => {
            if (event.key == "a") {
                this.direction -= InputDirection.left;
            } else if (event.key == "d") {
                this.direction -= InputDirection.right;
            }
            // console.log(this.direction);
        });
    }
    getAcceleration(object: PhysObject): Vec2 {
        return vec2.fromValues(this.direction * (this.force / object.mass), 0);
    }
}

export class Drag implements Force {
    coefficient: number
    constructor(coefficient: number) {
        this.coefficient = coefficient;
    }
    getAcceleration(object: PhysObject): Vec2 {
        return vec2.scale(vec2.negate(object.velocity), this.coefficient / object.mass);
    }
}

export class Pivot implements Constraint {
    object: PhysObject;
    position: Vec2;
    length: number;
    constructor(object: PhysObject, x: number, y: number) {
        this.position = vec2.fromValues(x,y);
        this.object = object;
        this.length = vec2.len(vec2.sub(this.position, this.object.position));
    }
    applyConstraint(acceleration: Vec2): Vec2 {
        // console.log(acceleration);
        const tensionDirection = vec2.normalize(vec2.sub(this.position, this.object.position));
        const tension = vec2.scale(tensionDirection, 
            -(vec2.dot(acceleration, tensionDirection) + (length * vec2.lenSq(this.object.velocity))));
        // console.log(tension);
        vec2.add(this.position, vec2.scale(tensionDirection, -this.length), this.object.position);
        return vec2.add(acceleration, tension);
    }
}