import type { EventMap } from "./types";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyEventCallback = (...args: any[]) => void;

const listeners: Record<string, AnyEventCallback[]> = {};

export function on<K extends keyof EventMap>(event: K, fn: (...args: EventMap[K]) => void): void {
    (listeners[event] ??= []).push(fn as AnyEventCallback);
}

export function off<K extends keyof EventMap>(event: K, fn: (...args: EventMap[K]) => void): void {
    const arr = listeners[event];
    if (arr) listeners[event] = arr.filter((f) => f !== fn);
}

export function emit<K extends keyof EventMap>(event: K, ...args: EventMap[K]): void {
    for (const fn of listeners[event] ?? []) fn(...args);
}
