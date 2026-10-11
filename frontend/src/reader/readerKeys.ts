/**
 * The reader's keyboard handler, registered from the component.
 *
 * `main.tsx` owns the one `keydown` listener for the whole app. The reader needs
 * bindings of its own -- four page views, and a spread that moves two pages at a
 * time -- but adding a second window listener would mean two handlers racing for
 * the same key, and the reader is a component that mounts and unmounts.
 *
 * So the reader registers here while it is mounted, and `main.tsx` offers it the
 * event first. `false` means "not mine", and the keys that mean the same thing
 * everywhere (Escape, F for fullscreen) stay with `main.tsx` rather than being
 * restated per layout.
 */

type ReaderKeyHandler = (event: KeyboardEvent) => boolean;

let handler: ReaderKeyHandler | null = null;

/** Called by the reader on mount, and with `null` on unmount. */
export function setReaderKeyHandler(next: ReaderKeyHandler | null): void {
    handler = next;
}

/**
 * Offer a keydown to the reader.
 *
 * Returns true when the reader consumed it, in which case the caller must not run
 * its own bindings. A throwing handler is treated as "not consumed" rather than
 * taking the app's only keyboard listener down with it.
 */
export function handleReaderKey(event: KeyboardEvent): boolean {
    if (!handler) return false;
    try {
        return handler(event);
    } catch {
        return false;
    }
}
