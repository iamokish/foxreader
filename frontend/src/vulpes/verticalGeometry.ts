/** Visible Vulpes dimensions used to constrain the two vertical splitters. */
export interface VerticalGeometry {
    /** Editor body height and the divider between Entries and Translators. */
    editor?: { height: number; splitter: number };
    /** Space shared by image/capture and the capture height needed for one line. */
    capture?: { available: number; minimum: number };
}
