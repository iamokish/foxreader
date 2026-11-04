import type { RefObject } from "preact";
import { useLayoutEffect } from "preact/hooks";
import type { VerticalGeometry } from "./verticalGeometry";

function pixels(value: string): number {
    const parsed = parseFloat(value);
    return Number.isFinite(parsed) ? parsed : 0;
}

function insets(style: CSSStyleDeclaration): number {
    return (
        pixels(style.paddingTop) +
        pixels(style.paddingBottom) +
        pixels(style.borderTopWidth) +
        pixels(style.borderBottomWidth)
    );
}

function margins(style: CSSStyleDeclaration): number {
    return pixels(style.marginTop) + pixels(style.marginBottom);
}

function minimumBorderBox(style: CSSStyleDeclaration): number {
    return pixels(style.minHeight) + (style.boxSizing === "border-box" ? 0 : insets(style));
}

function outerHeight(element: HTMLElement): number {
    return element.getBoundingClientRect().height + margins(getComputedStyle(element));
}

/** Include the wrappers that sit between the textarea and its section header. */
function wrappedMinimum(element: HTMLElement, content: number): number {
    const style = getComputedStyle(element);
    return Math.max(minimumBorderBox(style), content + insets(style)) + margins(style);
}

/** Measure stable containers; keep layout reads out of pointermove handlers. */
export function useVerticalPanelLimits(
    bodyRef: RefObject<HTMLDivElement>,
    enabled: boolean,
    setGeometry: (geometry: VerticalGeometry) => void,
): void {
    useLayoutEffect(() => {
        const body = bodyRef.current;
        if (!enabled || !body) return;
        const editorBody = body.querySelector<HTMLElement>(".vx-editor__body");
        const canvas = body.querySelector<HTMLElement>(".vx-canvas");
        const tools = body.querySelector<HTMLElement>(".vx-tools");
        const capture = body.querySelector<HTMLElement>(".vx-capture");
        const section = capture?.querySelector<HTMLElement>(".vx-section");
        const header = section?.querySelector<HTMLElement>(".vx-section__head");
        const sectionBody = section?.querySelector<HTMLElement>(".vx-section__body");
        const captureBody = sectionBody?.querySelector<HTMLElement>(".vx-capture__body");
        const wrapper = captureBody?.querySelector<HTMLElement>(".textarea-wrapper");
        const textarea = wrapper?.querySelector<HTMLTextAreaElement>("#textArea");
        let previous: VerticalGeometry = {};

        const measure = () => {
            // LayoutProvider applies this class in an effect. Earlier geometry
            // uses the shared styles and must not migrate saved pixel sizes.
            if (
                !document.body.classList.contains("layout-vulpes") ||
                document.body.classList.contains("workspace-reader")
            )
                return;
            const geometry: VerticalGeometry = {};
            const height = editorBody?.getBoundingClientRect().height ?? 0;
            if (editorBody && height > 0 && Number.isFinite(height)) {
                // Splitters can be removed/recreated when a section is toggled.
                const splitter = editorBody.querySelector<HTMLElement>(":scope > .ws-splitter--horizontal");
                const divider = splitter ? outerHeight(splitter) : 0;
                if (Number.isFinite(divider) && divider >= 0 && divider < height) {
                    geometry.editor = { height, splitter: divider };
                }
            }

            if (
                canvas &&
                tools &&
                capture &&
                section &&
                header &&
                sectionBody &&
                captureBody &&
                wrapper &&
                textarea &&
                !capture.classList.contains("is-collapsed") &&
                capture.getBoundingClientRect().height > 0 &&
                textarea.getBoundingClientRect().width > 0
            ) {
                const splitter = canvas.querySelector<HTMLElement>(":scope > .ws-splitter--horizontal");
                const available =
                    canvas.getBoundingClientRect().height - outerHeight(tools) - (splitter ? outerHeight(splitter) : 0);
                if (available > 0 && Number.isFinite(available)) {
                    const textStyle = getComputedStyle(textarea);
                    // Vulpes sets an explicit line-height. The fallback covers
                    // the first measurement before its stylesheet is applied.
                    const line = pixels(textStyle.lineHeight) || pixels(textStyle.fontSize) * 1.5;
                    let minimum =
                        Math.max(44, minimumBorderBox(textStyle), line + insets(textStyle)) + margins(textStyle);
                    minimum = wrappedMinimum(wrapper, minimum);
                    minimum = wrappedMinimum(captureBody, minimum);
                    minimum = wrappedMinimum(sectionBody, minimum);
                    minimum += outerHeight(header) + pixels(getComputedStyle(section).rowGap);
                    minimum = wrappedMinimum(section, minimum);
                    // Do not read capture.minHeight here: it is a result of this
                    // measurement, and feeding it back would prevent shrinking.
                    minimum = Math.ceil(minimum + insets(getComputedStyle(capture)));
                    if (Number.isFinite(minimum) && minimum > 0) {
                        geometry.capture = { available, minimum };
                    }
                }
            }

            if (!geometry.editor && !geometry.capture) return;
            if (
                geometry.editor?.height === previous.editor?.height &&
                geometry.editor?.splitter === previous.editor?.splitter &&
                geometry.capture?.available === previous.capture?.available &&
                geometry.capture?.minimum === previous.capture?.minimum
            )
                return;
            previous = geometry;
            setGeometry(geometry);
        };

        measure();
        let frame = 0;
        let disposed = false;
        const scheduleMeasure = () => {
            if (disposed || frame) return;
            frame = requestAnimationFrame(() => {
                frame = 0;
                measure();
            });
        };
        let textareaWidth = textarea?.getBoundingClientRect().width ?? 0;
        const observer =
            typeof ResizeObserver === "undefined"
                ? null
                : new ResizeObserver((entries) => {
                      let changed = entries.length === 0;
                      for (const entry of entries) {
                          if (entry.target !== textarea) changed = true;
                          else if (entry.contentRect.width !== textareaWidth) {
                              textareaWidth = entry.contentRect.width;
                              changed = true;
                          }
                      }
                      // An OCR drag changes only textarea height; it does not change its
                      // one-line minimum. Font/style changes are observed separately.
                      if (changed) scheduleMeasure();
                  });
        for (const element of [body, editorBody, tools, header, textarea]) {
            if (element) observer?.observe(element);
        }
        const textStyles =
            typeof window.MutationObserver === "undefined" ? null : new window.MutationObserver(scheduleMeasure);
        if (textarea) textStyles?.observe(textarea, { attributes: true, attributeFilter: ["style", "class"] });
        textStyles?.observe(document.body, { attributes: true, attributeFilter: ["class"] });
        document.fonts?.addEventListener("loadingdone", scheduleMeasure);
        if (!observer) window.addEventListener("resize", scheduleMeasure);
        return () => {
            disposed = true;
            observer?.disconnect();
            textStyles?.disconnect();
            if (frame) cancelAnimationFrame(frame);
            document.fonts?.removeEventListener("loadingdone", scheduleMeasure);
            if (!observer) window.removeEventListener("resize", scheduleMeasure);
        };
    }, [bodyRef, enabled, setGeometry]);
}
