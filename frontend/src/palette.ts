// Categorical palette for research directions, fixed slot order, never cycled (k is capped
// at 6 server-side). Slots 1-3 are the app's category colors; all six steps validated per
// mode against the app surfaces with the dataviz six-checks script (worst adjacent CVD
// dE 9.1 light / 8.4 dark). Light steps 4-6 sit under 3:1 contrast on the light surface,
// the documented relief case: legends are always shown, every mark carries a tooltip, and
// the direction lists repeat the same data as text.
export const SERIES = {
  light: ["#2a78d6", "#008300", "#d55181", "#eda100", "#1baf7a", "#eb6834"],
  dark: ["#3987e5", "#008300", "#d55181", "#c98500", "#199e70", "#d95926"],
};
