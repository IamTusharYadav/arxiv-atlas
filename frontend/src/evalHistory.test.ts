import { describe, expect, it } from "vitest";
import { axisFloor } from "./EvalHistory";

describe("axisFloor", () => {
  it("takes the tightest floor the data clears", () => {
    expect(axisFloor([4.6, 4.83, 5, 4.933])).toBe(4.5);
    expect(axisFloor([4, 4.2])).toBe(4);
    expect(axisFloor([3.1, 4.9])).toBe(3);
  });

  it("opens up as soon as one score drops below the band", () => {
    // The point of the check: a bad run has to stay inside the plot, not escape below it.
    expect(axisFloor([4.9, 4.4, 4.7])).toBe(4);
    expect(axisFloor([4.9, 2.8, 4.7])).toBe(1);
  });

  it("does not zoom on an empty history", () => {
    expect(axisFloor([])).toBe(1);
  });
});
