import { describe, expect, it } from "vitest";

import { evaluateLog } from "../src/core/evaluate";
import { LogFormatError, logToCsv, parseCsvRows, parseLogCsv, simulateLog } from "../src/core/logTable";
import { DEFAULT_WORLD } from "../src/core/world";

describe("CSV parsing", () => {
  it("handles quotes, CRLF and a byte order mark", () => {
    const rows = parseCsvRows('﻿a,b\r\n"x, y","say ""hi"""\r\n\r\n1,2');
    expect(rows).toEqual([["a", "b"], ["x, y", 'say "hi"'], ["1", "2"]]);
  });

  it("round-trips a simulated log exactly", () => {
    const log = simulateLog(DEFAULT_WORLD, 50, 0.3, 9);
    const back = parseLogCsv(logToCsv(log));
    expect(Array.from(back.propensity)).toEqual(Array.from(log.propensity));
    expect(Array.from(back.features.feats)).toEqual(Array.from(log.features.feats));
    expect(back.itemIds).toEqual(log.itemIds);
    expect(Array.from(back.rewards)).toEqual(Array.from(log.rewards));
  });

  it("explains what is wrong with a bad file", () => {
    const good = logToCsv(simulateLog(DEFAULT_WORLD, 5, 0.3, 1));
    expect(() => parseLogCsv("")).toThrow(LogFormatError);
    expect(() => parseLogCsv(good.split("\n")[0])).toThrow(/no data rows/);
    expect(() => parseLogCsv(good.replace("click_1", "clicked_1"))).toThrow(/Missing columns: click_1/);
    const lines = good.split("\n");
    const cells = lines[1].split(",");
    cells[cells.length - 1] = "0";
    expect(() => parseLogCsv([lines[0], cells.join(",")].join("\n"))).toThrow(/Row 2: propensity/);
    expect(() => parseLogCsv(good, { expectCtx: 4 })).toThrow(/4 context columns/);
  });
});

describe("speed", () => {
  it("evaluates the largest playground log in a few seconds", () => {
    const t0 = performance.now();
    const log = simulateLog(DEFAULT_WORLD, 20000, 0.2, 3);
    const ev = evaluateLog(log, ["current", "popularity", "relevance", "relevance_soft", "noisy_logging", "reversed"], {
      exploration: 0.2,
      clip: 20,
      nBoot: 200,
      seed: 3,
      nFolds: 2,
    });
    const ms = performance.now() - t0;
    expect(ev.policies.current!.estimates.ips.estimate).toBeCloseTo(ev.loggedMean, 9);
    expect(ms).toBeLessThan(15000);
    console.log(`20000-query evaluation took ${ms.toFixed(0)} ms`);
  });
});
