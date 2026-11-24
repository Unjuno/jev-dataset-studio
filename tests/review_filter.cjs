// Exercise the real async page loader with controlled response order.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../studio/static/app.js"), "utf8");
const binding = source.indexOf('setOptions($("target-mode")');
assert.ok(binding > 0);

for (const trigger of ["refresh", "review completion"]) {
  test(trigger + " during a filter change reloads the requested page", async () => {
    const requests = [];
    const filter = { value: "all" };
    const context = {
      document: { getElementById: () => filter },
      fetchPage: (url) => new Promise((resolve) => requests.push({ url, resolve })),
    };
    const load = vm.runInNewContext(source.slice(0, binding) + `
      activeJob = { id: "job" };
      reviewOffset = 50;
      currentAnnotation = { id: "previous" };
      api = fetchPage;
      renderAnnotations = () => {};
      loadAnnotations
    `, context);
    const switching = load({ offset: 0, status: "review" });
    const reloading = load(trigger === "refresh" ? {} : { preferred: "previous" });
    assert.equal(requests.length, 2);
    assert.equal(requests[0].url, "/jobs/job/annotations?status=review&offset=0&limit=50");
    assert.equal(requests[1].url, requests[0].url);
    assert.equal(vm.runInNewContext("reviewOffset", context), 50);
    assert.equal(vm.runInNewContext("reviewStatus", context), "all");
    requests[1].resolve({ items: [{ id: "fresh" }], total: 1 });
    assert.equal(await reloading, true);
    requests[0].resolve({ items: [{ id: "stale" }], total: 1 });
    assert.equal(await switching, false);
    assert.equal(filter.value, "review");
    assert.equal(vm.runInNewContext("reviewOffset", context), 0);
    assert.equal(vm.runInNewContext("currentAnnotation.id", context), "fresh");
  });
}
