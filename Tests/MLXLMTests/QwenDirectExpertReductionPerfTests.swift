import Foundation
import MLX
import MLXLMCommon
import Testing

/// Opt-in production-shape microbenchmark. This is not part of normal CI;
/// run with `MLX_QWEN_DIRECT_REDUCTION_PERF=1 swift test -c release --filter
/// QwenDirectExpertReductionPerfTests` when changing the reduction kernel.
@Suite(.serialized)
struct QwenDirectExpertReductionPerfTests {
    @Test func pairedDirectReductionVersusLegacy() {
        guard ProcessInfo.processInfo.environment["MLX_QWEN_DIRECT_REDUCTION_PERF"] == "1"
        else { return }

        let tokens = 2048
        let topK = 8
        let hidden = 2048
        let assignments = tokens * topK
        MLXRandom.seed(9127)

        let original = MLXRandom.uniform(
            low: -0.25, high: 0.25, [tokens, topK, hidden]
        ).asType(.bfloat16)
        let weights = softmax(
            MLXRandom.normal([tokens, topK]).asType(.bfloat16),
            axis: -1,
            precise: true)
        let expertIndexValues: [UInt32] = (0 ..< assignments).map { index in
            UInt32((index * 37 + index / topK * 11) % 256)
        }
        let order = argSort(MLXArray(expertIndexValues))
        let inverseOrder = argSort(order)
        let sorted = original.reshaped(assignments, hidden)[order]
        eval(original, weights, inverseOrder, sorted)

        func legacy() -> MLXArray { weightedExpertSum(original, weights) }
        func direct() -> MLXArray {
            weightedExpertUnsort(
                sortedOutputs: sorted,
                inverseOrder: inverseOrder,
                weights: weights)
        }

        for _ in 0 ..< 5 {
            eval(legacy())
            eval(direct())
        }

        var legacyTimes: [Double] = []
        var directTimes: [Double] = []
        for iteration in 0 ..< 25 {
            let first = iteration.isMultiple(of: 2) ? legacy : direct
            let second = iteration.isMultiple(of: 2) ? direct : legacy
            let firstStart = DispatchTime.now().uptimeNanoseconds
            eval(first())
            let firstElapsed = Double(DispatchTime.now().uptimeNanoseconds - firstStart) / 1e6
            let secondStart = DispatchTime.now().uptimeNanoseconds
            eval(second())
            let secondElapsed = Double(DispatchTime.now().uptimeNanoseconds - secondStart) / 1e6

            if iteration.isMultiple(of: 2) {
                legacyTimes.append(firstElapsed)
                directTimes.append(secondElapsed)
            } else {
                directTimes.append(firstElapsed)
                legacyTimes.append(secondElapsed)
            }
        }

        legacyTimes.sort()
        directTimes.sort()
        let median = legacyTimes.count / 2
        print(
            String(
                format: "[qwen-direct-reduction-perf] legacy=%.4fms direct=%.4fms speedup=%.3fx",
                legacyTimes[median], directTimes[median],
                legacyTimes[median] / directTimes[median]))
    }
}
