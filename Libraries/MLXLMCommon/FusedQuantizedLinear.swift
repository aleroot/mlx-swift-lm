// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Default-on rollback switch for the Qwen 3.5/3.6 four-projection GDN fusion.
package let qwen35FourGDNEnabled: Bool = {
    let raw = ProcessInfo.processInfo.environment["MLX_QWEN_FOUR_GDN"]?
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return raw != "0" && raw != "false" && raw != "off"
}()

/// A fused quantized projection and checkpoint-shaped views into its storage.
///
/// The views let a model keep its public/checkpoint module topology without
/// retaining a second physical copy of the quantized weights.
package struct FusedQuantizedLinearProjection {
    package let fused: QuantizedLinear
    package let sourceViews: [QuantizedLinear]
}

/// Coalesce compatible quantized linears along their output dimension.
///
/// This is intentionally stricter than merely casting to `QuantizedLinear`:
/// custom subclasses may add behavior, adapters, or state that cannot be
/// represented by one stock quantized matmul. Incompatible inputs return
/// `nil` and callers keep their original projections.
package func fuseQuantizedLinearProjections(
    _ linears: [Linear]
) -> FusedQuantizedLinearProjection? {
    guard linears.count > 1 else { return nil }

    let projections = linears.compactMap { $0 as? QuantizedLinear }
    guard projections.count == linears.count,
        zip(linears, projections).allSatisfy({ linear, projection in
            ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self)
                && linear === projection
        }),
        let first = projections.first,
        first.bias == nil,
        first.weight.ndim == 2,
        first.weight.dim(0) == first.shape.0,
        first.scales.ndim == 2,
        first.scales.dim(0) == first.shape.0,
        first.biases == nil || first.biases?.shape == first.scales.shape
    else {
        return nil
    }

    let hasQuantizationBiases = first.biases != nil
    guard
        projections.allSatisfy({ projection in
            projection.bias == nil
                && projection.bits == first.bits
                && projection.groupSize == first.groupSize
                && projection.mode == first.mode
                && projection.shape.1 == first.shape.1
                && projection.weight.ndim == 2
                && projection.weight.dim(0) == projection.shape.0
                && projection.weight.dim(1) == first.weight.dim(1)
                && projection.weight.dtype == first.weight.dtype
                && projection.scales.ndim == 2
                && projection.scales.dim(0) == projection.shape.0
                && projection.scales.dim(1) == first.scales.dim(1)
                && projection.scales.dtype == first.scales.dtype
                && (projection.biases != nil) == hasQuantizationBiases
                && (projection.biases == nil || projection.biases?.shape == projection.scales.shape)
                && projection.biases?.dtype == first.biases?.dtype
        })
    else {
        return nil
    }

    let fusedWeight = concatenated(projections.map(\.weight), axis: 0)
    let fusedScales = concatenated(projections.map(\.scales), axis: 0)
    let fusedBiases =
        hasQuantizationBiases
        ? concatenated(projections.compactMap(\.biases), axis: 0)
        : nil

    // Realize the concatenations once. The fused projection and the named
    // source views below then share this materialized backing storage.
    eval(fusedWeight, fusedScales)
    if let fusedBiases {
        eval(fusedBiases)
    }

    let fused = QuantizedLinear(
        weight: fusedWeight,
        bias: nil,
        scales: fusedScales,
        biases: fusedBiases,
        groupSize: first.groupSize,
        bits: first.bits,
        mode: first.mode)
    fused.freeze()

    var start = 0
    let sourceViews = projections.map { projection in
        let end = start + projection.shape.0
        defer { start = end }

        let rows = start ..< end
        let view = QuantizedLinear(
            weight: fusedWeight[rows],
            bias: nil,
            scales: fusedScales[rows],
            biases: fusedBiases.map { $0[rows] },
            groupSize: first.groupSize,
            bits: first.bits,
            mode: first.mode)
        view.freeze()
        return view
    }

    return FusedQuantizedLinearProjection(fused: fused, sourceViews: sourceViews)
}
