// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A module with derived inference state that must be built after checkpoint
/// parameters have been installed and before the model is published to callers.
///
/// Preparation deliberately belongs to the loading lifecycle rather than a
/// forward pass: implementations may materialize arrays or replace storage-
/// sharing module views and therefore must run while the loader or explicit
/// model-topology owner has exclusive access to the model.
package protocol InferenceStatePreparable: AnyObject {
    func prepareForInference() throws
}

package struct InferenceStatePreparationFailure {
    package let moduleType: String
    package let error: any Error
}

package struct InferenceStatePreparationReport {
    package let failures: [InferenceStatePreparationFailure]

    package var succeeded: Bool { failures.isEmpty }
}

private let inferenceStateLogger = Logger(
    subsystem: "mlx-swift-lm", category: "inference-state")

/// Prepare every participating module after checkpoint loading or an explicit
/// model-topology update.
///
/// Take a snapshot before invoking callbacks because a callback may replace
/// child modules while preserving their checkpoint-visible topology. A failed
/// optional optimization is logged and reported while the model remains usable
/// through its unfused path.
@discardableResult
package func prepareInferenceState(in model: Module) -> InferenceStatePreparationReport {
    let modules = model.modules()
    var failures: [InferenceStatePreparationFailure] = []
    for module in modules {
        guard let preparable = module as? any InferenceStatePreparable else { continue }
        do {
            try preparable.prepareForInference()
        } catch {
            let moduleType = String(reflecting: type(of: preparable))
            failures.append(.init(moduleType: moduleType, error: error))
            inferenceStateLogger.error(
                "Failed to prepare inference state for \(moduleType): \(String(describing: error))")
        }
    }
    return InferenceStatePreparationReport(failures: failures)
}

/// Prepare derived state and realize a fully loaded model before publication.
///
/// All custom checkpoint loaders should finalize through this function so
/// inference-only optimizations are applied consistently.
@discardableResult
package func materializeModelForInference(
    _ model: Module
) -> InferenceStatePreparationReport {
    let report = prepareInferenceState(in: model)
    eval(model)
    return report
}
