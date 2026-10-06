// Copyright © 2026 Apple Inc.

import MLX
import MLXNN

/// A part of a model that a load can leave out, such as the vision encoder of a VLM.
///
/// List components in ``ModelConfiguration/excludedComponents``. A model that conforms to
/// ``ExcludableComponentsProviding`` loads without them. Other models load every component.
public struct ModelComponent: Hashable, Sendable, CustomStringConvertible {
    public let name: String

    public init(_ name: String) {
        self.name = name
    }

    public var description: String { name }

    /// Image and video encoders, with the projectors that feed them to the language model.
    public static let vision = ModelComponent("vision")
}

/// A model that can load without some of its components.
///
/// Before the loader reads the checkpoint, it removes the modules of each excluded component.
/// It never reads the tensors in the component's namespaces. The loaded model is complete
/// without them and does not change afterwards. Inputs that need a missing component should
/// throw.
public protocol ExcludableComponentsProviding: BaseLanguageModel {
    /// The checkpoint components that make up each component this model can load without.
    ///
    /// Each ``CheckpointComponent/destination`` is the path of an optional module. Its
    /// ``CheckpointComponent/namespaces`` list every serialized prefix of the module's tensors,
    /// as they appear before ``BaseLanguageModel/prepareCheckpoint(_:)``.
    var excludableComponents: [ModelComponent: [CheckpointComponent]] { get }
}

/// Remove the modules of the `components` that `model` can load without.
///
/// Returns the serialized namespaces of the removed modules.
func removeExcludedComponents(
    _ components: Set<ModelComponent>, from model: BaseLanguageModel
) throws -> [String] {
    guard let model = model as? any ExcludableComponentsProviding else { return [] }
    let excluded = components.flatMap { model.excludableComponents[$0] ?? [] }
    for component in excluded {
        let path = component.destination.components(separatedBy: ".")
        var removal = NestedItem<String, Module>.none
        for key in path.dropFirst().reversed() {
            removal = .dictionary([key: removal])
        }
        try model.update(modules: ModuleChildren(values: [path[0]: removal]), verify: .all)
    }
    return excluded.flatMap(\.namespaces)
}
