// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

// MARK: - Contract

/// A kind of input that an embedding model places in the same vector space as text.
public enum EmbeddingModality: Hashable, Sendable {
    case text, image, video, audio
}

/// The task an embedding is prepared for. Each model renders it as its own instruction.
/// Use `.searchQuery` for queries and `.document` for corpus items; compared items must
/// share a task's space.
public enum EmbeddingTask: Hashable, Sendable {
    case searchQuery
    case document
    case clustering
    case classification
    case questionAnswering
    case factChecking
    case codeRetrieval
    case sentenceSimilarity
    /// The input is embedded as given, without a task instruction.
    case none
}

/// One independently embedded item: text and media in reading order.
public struct EmbeddingInput {

    /// One piece of an input.
    public enum Part {
        case text(String)
        case image(UserInput.Image)
        case video(UserInput.Video)
        case audio(UserInput.Audio)
    }

    public let parts: [Part]
    /// A document title or file name; models trained with titles use it for `.document`.
    public let title: String?

    public init(_ parts: [Part], title: String? = nil) {
        self.parts = parts
        self.title = title
    }

    /// Text followed by images.
    public init(text: String = "", title: String? = nil, images: [UserInput.Image] = []) {
        self.init([.text(text)] + images.map(Part.image), title: title)
    }
}

/// A loaded embedding model that maps text and media into one space of unit-length
/// float32 vectors. Each model owns its prompts, sequence layout, and batching.
public protocol MultimodalEmbedding: Sendable {

    /// The modalities that the loaded checkpoint accepts.
    var supportedModalities: Set<EmbeddingModality> { get async }

    /// Embeds independent items; returns one vector per item, in input order.
    func embed(_ inputs: sending [EmbeddingInput], task: EmbeddingTask) async throws -> [[Float]]
}

// MARK: - Factory

/// Loads a ``MultimodalEmbedding`` from a local checkpoint by its `model_type`, so callers
/// need no model-specific code.
///
/// ```swift
/// let embedding = try await MultimodalEmbeddingFactory.load(
///     modelDirectory: directory, tokenizerLoader: loader)
/// let vectors = try await embedding.embed(
///     [.init(text: "Aurora over Balestrino", images: [.url(photo)])], task: .document)
/// ```
public enum MultimodalEmbeddingFactory {

    /// The `model_type` values in `config.json` that the factory loads.
    public static var modelTypes: Set<String> { Set(loaders.keys) }

    /// Loads the checkpoint in `modelDirectory`. Media encoders load on first use, so the
    /// files must stay readable while the embedding is in use.
    /// - Throws: ``ModelFactoryError/unsupportedModelType(_:)`` for other model types.
    public static func load(
        modelDirectory: URL, tokenizerLoader: any TokenizerLoader
    ) async throws -> any MultimodalEmbedding {
        let configuration = try Data(contentsOf: modelDirectory.appending(component: "config.json"))
        let modelType = try JSONDecoder.json5().decode(
            BaseConfiguration.self, from: configuration
        ).modelType
        guard let load = loaders[modelType] else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try await load(modelDirectory, tokenizerLoader)
    }

    private typealias Loader =
        @Sendable (URL, any TokenizerLoader) async throws -> any MultimodalEmbedding

    private static let loaders: [String: Loader] = {
        let embeddingGemma2: Loader = {
            try await EmbeddingGemma2Embedding(modelDirectory: $0, tokenizerLoader: $1)
        }
        return ["embedding_gemma2": embeddingGemma2, "embedding_gemma2_text": embeddingGemma2]
    }()
}
