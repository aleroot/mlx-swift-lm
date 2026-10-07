// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing

/// Verifies that the public Gemma 4 vision encoder surface
/// (`Gemma4VisionModel.init` / `callAsFunction` and `Gemma4MultimodalEmbedder`) is
/// sufficient for a CLIENT module to run the vision tower and the `embed_vision`
/// projection on its own — e.g. a multimodal embedding model such as EmbeddingGemma 2,
/// which embeds images into the text space without any generation head.
///
/// Mirrors `Gemma4EncoderAccessTests`: deliberately imports `MLXVLM` without
/// `@testable`, so everything below compiles against declared API only, with no
/// internal access. Runs on a tiny deterministically-initialized model; no weights
/// are downloaded.
struct Gemma4VisionEncoderAccessTests {

    /// A small vision config: a 16×16 image makes a 4×4 patch grid that pools 2×2
    /// down to 4 soft tokens.
    private static let configJSON = """
        {
          "model_type": "gemma4_vision",
          "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
          "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16,
          "patch_size": 4, "pooling_kernel_size": 2, "position_embedding_size": 64,
          "default_output_length": 4, "rms_norm_eps": 1e-6,
          "use_clipped_linears": false, "standardize": false,
          "rope_parameters": { "rope_theta": 100.0, "rope_type": "axial" }
        }
        """

    private static func makeConfig() throws -> Gemma4VisionConfiguration {
        try JSONDecoder().decode(Gemma4VisionConfiguration.self, from: Data(configJSON.utf8))
    }

    @Test
    func towerEmbedsImagesWithoutAGenerationHead() throws {
        let tower = Gemma4VisionModel(config: try Self.makeConfig())
        try loadDeterministicWeights(into: tower)

        let pixels = MLXArray((0 ..< (3 * 16 * 16)).map { Float($0 % 97) / 97 }, [1, 3, 16, 16])
        let softTokens = tower(pixels)

        #expect(softTokens.shape == [1, 4, 32])
        #expect(softTokens.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test
    func embedderProjectsSoftTokensIntoTheTextSpace() throws {
        let embedder = Gemma4MultimodalEmbedder(embeddingDim: 32, textHiddenSize: 16, eps: 1e-6)
        try loadDeterministicWeights(into: embedder)

        let projected = embedder(
            MLXArray((0 ..< (4 * 32)).map { Float($0 % 13) / 13 }, [1, 4, 32]))

        #expect(projected.shape == [1, 4, 16])
        #expect(projected.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    /// Loads every parameter by its checkpoint name (`patch_embedder.*`,
    /// `encoder.layers.*`, `embedding_projection`), proving the weight path a real
    /// conversion uses also compiles and verifies.
    private func loadDeterministicWeights(into module: Module) throws {
        let weights = Dictionary(
            uniqueKeysWithValues: module.parameters().flattened().map { name, value in
                (name, MLXArray((0 ..< value.size).map { Float($0 % 17 - 8) * 0.05 }, value.shape))
            })
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(module)
    }
}
