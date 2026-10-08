// Copyright © 2026 Apple Inc.

import CoreMedia
import Foundation
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

@Suite(.serialized)
struct EmbeddingGemma2IntegrationTests {
    private struct Reference: Decodable {
        struct Record: Decodable {
            let name: String
            let task: String
            let parts: [[String]]
            let tokens: [Int]
            let embedding: [Float]
        }
        let cases: [Record]
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["EG2_TEST_MODEL_DIR"] != nil
                && ProcessInfo.processInfo.environment["EG2_TEST_GOLDEN_DIR"] != nil,
            "Requires the checkpoint and generated reference fixtures."))
    func publicActorAndFactoryMatchTransformersSDPA() async throws {
        let environment = ProcessInfo.processInfo.environment
        let directory = URL(filePath: try #require(environment["EG2_TEST_MODEL_DIR"]))
        let fixtures = URL(filePath: try #require(environment["EG2_TEST_GOLDEN_DIR"]))
        let reference = try JSONDecoder().decode(
            Reference.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("reference.json")))
        let loader = #huggingFaceTokenizerLoader()
        let embeddings = try await EmbeddingGemma2Embedding(
            modelDirectory: directory, tokenizerLoader: loader)
        for record in reference.cases {
            let input = try Self.input(record.parts, fixtures: fixtures)
            let vector = try await embeddings.embed(input, task: Self.task(record.task))
            #expect(vector.count == 768)
            #expect(Self.cosine(vector, record.embedding) > 0.999, Comment(rawValue: record.name))
        }
        let small = try await embeddings.embed(
            .init(text: "What causes the northern lights?"),
            task: .searchQuery, dimensions: 256)
        let query = try #require(reference.cases.first { $0.name == "query" })
        #expect(small.count == 256)
        #expect(Self.cosine(small, Array(query.embedding.prefix(256))) > 0.999)
        #expect(abs(small.reduce(0.0) { $0 + Double($1) * Double($1) } - 1) < 1e-5)

        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: directory, using: loader)
        for record in reference.cases where record.parts.allSatisfy({ $0[0] == "text" }) {
            let text = EmbeddingGemma2Embedding.prompt(
                text: record.parts.map { $0[1] }.joined(), title: nil, task: Self.task(record.task))
            let result = try await container.perform { context in
                let ids = context.tokenizer.encode(text: text)
                #expect(ids == record.tokens)
                let output = context.model(
                    MLXArray(ids.map(Int32.init), [1, ids.count]),
                    positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
                let vector = context.pooling(output, normalize: true)
                try MLX.checkedEval(vector)
                return vector.asArray(Float.self)
            }
            #expect(Self.cosine(result, record.embedding) > 0.999, Comment(rawValue: record.name))
        }
    }

    private static func task(_ value: String) -> EmbeddingGemma2Embedding.Task {
        switch value {
        case "searchQuery": .searchQuery
        case "document": .document
        default: .none
        }
    }

    private static func input(_ parts: [[String]], fixtures: URL) throws
        -> EmbeddingGemma2Embedding.Input
    {
        try .init(
            parts.map { part in
                switch part[0] {
                case "text": return .text(part[1])
                case "image": return .image(.url(fixtures.appendingPathComponent(part[1])))
                case "audio": return .audio(.url(fixtures.appendingPathComponent(part[1])))
                case "video":
                    return .video(
                        .frames(
                            (0 ..< 5).map { index in
                                UserInput.VideoFrame(
                                    image: .url(
                                        fixtures.appendingPathComponent("frame\(index).png")),
                                    timeStamp: CMTime(value: Int64(index), timescale: 1))
                            }))
                default: throw EmbeddingGemma2Embedding.Error.invalidMedia
                }
            })
    }

    private static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        let dot = zip(a, b).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
        let normA = a.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        let normB = b.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        return dot / (normA * normB)
    }
}
