// Copyright © 2026 Apple Inc.

/// Generation-loop adapter that owns a ``HarmonyFrameParser`` and a
/// ``HarmonyOutputRouter``.
///
/// This is the only Harmony type that knows about ``Generation``. The parser
/// stays pure; the router encodes policy; this type bridges both into the
/// token loop's emit callback.
struct HarmonyStreamAdapter {
    private var parser: HarmonyFrameParser
    private var router: HarmonyOutputRouter

    init?(tokenizer: any Tokenizer, tools: [[String: any Sendable]]?) {
        guard let parser = HarmonyFrameParser(tokenizer: tokenizer) else {
            return nil
        }
        self.parser = parser
        self.router = HarmonyOutputRouter(
            tokenizer: tokenizer,
            allowedToolNames: HarmonyOutputRouter.allowedToolNames(from: tools))
    }

    /// Feeds one token. Returns `false` when the consumer terminated.
    mutating func onToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        emitSteps(parser.push(token), emit: emit)
    }

    mutating func onGenerationEnd(
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) {
        _ = emitSteps(parser.finish(), emit: emit)
        _ = emitEvents(router.finish(), emit: emit)
    }

    private mutating func emitSteps(
        _ steps: [HarmonyParseStep],
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        for step in steps {
            if !emitEvents(router.route(step), emit: emit) {
                return false
            }
        }
        return true
    }

    private func emitEvents(
        _ events: [HarmonyOutputRouter.Event],
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        for event in events {
            let generation: Generation
            switch event {
            case .response(let text):
                generation = .chunk(text)
            case .toolCall(let call):
                generation = .toolCall(call)
            }
            if case .terminated = emit(generation) {
                return false
            }
        }
        return true
    }
}
