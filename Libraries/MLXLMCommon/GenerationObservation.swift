// Copyright © 2026 Apple Inc.

/// Observations to collect during one generation request.
public struct GenerationObservationOptions: Sendable, Equatable {
    /// Per-token log-probability reporting.
    public struct LogProbabilities: Sendable, Equatable {
        /// Number of top candidates, in addition to the chosen token.
        /// Values at or below zero report only the chosen token.
        public var topCount: Int

        public init(topCount: Int = 0) {
            self.topCount = topCount
        }
    }

    /// `nil` disables log-probability computation and readback for this request.
    public var logProbabilities: LogProbabilities?

    public init(logProbabilities: LogProbabilities? = nil) {
        self.logProbabilities = logProbabilities
    }
}

/// An ordered generation event or observation from a chat session.
///
/// Each event carries one payload. Read the typed properties to handle the
/// payloads you need. New observation properties can be added without changing
/// existing generation events.
public struct GenerationEvent: Sendable {
    /// Decoded output, a tool event, or completion metadata; otherwise `nil`.
    public let generation: Generation?

    /// One token's log probabilities; otherwise `nil`.
    /// These precede the token's decoded events. Text chunks can span tokens.
    public let logProbabilities: GenerateTokenLogProbabilities?

    init(generation: Generation) {
        self.generation = generation
        self.logProbabilities = nil
    }

    init(logProbabilities: GenerateTokenLogProbabilities) {
        self.generation = nil
        self.logProbabilities = logProbabilities
    }
}
